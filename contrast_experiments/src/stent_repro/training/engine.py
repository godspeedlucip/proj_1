"""Supervised training over explicitly prepared projection/volume pairs."""
from contextlib import nullcontext
from pathlib import Path
from copy import deepcopy
import json
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from stent_repro.configuration import resolve_config
from stent_repro.eulig2021.losses import soft_dice_A
from stent_repro.networks import UNet, checkpoint_forward
from stent_repro.voth2022.losses import soft_dice_B
from stent_repro.training.sampling import RunningStatistics, EpochDataset, numpy_rng_state, restore_numpy_rng


def training_loss(method, stage, dice_smoothing):
    if stage == "dte":
        return torch.nn.L1Loss()
    if method == "A":
        return soft_dice_A
    return lambda prediction, target: soft_dice_B(prediction, target, dice_smoothing)


class TrainingRuntime:
    def __init__(self, device, precision, activation_checkpointing):
        if precision not in ("float32", "amp_float16", "amp_bfloat16"):
            raise ValueError("unsupported training precision")
        if type(activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be boolean")
        if precision == "amp_float16" and device.type != "cuda":
            raise ValueError("amp_float16 requires CUDA")
        if precision == "amp_bfloat16" and device.type not in ("cpu", "cuda"):
            raise ValueError("amp_bfloat16 requires CPU or CUDA")
        if precision == "amp_bfloat16" and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError("amp_bfloat16 on CUDA requires native bfloat16 support")
        self.device = device
        self.activation_checkpointing = activation_checkpointing
        self.dtype = {"float32": None, "amp_float16": torch.float16, "amp_bfloat16": torch.bfloat16}[precision]
        self.scaler = None
        if precision == "amp_float16":
            # torch.amp.GradScaler is available in recent PyTorch releases.
            scaler = getattr(torch.amp, "GradScaler", None)
            self.scaler = torch.cuda.amp.GradScaler() if scaler is None else scaler("cuda")

    def forward(self, model, x):
        if self.device.type in ("cpu", "cuda"):
            context = torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype is not None)
        else:
            context = nullcontext()
        recompute = self.activation_checkpointing and model.training and torch.is_grad_enabled()
        with context:
            if recompute and isinstance(model, UNet):
                prediction = model(x, activation_checkpointing=True)
            elif recompute:
                prediction = checkpoint_forward(model, model, x)
            else:
                prediction = model(x)
        # The Dice sums and L1 reduction must use float32 in mixed precision.
        return prediction if self.dtype is None else prediction.float()

    def step(self, loss, optimizer):
        if self.scaler is None:
            loss.backward()
            optimizer.step()
            return True
        initial_scale = self.scaler.get_scale()
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        return self.scaler.get_scale() >= initial_scale

    def scaler_state(self):
        return None if self.scaler is None else self.scaler.state_dict()

    def restore_scaler(self, state):
        if self.scaler is not None:
            if not state:
                raise ValueError("float16 checkpoint is missing gradient scaler state")
            self.scaler.load_state_dict(state)
        elif state:
            raise ValueError("checkpoint gradient scaler does not match training precision")


def input_statistics(dataset):
    count = 0; mean = 0.0; m2 = 0.0
    for x, _ in dataset:
        x = x.double(); n = x.numel(); batch_mean = x.mean().item()
        delta = batch_mean - mean; total = count + n
        m2 += ((x-batch_mean)**2).sum().item() + delta*delta*count*n/total
        mean += delta*n/total; count = total
    if not count: raise ValueError('training data is empty')
    return {'mean':mean, 'std':max((m2/count)**0.5,1e-6), 'count':count}


def _atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,tmp)
    tmp.replace(path)


def _resume_contract(contract):
    result=deepcopy(contract)
    # These defaults reproduce the pre-v2 engine; other differences stay errors.
    for key,value in {'train_pairs':None,'validation_pairs':None,'normalization_mode':'frozen','selection_metric':'loss',
                      'dice_smoothing':1.0,'precision':'float32','activation_checkpointing':False}.items():
        result.setdefault(key,value)
    config=result.get('run_info',{}).get('resolved_config')
    if config is not None:
        config = resolve_config(config)
        # Extending the current stage's epoch horizon is the only mutable setting.
        config['training'][result['stage']]['epochs']=None
        result['run_info']['resolved_config'] = config
    return result


def train_prepared(train_dataset, validation_dataset, model, output_dir, *, method, stage, epochs, batch_size, learning_rate, seed, device, resume=None, run_info=None, train_pairs=None, validation_pairs=None, normalization_mode='frozen', selection_dataset=None, selection_metric='loss', dice_smoothing=1.0, precision='float32', activation_checkpointing=False):
    if method not in ('A','B') or stage not in ('dte','dtr'): raise ValueError('invalid method or stage')
    if epochs < 1 or batch_size < 1 or learning_rate <= 0: raise ValueError('invalid training parameters')
    if normalization_mode not in ('frozen','running','none'): raise ValueError('unsupported normalization mode')
    if normalization_mode=='running' and (method!='A' or stage!='dte'): raise ValueError('running statistics apply to A DTE only')
    if selection_metric not in ('loss','mae','dice'): raise ValueError('unsupported selection metric')
    for count in (train_pairs,validation_pairs):
        if count is not None and (type(count) is not int or count<1): raise ValueError('pair counts must be positive integers')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    runtime = TrainingRuntime(device, precision, activation_checkpointing)
    out = Path(output_dir)
    if resume is None and (out/'last.pt').exists(): raise FileExistsError('output already has a checkpoint; select a new directory or resume')
    out.mkdir(parents=True,exist_ok=True)
    run_info = {} if run_info is None else run_info
    contract = {'method':method,'stage':stage,'batch_size':batch_size,'learning_rate':learning_rate,'seed':seed,'run_info':run_info,
                'parameters':{name:list(value.shape) for name,value in model.state_dict().items()},
                'train_pairs':train_pairs,'validation_pairs':validation_pairs,'normalization_mode':normalization_mode,
                'selection_metric':selection_metric,'dice_smoothing':dice_smoothing,
                'precision':precision,'activation_checkpointing':activation_checkpointing}
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator().manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.999))
    start_epoch = 0; history = []; best = float('inf');best_selection=None
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if _resume_contract(checkpoint['contract']) != _resume_contract(contract): raise ValueError('checkpoint training contract differs')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        runtime.restore_scaler(checkpoint.get('scaler'))
        start_epoch = checkpoint['epoch']; history = checkpoint['history']; best = checkpoint['best_validation_loss']
        stats = checkpoint['normalization']; generator.set_state(checkpoint['loader_rng'].cpu())
        best_selection=checkpoint.get('best_selection',best if selection_metric=='loss' else None)
        if 'python_rng' in checkpoint: random.setstate(checkpoint['python_rng'])
        if 'numpy_rng' in checkpoint: restore_numpy_rng(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type == 'cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
        if epochs <= start_epoch: raise ValueError('epochs must exceed completed checkpoint epoch')
    else:
        stats = input_statistics(train_dataset) if method=='A' and stage=='dte' and normalization_mode=='frozen' else None
    running=RunningStatistics(stats) if normalization_mode=='running' else None
    loss_fn = training_loss(method, stage, dice_smoothing)
    for epoch in range(start_epoch,epochs):
        train_data=EpochDataset(train_dataset,train_pairs,seed,epoch,'train') if train_pairs is not None else train_dataset
        val_data=EpochDataset(validation_dataset,validation_pairs,seed,epoch,'validation') if validation_pairs is not None else validation_dataset
        train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=train_pairs is None,generator=generator,num_workers=0)
        val_loader=DataLoader(val_data,batch_size=batch_size,shuffle=False,num_workers=0)
        started = time.perf_counter(); losses = {};counts={};steps=0;skipped_steps=0
        for name, loader in (('train',train_loader),('validation',val_loader)):
            training = name == 'train'; model.train(training); total = 0.0; n = 0
            with torch.set_grad_enabled(training):
                for x,y in loader:
                    x=x.to(device); y=y.to(device)
                    if running is not None:
                        if training:running.update(x)
                        x=running.normalize(x)
                    elif stats is not None: x=(x-stats['mean'])/stats['std']
                    if training: optimizer.zero_grad(set_to_none=True)
                    pred = runtime.forward(model, x)
                    if pred.shape != y.shape: raise ValueError(f'prediction {tuple(pred.shape)} and target {tuple(y.shape)} differ')
                    loss = loss_fn(pred,y)
                    if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
                    if training:
                        if runtime.step(loss, optimizer):
                            steps += 1
                        else:
                            skipped_steps += 1
                    total += loss.item()*len(x); n += len(x)
            if not n: raise ValueError(f'{name} dataset is empty')
            losses[name] = total/n
            counts[name]=n
        if running is not None:stats=running.state_dict()
        selection=losses['validation']
        if selection_dataset is not None and hasattr(selection_dataset,'evaluate'):
            model.eval()
            with torch.inference_mode():selection=float(selection_dataset.evaluate(model,stats,device))
        elif selection_metric!='loss':
            panel=selection_dataset if selection_dataset is not None else EpochDataset(validation_dataset,validation_pairs or len(validation_dataset),seed,0,'selection')
            model.eval();values=[]
            with torch.inference_mode():
                for x,y in DataLoader(panel,batch_size=batch_size,shuffle=False):
                    x=x.to(device);y=y.to(device)
                    if stats is not None:x=(x-stats['mean'])/stats['std']
                    pred=model(x)
                    axes=tuple(range(1,pred.ndim))
                    if selection_metric=='mae':score=(pred-y).abs().mean(dim=axes)
                    else:
                        binary=pred>=.5;den=binary.sum(dim=axes)+y.sum(dim=axes)
                        score=torch.where(den>0,2*(binary*y).sum(dim=axes)/den.clamp(min=1),torch.ones_like(den,dtype=torch.float32))
                    values.extend(score.cpu().tolist())
            selection=float(np.mean(values))
        if not np.isfinite(selection):raise RuntimeError('non-finite selection metric')
        improved=best_selection is None or (selection>best_selection if selection_metric=='dice' else selection<best_selection)
        if improved:best_selection=selection
        best=min(best,losses['validation'])
        history.append({'epoch':epoch+1,'train_loss':losses['train'],'validation_loss':losses['validation'],
                        'selection_metric':selection_metric,'selection_value':selection,'train_samples':counts['train'],
                        'validation_samples':counts['validation'],'optimizer_steps':steps,
                        'skipped_optimizer_steps':skipped_steps,'seconds':time.perf_counter()-started})
        state = {'contract':contract,'epoch':epoch+1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                 'scaler':runtime.scaler_state(),
                 'normalization':stats,'history':history,'best_validation_loss':best,'best_selection':best_selection,'loader_rng':generator.get_state(),
                 'python_rng':random.getstate(),'numpy_rng':numpy_rng_state(),
                 'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                 'peak_cuda_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
        _atomic_save(state,out/'last.pt')
        if improved: _atomic_save(state,out/'best.pt')
        (out/'metrics.json').write_text(json.dumps({'contract':contract,'normalization':stats,'history':history,'best_validation_loss':best,'best_selection':best_selection,'peak_cuda_bytes':state['peak_cuda_bytes']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'history':history,'normalization':stats,'best_validation_loss':best,'best_selection':best_selection}
