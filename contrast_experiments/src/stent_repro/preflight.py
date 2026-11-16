"""Separate executable-shape evidence from scientific reproduction readiness."""
import gc
import json
import platform
from pathlib import Path
import time
import numpy as np
import torch
from stent_repro.configuration import resolve_config
from stent_repro.geometry import Grid,circular_view,project_points,forward_project
from stent_repro.model_registry import build_model,model_spec,architecture_report
from stent_repro.runtime import benchmark_forward
from stent_repro.training.engine import TrainingRuntime, training_loss


def resource_contract(config,stage,device):
    device=torch.device(device);_,channels,_=model_spec(config,stage)
    training=config['training'][stage]
    hardware=torch.cuda.get_device_name(device) if device.type=='cuda' and torch.cuda.is_available() else platform.machine()+' '+platform.processor()
    fields=['architecture','bn_eps','bn_momentum','conv_bias','initialization','compact_base',f'{stage}_activation']
    if stage=='dtr':fields.append('dice_smoothing')
    return {'method':config['method'],'stage':stage,'network':{key:config['network'][key] for key in fields},
            'input_shape':[training['batch_size'],channels,*training['patch_shape']],
            'device':str(device),'hardware':hardware,'torch':str(torch.__version__),'cuda':torch.version.cuda,
            'precision':training.get('precision','float32'),'activation_checkpointing':training.get('activation_checkpointing',False),
            'optimizer':'Adam','loss':'L1' if stage=='dte' else 'soft_dice_'+config['method']}


def validate_resource_report(config,stage,device,path):
    report=json.loads(Path(path).read_text(encoding='utf8'))
    measured=report.get('resource_contract')
    if isinstance(measured,dict):measured={'activation_checkpointing':False,**measured}
    if measured!=resource_contract(config,stage,device):raise ValueError('preflight report does not match configuration, stage, device, or runtime hardware')
    if not report.get('full_step_measured') or not report.get('gradient_finite') or report.get('optimizer_state_tensors',0)<=0:
        raise ValueError('preflight report must prove a real forward/backward/optimizer step')
    if not report.get('formal_task_ready'):raise ValueError('preflight report did not pass formal task readiness')
    return report


def geometry_validation(config):
    g=config['geometry'];view=circular_view(0,g['sdd_mm'],g['sid_mm'],g['detector_shape'],g['pixel_size_mm'])
    point=np.array([[1.,2.,3.]]);projected=project_points(point,view)[0]
    scale=g['sdd_mm']/(g['sid_mm']+2)/g['pixel_size_mm']
    expected=np.array([(g['detector_shape'][1]-1)/2+scale,(g['detector_shape'][0]-1)/2+3*scale])
    grid=Grid((9,33,33),(.5,.5,.5),(-8.,-8.,-2.))
    z,y,x=np.meshgrid(np.arange(9),np.arange(33)*.5-8,np.arange(33)*.5-8,indexing='ij')
    cylinder=((x*x+y*y)<4.25**2).astype(np.float32)
    central=[circular_view(0,200,100,(1,1),1)]
    first=float(forward_project(cylinder,grid,central,step_mm=.125)[0,0,0])
    second=float(forward_project(cylinder,grid,central,step_mm=.0625)[0,0,0])
    return {'scope':'analytic_cpu_reference_not_system_calibration','point_error_pixels':float(np.linalg.norm(projected-expected)),
            'central_ray_expected':8.5,'central_ray_observed':first,'central_ray_relative_error':abs(first-8.5)/8.5,
            'step_halving_change':abs(first-second)/max(abs(second),1e-12)}


def preflight(config,*,stage=None,device='cpu',measure=False):
    c=resolve_config(config);stage=stage or 'dtr';device=torch.device(device);available=device.type=='cpu' or (device.type=='cuda' and torch.cuda.is_available())
    report=architecture_report(c,stage);limitations=[]
    if c['method']=='A':limitations.append('A stage interpretation is not author-verified')
    else:limitations.append('TensorFlow author parameters are not verified; explicit PyTorch port decisions apply')
    if c['network']['architecture']=='compact-g0':limitations.append('compact resource-adapted G0 architecture')
    if not c['provenance']['geometry_calibrated']:limitations.append('actual system geometry has not been calibrated')
    if not c['provenance']['author_data']:limitations.append('original author data unavailable')
    if not available:limitations.append('requested device unavailable')
    result={'method':c['method'],'stage':stage,'device':str(device),'device_available':available,'full_step_measured':False,
            'gradient_finite':None,'optimizer_state_tensors':0,'estimated_parameter_optimizer_bytes':report['parameter_count']*16,
            'estimate_scope':'float32 parameters, gradients and two Adam moments; excludes activations/workspaces',
            'peak_cuda_bytes':None,'formal_task_ready':False,'paper_numerical_reproduction_ready':False,'limitations':limitations,
            'architecture':report,'geometry_validation':geometry_validation(c),'resource_contract':resource_contract(c,stage,device)}
    if measure and available:
        training = c['training'][stage]
        _, channels, widths = model_spec(c, stage)
        shape = [training['batch_size'], channels, *training['patch_shape']]
        stride = 2 ** (len(widths) - 1)
        model = optimizer = x = y = pred = loss = None
        phase = 'validating_runtime'
        track_peak = False
        try:
            if any(size % stride for size in shape[2:]):
                raise ValueError('training patch dimensions must be multiples of network stride')
            contract = result['resource_contract']
            runtime = TrainingRuntime(device, contract['precision'], contract['activation_checkpointing'])
            loss_fn = training_loss(c['method'], stage, c['network']['dice_smoothing'])
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                track_peak = True
            phase = 'building_model'
            model = build_model(c, stage).to(device).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=training['learning_rate'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            phase = 'allocating_inputs'
            started = time.perf_counter()
            x = torch.randn(shape, device=device)
            y = torch.rand([shape[0], 1, *shape[2:]], device=device)
            phase = 'forward_and_loss'
            pred = runtime.forward(model, x)
            loss = loss_fn(pred, y)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite preflight loss')
            phase = 'backward_and_optimizer'
            updated = runtime.step(loss, optimizer)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            phase = 'checking_gradients'
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
            nonzero = finite and any(bool(torch.any(g != 0)) for g in gradients)
            result.update(
                full_step_measured=finite and nonzero and updated,
                gradient_finite=finite,
                gradient_nonzero=nonzero,
                measured_input_shape=shape,
                step_seconds=seconds,
                optimizer_step_applied=updated,
                loss_value=float(loss.detach()),
                step_timing_scope=(
                    'single synchronized training step including synthetic input allocation, '
                    'excluding model construction and gradient diagnostics'
                ),
                optimizer_state_tensors=sum(
                    torch.is_tensor(value)
                    for state in optimizer.state.values()
                    for value in state.values()
                ),
            )
            result['formal_task_ready'] = bool(
                result['full_step_measured']
                and c['network']['architecture'] != 'compact-g0'
                and c['provenance']['geometry_calibrated']
                and c['geometry']['calibration_id']
                and c['provenance']['dataset_version'] not in ('unspecified', 'REPLACE_WITH_FROZEN_DATA_VERSION')
            )
        except (RuntimeError,ValueError,MemoryError) as exc:
            result['measurement_error'] = str(exc)
            result['measurement_phase'] = phase
        finally:
            if track_peak:
                result['peak_cuda_bytes'] = torch.cuda.max_memory_allocated(device)
            del model, optimizer, x, y, pred, loss
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    if device.type=='cpu':result['limitations'].append('CPU full step can be checked; CUDA peak memory and GPU timing remain unmeasured')
    return result


def benchmark_stage(
    config, *, stage, device='cpu', tile_shape=None, overlap=None,
    warmup=2, repeats=5, compare_whole=True,
):
    """Measure a configured stage on a reproducible synthetic patch batch."""
    config=resolve_config(config)
    _,channels,widths=model_spec(config,stage)
    training=config['training'][stage]
    shape=(training['batch_size'],channels,*training['patch_shape'])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config['seed'])
        model=build_model(config,stage).eval()
        x=torch.randn(shape)
    report=benchmark_forward(
        model,x,device=device,stride=2**(len(widths)-1),tile_shape=tile_shape,
        overlap=overlap,warmup=warmup,repeats=repeats,compare_whole=compare_whole,
    )
    report.update(method=config['method'],stage=stage,architecture=config['network']['architecture'],
                  precision='float32',batch_source='synthetic_training_patch')
    return report
