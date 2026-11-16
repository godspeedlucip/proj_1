"""Explicit architectures; reports never allocate the large B-DTE weights."""
import torch
from stent_repro.networks import UNet


def model_spec(config, stage):
    if stage not in ('dte','dtr'): raise ValueError('stage must be dte or dtr')
    n=config['network']; a=config['method']=='A'
    dims=3 if not a and stage=='dtr' else 2
    inputs=1 if stage=='dte' else (n['slice_channels'] if a else n['input_channels'])
    if n['architecture']=='compact-g0': channels=tuple(n['compact_base']*2**i for i in range(3))
    elif a: channels=(64,128,256,512)+( (1024,) if n['architecture']=='four-downsample' else () )
    else: channels=(64,128,256,512,1024,2048,4096) if stage=='dte' else (16,32,64,128,256)
    return dims,inputs,channels


def build_model(config, stage):
    dims,inputs,channels=model_spec(config,stage); n=config['network']
    return UNet(dims,inputs,channels,dropout=0.2 if config['method']=='A' else 0,
                final=n[f'{stage}_activation'],bn_eps=n['bn_eps'],bn_momentum=n['bn_momentum'],bias=n['conv_bias'])


def architecture_report(config, stage):
    dims,inputs,channels=model_spec(config,stage); layers=[]
    with torch.device('meta'):
        model=build_model(config,stage).eval()
        hooks=[]
        for name,module in model.named_modules():
            if not list(module.children()):
                def capture(mod,args,out,label=name):
                    layers.append({'name':label,'type':type(mod).__name__,'input_shape':list(args[0].shape),'output_shape':list(out.shape)})
                hooks.append(module.register_forward_hook(capture))
        shape=[1,inputs]+config['training'][stage]['patch_shape']
        stride=2**(len(channels)-1)
        shape=[*shape[:2],*[max(stride,((x+stride-1)//stride)*stride) for x in shape[2:]]]
        output=model(torch.empty(shape,device='meta'))
        for hook in hooks: hook.remove()
        count=sum(p.numel() for p in model.parameters())
    return {'architecture':config['network']['architecture'],
            'architecture_status':'resource_adapted_g0' if config['network']['architecture']=='compact-g0' else 'interpretation_not_author_verified',
            'method':config['method'],'stage':stage,'encoder_stages':len(channels),'decoder_stages':len(channels)-1,
            'downsampling_steps':len(channels)-1,'channels':list(channels),'parameter_count':count,
            'input_shape':shape,'output_shape':list(output.shape),'layers':layers,'measurement':'meta_shapes_only',
            'bn':{'epsilon':config['network']['bn_eps'],'momentum':config['network']['bn_momentum'],
                  'update':'running = (1-momentum)*running + momentum*batch'}}
