import json
from pathlib import Path
import numpy as np
import torch
from .heatmap import make_heatmap
from .models.unet2d import MarkerUNet2D


def _image_tensor(image):
    a=np.asarray(image,dtype=np.float32)
    lo=float(np.min(a)); hi=float(np.max(a))
    if hi>lo: a=(a-lo)/(hi-lo)
    else: a=np.zeros_like(a)
    return torch.from_numpy(a[None,None])


def train_detector(cases, config, out_dir):
    cfg=dict(config); seed=int(cfg.get("seed",0)); torch.manual_seed(seed); np.random.seed(seed)
    device=torch.device(cfg.get("device","cpu"))
    model=MarkerUNet2D(widths=tuple(cfg.get("widths",(32,64,128,256)))).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(cfg.get("learning_rate",1e-3)))
    examples=[]; sigma=float(cfg.get("sigma",2.0))
    for case in cases:
        for view in case.views:
            image=view.image
            examples.append((_image_tensor(image).to(device),torch.from_numpy(make_heatmap(view.marker_xy,image.shape,sigma)[None,None]).to(device)))
    if not examples: raise ValueError("training requires at least one view")
    epochs=int(cfg.get("epochs",1)); batch_size=max(1,int(cfg.get("batch_size",4))); alpha=float(cfg.get("alpha",0)); lambda_fn=float(cfg.get("lambda_fn",0))
    losses=[]
    for _ in range(epochs):
        order=np.random.permutation(len(examples)); epoch_loss=[]; model.train()
        for start in range(0,len(order),batch_size):
            batch=[examples[i] for i in order[start:start+batch_size]]
            x=torch.cat([z[0] for z in batch]); y=torch.cat([z[1] for z in batch])
            optimizer.zero_grad(set_to_none=True); pred=torch.sigmoid(model(x))
            loss=torch.mean((1+alpha*y)*(pred-y).square())
            if lambda_fn: loss=loss+lambda_fn*torch.relu(y-pred).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
            loss.backward(); optimizer.step(); epoch_loss.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_loss)))
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    fit_groups=sorted({str(c.group_id) for c in cases})
    checkpoint={"state_dict":model.cpu().state_dict(),"config":cfg,"fit_groups":fit_groups,"loss":losses[-1],"format":"marker_unet_checkpoint_v1"}
    torch.save(checkpoint,out/"checkpoint.pt")
    (out/"config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    (out/"training_groups.json").write_text(json.dumps(fit_groups,indent=2),encoding="utf-8")
    return {"loss":losses[-1],"losses":losses,"fit_groups":fit_groups,"checkpoint":str(out/"checkpoint.pt")}


def predict_heatmap(checkpoint, image):
    ckpt=torch.load(checkpoint,map_location="cpu",weights_only=False)
    cfg=ckpt.get("config",{})
    model=MarkerUNet2D(widths=tuple(cfg.get("widths",(32,64,128,256))))
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad(): return torch.sigmoid(model(_image_tensor(image)))[0,0].numpy()
