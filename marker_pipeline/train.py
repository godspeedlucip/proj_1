import json
from pathlib import Path
import numpy as np
import torch
from .heatmap import make_heatmap
from .models.unet2d import MarkerUNet2D
from .models.joint_epipolar import JointEpipolarMarkerNet
from .epipolar import EpipolarSupport, fundamentals_from_projections, validate_relative_geometry
from .view_order import canonical_triplet_indices


def _image_tensor(image):
    a=np.asarray(image,dtype=np.float32)
    lo=float(np.min(a)); hi=float(np.max(a))
    if hi>lo: a=(a-lo)/(hi-lo)
    else: a=np.zeros_like(a)
    return torch.from_numpy(a[None,None])


def train_detector(cases, config, out_dir):
    if config.get("model_type", "single_view_v1") == "joint_epipolar_v1":
        return train_joint_detector(cases, config, out_dir)
    if config.get("model_type", "single_view_v1") != "single_view_v1":
        raise ValueError("unknown model_type")
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


def _joint_model(config, reference_F):
    backbone = MarkerUNet2D(widths=tuple(config.get("widths", (32, 64, 128, 256))))
    support = EpipolarSupport(
        reference_F,
        stride=int(config.get("support_stride", 4)),
        samples=int(config.get("support_samples", 32)),
        chunk_rows=int(config.get("support_chunk_rows", 32)),
        temperature=float(config.get("support_temperature", 0.1)),
    )
    return JointEpipolarMarkerNet(backbone, support)


def _ordered_views(case):
    order = canonical_triplet_indices([view.angle_deg for view in case.views])
    return order, [case.views[index] for index in order]


def train_joint_detector(cases, config, out_dir):
    cfg = dict(config)
    cfg["model_type"] = "joint_epipolar_v1"
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not cases:
        raise ValueError("training requires at least one case")
    _, reference_views = _ordered_views(cases[0])
    image_shape = tuple(reference_views[0].image.shape)
    geometry_id = cases[0].metadata.get("geometry_id", "sim_geometry_v1")
    reference_F = fundamentals_from_projections([view.P for view in reference_views])
    tolerance = float(cfg.get("geometry_tolerance_px", 0.25))
    sigma = float(cfg.get("sigma", 2.0))
    examples = []
    for case in cases:
        _, views = _ordered_views(case)
        if case.metadata.get("geometry_id", "sim_geometry_v1") != geometry_id:
            raise ValueError(f"case {case.case_id} has a different geometry_id")
        if any(tuple(view.image.shape) != image_shape for view in views):
            raise ValueError(f"case {case.case_id} has a different detector pixel grid")
        try:
            validate_relative_geometry(reference_F, [view.P for view in views], image_shape,
                                       tolerance_px=tolerance)
        except ValueError as exc:
            raise ValueError(f"case {case.case_id}: {exc}") from exc
        images = torch.stack([_image_tensor(view.image)[0] for view in views])
        labels = torch.stack([
            torch.from_numpy(make_heatmap(view.marker_xy, image_shape, sigma)[None])
            for view in views
        ])
        examples.append((images, labels))
    device = torch.device(cfg.get("device", "cpu"))
    model = _joint_model(cfg, reference_F).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    epochs = int(cfg.get("epochs", 1))
    if epochs < 1:
        raise ValueError("epochs must be positive")
    batch_size = max(1, int(cfg.get("batch_size", 1)))
    alpha = float(cfg.get("alpha", 0.0))
    lambda_fn = float(cfg.get("lambda_fn", 0.0))
    aux_weight = float(cfg.get("aux_loss_weight", 0.25))
    losses = []
    for _ in range(epochs):
        order = np.random.permutation(len(examples))
        epoch_losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            group = [examples[index] for index in order[start:start + batch_size]]
            images = torch.stack([item[0] for item in group]).to(device)
            targets = torch.stack([item[1] for item in group]).to(device)
            optimizer.zero_grad(set_to_none=True)
            initial_logits, refined_logits = model(images)
            def heat_loss(logits):
                prediction = torch.sigmoid(logits)
                loss = torch.mean((1 + alpha * targets) * (prediction - targets).square())
                if lambda_fn:
                    loss = loss + lambda_fn * torch.relu(targets - prediction).square().mean()
                return loss
            loss = heat_loss(refined_logits) + aux_weight * heat_loss(initial_logits)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite joint training loss")
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fit_groups = sorted({str(case.group_id) for case in cases})
    checkpoint = {
        "state_dict": model.cpu().state_dict(), "config": cfg, "fit_groups": fit_groups,
        "reference_F": reference_F, "image_shape": image_shape, "geometry_id": geometry_id,
        "loss": losses[-1], "format": "marker_joint_epipolar_checkpoint_v1",
    }
    torch.save(checkpoint, out / "checkpoint.pt")
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out / "training_groups.json").write_text(json.dumps(fit_groups, indent=2), encoding="utf-8")
    return {"loss": losses[-1], "losses": losses, "fit_groups": fit_groups,
            "checkpoint": str(out / "checkpoint.pt")}


def predict_joint_heatmaps(checkpoint, case):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("format") != "marker_joint_epipolar_checkpoint_v1":
        raise ValueError("checkpoint is not a joint epipolar model")
    cfg = ckpt["config"]
    order, views = _ordered_views(case)
    image_shape = tuple(ckpt["image_shape"])
    if any(tuple(view.image.shape) != image_shape for view in views):
        raise ValueError(f"case {case.case_id} has a different detector pixel grid")
    if case.metadata.get("geometry_id", "sim_geometry_v1") != ckpt["geometry_id"]:
        raise ValueError(f"case {case.case_id} has a different geometry_id")
    try:
        validate_relative_geometry(ckpt["reference_F"], [view.P for view in views], image_shape,
                                   tolerance_px=float(cfg.get("geometry_tolerance_px", 0.25)))
    except ValueError as exc:
        raise ValueError(f"case {case.case_id}: {exc}") from exc
    model = _joint_model(cfg, ckpt["reference_F"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    images = torch.stack([_image_tensor(view.image)[0] for view in views])[None]
    with torch.no_grad():
        initial, refined = model(images)
    initial = torch.sigmoid(initial)[0, :, 0].numpy()
    refined = torch.sigmoid(refined)[0, :, 0].numpy()
    inverse = np.argsort(order)
    return initial[inverse], refined[inverse]
