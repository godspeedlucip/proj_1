"""Local entry for the native Dataset504 marker/depth Trainer and paired prediction."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
STORE = PROJECT / "training_data/training/nnunet/phase2"
DATASET = "Dataset504_MarkerDepth"
ROOT = STORE / DATASET
PREP = ROOT / "preprocessed" / DATASET
PLANS = "DepthCascadePlans"
BASE = "nnUNetTrainerMarkerGuidedDepth"
SOURCE = (STORE / "Dataset503_MarkerGuided_fold_0/results/pretrained_from_Dataset501" /
          "nnUNetTrainerMarkerGuidedFeatureFusion__nnUNetPlans__2d")
SOURCE_CKPT = SOURCE / "fold_0/checkpoint_final.safetensors"
SOURCE_SHA = "92cedb46947e96017ef32a72aafbdf07e87d2b54642eea7c3890e29641340902"
sys.path[:0] = [str(HERE), str(HERE.parent), str(PROJECT / "model_outputs/models/phase2_nnunet/deps")]


def sha256(path):
    with path.open("rb") as f:
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_config(path, payload):
    data = json.dumps(payload, indent=2) + "\n"
    if path.is_file():
        if path.read_text() != data:
            raise RuntimeError(f"Prepared configuration changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(data)
        temp.replace(path)


def bootstrap():
    from nnunetv2.training.nnUNetTrainer.variants.marker_guided_depth.data import prepare_prior_cache
    path = ROOT / "manifest.json"
    manifest = json.loads(path.read_text())
    rows, split = manifest["samples"], manifest["fold"]
    counts = {k: sum(r["split"] == k for r in rows) for k in ("train", "val", "test")}
    if counts != {"train": 5040, "val": 720, "test": 60} or sha256(SOURCE_CKPT) != SOURCE_SHA:
        raise ValueError("Unexpected physical cohort or segmentation initialization")
    if manifest["validation"]["full_images_checked"] != len(rows):
        raise ValueError("Full image audit incomplete")
    old_split_path = STORE / "preprocessed/Dataset501_MarkerGuided_fold_0/splits_final.json"
    old = json.loads(old_split_path.read_text())[0]
    groups = {p: sorted({re.fullmatch(r"topcow_ct_\d+_(.+)_\d+_v[0-9_]+v\d+", k).group(1)
                         for k in old[p]}) for p in ("train", "val")}
    if any(set(groups[s]) != set(split[s]) for s in groups):
        raise ValueError("Pretrained fold group mismatch")
    plans = json.loads((SOURCE / "plans.json").read_text())
    ds = json.loads((SOURCE / "dataset.json").read_text())
    ds.update(name=DATASET, numTraining=counts["train"] + counts["val"],
              description="Full detector projections, oracle E5-U marker priors and floating depth supervision")
    plans.update(dataset_name=DATASET, plans_name=PLANS)
    plans["configurations"]["2d"].update(patch_size=[1280, 1280], batch_size=1,
                                          data_identifier="DepthCascadePlans_2d")
    plans["joint_depth"] = {
        "manifest": str(path), "manifest_sha256": sha256(path), "target_scale": manifest["target_scale"],
        "depth_definition": manifest["depth_definition"], "marker_source": manifest["marker_source"],
        "init_seg_checkpoint": str(SOURCE_CKPT), "init_seg_sha256": SOURCE_SHA,
        "init_seg_dataset": "Dataset501_MarkerGuided_fold_0", "init_seg_train_val_groups": groups,
        "seed": 20261118, "gate": "hard_mask", "threshold": 0.5, "pad_multiple": 256,
        "preprocessing": "zscore_native_canvas_then_bottom_right_padding",
    }
    for folder in (ROOT / "raw" / DATASET, PREP):
        write_config(folder / "dataset.json", ds)
    write_config(PREP / (PLANS + ".json"), plans)
    write_config(PREP / "splits_final.json", [{k: [r["sample_id"] for r in rows if r["split"] == k]
                                               for k in ("train", "val")}])
    write_config(PREP / "dataset_fingerprint.json", {"source": str(path), "sha256": sha256(path),
                 "counts": counts, "shape": [1056, 1056], "support_check": manifest["validation"]})
    (PREP / "DepthCascadePlans_2d").mkdir(exist_ok=True)
    prepare_prior_cache(ROOT / "prior_cache", rows, (1056, 1056))
    print(f"Prepared {DATASET}: {counts}, {len(list((ROOT / 'prior_cache').glob('*.npz')))} marker pairs",
          flush=True)


def setup(device):
    for var, folder in (("nnUNet_raw", "raw"), ("nnUNet_preprocessed", "preprocessed"), ("nnUNet_results", "results")):
        os.environ[var] = str(ROOT / folder)
    os.environ["nnUNet_compile"] = "false"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if device is None:
        return None
    match = re.fullmatch(r"cuda:(\d+)", device)
    if not match:
        raise ValueError("--device must be cuda:<physical GPU index>")
    uuid = subprocess.check_output(["nvidia-smi", "--id=" + match[1], "--query-gpu=uuid",
                                    "--format=csv,noheader"], text=True).strip()
    if not re.fullmatch(r"GPU-[a-zA-Z0-9-]+", uuid):
        raise RuntimeError(f"Invalid device UUID {uuid!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Cannot see selected GPU")
    torch.set_num_threads(2)
    print(f"Physical {device} ({uuid}), process cuda:0, {torch.cuda.get_device_name(0)}", flush=True)
    return torch.device("cuda", 0)


def train(args):
    trainer = BASE + ("NoMarker" if args.marker == "off" else "") + ("Smoke" if args.smoke else "")
    folder = ROOT / "results" / DATASET / (trainer + "__" + PLANS + "__2d") / "fold_0"
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    if args.background:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *(v for v in sys.argv[1:] if v != "--background")]
        with (logs / (trainer + ".log")).open("ab") as output:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        print(f"Spawned PID {proc.pid}; {output.name}", flush=True)
        return
    with (logs / (trainer + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"{trainer} already running") from exc
        if args.resume:
            if not any((folder / n).is_file() for n in ("checkpoint_latest.safetensors", "checkpoint_final.safetensors")):
                raise RuntimeError("Resume requires an existing latest or final checkpoint")
        elif any(folder.glob("checkpoint_*.safetensors")):
            raise RuntimeError(f"Existing checkpoints at {folder}. Use --resume.")
        device = setup(args.device)
        (logs / (trainer + ".pid")).write_text(str(os.getpid()) + "\n")
        from nnunetv2.run.run_training import run_training
        run_training(DATASET, "2d", 0, trainer_class_name=trainer, plans_identifier=PLANS,
                     continue_training=args.resume, val_with_best=True, device=device)


def predict(args):
    device = setup(args.device)
    from nnunetv2.training.nnUNetTrainer.variants.marker_guided_depth.inference import predict_frame, restore_network, save_frame
    import tifffile
    ck = args.checkpoint.resolve(strict=True)
    net, joint = restore_network(ck.parent.parent, ck, device)
    img = tifffile.imread(args.image)
    pts = [args.marker_xy[:2], args.marker_xy[2:]] if args.marker_xy is not None else None
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    vals = predict_frame(net, img, pts, pts is not None, joint["target_scale"], device)
    save_frame(out, args.image.stem, vals, True)
    write_config(out / "projection_metadata.json", {"image": str(args.image.resolve()), "shape": list(img.shape),
                 "checkpoint": str(ck), "checkpoint_sha256": sha256(ck), "marker_xy": pts, "marker_enabled": pts is not None,
                 "target_scale": joint["target_scale"], "depth_definition": joint["depth_definition"], "threshold": 0.5,
                 "manifest_sha256": joint["manifest_sha256"], "geometry": None,
                 "predicted_foreground": int(vals["masks"].sum())})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    tr = sub.add_parser("train")
    tr.add_argument("--device", required=True)
    tr.add_argument("--marker", choices=["on", "off"], default="on")
    tr.add_argument("--smoke", action="store_true")
    tr.add_argument("--background", action="store_true")
    tr.add_argument("--resume", action="store_true")
    pr = sub.add_parser("predict")
    pr.add_argument("--device", required=True)
    pr.add_argument("--checkpoint", type=Path, required=True)
    pr.add_argument("--image", type=Path, required=True)
    pr.add_argument("--output", type=Path, required=True)
    group = pr.add_mutually_exclusive_group(required=True)
    group.add_argument("--marker-xy", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    group.add_argument("--no-marker", action="store_true")
    args = cli.parse_args()
    if args.command == "prepare":
        setup(None)
        with (ROOT / "manifest.json").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            bootstrap()
    elif args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
