"""Dispatch each smoke-qualified Phase 3 arm on a free GPU and monitor it."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ARMS = ("a60_n3", "a45_n3", "a75_n3", "a60_n2", "a60_n5")
INTERVAL_SECONDS = 1800


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def smoke_passed(root: Path, arm: str) -> bool:
    base = root / arm / "smoke"
    if not all((base / name).is_file() for name in
               ("passed.json", "checkpoint_audit.json", "results_audit.json")):
        return False
    return (read_json(base / "passed.json").get("passed") is True
            and read_json(base / "checkpoint_audit.json").get("complete") is True
            and read_json(base / "results_audit.json").get("complete") is True)


def smoke_ready(root: Path) -> bool:
    return all(smoke_passed(root, arm) for arm in ARMS)


def available_gpus(reserved: set[int]) -> list[int]:
    lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    usage = []
    for line in lines:
        index, memory = (int(piece.strip()) for piece in line.split(","))
        if memory < 1024 and index not in reserved:
            usage.append((memory, index))
    return [index for _, index in sorted(usage, key=lambda row:
            (0 if row[1] in (4, 5) else 1, row[0], row[1]))]


def gpu_memory_mib() -> dict[int, int]:
    lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    return {int(parts[0].strip()): int(parts[1].strip())
            for parts in (line.split(",") for line in lines)}


def live_command(pid: int) -> str | None:
    path = Path(f"/proc/{pid}/cmdline")
    try:
        return path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        return None


def model_folder(root: Path, arm: str) -> Path:
    prepared = read_json(root / arm / "prepared/prepared.json")
    return (Path(prepared["results_dir"])
            / f"nnUNetTrainerPhase3__{prepared['plans_name']}__3d_fullres/fold_0")


def progress_snapshot(root: Path, running: dict, completed: set[str]) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    rows = {}
    for arm in ARMS:
        base = root / arm
        status_path = base / "status.json"
        status = read_json(status_path) if status_path.exists() else {}
        fold = model_folder(root, arm) if (base / "prepared/prepared.json").is_file() else None
        epochs = sorted(int(p.stem.rsplit("_", 1)[-1]) for p in fold.glob("checkpoint_epoch_*.pth")) if fold else []
        previews = sorted(int(p.name.rsplit("_", 1)[-1])
                          for p in (fold / "validation_previews").glob("epoch_*") if p.is_dir()) if fold else []
        events = list((fold / "tensorboard").glob("events.out.tfevents.*")) if fold else []
        tensorboard = {}
        if events:
            accumulator = EventAccumulator(str(fold / "tensorboard"),
                                           size_guidance={"scalars": 1, "images": 1})
            accumulator.Reload()
            tags = accumulator.Tags()
            for tag in ("train/loss", "validation/patch_loss", "validation/full/dice"):
                if tag in tags["scalars"] and accumulator.Scalars(tag):
                    event = accumulator.Scalars(tag)[-1]
                    tensorboard[tag] = dict(step=event.step, value=event.value)
            tensorboard["validation_image_tags"] = [tag for tag in tags["images"]
                                                       if tag.startswith("validation/preview/")]
        rows[arm] = dict(stage=status.get("stage"), state=status.get("state"),
                         pid=running.get(arm, {}).get("pid"), gpu=running.get(arm, {}).get("gpu"),
                         completed=arm in completed, checkpoint_epochs=epochs,
                         preview_epochs=previews, tensorboard_event_files=len(events),
                         tensorboard=tensorboard,
                         formal_test_audited=(base / "results_audit.json").is_file())
    return dict(timestamp_unix=time.time(), gpu_memory_mib=gpu_memory_mib(), arms=rows)


def audit_training_artifacts(root: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    audit = {}
    expected = list(range(1, 71))
    for arm in ARMS:
        fold = model_folder(root, arm)
        epochs = sorted(int(p.stem.rsplit("_", 1)[-1]) for p in fold.glob("checkpoint_epoch_*.pth"))
        if epochs != expected:
            raise ValueError(f"{arm}: expected all 70 epoch checkpoints, got {epochs}")
        for epoch in expected:
            if not any((fold / "validation_previews" / f"epoch_{epoch:04d}").glob("*.png")):
                raise ValueError(f"{arm}: validation image missing for epoch {epoch}")
            if not (fold / "validation_metrics" / f"epoch_{epoch:04d}.json").is_file():
                raise ValueError(f"{arm}: validation metrics missing for epoch {epoch}")
        events = EventAccumulator(str(fold / "tensorboard"), size_guidance={"scalars": 0, "images": 0})
        events.Reload()
        tags = events.Tags()
        for tag in ("train/loss", "validation/patch_loss", "validation/full/dice"):
            if tag not in tags["scalars"] or len(events.Scalars(tag)) < 70:
                raise ValueError(f"{arm}: TensorBoard scalar {tag} has fewer than 70 epochs")
        image_tags = [tag for tag in tags["images"] if tag.startswith("validation/preview/")]
        if not image_tags or sum(len(events.Images(tag)) for tag in image_tags) < 70:
            raise ValueError(f"{arm}: TensorBoard validation images have fewer than 70 epochs")
        audit[arm] = dict(checkpoint_epochs=len(epochs), preview_epochs=70,
                          train_loss_events=len(events.Scalars("train/loss")),
                          validation_image_events=sum(len(events.Images(tag)) for tag in image_tags),
                          checkpoint_best=(fold / "checkpoint_best.pth").is_file(),
                          checkpoint_final=(fold / "checkpoint_final.pth").is_file())
        if not audit[arm]["checkpoint_best"] or not audit[arm]["checkpoint_final"]:
            raise ValueError(f"{arm}: best/final checkpoint missing")
    return audit


def run(root: Path) -> None:
    root = root.resolve()
    info = read_json(root / "run.json")
    entry = Path(info["code_root"]) / "acquisition_launch.py"
    with (root / "supervisor.lock").open("a+") as supervisor_lock:
        fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / "formal_scheduler.lock").open("a+") as scheduler_lock:
            fcntl.flock(scheduler_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (root / "formal_scheduler.pid").write_text(str(os.getpid()) + "\n")
            code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            status_path = root / "formal_scheduler_status.json"
            running: dict[str, dict] = {}
            completed: set[str] = set()
            last_snapshot = 0.0
            while True:
                for arm in ARMS:
                    if (root / arm / "results_audit.json").is_file() and read_json(root / arm / "status.json").get("stage") == "complete":
                        completed.add(arm)
                    if arm in completed or arm in running:
                        continue
                    pid_path = root / arm / "formal_worker.pid"
                    if pid_path.is_file():
                        pid = int(pid_path.read_text())
                        cmd = live_command(pid)
                        if cmd and "acquisition_launch.py worker" in cmd and f"--arm {arm}" in cmd and "--stop-after complete" in cmd:
                            gpu = read_json(root / arm / "formal_launch.json")["gpu"]
                            running[arm] = dict(pid=pid, gpu=gpu, process=None)
                for arm, row in list(running.items()):
                    process = row["process"]
                    code = process.poll() if process is not None else (None if live_command(row["pid"]) else 0)
                    if code is None:
                        continue
                    del running[arm]
                    if (code != 0 or not (root / arm / "results_audit.json").is_file()
                            or read_json(root / arm / "status.json").get("stage") != "complete"):
                        raise RuntimeError(f"{arm} formal worker failed; inspect worker and stage logs")
                    completed.add(arm)
                reserved = {row["gpu"] for row in running.values()}
                for arm in ARMS:
                    if arm in completed or arm in running:
                        continue
                    if not smoke_passed(root, arm):
                        status = read_json(root / arm / "status.json")
                        if status.get("state") == "failed":
                            raise RuntimeError(f"{arm} smoke failed; formal training withheld for that arm")
                        continue
                    free = available_gpus(reserved)
                    if not free:
                        break
                    gpu = free[0]
                    command = [info["phase3_python"], "-u", str(entry), "worker",
                               "--run-dir", str(root), "--arm", arm, "--gpu", str(gpu),
                               "--stop-after", "complete"]
                    with (root / arm / "formal_worker.log").open("a") as log:
                        process = subprocess.Popen(command, cwd=info["code_root"],
                                                   stdout=log, stderr=subprocess.STDOUT,
                                                   stdin=subprocess.DEVNULL, start_new_session=True)
                    running[arm] = dict(pid=process.pid, gpu=gpu, process=process)
                    reserved.add(gpu)
                    (root / arm / "formal_worker.pid").write_text(str(process.pid) + "\n")
                    write_json(root / arm / "formal_launch.json",
                               dict(command=command, pid=process.pid, gpu=gpu,
                                    launched_unix=time.time(), scheduler_sha256=code_hash))
                state = "formal_running" if running else ("waiting_for_gpu" if smoke_ready(root) else "waiting_for_smoke")
                serializable = {arm: {key: value for key, value in row.items() if key != "process"}
                                for arm, row in running.items()}
                write_json(status_path, dict(state=state, running=serializable,
                                             completed=sorted(completed), updated_unix=time.time(),
                                             scheduler_sha256=code_hash))
                if time.monotonic() - last_snapshot >= INTERVAL_SECONDS:
                    snapshot = progress_snapshot(root, serializable, completed)
                    write_json(root / "progress_latest.json", snapshot)
                    with (root / "progress_30min.jsonl").open("a") as stream:
                        stream.write(json.dumps(snapshot, allow_nan=False) + "\n")
                    last_snapshot = time.monotonic()
                if len(completed) == len(ARMS):
                    audit = audit_training_artifacts(root)
                    write_json(root / "formal_artifact_audit.json", audit)
                    sys.path.insert(0, info["code_root"])
                    from nnunetv2.phase3.acquisition_run import compare
                    compare(root)
                    write_json(status_path, dict(state="complete", completed=list(ARMS),
                                                 updated_unix=time.time(), scheduler_sha256=code_hash))
                    return
                time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.run_dir)
    except BaseException as exc:
        root = args.run_dir.resolve()
        if root.is_dir():
            write_json(root / "formal_scheduler_status.json",
                       dict(state="failed", error=str(exc), updated_unix=time.time()))
        raise


if __name__ == "__main__":
    main()
