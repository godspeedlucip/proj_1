"""Synchronized full-detector inputs and floating-point supervision for Dataset504."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset



def prior_key(points, shape):
    key = json.dumps([list(shape), points, "marker_prior_v1_oracle_r0"], sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()


def oracle_prior(points, shape):
    from tools.marker_guided.build_prior_channels import build_marker_channels
    view = {"marker_xy": points, "pair_status": "reliable", "error_radius_px": [0, 0]}
    return build_marker_channels(view, shape)[:2]


def prepare_prior_cache(folder, records, shape):
    folder.mkdir(parents=True, exist_ok=True)
    for row in records:
        points = row["marker_xy"]
        file = folder / (prior_key(points, shape) + ".npz")
        if not file.exists():
            point, band = oracle_prior(points, shape)
            with file.with_suffix(".part").open("wb") as f:
                np.savez_compressed(f, point=point, band=band)
            file.with_suffix(".part").replace(file)


@lru_cache(maxsize=4)
def _read_prior(file):
    with np.load(file, allow_pickle=False) as maps:
        return maps["point"], maps["band"]


def prepare_image(image, points, marker_enabled, cache_dir=None):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) == 0 or not np.isfinite(image).all():
        raise ValueError("Expected one finite 2D detector image")
    height, width = image.shape
    channels = np.zeros((4, height, width), dtype=np.float32)
    channels[0] = (image - image.mean(dtype=np.float64)) / max(image.std(dtype=np.float64), 1e-8)
    if marker_enabled:
        file = cache_dir / (prior_key(points, image.shape) + ".npz") if cache_dir else None
        channels[1], channels[2] = _read_prior(file) if file and file.is_file() else oracle_prior(points, image.shape)
        channels[3].fill(1)
    return channels


def pad_frame(array, *, flip_y=False, flip_x=False):
    if flip_y:
        array = np.flip(array, axis=-2)
    if flip_x:
        array = np.flip(array, axis=-1)
    h, w = array.shape[-2:]
    return torch.from_numpy(np.pad(array, ((0, 0), (0, -h % 256), (0, -w % 256))))


class FullFrameDataset(Dataset):
    def __init__(self, root, records, scale, marker_enabled, flips=None):
        self.root, self.records, self.scale = root, records, scale
        self.marker_enabled, self.flips = marker_enabled, flips

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = tifffile.imread(self.root / row["image"])
        inputs = prepare_image(image, row["marker_xy"], self.marker_enabled, self.root / "prior_cache")
        target = tifffile.imread(self.root / row["mask"])
        depth = tifffile.imread(self.root / row["depth"])
        if target.shape != image.shape or depth.shape != image.shape or depth.dtype != np.float32:
            raise ValueError(f"Misaligned joint labels for {row['sample_id']}")
        flip_y, flip_x = self.flips[index] if self.flips is not None else (False, False)
        def tensor(array):
            return pad_frame(array, flip_y=flip_y, flip_x=flip_x)
        return {
            "data": tensor(inputs),
            "target": tensor(target.astype(np.uint8, copy=False)[None]),
            "depth_target": tensor((depth / np.float32(self.scale))[None]),
            "valid_pixels": tensor(np.ones((1, *image.shape), dtype=bool)),
            "keys": row["sample_id"],
        }


def _train_epoch(records, seed, epoch, count):
    # Traverse each shuffled cycle once, including when an epoch straddles two cycles.
    indices = []
    offset = epoch * count
    while len(indices) < count:
        cycle, start = divmod(offset, len(records))
        order = np.random.default_rng([seed, cycle, 0]).permutation(len(records))
        selected = order[start:start + count - len(indices)]
        indices.extend(selected.tolist())
        offset += len(selected)
    flips = np.random.default_rng([seed, epoch, 1]).integers(2, size=(count, 2))
    return [records[i] for i in indices], flips


def make_loaders(manifest_path, seed, smoke, epoch, marker_enabled):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    train = [s for s in manifest["samples"] if s["split"] == "train"]
    val = [s for s in manifest["samples"] if s["split"] == "val"]
    if smoke:
        val = val[:2]
    train, flips = _train_epoch(train, int(seed), int(epoch), 4 if smoke else 250)
    def loader(rows, augmentation):
        dataset = FullFrameDataset(path.parent, rows, manifest["target_scale"], marker_enabled, augmentation)
        return DataLoader(dataset, batch_size=1, num_workers=2, prefetch_factor=1, pin_memory=True)
    # The iterators are created lazily: validation workers should not fork while training is running.
    def iterate(rows, augmentation):
        yield from loader(rows, augmentation)
    return iterate(train, flips), iterate(val, None), len(val)
