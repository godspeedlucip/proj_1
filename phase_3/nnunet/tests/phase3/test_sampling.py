import json

import numpy as np
import pytest
import torch

from nnunetv2.phase3.config import resolve_config
from nnunetv2.phase3.data_conversion import prepare_raw_fold
from nnunetv2.phase3.dataloading import Phase3DataLoader
from nnunetv2.phase3.manifest import load_manifest, load_splits
from nnunetv2.phase3.planning import prepare_training_fold
from nnunetv2.phase3.storage import FDKDatasetBlosc2
from nnunetv2.phase3.transforms import build_transforms
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from .fixtures import make_config, synthetic_dataset


@pytest.fixture
def loader_source(tmp_path):
    mp, sp = synthetic_dataset(tmp_path / "input", count=5)
    raw = json.loads(mp.read_text(encoding="utf-8"))
    raw["cases"][4]["group_id"] = "g0"
    mp.write_text(json.dumps(raw), encoding="utf-8")
    prepared = prepare_raw_fold(load_manifest(mp, require_labels=True), load_splits(sp),
                                source_fold=0, dataset_id=903, output=tmp_path / "fold")
    cfg = resolve_config(make_config(run_kind="smoke", cnn__patch_size=[32, 32, 32],
                                     cnn__batch_size=1, training__max_optimizer_steps=4,
                                     training__updates_per_epoch=2))
    completed = prepare_training_fold(prepared, cfg)
    plans = PlansManager(str(completed.preprocessed_dir / f"{completed.plans_name}.json"))
    dataset_json = json.loads((completed.preprocessed_dir / "dataset.json").read_text(encoding="utf-8"))
    data_folder = completed.preprocessed_dir / plans.get_configuration("3d_fullres").data_identifier
    dataset = FDKDatasetBlosc2(str(data_folder), list(completed.train_ids))
    label_manager = plans.get_label_manager(dataset_json)
    return dataset, label_manager, completed.case_groups


def _loader(source, seed, *, training=True, ds=True):
    dataset, label_manager, groups = source
    return Phase3DataLoader(dataset, batch_size=1, patch_size=(32, 32, 32),
                            final_patch_size=(32, 32, 32), label_manager=label_manager,
                            transforms=build_transforms(training=training, deep_supervision=ds),
                            case_groups=groups, seed=seed)


def test_group_equal_case_probabilities(loader_source):
    loader = _loader(loader_source, 123)
    counts = {key: 0 for key in loader_source[0].identifiers}
    for _ in range(3000):
        counts[loader.get_indices()[0]] += 1
    assert 0.21 < counts["case_0"] / 3000 < 0.29
    assert 0.21 < counts["case_4"] / 3000 < 0.29
    assert 0.46 < counts["case_1"] / 3000 < 0.54


def test_mirror_padding_and_nearest_ds_keep_channels_aligned():
    image = torch.zeros((2, 32, 32, 32), dtype=torch.float32)
    image[:, 3, 5, 7] = 1
    seg = torch.zeros((1, 32, 32, 32), dtype=torch.int16)
    seg[0, 3, 5, 7] = 1
    seg[0, :2] = -1
    torch.manual_seed(17)
    transformed = build_transforms(training=True, deep_supervision=True)(image=image, segmentation=seg)
    targets = transformed["segmentation"]
    assert [tuple(t.shape[-3:]) for t in targets] == [
        (32, 32, 32), (16, 16, 16), (8, 8, 8), (4, 4, 4)]
    assert all(set(t.unique().tolist()) <= {0, 1, 2} for t in targets)
    assert torch.equal(transformed["image"][0] > 0, transformed["image"][1] > 0)
    assert torch.equal(transformed["image"][0] > 0, targets[0][0] == 1)
    assert int((targets[0] == 2).sum()) == 2 * 32 * 32


def test_loader_state_restores_patch_and_keys_after_interleaved_validation(loader_source):
    train = _loader(loader_source, 2026)
    validation = _loader(loader_source, 99, training=False)
    first = train.generate_train_batch()
    assert first["data"].shape == (1, 2, 32, 32, 32)
    assert [tuple(t.shape[-3:]) for t in first["target"]] == [
        (32, 32, 32), (16, 16, 16), (8, 8, 8), (4, 4, 4)]
    assert all(set(t.unique().tolist()) <= {0, 1, 2} for t in first["target"])
    state = train.state_dict()
    expected = [train.generate_train_batch() for _ in range(3)]
    _ = validation.generate_train_batch()
    resumed = _loader(loader_source, 2026)
    resumed.load_state_dict(state)
    actual = [resumed.generate_train_batch() for _ in range(3)]
    for left, right in zip(expected, actual):
        assert left["keys"] == right["keys"]
        assert torch.equal(left["data"], right["data"])
        assert all(torch.equal(a, b) for a, b in zip(left["target"], right["target"]))


def test_background_only_is_valid_but_all_ignore_is_rejected(loader_source):
    loader = _loader(loader_source, 42, ds=False)
    batch = loader.generate_train_batch()
    assert isinstance(batch["target"], torch.Tensor)
    assert bool((batch["target"] != 2).any())
    dataset, label_manager, groups = loader_source
    class BackgroundDataset:
        identifiers = ["background"]

        def load_case(self, identifier):
            data = np.zeros((2, 32, 32, 32), np.float32)
            seg = np.zeros((1, 32, 32, 32), np.int8)
            return data, seg, None, {"class_locations": {1: [], (-1, 0, 1): np.array([[0, 8, 8, 8]])}}

    background_loader = Phase3DataLoader(
        BackgroundDataset(), batch_size=1, patch_size=(32, 32, 32),
        final_patch_size=(32, 32, 32), label_manager=label_manager,
        transforms=build_transforms(training=False, deep_supervision=False),
        case_groups={"background": "g"}, seed=1)
    background = background_loader.generate_train_batch()
    assert torch.equal(background["target"], torch.zeros_like(background["target"]))

    class AllInvalidDataset:
        identifiers = ["invalid"]

        def load_case(self, identifier):
            data = np.zeros((2, 8, 8, 8), np.float32)
            seg = np.full((1, 8, 8, 8), 2, np.int8)
            return data, seg, None, {"class_locations": {1: [], (-1, 0, 1): []}}

    with pytest.raises(ValueError, match="valid"):
        Phase3DataLoader(AllInvalidDataset(), batch_size=1, patch_size=(32, 32, 32),
                         final_patch_size=(32, 32, 32), label_manager=label_manager,
                         transforms=build_transforms(training=False, deep_supervision=False),
                         case_groups={"invalid": "g"}, seed=1)
