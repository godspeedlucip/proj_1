import json

import nibabel as nib
import numpy as np
import pytest

from nnunetv2.phase3.data_conversion import make_views, prepare_raw_fold
from nnunetv2.phase3.manifest import FoldSplit, load_manifest, load_splits
from nnunetv2.phase3.nifti_io import load_volume, read_nifti
from nnunetv2.phase3.prior_stats import fit_prior_stats, validate_prior_stats
from .fixtures import synthetic_dataset


def _replace_image(path, value):
    nii = nib.load(path)
    changed = nib.Nifti1Image(np.full(nii.shape, value, np.float32), nii.affine, nii.header)
    changed.header.set_xyzt_units("mm")
    nib.save(changed, path)


def test_stats_ignore_validation_and_test_intensities(tmp_path):
    mp, sp = synthetic_dataset(tmp_path)
    manifest = load_manifest(mp, require_labels=True)
    split = load_splits(sp)[0]
    before = fit_prior_stats(manifest, split, seed=2026)
    for case in manifest.cases:
        if case.group_id in ("g2", "g3"):
            _replace_image(case.image, 100000.0)
    after = fit_prior_stats(load_manifest(mp, require_labels=True), split, seed=2026)
    assert (before.lo, before.hi, before.input_digest) == (after.lo, after.hi, after.input_digest)
    assert before.sample_count == 2 * 65536
    with pytest.raises(ValueError):
        validate_prior_stats(before, manifest, FoldSplit(("g1",), ("g0", "g2"), ("g3",)))


def test_training_groups_have_equal_sample_weight_despite_case_count(tmp_path):
    mp, sp = synthetic_dataset(tmp_path, count=4)
    raw = json.loads(mp.read_text(encoding="utf-8"))
    raw["cases"][2]["group_id"] = "g0"
    raw["cases"][2]["source_model_ids"] = ["source_2"]
    raw["cases"][3]["group_id"] = "g0"
    raw["cases"][3]["source_model_ids"] = ["source_3"]
    raw["group_provenance"] = {"g0": "synthetic", "g1": "synthetic"}
    mp.write_text(json.dumps(raw), encoding="utf-8")
    sp.write_text(json.dumps({"schema_version": "fdk_stent_splits_v1",
                              "folds": {"0": {"train": ["g0", "g1"],
                                               "validation": ["g0"], "test": ["g1"]}}}), encoding="utf-8")
    # Isolate the sampler contract directly; split validation separately rejects this overlap.
    manifest = load_manifest(mp, require_labels=True)
    for case in manifest.cases:
        _replace_image(case.image, 0.0 if case.group_id == "g0" else 10.0)
    stats = fit_prior_stats(manifest, FoldSplit(("g0", "g1"), (), ()), seed=2026)
    assert stats.sample_count == 2 * 65536
    assert stats.lo == 0.0 and stats.hi == 10.0


def test_make_views_matches_legacy_formula_and_does_not_quantize(tmp_path):
    mp, sp = synthetic_dataset(tmp_path)
    manifest = load_manifest(mp, require_labels=True)
    stats = fit_prior_stats(manifest, load_splits(sp)[0], seed=2026)
    volume = load_volume(manifest.cases[0], require_label=True)
    views = make_views(volume, stats)
    raw = volume.image_raw
    expected_z = ((raw.astype(np.float64) - raw.mean(dtype=np.float64)) /
                  max(raw.std(dtype=np.float64, ddof=0), 1e-6)).astype(np.float32)
    expected_percentile = np.clip((raw - stats.lo) / (stats.hi - stats.lo), 0, 1).astype(np.float32)
    assert views.shape == (2, *raw.shape)
    np.testing.assert_allclose(views[0], expected_z, atol=1e-6)
    np.testing.assert_allclose(views[1], expected_percentile, atol=1e-6)
    assert views.dtype == np.float32


def test_prepare_fold_only_exposes_train_and_validation(tmp_path):
    mp, sp = synthetic_dataset(tmp_path / "input")
    manifest, splits = load_manifest(mp, require_labels=True), load_splits(sp)
    output = tmp_path / "prepared"
    prepared = prepare_raw_fold(manifest, splits, source_fold=0, dataset_id=901, output=output)
    assert prepared.source_fold == 0 and prepared.nnunet_fold == 0
    assert set(prepared.train_ids).isdisjoint(prepared.validation_ids)
    assert prepared.test_ids == ("case_3",)
    assert prepared.stats.seed == 2026
    raw = prepared.raw_dir
    assert {p.name for p in (raw / "imagesTr").glob("*.nii.gz")} == {
        f"case_{i}_{channel:04d}.nii.gz" for i in range(3) for channel in range(2)}
    assert {p.name for p in (raw / "labelsTr").glob("*.nii.gz")} == {f"case_{i}.nii.gz" for i in range(3)}
    assert not (raw / "imagesTs").exists()
    dataset = json.loads((raw / "dataset.json").read_text(encoding="utf-8"))
    assert dataset["labels"] == {"background": 0, "stent_struts": 1, "ignore": 2}
    assert dataset["overwrite_image_reader_writer"] == "FDKNiftiIO"
    assert len(dataset["channel_names"]) == 2
    split = json.loads((prepared.preprocessed_dir / "splits_final.json").read_text(encoding="utf-8"))
    assert split == [{"train": ["case_0", "case_1"], "val": ["case_2"]}]
    original = load_volume(manifest.cases[0], require_label=True)
    first, meta = read_nifti(raw / "imagesTr" / "case_0_0000.nii.gz")
    second, _ = read_nifti(raw / "imagesTr" / "case_0_0001.nii.gz")
    np.testing.assert_allclose(first, make_views(original, prepared.stats)[0], atol=1e-6)
    np.testing.assert_allclose(second, make_views(original, prepared.stats)[1], atol=1e-6)
    np.testing.assert_allclose(meta.affine, original.meta.affine)
    with pytest.raises(FileExistsError):
        prepare_raw_fold(manifest, splits, source_fold=0, dataset_id=901, output=output)
