import copy
import itertools
from types import SimpleNamespace

import pytest

from nnunetv2.phase3.cli import load_preset, main
from nnunetv2.phase3.data_conversion import prepare_raw_fold
from nnunetv2.phase3.launch import launch_training, model_folder
from nnunetv2.phase3.manifest import load_manifest, load_splits
from nnunetv2.phase3.planning import prepare_training_fold
from .fixtures import synthetic_dataset


PRESETS = ("baseline", "swin", "tubular", "projection", "swin_tubular",
           "swin_projection", "tubular_projection", "full", "random_prior",
           "h00", "h01", "h10", "h11", "no_ds", "smoke")
CORE = PRESETS[:8]


def test_all_presets_resolve_and_eight_core_switches_cover_cube():
    configs = {name: load_preset(name) for name in PRESETS}
    assert len(configs) == 15
    combinations = {(cfg.swin.enabled, cfg.tubular.enabled, cfg.projection.enabled)
                    for name, cfg in configs.items() if name in CORE}
    assert combinations == set(itertools.product((False, True), repeat=3))
    for name in CORE:
        cfg = configs[name]
        assert cfg.run_kind == "formal"
        assert cfg.cnn == configs["baseline"].cnn
        assert cfg.training == configs["baseline"].training
        assert cfg.deep_supervision == configs["baseline"].deep_supervision
    assert configs["smoke"].run_kind == "smoke"
    assert configs["random_prior"].swin.init == "random"


def test_h_matrix_and_no_ds_contract():
    configs = {name: load_preset(name) for name in PRESETS}
    for name in ("h00", "h01"):
        assert configs[name].swin.execute_through == "layers2"
        assert not configs[name].fusion.bottleneck_enabled
    for name in ("h10", "h11"):
        assert configs[name].swin.execute_through == "layers3"
        assert configs[name].fusion.bottleneck_enabled
    assert configs["h11"].fusion == configs["full"].fusion
    assert not configs["h00"].fusion.decoder_conditioning
    assert configs["h01"].fusion.decoder_conditioning
    assert not configs["h10"].fusion.decoder_conditioning
    assert configs["h11"].fusion.decoder_conditioning
    assert not configs["no_ds"].deep_supervision.enabled


def test_unknown_preset_and_conflicting_config_fail():
    with pytest.raises(ValueError, match="Unknown preset"):
        load_preset("missing")
    with pytest.raises(ValueError, match="conflict"):
        load_preset("baseline", overrides={"swin": {"enabled": True}})
    with pytest.raises(SystemExit):
        main(["train", "--unknown-flag"])


def test_launcher_uses_isolated_env_and_rejects_existing_result(tmp_path, monkeypatch):
    mp, sp = synthetic_dataset(tmp_path / "中文 空格" / "input")
    prepared = prepare_raw_fold(load_manifest(mp, require_labels=True), load_splits(sp),
                                source_fold=0, dataset_id=913, output=tmp_path / "中文 空格" / "prepared")
    config = load_preset("smoke")
    prepared = prepare_training_fold(prepared, config)
    seen = {}
    def fake_run(args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("nnunetv2.phase3.launch.subprocess.run", fake_run)
    monkeypatch.setenv("nnUNet_raw", "wrong")
    monkeypatch.setenv("PYTHONPATH", "wrong-package")
    assert launch_training(prepared, config, seed=config.seed, device="cpu",
                           prior_artifact=None, resume=False) == 0
    assert isinstance(seen["args"], list) and "--dataset" in seen["args"]
    assert seen["kwargs"]["env"]["nnUNet_raw"] == str(prepared.raw_root)
    assert seen["kwargs"]["env"]["PYTHONPATH"].endswith("phase_3\\nnunet")
    assert not model_folder(prepared).exists()
    model_folder(prepared).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        launch_training(prepared, config, seed=config.seed, device="cpu",
                        prior_artifact=None, resume=False)
    with pytest.raises(FileNotFoundError):
        launch_training(prepared, config, seed=config.seed, device="cpu",
                        prior_artifact=None, resume=True)
