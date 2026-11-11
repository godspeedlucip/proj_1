import json
from pathlib import Path
import subprocess
import sys

import pytest

from nnunetv2.phase3.config import config_digest, resolve_config
from nnunetv2.phase3.runtime import verify_runtime
from .fixtures import make_config


def test_formal_config_roundtrips_without_mutable_aliases():
    cfg = resolve_config(make_config())
    assert cfg.cnn.channels == (32, 64, 128, 256, 320)
    assert cfg.projection.operator_version == "label_parallel_v1"
    assert cfg.deep_supervision_weights == (4/7, 2/7, 1/7, 0)
    as_dict = cfg.to_dict()
    as_dict["cnn"]["channels"][0] = 1
    assert cfg.cnn.channels == (32, 64, 128, 256, 320)
    assert config_digest(cfg) == config_digest(resolve_config(cfg.to_dict()))


@pytest.mark.parametrize("change", [
    {"unknown_field": 1},
    {"projection__weight": float("nan")},
    {"projection__weight": -0.1},
    {"projection__enabled": False},
    {"cnn__dimension": "2d"},
    {"cnn__dimension": "3d_cascade_fullres"},
    {"cnn__num_output_channels": 3},
    {"cnn__patch_size": [96, 128, 128]},
])
def test_rejects_unsupported_or_conflicting_config(change):
    with pytest.raises(ValueError):
        resolve_config(make_config(**change))


def test_smoke_requires_distinct_budget_and_identity():
    with pytest.raises(ValueError):
        resolve_config(make_config(run_kind="formal", cnn__patch_size=[32, 32, 32]))
    with pytest.raises(ValueError):
        resolve_config(make_config(run_kind="smoke", cnn__patch_size=[32, 32, 32]))
    cfg = resolve_config(make_config(run_kind="smoke", cnn__patch_size=[32, 32, 32],
                                     cnn__batch_size=1, training__max_optimizer_steps=4,
                                     training__updates_per_epoch=2))
    assert cfg.run_kind == "smoke"


def test_import_root_checked_in_fresh_subprocess():
    root = Path(__file__).resolve().parents[2]
    report = verify_runtime(root)
    assert Path(report["package_file"]).is_relative_to(root)
    script = "from nnunetv2.phase3.runtime import verify_runtime; verify_runtime(r'{}')".format(root.parent)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode != 0
    assert "root" in result.stderr.lower()
