import json
import os
from pathlib import Path

import pytest
import torch

from nnunetv2.phase3.prior_weights import file_sha256, load_ssl_weights, prepare_prior_artifact
from nnunetv2.phase3.swin_prior import SwinPriorEncoder


def test_audited_ssl_load_and_reject_shape_conflict(tmp_path):
    encoder = SwinPriorEncoder(execute_through="layers2")
    weights = tmp_path / "weights.pt"
    state = encoder.swinViT.state_dict()
    torch.save({"state_dict": {f"module.swin_vit.{k}": v for k, v in state.items()}}, weights)
    report = load_ssl_weights(encoder, weights, expected_sha256=file_sha256(weights))
    assert report.loaded_parameter_fraction == 1.0
    artifact = prepare_prior_artifact(weights, tmp_path / "artifact.json")
    assert json.loads(artifact.read_text(encoding="utf-8"))["sha256"] == report.sha256
    bad = dict(state)
    bad["patch_embed.proj.weight"] = bad["patch_embed.proj.weight"][:1]
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="shape mismatch"):
        load_ssl_weights(encoder, tmp_path / "bad.pt")


@pytest.mark.prior_artifact
def test_official_ssl_weights_match_complete_monai_backbone():
    default = Path(__file__).resolve().parents[4] / "tmp/fdk_v3/official_ssl/model_swinvit.pt"
    weights = Path(os.getenv("PHASE3_SSL_WEIGHTS", str(default)))
    if not weights.is_file():
        pytest.skip("Official SSL weights not present")
    report = load_ssl_weights(SwinPriorEncoder(execute_through="layers3"), weights)
    assert report.loaded_parameter_fraction == 1.0
    assert not report.missing and not report.unexpected
    assert len(report.loaded) > 100
