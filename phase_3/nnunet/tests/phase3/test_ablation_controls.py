import pytest
import torch

from nnunetv2.phase3.config import resolve_config
from nnunetv2.phase3.data_conversion import prepare_raw_fold
from nnunetv2.phase3.losses import Phase3CompositeLoss
from nnunetv2.phase3.manifest import load_manifest, load_splits
from nnunetv2.phase3.network import build_phase3_network
from nnunetv2.phase3.planning import make_phase3_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from .fixtures import make_config, synthetic_dataset


@pytest.mark.parametrize("unetr,joint", [(True, True), (False, True),
                                          (True, False), (False, False)])
def test_gdl_ablation_switches_reach_network_and_loss(tmp_path, unetr, joint):
    torch.set_num_threads(2)
    manifest_path, splits_path = synthetic_dataset(tmp_path / "source")
    prepared = prepare_raw_fold(load_manifest(manifest_path, require_labels=True),
                                load_splits(splits_path), source_fold=0,
                                dataset_id=934, output=tmp_path / "prepared")
    config = resolve_config(make_config(
        run_kind="smoke", cnn__patch_size=[32, 32, 32], cnn__batch_size=1,
        training__max_optimizer_steps=4, training__updates_per_epoch=2,
        segmentation__loss="generalized_dice_bce",
        swin__enabled=unetr, swin__execute_through="layers3" if unetr else "layers2",
        fusion__enabled=unetr, fusion__bottleneck_enabled=unetr,
        tubular__enabled=joint, projection__enabled=joint,
        projection__weight=0.1 if joint else 0.0))
    manager = PlansManager(make_phase3_plans(prepared, config))
    network = build_phase3_network(manager, manager.get_configuration("3d_fullres"), 2, 2)
    loss = Phase3CompositeLoss(config)

    assert (network.prior is not None) is unetr
    assert (network.bottleneck is not None) is unetr
    assert bool(network.skip_fusions) is unetr
    assert (network.tubular is not None) is joint
    assert (loss.projection is not None) is joint
    assert loss.segmentation.__class__.__name__ == "MaskedGeneralizedDiceBCE"

    with torch.no_grad():
        outputs = network(torch.zeros((1, 2, 32, 32, 32)))
        targets = [torch.zeros((1, 1, *output.shape[2:])) for output in outputs]
        breakdown = loss.breakdown(outputs, targets)
    assert torch.isfinite(breakdown.total)
    if not joint:
        assert breakdown.projection.item() == 0
