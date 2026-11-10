import torch
from torch.nn import functional as F

from phase_3.fdk_unet.model import UNet3D


def test_unet_preserves_odd_volume_size_and_trains_all_levels():
    model = UNet3D(in_channels=1, num_classes=2, widths=(4, 8, 12, 16, 20))
    image = torch.randn(1, 1, 33, 35, 37)
    target = torch.zeros(1, 33, 35, 37, dtype=torch.long)
    target[:, 9:25, 12:23, 15:21] = 1

    logits = model(image)
    assert logits.shape == (1, 2, 33, 35, 37)

    F.cross_entropy(logits, target).backward()
    assert model.encoder[0][0].weight.grad is not None
    assert model.encoder[-1][0].weight.grad is not None
    assert model.head.weight.grad is not None


def test_encoder_exposes_stable_scales_for_later_feature_fusion():
    model = UNet3D(in_channels=1, num_classes=2, widths=(4, 8, 12, 16, 20))
    with torch.no_grad():
        features = model.encode(torch.randn(1, 1, 32, 32, 32))

    assert [feature.shape[2:] for feature in features] == [
        (32, 32, 32),
        (16, 16, 16),
        (8, 8, 8),
        (4, 4, 4),
        (2, 2, 2),
    ]
