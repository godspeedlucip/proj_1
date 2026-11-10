"""Standalone FDK segmentation; ordinary models do not import MONAI."""
from .model import UNet3D
from .plain_unet import PlainUNet3D

__all__ = ["UNet3D", "PlainUNet3D"]
