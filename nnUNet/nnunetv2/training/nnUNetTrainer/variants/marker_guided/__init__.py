from .marker_guided_network import MarkerGuidedPlainConvUNet
from .residual_prior_adapter import ResidualPriorAdapter


def __getattr__(name):
    if name in ("nnUNetTrainerMarkerGuidedFeatureFusion", "nnUNetTrainerMarkerGuidedSmoke"):
        from .marker_guided_trainer import nnUNetTrainerMarkerGuidedFeatureFusion, nnUNetTrainerMarkerGuidedSmoke
        return {"nnUNetTrainerMarkerGuidedFeatureFusion": nnUNetTrainerMarkerGuidedFeatureFusion,
                "nnUNetTrainerMarkerGuidedSmoke": nnUNetTrainerMarkerGuidedSmoke}[name]
    raise AttributeError(name)


__all__ = ["MarkerGuidedPlainConvUNet", "ResidualPriorAdapter", "nnUNetTrainerMarkerGuidedFeatureFusion",
           "nnUNetTrainerMarkerGuidedSmoke"]
