import numpy as np
import torch

from phase_1.marker_pipeline.heatmap import make_heatmap
from phase_1.marker_pipeline.models.unet2d import MarkerUNet2D
from phase_1.marker_pipeline.train import train_detector, predict_heatmap


def test_unet_preserves_non_multiple_of_eight_shape():
    model = MarkerUNet2D(widths=(8, 16, 32, 64))
    assert model(torch.zeros(1, 1, 31, 37)).shape == (1, 1, 31, 37)


def test_heatmap_contains_two_distinct_peaks():
    h = make_heatmap([[5.2, 6.1], [24.0, 20.0]], (32, 32), sigma=1.2)
    assert h.shape == (32, 32)
    assert h[6, 5] > .9 and h[20, 24] > .9
    assert h[6, 24] < .1


def test_training_updates_weights_and_checkpoint_predicts(tmp_path):
    class V:
        image=np.zeros((32,32),dtype=np.float32)
        marker_xy=np.array([[8.,9.],[23.,22.]])
    class C:
        group_id="g1"; views=[V(), V(), V()]
    config={"epochs":1,"batch_size":3,"widths":[8,16,32,64],"seed":4}
    result=train_detector([C()],config,tmp_path)
    assert np.isfinite(result["loss"])
    pred=predict_heatmap(tmp_path/"checkpoint.pt", V.image)
    assert pred.shape == V.image.shape and np.isfinite(pred).all()
