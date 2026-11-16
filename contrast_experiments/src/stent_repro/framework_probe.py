"""Check matched CPU operators and losses with fixed, shared parameters."""

import argparse
import importlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from stent_repro.eulig2021.losses import soft_dice_A
from stent_repro.voth2022.losses import soft_dice_B


_EPSILON = 1e-5
_FACTORS = (2, 3)
_SMOOTHING_A = 1.0
_SMOOTHING_B = 0.7
_ATOL = 2e-6
_RTOL = 2e-6


def _inputs():
    images = ((np.arange(210, dtype=np.float32) % 29 - 14) / 13).reshape(2, 5, 7, 3)
    kernel = ((np.arange(54, dtype=np.float32) % 17 - 8) / 19).reshape(3, 3, 3, 2)
    bias = np.array([0.25, -0.375], dtype=np.float32)
    gamma, beta, mean, variance = np.array(
        [[1.5, 0.75], [0.02, -0.125], [0.3, -0.2], [0.5, 1.6]], dtype=np.float32
    )
    pixels = np.arange(420).reshape(10, 21, 2)
    target = np.stack([pixels % 7 == 0, pixels % 3 != 0]).astype(np.float32)
    return images, kernel, bias, gamma, beta, mean, variance, target


def _numpy_outputs(images, kernel, bias, gamma, beta, mean, variance, target):
    n, height, width, _ = images.shape
    padded = np.pad(images, ((0, 0), (1, 1), (1, 1), (0, 0)))
    conv = np.empty((n, height, width, kernel.shape[-1]), dtype=np.float32)
    for row in range(height):
        for column in range(width):
            patch = padded[:, row:row + 3, column:column + 3, :, None]
            conv[:, row, column, :] = np.sum(patch * kernel, axis=(1, 2, 3)) + bias
    normalized = (conv - mean) * gamma / np.sqrt(variance + _EPSILON) + beta
    relu = np.maximum(normalized, 0)
    upsampled = relu.repeat(_FACTORS[0], axis=1).repeat(_FACTORS[1], axis=2)
    probabilities = 1 / (1 + np.exp(-upsampled))
    overlap = 2 * np.sum(probabilities * target)
    total = np.sum(probabilities) + np.sum(target)
    return {
        'conv': conv, 'batch_norm': normalized, 'relu': relu, 'upsampled': upsampled,
        'probabilities': probabilities,
        'soft_dice_loss_a': 1 - (overlap + _SMOOTHING_A) / (total + _SMOOTHING_A),
        'soft_dice_loss_b': 1 - (overlap + _SMOOTHING_B) / (total + _SMOOTHING_B),
        'mae_loss': np.abs(probabilities - target).mean(),
    }


def _torch_outputs(images, kernel, bias, gamma, beta, mean, variance, target):
    conv_layer = nn.Conv2d(3, 2, kernel_size=3, padding=1, device='cpu', dtype=torch.float32)
    bn = nn.BatchNorm2d(2, eps=_EPSILON, device='cpu', dtype=torch.float32).eval()
    with torch.inference_mode():
        conv_layer.weight.copy_(torch.from_numpy(kernel.transpose(3, 2, 0, 1)))
        conv_layer.bias.copy_(torch.from_numpy(bias))
        bn.weight.copy_(torch.from_numpy(gamma))
        bn.bias.copy_(torch.from_numpy(beta))
        bn.running_mean.copy_(torch.from_numpy(mean))
        bn.running_var.copy_(torch.from_numpy(variance))
        conv = conv_layer(torch.from_numpy(images).permute(0, 3, 1, 2))
        normalized = bn(conv)
        relu = F.relu(normalized)
        upsampled = F.interpolate(relu, scale_factor=_FACTORS, mode='nearest')
        probabilities = torch.sigmoid(upsampled)
        truth = torch.from_numpy(target).permute(0, 3, 1, 2)
        outputs = {
            'conv': conv, 'batch_norm': normalized, 'relu': relu, 'upsampled': upsampled,
            'probabilities': probabilities,
            'soft_dice_loss_a': soft_dice_A(probabilities, truth),
            'soft_dice_loss_b': soft_dice_B(probabilities, truth, _SMOOTHING_B),
            'mae_loss': F.l1_loss(probabilities, truth),
        }
    return {key: (val.permute(0, 2, 3, 1) if val.ndim == 4 else val).numpy()
            for key, val in outputs.items()}


def _tensorflow_outputs(tf, images, kernel, bias, gamma, beta, mean, variance, target):
    with tf.device('/CPU:0'):
        conv = tf.nn.bias_add(tf.nn.conv2d(images, kernel, strides=1, padding='SAME'), bias)
        bn = tf.keras.layers.BatchNormalization(axis=-1, epsilon=_EPSILON, dtype='float32')
        bn.build(conv.shape)
        bn.set_weights([gamma, beta, mean, variance])
        normalized = bn(conv, training=False)
        relu = tf.nn.relu(normalized)
        upsampled = tf.keras.layers.UpSampling2D(
            size=_FACTORS, data_format='channels_last', interpolation='nearest', dtype='float32'
        )(relu)
        probabilities = tf.math.sigmoid(upsampled)
        overlap = 2 * tf.reduce_sum(probabilities * target)
        total = tf.reduce_sum(probabilities) + tf.reduce_sum(target)
        outputs = {
            'conv': conv, 'batch_norm': normalized, 'relu': relu, 'upsampled': upsampled,
            'probabilities': probabilities,
            'soft_dice_loss_a': 1 - (overlap + _SMOOTHING_A) / (total + _SMOOTHING_A),
            'soft_dice_loss_b': 1 - (overlap + _SMOOTHING_B) / (total + _SMOOTHING_B),
            'mae_loss': tf.reduce_mean(tf.abs(probabilities - target)),
        }
    return {key: val.numpy() for key, val in outputs.items()}


def _comparison(reference, pytorch):
    checks = {}
    for key, expected in reference.items():
        expected = np.asarray(expected)
        actual = pytorch[key]
        comparable = (actual.shape == expected.shape and np.isfinite(actual).all()
                      and np.isfinite(expected).all())
        check = {
            'shape': list(expected.shape), 'pytorch_shape': list(actual.shape),
            'passed': bool(comparable and np.allclose(actual, expected, atol=_ATOL, rtol=_RTOL)),
            'max_abs_error': float(np.abs(actual - expected).max()) if comparable else None,
        }
        if comparable and expected.ndim == 0:
            check['reference_value'] = float(expected)
            check['pytorch_value'] = float(actual)
        checks[key] = check
    return {'status': 'passed' if all(check['passed'] for check in checks.values()) else 'failed',
            'checks': checks}


def run_framework_probe(*, require_tensorflow=False):
    """Return stage-by-stage same-weight parity in a deterministic CPU fixture.

    A missing TensorFlow package or import dependency is reported as
    unavailable. With ``require_tensorflow=True``, unavailable TF causes
    overall failure. The NumPy comparison always runs.
    """
    params = _inputs()
    pytorch = _torch_outputs(*params)
    comparisons = {'numpy_vs_pytorch': _comparison(_numpy_outputs(*params), pytorch)}
    versions = {'numpy': np.__version__, 'pytorch': str(torch.__version__)}
    try:
        tensorflow = importlib.import_module('tensorflow')
    except (ImportError, OSError) as exc:
        comparisons['tensorflow_vs_pytorch'] = {
            'status': 'unavailable', 'reason': f'{type(exc).__name__}: {exc}',
        }
    else:
        versions['tensorflow'] = tensorflow.__version__
        comparisons['tensorflow_vs_pytorch'] = _comparison(
            _tensorflow_outputs(tensorflow, *params), pytorch
        )
    accepted = {'passed'} if require_tensorflow else {'passed', 'unavailable'}
    return {
        'status': 'passed' if all(item['status'] in accepted for item in comparisons.values()) else 'failed',
        'device': 'cpu', 'dtype': 'float32', 'versions': versions,
        'atol': _ATOL, 'rtol': _RTOL,
        'conventions': {
            'conv': '3x3 stride-1 cross-correlation, symmetric zero padding, bias',
            'tf_kernel_layout': 'HWIO', 'torch_kernel_layout': 'OIHW',
            'comparison_activation_layout': 'NHWC',
            'batch_norm': 'eval with copied gamma, beta, running mean and variance',
            'batch_norm_epsilon': _EPSILON, 'nearest_scale_hw': list(_FACTORS),
            'dice_reduction': 'global over batch, channels and spatial dimensions',
            'dice_smoothing_a': _SMOOTHING_A, 'dice_smoothing_b': _SMOOTHING_B,
            'mae_reduction': 'global mean',
        },
        'comparisons': comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-tensorflow', action='store_true')
    args = parser.parse_args()
    report = run_framework_probe(require_tensorflow=args.require_tensorflow)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
