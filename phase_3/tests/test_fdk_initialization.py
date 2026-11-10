import random
import numpy as np
import torch


def test_named_seed_and_rng_roundtrip():
    from phase_3.fdk_unet.initialization import seed_for, capture_rng_state, restore_rng_state, named_rng

    original = seed_for(2026, 'unet')
    assert seed_for(2026, 'extra') != original
    assert seed_for(2026, 'unet') == original
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng_state(state)
    with named_rng(2026, 'module'):
        torch.randn(300)
        np.random.random(200)
        random.random()
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
