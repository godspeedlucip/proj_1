import pytest

from phase_1.marker_pipeline.view_order import canonical_triplet_indices


def test_canonical_triplet_handles_wrap_and_manifest_order():
    assert canonical_triplet_indices([30, 330, 0]) == (1, 2, 0)
    assert canonical_triplet_indices([60, 0, 30]) == (1, 2, 0)
    assert canonical_triplet_indices([0, 30, 60]) == (0, 1, 2)


@pytest.mark.parametrize("angles", [[0, 0, 30], [0, 20, 60], [1, 31, 61], [0, 30, float("nan")], [0, 30]])
def test_canonical_triplet_rejects_invalid_angles(angles):
    with pytest.raises(ValueError):
        canonical_triplet_indices(angles)
