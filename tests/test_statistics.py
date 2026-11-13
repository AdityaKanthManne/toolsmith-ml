import pytest
from toolsmith.statistics import paired_accuracy_delta


def test_paired_bootstrap_is_reproducible():
    args = ([True, False, False, True], [True, True, True, True])
    a = paired_accuracy_delta(*args, resamples=200, seed=11)
    b = paired_accuracy_delta(*args, resamples=200, seed=11)
    assert a == b
    assert a["delta"] == pytest.approx(.5)
    assert a["baseline_accuracy"] == pytest.approx(.5)
    assert a["tuned_accuracy"] == pytest.approx(1)


def test_pair_count_validation():
    with pytest.raises(ValueError):
        paired_accuracy_delta([True], [], resamples=100)
