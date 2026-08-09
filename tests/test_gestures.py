from types import SimpleNamespace

import pytest

from handportal.gestures import PinchTrigger, normalized_distance, pinch_distance


def point(x, y):
    return SimpleNamespace(x=x, y=y)


def test_normalized_distance():
    assert normalized_distance(point(0, 0), point(0.03, 0.04)) == pytest.approx(0.05)


def test_pinch_distance_uses_thumb_and_index():
    hand = [point(0, 0) for _ in range(21)]
    hand[4], hand[8] = point(0.1, 0.2), point(0.4, 0.6)
    assert pinch_distance(hand) == pytest.approx(0.5)


def test_trigger_fires_once_until_full_release():
    trigger = PinchTrigger(0.055, 0.070)
    assert trigger.update([0.05]) is True
    assert trigger.update([0.04]) is False
    assert trigger.update([0.06]) is False
    assert trigger.update([0.08]) is False
    assert trigger.update([0.05]) is True


def test_trigger_validates_thresholds():
    with pytest.raises(ValueError):
        PinchTrigger(0.08, 0.07)
