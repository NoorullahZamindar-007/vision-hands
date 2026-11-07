"""Gesture primitives independent from MediaPipe's runtime."""

import math
from dataclasses import dataclass


def normalized_distance(a, b):
    """Return the 2D Euclidean distance between landmark-like objects."""
    return math.hypot(a.x - b.x, a.y - b.y)


def pinch_distance(hand, thumb_tip=4, index_tip=8):
    return normalized_distance(hand[thumb_tip], hand[index_tip])


@dataclass
class PinchTrigger:
    """One-shot pinch detector with hysteresis to reject landmark jitter."""

    enter_threshold: float = 0.055
    release_threshold: float = 0.070
    armed: bool = True

    def __post_init__(self):
        if not 0 < self.enter_threshold < self.release_threshold:
            raise ValueError("pinch thresholds must satisfy 0 < enter < release")
        self._initial_enter = self.enter_threshold
        self._initial_release = self.release_threshold

    def update(self, distances):
        # A single reading (or None for "no hand") is a natural call:
        # without this the generator below raises TypeError on a
        # non-iterable instead of treating it as one reading (or none).
        if distances is None:
            distances = []
        elif isinstance(distances, (int, float)) and not isinstance(distances, bool):
            distances = [distances]
        # A non-finite reading (NaN/inf from a glitchy landmark) is not a
        # pinch, but it is also not a release: comparing against it is
        # False either way, so a disarmed trigger would never re-arm and
        # pinch-to-switch would die silently. Drop such readings, which
        # makes them behave like a missing hand (no pinch, and a disarmed
        # trigger re-arms on the empty set, as before). A non-numeric
        # reading (None, or a stray string from a caller passing raw
        # landmark data) used to raise TypeError inside math.isfinite
        # and kill the gesture loop; drop those the same way.
        values = tuple(
            d for d in distances
            if isinstance(d, (int, float)) and not isinstance(d, bool) and math.isfinite(d)
        )
        if self.armed and any(d < self.enter_threshold for d in values):
            self.armed = False
            return True
        if not self.armed and (not values or all(d > self.release_threshold for d in values)):
            self.armed = True
        return False

    def adjust(self, delta):
        width = self.release_threshold - self.enter_threshold
        self.enter_threshold = min(0.20, max(0.005, self.enter_threshold + delta))
        self.release_threshold = min(0.22, self.enter_threshold + width)

    def reset(self):
        """Restore the thresholds this trigger was constructed with."""
        self.enter_threshold = self._initial_enter
        self.release_threshold = self._initial_release
        self.armed = True
