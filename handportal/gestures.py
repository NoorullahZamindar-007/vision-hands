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
        values = tuple(distances)
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
