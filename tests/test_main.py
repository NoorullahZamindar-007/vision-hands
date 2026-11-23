"""Tests for portal_app.main() startup/exit paths, without a webcam.

Only the camera-drop exit is covered here: the rest of main() is the
live loop, which the render tests cover function by function.
"""

from pathlib import Path

import cv2
import numpy as np

import portal_app


class _FakeCapture:
    """A camera that opens, reports 1280x720, then stops returning frames."""

    def __init__(self, index=0):
        self.index = index

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 1280.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 720.0
        return 0.0

    def read(self):
        return False, None

    def release(self):
        return None


class _FakeLandmarker:
    def close(self):
        return None


def test_camera_stopping_mid_run_exits_nonzero(monkeypatch):
    # A camera that stops returning frames used to exit 0, as if the
    # user had pressed 'q' — while the camera-open failure already
    # returns 1. Supervisors and scripts could not tell the failure
    # apart from a clean quit.
    for name in ("CAM_INDEX", "FRAME_W", "FRAME_H", "FULLSCREEN",
                 "DETECT_DOWNSCALE", "VIGNETTE", "MODEL_PATH"):
        monkeypatch.setattr(portal_app, name, getattr(portal_app, name))
    monkeypatch.setattr(portal_app, "ensure_model_downloaded",
                        lambda: Path("fake.task"))
    monkeypatch.setattr(portal_app, "_detect_screen_resolution",
                        lambda: (1920, 1080))
    monkeypatch.setattr(portal_app.cv2, "VideoCapture", _FakeCapture)
    monkeypatch.setattr(portal_app.cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(portal_app.cv2, "resizeWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(portal_app.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(
        portal_app.mp_vision.HandLandmarker,
        "create_from_options",
        staticmethod(lambda options: _FakeLandmarker()),
    )

    assert portal_app.main(["--windowed"]) == 1


class _FakeFrameCapture(_FakeCapture):
    """A camera that returns one real frame, then stops returning frames."""

    def __init__(self, index=0):
        super().__init__(index)
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.reads == 1:
            return True, np.zeros((720, 1280, 3), dtype=np.uint8)
        return False, None


class _FailingLandmarker:
    def detect_for_video(self, image, timestamp_ms):
        raise RuntimeError("timestamp must be monotonically increasing")

    def close(self):
        return None


def test_detection_failure_mid_run_exits_nonzero(monkeypatch, capsys):
    # A detection failure (a repeated timestamp, a model error) used to
    # kill the app with a bare traceback instead of exiting 1 with a
    # clear message, like the camera failures do.
    for name in ("CAM_INDEX", "FRAME_W", "FRAME_H", "FULLSCREEN",
                 "DETECT_DOWNSCALE", "VIGNETTE", "MODEL_PATH"):
        monkeypatch.setattr(portal_app, name, getattr(portal_app, name))
    monkeypatch.setattr(portal_app, "ensure_model_downloaded",
                        lambda: Path("fake.task"))
    monkeypatch.setattr(portal_app, "_detect_screen_resolution",
                        lambda: (1920, 1080))
    monkeypatch.setattr(portal_app.cv2, "VideoCapture", _FakeFrameCapture)
    monkeypatch.setattr(portal_app.cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(portal_app.cv2, "resizeWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(portal_app.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(
        portal_app.mp_vision.HandLandmarker,
        "create_from_options",
        staticmethod(lambda options: _FailingLandmarker()),
    )

    assert portal_app.main(["--windowed"]) == 1
    assert "Hand detection failed" in capsys.readouterr().out
