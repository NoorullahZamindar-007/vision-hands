import numpy as np

import portal_app


def test_effects_preserve_shape_and_dtype():
    image = np.full((80, 120, 3), 127, dtype=np.uint8)
    for effect in portal_app.EFFECTS:
        result = effect(image)
        assert result.shape == image.shape
        assert result.dtype == np.uint8


def test_fill_window_covers_requested_dimensions():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = portal_app._fill_window(image, 800, 800)
    assert result.shape == (800, 800, 3)


def test_vignette_shape_and_type():
    result = portal_app.make_vignette_layer(320, 180)
    assert result.shape == (180, 320, 3)
    assert result.dtype == np.uint8


def test_quad_smoother_uses_exponential_average():
    smoother = portal_app.QuadSmoother(alpha=0.5)
    first = np.zeros((4, 2), dtype=np.float32)
    second = np.full((4, 2), 10, dtype=np.float32)
    smoother.update(first)
    assert np.allclose(smoother.update(second), 5)
