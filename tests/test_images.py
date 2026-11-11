import numpy as np
import pytest

import portal_app


def test_effects_preserve_shape_and_dtype():
    image = np.full((80, 120, 3), 127, dtype=np.uint8)
    for effect in portal_app.EFFECTS:
        result = effect(image)
        assert result.shape == image.shape
        assert result.dtype == np.uint8


def test_effects_handle_realistic_noise_image():
    # The flat-gray image in test_effects_preserve_shape_and_dtype never
    # exercises the real paths: edge_mesh finds zero edges,
    # grayscale_sketch's cv2.divide never sees a near-zero denominator,
    # pixel_mosaic's seam lines are invisible, and np.roll wrap-around in
    # invert_glitch/chromatic is undetectable. Noise exercises all of them.
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(80, 120, 3), dtype=np.uint8)
    for effect in portal_app.EFFECTS:
        result = effect(image)
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        assert np.isfinite(result.astype(np.float64)).all()
        assert not np.array_equal(result, image)


def test_font_respects_bold_flag():
    if not portal_app.PIL_AVAILABLE or portal_app.FONT_PATH is None:
        pytest.skip("no PIL fonts available")
    bold = portal_app._font(12, bold=True)
    regular = portal_app._font(12)
    assert bold is not None and regular is not None
    assert bold.getname()[1] == "Bold"
    assert regular.getname()[1] == "Book"


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
