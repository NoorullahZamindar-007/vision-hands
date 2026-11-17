"""Tests for the portal compositing path — the code that produces the
app's entire visual output.

These functions had zero coverage: effects, gestures, CLI, model
download and fonts were all tested, but the crop -> effect -> feathered
mask -> glow -> brackets -> particles composition was only exercised by
running the webcam app, so a compositing regression shipped silently.
All tests below are webcam-independent (synthetic frames only).
"""

from types import SimpleNamespace

import pytest
import numpy as np

import portal_app


def _quad(x0=50, y0=50, x1=150, y1=150):
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
    )


def _solid(value):
    return lambda crop: np.full_like(crop, value)


def test_render_portal_changes_inside_quad_only():
    frame = np.full((200, 200, 3), 100, dtype=np.uint8)
    before = frame.copy()
    portal_app.render_portal_into_quad(frame, _solid(200), _quad())

    # Well inside the quad the effect lands (mask is 255 there).
    assert np.all(frame[100, 100] == 200)
    # Far outside the quad — beyond the feather band — nothing changes.
    assert np.array_equal(frame[5, 5], before[5, 5])
    assert np.array_equal(frame[195, 195], before[195, 195])
    assert np.array_equal(frame[5, 195], before[5, 195])


def test_render_portal_alpha_zero_leaves_frame_unchanged():
    frame = np.full((200, 200, 3), 100, dtype=np.uint8)
    before = frame.copy()
    portal_app.render_portal_into_quad(frame, _solid(200), _quad(), alpha=0.0)

    assert np.array_equal(frame, before)


def test_render_portal_crossfade_blend_endpoints():
    # blend=0 must show the previous effect, blend=1 the new one;
    # the centre pixel is far from the feathered edge so the mask is 255.
    for blend, expected in ((0.0, 50), (1.0, 200)):
        frame = np.full((200, 200, 3), 100, dtype=np.uint8)
        portal_app.render_portal_into_quad(
            frame, _solid(200), _quad(), prev_effect_fn=_solid(50), blend=blend
        )
        assert np.all(frame[100, 100] == expected)


def test_render_portal_crossfade_blend_clamped():
    # blend is a 0..1 weight: above 1 it overshot the new effect to
    # white (255) and below 0 it undershot the old one to black,
    # because addWeighted saturates instead of erroring.
    for blend, expected in ((1.5, 200), (-0.5, 50)):
        frame = np.full((200, 200, 3), 100, dtype=np.uint8)
        portal_app.render_portal_into_quad(
            frame, _solid(200), _quad(), prev_effect_fn=_solid(50), blend=blend
        )
        assert np.all(frame[100, 100] == expected)


def _fake_hand(tip_x, tip_y):
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    landmarks[portal_app.INDEX_TIP] = SimpleNamespace(x=tip_x, y=tip_y)
    return landmarks


def test_build_quad_orders_corners_tl_tr_br_bl():
    quad = portal_app.build_quad_from_index_fingers(
        _fake_hand(0.2, 0.3), _fake_hand(0.6, 0.7), 200, 100
    )

    # Tips at (40, 30) and (120, 70) on a 200x100 frame.
    expected = np.array(
        [[40, 30], [120, 30], [120, 70], [40, 70]], dtype=np.float32
    )
    assert np.allclose(quad, expected)


def test_build_quad_enforces_minimum_box():
    # Both fingertips at the same spot: the box must not collapse to a
    # sliver — it is floored to a 20 px box centred on the point.
    quad = portal_app.build_quad_from_index_fingers(
        _fake_hand(0.5, 0.5), _fake_hand(0.5, 0.5), 200, 100
    )

    assert quad[0][0] == 90 and quad[1][0] == 110  # 20 px wide
    assert quad[0][1] == 40 and quad[2][1] == 60  # 20 px tall


def test_portal_glow_only_brightens():
    frame = np.full((200, 200, 3), 100, dtype=np.uint8)
    before = frame.copy()
    portal_app.draw_portal_glow(frame, _quad(), (0, 0, 255), intensity=1.0)

    # Glow is additive: no pixel may get darker.
    assert np.all(frame.astype(np.int16) >= before.astype(np.int16))
    # And it actually does something near the edge.
    assert np.any(frame != before)


def test_corner_accents_draw_at_corners():
    frame = np.full((200, 200, 3), 100, dtype=np.uint8)
    portal_app.draw_corner_accents(frame, _quad(), (0, 255, 0), t=0.0)

    # Brackets are drawn at the quad corners, not in the middle.
    assert np.any(frame[50:70, 50:70] != 100)
    assert np.array_equal(frame[100, 100], np.array([100, 100, 100], dtype=np.uint8))


def test_particles_spawn_drift_and_expire():
    particles = []
    portal_app.spawn_particles(particles, _quad(), 5)

    assert len(particles) == 5
    # All spawn at a quad corner.
    assert all(
        (p.x, p.y) in {(50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)}
        for p in particles
    )

    first = particles[0]
    x0, y0 = first.x, first.y
    alive = portal_app.update_particles(particles, dt=0.1)
    # A short step moves them but keeps them alive.
    assert len(alive) == 5
    assert (first.x, first.y) != (x0, y0)

    # A long step expires every particle (max life is 1.0 s).
    assert portal_app.update_particles(alive, dt=2.0) == []


def test_draw_particles_marks_the_frame():
    frame = np.full((200, 200, 3), 100, dtype=np.uint8)
    particles = [portal_app.Particle(100.0, 100.0, 0.0, 0.0, 1.0, 0.4)]
    portal_app.draw_particles(frame, particles, (0, 255, 0))

    assert np.any(frame[95:106, 95:106] != 100)


def test_render_portal_negative_alpha_leaves_frame_unchanged():
    # A negative alpha wrapped uint8 (alpha=-0.5 rendered the portal
    # half-visible instead of hidden); it must behave exactly like 0.
    frame = np.full((200, 200, 3), 100, dtype=np.uint8)
    before = frame.copy()
    portal_app.render_portal_into_quad(frame, _solid(200), _quad(), alpha=-0.5)

    assert np.array_equal(frame, before)


def test_render_portal_alpha_above_one_is_fully_opaque():
    # Above 1 the portal must stay fully opaque, never wrap back toward
    # transparent as the uint8 mask scaling used to do.
    frame = np.full((200, 200, 3), 100, dtype=np.uint8)
    portal_app.render_portal_into_quad(frame, _solid(200), _quad(), alpha=2.0)

    assert np.all(frame[100, 100] == 200)


def test_portal_glow_intensity_above_one_matches_full_intensity():
    # An intensity above 1 wrapped uint8 (1.004 -> 256 -> 0 made the
    # glow vanish, 1.5 came back dimmer than full); it must saturate.
    full = np.full((200, 200, 3), 100, dtype=np.uint8)
    portal_app.draw_portal_glow(full, _quad(), (0, 0, 255), intensity=1.0)
    over = np.full((200, 200, 3), 100, dtype=np.uint8)
    portal_app.draw_portal_glow(over, _quad(), (0, 0, 255), intensity=1.5)

    assert np.array_equal(over, full)


def test_portal_glow_negative_intensity_matches_zero():
    # A negative intensity must behave exactly like no glow at all.
    zero = np.full((200, 200, 3), 100, dtype=np.uint8)
    portal_app.draw_portal_glow(zero, _quad(), (0, 0, 255), intensity=0.0)
    negative = np.full((200, 200, 3), 100, dtype=np.uint8)
    portal_app.draw_portal_glow(negative, _quad(), (0, 0, 255), intensity=-0.5)

    assert np.array_equal(negative, zero)


def test_blend_panel_alpha_above_one_is_fully_opaque():
    # An alpha above 1 wrapped uint8 arithmetic (alpha=2.0 turned a
    # 200-grey panel into 44) instead of saturating at full opacity.
    panel = np.zeros((20, 20, 4), dtype=np.uint8)
    panel[:, :, :3] = 200
    panel[:, :, 3] = 255
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)
    portal_app.blend_panel(frame, panel, 10, 10, alpha=2.0)

    assert np.all(frame[15, 15] == 200)


def _opaque_panel(value=200):
    panel = np.zeros((20, 20, 4), dtype=np.uint8)
    panel[:, :, :3] = value
    panel[:, :, 3] = 255
    return panel


def test_blend_panel_clips_negative_offsets():
    # A panel hung off the top-left corner must composite its visible
    # part at the frame origin, not shift it or crash on the slice.
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)
    portal_app.blend_panel(frame, _opaque_panel(), -10, -10, alpha=1.0)

    assert np.all(frame[0, 0] == 200)   # visible part lands at the origin
    assert np.all(frame[50, 50] == 100)  # far from the panel, untouched


def test_blend_panel_fully_outside_frame_leaves_frame_unchanged():
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)
    before = frame.copy()
    portal_app.blend_panel(frame, _opaque_panel(), 500, 500, alpha=1.0)

    assert np.array_equal(frame, before)


def test_hud_panels_return_rgba_arrays():
    # The PIL HUD builders had no coverage: a wrong height, a missing
    # alpha channel, or an empty panel would only show up on a live run.
    if not portal_app.PIL_AVAILABLE:
        pytest.skip("PIL is unavailable")

    effect = portal_app.build_effect_panel(0, "Thermal", None, 1.0, (255, 170, 64))
    assert effect.shape[0] == 62 and effect.shape[1] >= 300 and effect.shape[2] == 4
    assert effect.dtype == np.uint8
    assert np.any(effect[:, :, 3] > 0)  # the panel is not fully transparent

    fps = portal_app.build_fps_pill(59.4)
    assert fps.shape[0] == 24 and fps.shape[2] == 4 and fps.dtype == np.uint8
    assert np.any(fps[:, :, 3] > 0)

    hint = portal_app.build_hint_panel()
    assert hint.shape[0] == 28 and hint.shape[2] == 4 and hint.dtype == np.uint8
    assert np.any(hint[:, :, 3] > 0)


def test_hud_fallback_draws_without_pil():
    # The OpenCV fallback HUD is only used when PIL is missing; it must
    # still draw the effect name and FPS instead of crashing.
    frame = np.full((200, 200, 3), 100, dtype=np.uint8)
    portal_app._hud_fallback(frame, "Thermal", 30.0)

    assert np.any(frame[:40, :] != 100)  # text lands in the top strip
