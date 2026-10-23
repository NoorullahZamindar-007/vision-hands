"""HandPortal reusable computer-vision components.

The light submodules (config, gestures, model) are importable directly, so
``from handportal import PinchTrigger`` works without pulling in OpenCV::

    from handportal import PinchTrigger, ensure_model

The OpenCV-based effects stay lazy: they are only imported on first
attribute access, keeping ``import handportal`` cheap on machines without
a camera stack installed.
"""

import importlib
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("handportal")
except PackageNotFoundError:  # not installed (e.g. running from a bare checkout)
    __version__ = "0.0.0+dev"

from .config import (
    CAM_INDEX,
    DETECT_DOWNSCALE,
    FRAME_H,
    FRAME_W,
    FULLSCREEN,
    PINCH_ENTER_THRESHOLD,
    PINCH_RELEASE_THRESHOLD,
    VIGNETTE,
)
from .gestures import PinchTrigger, normalized_distance, pinch_distance
from .model import MODEL_SHA256, MODEL_URL, ensure_model, file_sha256

# Names that live in handportal.effects (imports cv2) and are therefore
# resolved lazily through __getattr__ below.
_LAZY_EFFECTS = {
    "EFFECTS",
    "EFFECT_NAMES",
    "chromatic",
    "duotone",
    "edge_mesh",
    "grayscale_sketch",
    "invert_glitch",
    "pixel_mosaic",
    "posterize_neon",
    "thermal",
}

__all__ = [
    "__version__",
    "CAM_INDEX",
    "DETECT_DOWNSCALE",
    "FRAME_H",
    "FRAME_W",
    "FULLSCREEN",
    "MODEL_SHA256",
    "MODEL_URL",
    "PINCH_ENTER_THRESHOLD",
    "PINCH_RELEASE_THRESHOLD",
    "VIGNETTE",
    "PinchTrigger",
    "ensure_model",
    "file_sha256",
    "normalized_distance",
    "pinch_distance",
    *sorted(_LAZY_EFFECTS),
]


def __getattr__(name):
    """Lazily expose the cv2-based effects without importing cv2 eagerly."""
    if name in _LAZY_EFFECTS:
        module = importlib.import_module(".effects", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
