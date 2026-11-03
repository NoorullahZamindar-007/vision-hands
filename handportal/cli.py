"""Command-line interface for HandPortal."""

import argparse
import os

from . import config


def build_parser():
    parser = argparse.ArgumentParser(description="Stretch visual effects between your fingertips.")
    parser.add_argument("--camera", type=int, default=config.CAM_INDEX, help="camera device index")
    parser.add_argument("--width", type=int, default=config.FRAME_W, help="requested frame width")
    parser.add_argument("--height", type=int, default=config.FRAME_H, help="requested frame height")
    parser.add_argument("--pinch-threshold", type=float, default=config.PINCH_ENTER_THRESHOLD)
    parser.add_argument("--windowed", action="store_true", help="start in a resizable window")
    parser.add_argument("--downscale", action="store_true", help="detect hands at half resolution")
    parser.add_argument("--no-vignette", action="store_true", help="disable the vignette")
    parser.add_argument("--model", help="path to hand_landmarker.task")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.camera < 0:
        parser.error("camera index must be non-negative")
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    if not 0.005 <= args.pinch_threshold <= 0.20:
        parser.error("pinch threshold must be between 0.005 and 0.20")
    if args.model is not None:
        # An explicit --model path is a user choice, not a download target:
        # without this check a typo silently downloads a fresh model to the
        # mistyped path instead of using the intended file.
        model_path = os.path.expanduser(args.model)
        if not os.path.isfile(model_path):
            parser.error(f"model file does not exist: {args.model}")
    return args
