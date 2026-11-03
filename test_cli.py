import pytest

from handportal.cli import parse_args


def test_cli_overrides_runtime_options():
    args = parse_args(["--camera", "2", "--width", "640", "--height", "360", "--windowed"])
    assert (args.camera, args.width, args.height) == (2, 640, 360)
    assert args.windowed is True


def test_cli_rejects_invalid_dimensions():
    with pytest.raises(SystemExit):
        parse_args(["--width", "0"])


def test_cli_rejects_negative_camera_index():
    with pytest.raises(SystemExit):
        parse_args(["--camera", "-1"])


def test_cli_rejects_out_of_range_pinch_threshold():
    with pytest.raises(SystemExit):
        parse_args(["--pinch-threshold", "0.5"])


def test_cli_rejects_missing_model_file():
    # A typo'd --model path used to silently download a fresh model to the
    # mistyped path instead of failing fast.
    with pytest.raises(SystemExit):
        parse_args(["--model", "no-such-model.task"])


def test_cli_accepts_existing_model_file(tmp_path):
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"fake-model")
    args = parse_args(["--model", str(model)])
    assert args.model == str(model)
