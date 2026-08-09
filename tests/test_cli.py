import pytest

from handportal.cli import parse_args


def test_cli_overrides_runtime_options():
    args = parse_args(["--camera", "2", "--width", "640", "--height", "360", "--windowed"])
    assert (args.camera, args.width, args.height) == (2, 640, 360)
    assert args.windowed is True


def test_cli_rejects_invalid_dimensions():
    with pytest.raises(SystemExit):
        parse_args(["--width", "0"])
