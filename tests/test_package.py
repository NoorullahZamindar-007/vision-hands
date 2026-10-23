"""Package-level checks for the handportal distribution."""

import re
from pathlib import Path

import handportal


def test_version_is_single_sourced_from_pyproject():
    """__version__ must track the version declared in pyproject.toml."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
    assert declared, "pyproject.toml must declare a version"
    assert handportal.__version__ == declared.group(1)
    assert "__version__" in handportal.__all__
