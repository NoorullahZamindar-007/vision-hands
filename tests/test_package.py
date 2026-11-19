"""Package-level checks for the handportal distribution."""

import re
from pathlib import Path

import pytest

import handportal
from handportal import effects


def test_version_is_single_sourced_from_pyproject():
    """__version__ must track the version declared in pyproject.toml."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
    assert declared, "pyproject.toml must declare a version"
    assert handportal.__version__ == declared.group(1)
    assert "__version__" in handportal.__all__


def test_lazy_effect_imports_resolve_to_the_effects_module():
    # The README documents `from handportal import thermal`: the lazy
    # __getattr__ must hand back the real effect functions, not a
    # wrapper, and the effect registry must resolve the same way.
    assert handportal.thermal is effects.thermal
    assert handportal.EFFECTS is effects.EFFECTS
    assert handportal.EFFECT_NAMES is effects.EFFECT_NAMES
    assert len(handportal.EFFECTS) == len(handportal.EFFECT_NAMES) == 8


def test_lazy_effect_names_are_exported():
    # Every lazily-resolved name must be importable and listed in
    # __all__, or `from handportal import <effect>` breaks silently
    # for star-imports and linters.
    for name in sorted(handportal._LAZY_EFFECTS):
        assert name in handportal.__all__
        assert getattr(handportal, name) is not None


def test_unknown_attribute_raises_attribute_error():
    # __getattr__ must not swallow typos: an unknown name is an error,
    # not a silent None.
    with pytest.raises(AttributeError):
        _missing = handportal.no_such_effect
