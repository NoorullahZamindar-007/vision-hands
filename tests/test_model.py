from hashlib import sha256

import pytest

from handportal.model import ensure_model, file_sha256


def test_file_sha256(tmp_path):
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"handportal")
    assert file_sha256(sample) == sha256(b"handportal").hexdigest()


def test_existing_verified_model_is_reused(tmp_path):
    model = tmp_path / "model.task"
    model.write_bytes(b"valid model")
    digest = sha256(b"valid model").hexdigest()
    assert ensure_model(model, expected_sha256=digest) == model


def test_existing_corrupt_model_is_rejected(tmp_path):
    model = tmp_path / "model.task"
    model.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        ensure_model(model, expected_sha256="0" * 64)
