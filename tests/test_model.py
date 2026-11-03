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


def test_model_path_that_is_a_directory_is_rejected(tmp_path):
    # A directory at the model path used to crash inside the checksum read
    # with a raw IsADirectoryError instead of a clear startup error.
    model_dir = tmp_path / "model.task"
    model_dir.mkdir()
    with pytest.raises(RuntimeError, match="not a file"):
        ensure_model(model_dir, expected_sha256="0" * 64)
