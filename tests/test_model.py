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


def test_unreadable_existing_model_reports_a_clear_error(monkeypatch, tmp_path):
    # The checksum read used to sit outside the friendly-error handling,
    # so an unreadable model file escaped as a bare OSError traceback.
    import handportal.model as model_module

    model = tmp_path / "model.task"
    model.write_bytes(b"valid model")

    def _boom(_path):
        raise OSError("permission denied")

    monkeypatch.setattr(model_module, "file_sha256", _boom)
    with pytest.raises(RuntimeError, match="Could not read the hand model"):
        ensure_model(model, expected_sha256="0" * 64)


def test_uncreatable_model_directory_reports_a_clear_error(monkeypatch, tmp_path):
    # The parent mkdir used to sit outside the friendly-error handling,
    # so an uncreatable folder escaped as a bare OSError traceback.
    from pathlib import Path

    real_mkdir = Path.mkdir

    def _boom(self, *args, **kwargs):
        if self == tmp_path / "nested":
            raise OSError("permission denied")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _boom)
    with pytest.raises(RuntimeError, match="Could not create the model directory"):
        ensure_model(tmp_path / "nested" / "model.task", expected_sha256="0" * 64)
