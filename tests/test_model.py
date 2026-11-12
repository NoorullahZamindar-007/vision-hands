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


def test_transient_download_failure_is_retried_once(monkeypatch, tmp_path):
    # One brief network blip at startup used to abort the app, even
    # though the download is atomic and checksum-verified — so a single
    # failure is retried once before the friendly error is raised.
    import http.client
    from io import BytesIO

    import handportal.model as model_module
    from handportal.model import file_sha256

    payload = b"fake-model-bytes"
    calls = {"n": 0}

    class FakeResponse:
        def __enter__(self):
            return BytesIO(payload)

        def __exit__(self, *args):
            return False

    def _flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise http.client.IncompleteRead(b"partial", 100)
        return FakeResponse()

    monkeypatch.setattr(model_module.urllib.request, "urlopen", _flaky)
    monkeypatch.setattr(model_module.time, "sleep", lambda seconds: None)

    import hashlib

    target = tmp_path / "model.task"
    result = ensure_model(
        target, expected_sha256=hashlib.sha256(payload).hexdigest()
    )

    assert result == target
    assert target.read_bytes() == payload
    assert calls["n"] == 2


def test_interrupted_download_reports_a_clear_error(monkeypatch, tmp_path):
    # A connection that drops mid-download raises IncompleteRead, which
    # is an http.client.HTTPException rather than a URLError, so it used
    # to escape ensure_model as a bare traceback instead of the
    # friendly "Could not download" error.
    import http.client

    import handportal.model as model_module

    def _boom(*args, **kwargs):
        raise http.client.IncompleteRead(b"partial", 100)

    monkeypatch.setattr(model_module.urllib.request, "urlopen", _boom)
    with pytest.raises(RuntimeError, match="Could not download the hand model"):
        ensure_model(tmp_path / "model.task", expected_sha256="0" * 64)
