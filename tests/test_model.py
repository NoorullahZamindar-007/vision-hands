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


def test_unreadable_downloaded_model_reports_a_clear_error(monkeypatch, tmp_path):
    # The post-download checksum read used to sit outside the
    # friendly-error handling, so an unreadable temp file escaped as a
    # bare OSError traceback instead of a clear startup error.
    import hashlib
    from io import BytesIO

    import handportal.model as model_module

    payload = b"fake-model-bytes"

    class FakeResponse:
        def __enter__(self):
            return BytesIO(payload)

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(model_module.urllib.request, "urlopen", lambda *a, **k: FakeResponse())

    def _boom(_path):
        raise OSError("permission denied")

    monkeypatch.setattr(model_module, "file_sha256", _boom)
    with pytest.raises(RuntimeError, match="Could not install the downloaded hand model"):
        ensure_model(tmp_path / "model.task", expected_sha256=hashlib.sha256(payload).hexdigest())


def test_failed_model_install_reports_a_clear_error(monkeypatch, tmp_path):
    # Same for the move into place: a failed replace used to escape as
    # a bare OSError instead of a clear startup error.
    import hashlib
    from io import BytesIO
    from pathlib import Path

    import handportal.model as model_module

    payload = b"fake-model-bytes"

    class FakeResponse:
        def __enter__(self):
            return BytesIO(payload)

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(model_module.urllib.request, "urlopen", lambda *a, **k: FakeResponse())

    real_replace = Path.replace

    def _boom(self, target):
        if self.suffix == ".download":
            raise OSError("permission denied")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _boom)
    with pytest.raises(RuntimeError, match="Could not install the downloaded hand model"):
        ensure_model(tmp_path / "model.task", expected_sha256=hashlib.sha256(payload).hexdigest())


def test_transient_download_failure_is_retried_once(monkeypatch, tmp_path):
    # One brief network blip at startup used to abort the app, even
    # though the download is atomic and checksum-verified — so a single
    # failure is retried once before the friendly error is raised.
    import http.client
    from io import BytesIO

    import handportal.model as model_module

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


def test_malformed_download_url_reports_a_clear_error(monkeypatch, tmp_path):
    # A malformed URL raises ValueError from urlopen ("unknown url
    # type") instead of a URLError, so it used to escape ensure_model
    # as a bare traceback instead of the friendly "Could not download"
    # error. The retry still runs once, so the sleep is stubbed out.
    import handportal.model as model_module

    monkeypatch.setattr(model_module.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="Could not download the hand model"):
        ensure_model(tmp_path / "model.task", url="notaurl", expected_sha256="0" * 64)


def test_oversized_model_download_is_refused(monkeypatch, tmp_path):
    # The download loop used to read until EOF with no total limit, so
    # a misbehaving server could stream gigabytes to disk until it
    # filled. The real model is ~7 MB; refuse anything over the 64 MB
    # cap instead of retrying (an oversized stream will not shrink).
    import handportal.model as model_module

    class OversizedResponse:
        """Yields 1 MB chunks until the caller stops reading."""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            return b"x" * (1024 * 1024)

    monkeypatch.setattr(
        model_module.urllib.request, "urlopen", lambda *a, **k: OversizedResponse()
    )

    with pytest.raises(RuntimeError, match="larger than expected"):
        ensure_model(tmp_path / "model.task", expected_sha256="0" * 64)

    assert not (tmp_path / "model.task").exists()
    assert not list(tmp_path.glob("*.download"))
