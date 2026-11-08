"""Safe acquisition of the MediaPipe hand landmark model."""

import tempfile
import urllib.error
import urllib.request
from hashlib import sha256
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(path, url=MODEL_URL, expected_sha256=MODEL_SHA256):
    """Return a verified model path, downloading atomically when absent."""
    target = Path(path)
    if target.exists():
        if not target.is_file():
            raise RuntimeError(f"Model path is not a file: {target}.")
        # The checksum read used to sit outside the friendly-error
        # handling below, so an unreadable model file escaped as a bare
        # OSError traceback at startup instead of a clear message.
        try:
            digest = file_sha256(target)
        except OSError as exc:
            raise RuntimeError(
                f"Could not read the hand model at {target}. "
                "Check permissions, or delete it and retry."
            ) from exc
        if digest != expected_sha256:
            raise RuntimeError(f"Model checksum failed: {target}. Delete it and retry.")
        return target
    # Same for the parent directory: an uncreatable folder escaped as a
    # bare OSError instead of the "Could not download..." message.
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Could not create the model directory {target.parent}. "
            f"Check permissions or place the model at {target}."
        ) from exc
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".download", delete=False) as out:
            temporary = Path(out.name)
            with urllib.request.urlopen(url, timeout=30) as response:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
        if file_sha256(temporary) != expected_sha256:
            raise RuntimeError("Downloaded MediaPipe model failed checksum verification.")
        temporary.replace(target)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Could not download the hand model. Check your connection or place it at {target}."
        ) from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target
