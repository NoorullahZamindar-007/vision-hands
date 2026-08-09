"""Safe acquisition of the MediaPipe hand landmark model."""

from hashlib import sha256
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

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
        if file_sha256(target) != expected_sha256:
            raise RuntimeError(f"Model checksum failed: {target}. Delete it and retry.")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
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
