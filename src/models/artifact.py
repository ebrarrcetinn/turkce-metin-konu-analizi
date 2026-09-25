"""Model dosyasının kaydedilmesi ve SHA-256 doğrulamasıyla yüklenmesi.

joblib dosyaları yüklenirken kod çalıştırabileceği için model yalnızca metadata'daki özet
eşleşirse açılır.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

REQUIRED_METADATA_KEYS = (
    "model_name", "model_version", "training_timestamp", "dataset", "labels",
    "preprocessing", "hyperparameters", "thresholds", "metrics", "artifact_sha256",
)


class ModelArtifactError(RuntimeError):
    """Model bulunamadı, bozuk veya doğrulanamadı."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(target: Path, writer) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=target.name, suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        writer(tmp)
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_artifact(model: Any, metadata: dict, model_path: Path) -> dict:
    """Modeli ve metadata'yı atomik olarak yazar; metadata'ya SHA-256 ekler."""
    _atomic_write_bytes(model_path, lambda p: joblib.dump(model, p, compress=3))
    metadata = {**metadata, "artifact_file": model_path.name,
                "artifact_sha256": _sha256(model_path)}
    write_metadata(metadata, model_path.with_suffix(".json"))
    return metadata


def write_metadata(metadata: dict, metadata_path: Path) -> None:
    payload = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write_bytes(metadata_path, lambda p: p.write_bytes(payload))


def read_metadata(metadata_path: Path) -> dict:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelArtifactError(f"Model metadata bulunamadı: {metadata_path}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelArtifactError(f"Model metadata bozuk: {metadata_path}: {exc}") from exc
    missing = [k for k in REQUIRED_METADATA_KEYS if k not in metadata]
    if missing:
        raise ModelArtifactError(f"Model metadata eksik alan(lar): {missing}")
    return metadata


def load_artifact(model_path: Path) -> tuple[Any, dict]:
    """Doğrulanmış artifact'ı yükler. Her türlü tutarsızlıkta ModelArtifactError fırlatır."""
    if not model_path.exists():
        raise ModelArtifactError(
            f"Model dosyası bulunamadı: {model_path}\n"
            "Modeli üretmek için:\n"
            "  python -m scripts.download_data\n"
            "  python -m scripts.build_dataset\n"
            "  python train.py"
        )
    metadata = read_metadata(model_path.with_suffix(".json"))
    actual = _sha256(model_path)
    if actual != metadata["artifact_sha256"]:
        raise ModelArtifactError(
            f"Model dosyasının SHA-256 özeti metadata ile uyuşmuyor ({actual[:12]}… != "
            f"{metadata['artifact_sha256'][:12]}…). Dosya bozulmuş veya değiştirilmiş olabilir; "
            "`python train.py` ile yeniden üretin."
        )
    try:
        model = joblib.load(model_path)
    except Exception as exc:  # joblib birçok farklı hata tipi fırlatabilir
        raise ModelArtifactError(f"Model yüklenemedi: {exc}") from exc
    classes = [str(c) for c in getattr(model, "classes_", [])]
    if classes != metadata["labels"]["joint"]:
        raise ModelArtifactError("Model sınıfları metadata ile uyuşmuyor.")
    logger.info("Model yüklendi: %s v%s", metadata["model_name"], metadata["model_version"])
    return model, metadata
