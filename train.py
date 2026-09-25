"""Final modeli eğitir ve models/ altına kaydeder.

Önce veri hazırlanmalıdır:
    python -m scripts.download_data
    python -m scripts.build_dataset
"""

from __future__ import annotations

import json
import sys

from src.data.build import DatasetError
from src.logging_setup import setup_logging
from src.models.training import train_final


def main() -> int:
    setup_logging("INFO")
    try:
        result = train_final()
    except DatasetError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    meta, report = result["metadata"], result["report"]
    print(json.dumps({
        "model": meta["model_name"],
        "version": meta["model_version"],
        "calibration": report["selected_calibration"],
        "thresholds": meta["thresholds"],
        "validation_argmax": meta["metrics"]["validation_argmax"],
        "validation_with_threshold": meta["metrics"]["validation_with_threshold"],
        "validation_ece": meta["metrics"]["validation_ece"],
        "artifact_sha256": meta["artifact_sha256"][:16] + "…",
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
