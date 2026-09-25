"""Ham veriden işlenmiş train/val/test/ood/external setlerini üretir."""

import json
import sys

from src.data.build import DatasetError, build
from src.logging_setup import setup_logging


def main() -> int:
    setup_logging("INFO")
    try:
        result = build()
    except DatasetError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    summary = {k: v for k, v in result.report.items()
               if k not in {"mapped_categories", "excluded_categories", "label_distribution"}}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
