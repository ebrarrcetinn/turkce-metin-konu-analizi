"""Veri kaynaklarını sabit commit'lerden indirir ve provenance kaydını yazar.

Kullanım: python -m scripts.download_data [--force]
"""

import argparse
import sys

from src.data.sources import DataAcquisitionError, fetch_all
from src.logging_setup import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Var olan dosyaları yeniden indir")
    args = parser.parse_args()
    setup_logging("INFO")
    try:
        provenance = fetch_all(force=args.force)
    except DataAcquisitionError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    for key, info in provenance.items():
        files = info["files"]
        print(f"{key:8s} {info['license']:5s} {len(files)} dosya, "
              f"{sum(f['bytes'] for f in files):,} bayt, sha256 doğrulandı="
              f"{all(f['sha256_verified'] for f in files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
