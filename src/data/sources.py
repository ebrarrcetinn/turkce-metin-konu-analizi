"""
Dosya   : src/data/sources.py
Konu    : Veri Kaynaklarının İndirilmesi
Açıklama: Bu dosyada amacım eğitim ve değerlendirme verilerini sabit commit adreslerinden
          indirmek, her dosyayı SHA-256 ile doğrulamak ve kaynak/lisans bilgisini
          provenance.json dosyasına yazmak.
Yazar   : Ebrar Cemre Çetin
Tarih   : 23.09.2026
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from src.config import RAW_DIR, USER_AGENT

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = 60  # saniye
DOWNLOAD_RETRIES = 3
# İndirdiğim her dosyanın adresini, özetini, lisansını ve indirme zamanını bu dosyaya yazıyorum;
# böylece hangi verinin hangi sürümüyle eğittiğimi sonradan bilebiliyorum.
PROVENANCE_FILE = "provenance.json"


class DataAcquisitionError(RuntimeError):
    """Veriyi indiremediğimde veya doğrulayamadığımda bu hatayı veriyorum."""


@dataclass(frozen=True)
class RemoteFile:
    """İndireceğim tek dosya: adresi, yerel adı ve beklediğim SHA-256 özeti."""

    url: str
    local_name: str
    sha256: str


@dataclass(frozen=True)
class DataSource:
    """Kullandığım bir veri seti: kaynağı, lisansı, sabitlediğim commit'i ve dosyaları."""

    key: str
    name: str
    homepage: str
    license: str
    commit: str
    role: str
    files: tuple[RemoteFile, ...]


# Dosyaları GitHub'ın ham dosya sunucusundan indiriyorum. Adreslerde dal adı ("main") yerine
# commit kimliğini kullandım: veri seti sahibi dosyaları sonradan değiştirse bile her zaman
# aynı sürümü indiriyorum ve sonuçlar tekrar üretilebilir kalıyor.
_RAW = "https://raw.githubusercontent.com"
# 1) Türkçe Tabu Veri Seti: 150 kategori, 37.278 kavram kartı ("kelime" + "aciklama").
#    Eğitim, doğrulama ve testin tamamını buradan alıyorum.
_TABOO_COMMIT = "f621b460f4511f2afb1a89f87b2d94af9440b09c"
# 150 dosyanın SHA-256 özetlerini ayrı bir JSON dosyasında tutuyorum (data/*.json -> özet).
_TABOO_MANIFEST: dict[str, str] = json.loads(
    (Path(__file__).with_name("taboo_manifest.json")).read_text(encoding="utf-8")
)
# 2) Turkish BQuAD: lise biyoloji ders kitabı paragrafları. Yalnızca harici test için kullanıyorum.
_BQUAD_COMMIT = "30a3e070e5eeb3ba04da89cac3a426eb858df032"
# 3) Osmanlı tarihi okuma-anlama veri seti. Yalnızca harici test için kullanıyorum.
_OTTOMAN_COMMIT = "c6852f86f25e093ead13d8e2d0d87d92deadf727"
_OTTOMAN_BASE = (
    "https://raw.githubusercontent.com/okanvk/"
    f"Turkish-Reading-Comprehension-Question-Answering-Dataset/{_OTTOMAN_COMMIT}"
    "/data/2020-enelpi-squad-dataset/"
)
_OTTOMAN_FILES = {
    "1299-1451/train_data.json": "2edb403631f92201fdc4da1065412ad425c8f78daabae66ba1d344e89a5884a4",
    "1451-1579/train_data.json": "9ef3eb3818e1ca31f0db6600fa603bf1e0dd8079bad25e416e0748b3d183a82a",
    "1451-1579/test_data.json": "ae9ee0ea113be5124d3a677d45599feff4de7d9363d0e7336110946ca3639a70",
    "1579-1792/train_data_v2.json":
        "ffacba321ea3aa2bf72b2f44c0db1fb200c26e510ffff336fb675cc8b991c96b",
    "1792-1922/train_data.json": "9ff08b7d7e1b955195755432e202abc7284d283ae6bb52f8da9efae340d11b77",
    "1792-1922/test_data.json": "13a0b72088bed5b90b7b47bd23cf27b711fc9eda98ea37dde69f3e92227d630e",
}

SOURCES: tuple[DataSource, ...] = (
    DataSource(
        key="taboo",
        name="Türkçe Tabu Veri Seti (wleeaf/taboo-dataset)",
        homepage="https://github.com/wleeaf/taboo-dataset",
        license="MIT",
        commit=_TABOO_COMMIT,
        role="Birincil eğitim/validation/test verisi (150 kategori, 37.278 kart)",
        files=tuple(
            RemoteFile(
                url=f"{_RAW}/wleeaf/taboo-dataset/{_TABOO_COMMIT}/data/{name}",
                local_name=f"taboo/data/{name}",
                sha256=sha,
            )
            for name, sha in sorted(_TABOO_MANIFEST.items())
        ),
    ),
    DataSource(
        key="bquad",
        name="Turkish BQuAD — MEB lise biyoloji paragrafları (TurQuest/turkish-bquad)",
        homepage="https://github.com/TurQuest/turkish-bquad",
        license="MIT",
        commit=_BQUAD_COMMIT,
        role="Harici (cross-domain) değerlendirme: insan yazımı Biyoloji metni",
        files=tuple(
            RemoteFile(
                url=f"{_RAW}/TurQuest/turkish-bquad/{_BQUAD_COMMIT}/{name}",
                local_name=f"bquad/{name}",
                sha256=sha,
            )
            for name, sha in (
                ("turkish-bquad-train-v1.json",
                 "0c8f9510444446d4b3d0a2e740aee897c3a645d526182880d6d25c1ce88899b0"),
                ("turkish-bquad-valid-v1.json",
                 "2b4e6d26d6aa64469ec088f3f854144b77ce404266a063f325c4e5c0078246a7"),
            )
        ),
    ),
    DataSource(
        key="ottoman",
        name="Osmanlı Tarihi Okuduğunu Anlama Veri Seti (okanvk, 2020-enelpi alt kümesi)",
        homepage="https://github.com/okanvk/Turkish-Reading-Comprehension-Question-Answering-Dataset",
        license="MIT",
        commit=_OTTOMAN_COMMIT,
        role="Harici (cross-domain) değerlendirme: insan yazımı Tarih/Osmanlı metni",
        files=tuple(
            RemoteFile(url=_OTTOMAN_BASE + rel, local_name=f"ottoman/{rel}", sha256=sha)
            for rel, sha in _OTTOMAN_FILES.items()
        ),
    ),
)


def sha256_file(path: Path) -> str:
    """Dosyanın SHA-256 özetini hesaplıyorum. Tek bir bayt bile değişse özet tamamen değişir;
    böylece indirdiğim dosyanın bozuk veya farklı olmadığını kesin olarak anlıyorum."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):  # 64 KB parçalar halinde okuyorum
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path, retries: int = DOWNLOAD_RETRIES) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # Önce geçici .part dosyasına yazıyorum, tamamlanınca asıl ada taşıyorum. İndirme yarıda
    # kesilirse yarım dosya asıl dosya gibi görünüp sonraki çalıştırmada kullanılmıyor.
    tmp = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers={"User-Agent": USER_AGENT}
            ) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in resp.iter_content(1 << 16):
                        handle.write(chunk)
            tmp.replace(target)
            return
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            logger.warning("İndirme denemesi %d/%d başarısız: %s (%s)", attempt, retries, url, exc)
            time.sleep(1.5 * attempt)  # her denemede biraz daha uzun bekliyorum
    tmp.unlink(missing_ok=True)
    raise DataAcquisitionError(f"İndirilemedi: {url}: {last_error}")


def fetch_all(raw_dir: Path = RAW_DIR, force: bool = False) -> dict[str, dict]:
    """Burada amacım tüm kaynakları indirmek, doğrulamak ve provenance kaydını yazmak."""
    provenance: dict[str, dict] = {}
    for source in SOURCES:
        logger.info("%s: %d dosya kontrol ediliyor", source.key, len(source.files))
        entries = []
        for remote in source.files:
            target = raw_dir / remote.local_name
            # Hedef yolun data/raw dışına çıkmadığını denetliyorum ("../" içeren adlara karşı).
            if raw_dir.resolve() not in target.resolve().parents:
                raise DataAcquisitionError(f"Güvensiz hedef yol: {remote.local_name}")
            # Dosya zaten varsa yeniden indirmiyorum; yalnızca özetini yeniden doğruluyorum.
            if force or not target.exists():
                logger.debug("İndiriliyor: %s", remote.url)
                _download(remote.url, target)
            digest = sha256_file(target)
            if digest != remote.sha256:
                target.unlink(missing_ok=True)
                raise DataAcquisitionError(
                    f"SHA-256 uyuşmazlığı: {remote.local_name} ({digest} != {remote.sha256})"
                )
            entries.append(
                {"url": remote.url, "file": remote.local_name, "sha256": digest,
                 "bytes": target.stat().st_size}
            )
        provenance[source.key] = {
            "name": source.name,
            "homepage": source.homepage,
            "license": source.license,
            "commit": source.commit,
            "role": source.role,
            "files": entries,
            "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / PROVENANCE_FILE).write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return provenance
