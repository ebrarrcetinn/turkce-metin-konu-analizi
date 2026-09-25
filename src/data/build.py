"""Ham kaynaklardan train/val/test, OOD ve harici değerlendirme setlerini üretir.

Bölme terim bazında gruplanır; böylece aynı terimin varyasyonları farklı bölmelere düşmez.
Taksonomi dışı kategorilerin bir kısmı eğitimden tamamen ayrılarak OOD testi için saklanır.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    OTHER_LABEL,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    TEST_SIZE,
    UNSEEN_OOD_FRACTION,
    VAL_SIZE,
)
from src.models.taxonomy import CATEGORY_TO_TAXONOMY, EXCLUDED_CATEGORIES, map_card
from src.preprocessing.text import normalize

logger = logging.getLogger(__name__)

EXPECTED_TABOO_CARDS = 37_278
REQUIRED_CARD_KEYS = ("kategori", "kelime", "aciklama")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
EXTERNAL_MIN_WORDS, EXTERNAL_MAX_WORDS = 6, 40
EXTERNAL_SAMPLE_PER_SOURCE = 400
NEAR_DUP_COSINE = 0.9
SPLIT_COLUMNS = ["text", "general", "subtopic", "category", "term", "group", "source"]


class DatasetError(RuntimeError):
    """Beklenmeyen veri formatı / eksik veri."""


@dataclass(frozen=True)
class BuildResult:
    frames: dict[str, pd.DataFrame]
    report: dict


def _find_taboo_data_dir(raw_dir: Path) -> Path:
    data_dir = raw_dir / "taboo" / "data"
    if not any(data_dir.glob("*.json")):
        raise DatasetError(
            f"Tabu verisi bulunamadı: {data_dir}. "
            "Önce `python -m scripts.download_data` çalıştırın."
        )
    return data_dir


def load_taboo_cards(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    data_dir = _find_taboo_data_dir(raw_dir)
    records: list[dict] = []
    for path in sorted(data_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetError(f"Bozuk JSON/encoding: {path.name}: {exc}") from exc
        if not isinstance(payload, list):
            logger.warning("Liste olmayan dosya atlandı: %s", path.name)
            continue
        for card in payload:
            missing = [k for k in REQUIRED_CARD_KEYS if k not in card]
            if missing:
                raise DatasetError(f"{path.name}: eksik alan(lar) {missing}")
            records.append(
                {
                    "category": str(card["kategori"]).strip(),
                    "term": str(card["kelime"]).strip(),
                    "definition": str(card["aciklama"]).strip(),
                    "difficulty": card.get("zorluk"),
                    "source": "taboo",
                    "source_file": path.name,
                }
            )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != EXPECTED_TABOO_CARDS:
        logger.warning("Beklenen kart sayısı %d, bulunan %d", EXPECTED_TABOO_CARDS, len(frame))
    return frame


def term_group(term: str) -> str:
    """Gruplama anahtarı: normalize terim; tekil/çoğul ve çekim farklarını azaltmak için
    her sözcüğün ilk 6 harfi kullanılır."""
    return " ".join(tok[:6] for tok in normalize(term).split())


def prepare_taboo(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"raw_rows": int(len(frame))}
    frame = frame.copy()
    empty = frame["definition"].str.len() == 0
    stats["empty_definition_removed"] = int(empty.sum())
    frame = frame[~empty]

    mapped = [map_card(c, t) for c, t in zip(frame["category"], frame["term"], strict=True)]
    frame["general"] = [m[0] if m else None for m in mapped]
    frame["subtopic"] = [m[1] if m else None for m in mapped]
    stats["excluded_ambiguous_rows"] = int(frame["general"].isna().sum())
    frame = frame[frame["general"].notna()].copy()

    frame["text"] = frame["term"] + " " + frame["definition"]
    frame["norm"] = frame["text"].map(normalize)
    frame["group"] = frame["term"].map(term_group)

    before = len(frame)
    frame = frame.drop_duplicates(subset=["norm", "general", "subtopic"])
    stats["exact_duplicates_removed"] = int(before - len(frame))
    conflict_mask = frame.duplicated(subset=["norm"], keep=False)
    stats["label_conflicts_removed"] = int(conflict_mask.sum())
    frame = frame[~conflict_mask]
    stats["rows_after_cleaning"] = int(len(frame))
    return frame.reset_index(drop=True), stats


def _choose_unseen_other_categories(frame: pd.DataFrame, seed: int) -> set[str]:
    other_categories = sorted(frame.loc[frame["general"] == OTHER_LABEL, "category"].unique())
    rng = random.Random(seed)
    return set(rng.sample(other_categories, round(len(other_categories) * UNSEEN_OOD_FRACTION)))


def _group_holdout(
    frame: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_splits = max(2, round(1 / fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, hold_idx = next(splitter.split(frame, frame["category"], frame["group"]))
    return frame.iloc[keep_idx], frame.iloc[hold_idx]


def split_frames(frame: pd.DataFrame, seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    unseen = _choose_unseen_other_categories(frame, seed)
    ood_mask = frame["category"].isin(unseen)
    seen = frame[~ood_mask]
    ood = frame[ood_mask]
    # Eğitimde görülebilecek bir terimle aynı grupta olan OOD kartları atılır (sızıntı).
    ood = ood[~ood["group"].isin(set(seen["group"]))]

    train_val, test = _group_holdout(seen, TEST_SIZE, seed)
    train, val = _group_holdout(train_val, VAL_SIZE / (1 - TEST_SIZE), seed + 1)
    return {"train": train, "val": val, "test": test, "ood_unseen": ood}


def _squad_contexts(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [p["context"] for doc in payload.get("data", []) for p in doc.get("paragraphs", [])]


def _sentences(contexts: list[str]) -> list[str]:
    out: list[str] = []
    for context in contexts:
        for sentence in SENTENCE_SPLIT_RE.split(context.replace("\n", " ")):
            sentence = sentence.strip()
            if EXTERNAL_MIN_WORDS <= len(sentence.split()) <= EXTERNAL_MAX_WORDS:
                out.append(sentence)
    return out


def build_external(raw_dir: Path = RAW_DIR, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """İnsan yazımı, farklı alan/üsluptaki metinlerden harici değerlendirme seti."""
    specs = [
        ("bquad", sorted((raw_dir / "bquad").glob("*.json")), "Biyoloji", None),
        ("ottoman", sorted((raw_dir / "ottoman").rglob("*.json")), "Tarih", "Osmanlı Tarihi"),
    ]
    rng = random.Random(seed)
    rows: list[dict] = []
    for source, paths, general, subtopic in specs:
        if not paths:
            logger.warning("Harici kaynak bulunamadı: %s", source)
            continue
        sentences = sorted(set(_sentences([c for p in paths for c in _squad_contexts(p)])))
        sample = rng.sample(sentences, min(EXTERNAL_SAMPLE_PER_SOURCE, len(sentences)))
        rows += [
            {"text": s, "general": general, "subtopic": subtopic, "source": source,
             "category": source, "term": "", "group": f"{source}:{i}"}
            for i, s in enumerate(sample)
        ]
    frame = pd.DataFrame(rows, columns=SPLIT_COLUMNS)
    frame["norm"] = frame["text"].map(normalize)
    return frame


def near_duplicate_rate(reference: pd.DataFrame, query: pd.DataFrame,
                        threshold: float = NEAR_DUP_COSINE) -> float:
    """query örneklerinden, reference içinde char 3-5gram TF-IDF kosinüs benzerliği
    >= threshold olan bir komşusu bulunanların oranı. (Yalnızca analiz amaçlı.)"""
    if reference.empty or query.empty:
        return 0.0
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(reference["norm"])
    ref, qry = vec.transform(reference["norm"]), vec.transform(query["norm"])
    hits = 0
    for start in range(0, qry.shape[0], 1000):
        sims = qry[start : start + 1000] @ ref.T
        hits += int((np.asarray(sims.max(axis=1).todense()).ravel() >= threshold).sum())
    return hits / qry.shape[0]


def build(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> BuildResult:
    cards = load_taboo_cards(raw_dir)
    prepared, stats = prepare_taboo(cards)
    frames = split_frames(prepared)
    frames["external"] = build_external(raw_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame[SPLIT_COLUMNS].to_json(out_dir / f"{name}.jsonl", orient="records", lines=True,
                                     force_ascii=False)

    train = frames["train"]
    train_groups = set(train["group"])
    report = {
        "cleaning": stats,
        "split_sizes": {k: int(len(v)) for k, v in frames.items()},
        "unseen_ood_categories": sorted(frames["ood_unseen"]["category"].unique().tolist()),
        "leakage": {
            "group_overlap_train_val": len(train_groups & set(frames["val"]["group"])),
            "group_overlap_train_test": len(train_groups & set(frames["test"]["group"])),
            "exact_text_overlap_train_test": len(set(train["norm"]) & set(frames["test"]["norm"])),
            "exact_text_overlap_train_val": len(set(train["norm"]) & set(frames["val"]["norm"])),
            f"near_dup_rate_val_vs_train_cos{NEAR_DUP_COSINE}":
                round(near_duplicate_rate(train, frames["val"]), 4),
            f"near_dup_rate_test_vs_train_cos{NEAR_DUP_COSINE}":
                round(near_duplicate_rate(train, frames["test"]), 4),
            f"near_dup_rate_external_vs_train_cos{NEAR_DUP_COSINE}":
                round(near_duplicate_rate(train, frames["external"]), 4),
        },
        "label_distribution": {k: dict(Counter(v["general"])) for k, v in frames.items()},
        "mapped_categories": sorted(CATEGORY_TO_TAXONOMY),
        "excluded_categories": sorted(EXCLUDED_CATEGORIES),
    }
    (out_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return BuildResult(frames=frames, report=report)


def load_split(name: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    path = processed_dir / f"{name}.jsonl"
    if not path.exists():
        raise DatasetError(f"{path} yok. Önce `python -m scripts.build_dataset` çalıştırın.")
    frame = pd.read_json(path, orient="records", lines=True, dtype={"subtopic": "object"})
    frame["subtopic"] = frame["subtopic"].astype(object).where(frame["subtopic"].notna(), None)
    return frame
