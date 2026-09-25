"""Çalışma zamanı çıkarımı.

Model "Genel > Alt" birleşik sınıflarıyla eğitilir; genel konu olasılığı alt konu
olasılıklarının toplamıdır, bu yüzden seçilen alt konu her zaman genel konuya aittir.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_SUBTOPIC_THRESHOLD,
    KEYWORD_MIN_WEIGHT,
    MAX_INPUT_CHARS,
    MAX_SUBTOPICS,
    MODEL_PATH,
    OTHER_LABEL,
    SHORT_INPUT_MAX_TOKENS,
    UNCERTAIN_LABEL,
)
from src.models.artifact import load_artifact
from src.preprocessing.text import TURKISH_STOPWORDS, content_tokens, f5_stem, normalize

logger = logging.getLogger(__name__)

JOINT_SEPARATOR = " > "

STATUS_OK = "ok"
STATUS_UNCERTAIN = "uncertain"
STATUS_OUT_OF_SCOPE = "out_of_scope"
STATUS_EMPTY = "empty"


def split_joint(label: str) -> tuple[str, str | None]:
    if JOINT_SEPARATOR in label:
        general, sub = label.split(JOINT_SEPARATOR, 1)
        return general, sub
    return label, None


def joint_label(general: str, subtopic: str | None) -> str:
    return f"{general}{JOINT_SEPARATOR}{subtopic}" if subtopic else general


@dataclass
class Prediction:
    text: str
    status: str
    general: str | None
    confidence: float
    general_scores: dict[str, float] = field(default_factory=dict)
    subtopics: list[tuple[str, float]] = field(default_factory=list)
    top_guess: str | None = None
    latency_ms: float = 0.0
    truncated: bool = False
    short_input: bool = False
    joint_subtopic_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["subtopics"] = [{"name": n, "score": s} for n, s in self.subtopics]
        return payload


def aggregate(joint_classes: list[str], joint_proba: np.ndarray) -> tuple[dict, dict]:
    """Birleşik olasılıkları genel konu marjinallerine ve koşullu alt konu dağılımlarına çevirir."""
    general_scores: dict[str, float] = defaultdict(float)
    sub_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for label, p in zip(joint_classes, joint_proba, strict=True):
        general, sub = split_joint(label)
        general_scores[general] += float(p)
        if sub:
            sub_scores[general][sub] = float(p)
    conditional = {
        g: {s: v / general_scores[g] for s, v in subs.items()} if general_scores[g] > 0 else {}
        for g, subs in sub_scores.items()
    }
    return dict(general_scores), conditional


def select_subtopics(conditional: dict[str, float], threshold: float,
                     max_items: int = MAX_SUBTOPICS) -> list[tuple[str, float]]:
    """En olası alt konu her zaman; ek alt konular yalnızca eşiği geçerse (en fazla max_items)."""
    ranked = sorted(conditional.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return []
    chosen = [ranked[0]] + [kv for kv in ranked[1:] if kv[1] >= threshold]
    return [(name, round(score, 4)) for name, score in chosen[:max_items]]


def decide(general_scores: dict[str, float],
           min_confidence: float | dict[str, float]) -> tuple[str, str, float]:
    """(durum, gösterilecek etiket, güven) döndürür.

    ``min_confidence`` tek bir eşik ya da genel konu başına eşik sözlüğü olabilir.
    """
    top, conf = max(general_scores.items(), key=lambda kv: kv[1])
    if top == OTHER_LABEL:
        return STATUS_OUT_OF_SCOPE, OTHER_LABEL, conf
    threshold = (min_confidence.get(top, DEFAULT_MIN_CONFIDENCE)
                 if isinstance(min_confidence, dict) else min_confidence)
    if conf < threshold:
        return STATUS_UNCERTAIN, UNCERTAIN_LABEL, conf
    return STATUS_OK, top, conf


class TopicClassifier:
    """Eğitilmiş modeli yükleyip tek metin ya da toplu tahmin yapar."""

    def __init__(self, model: Any, metadata: dict):
        self.model = model
        self.metadata = metadata
        self.classes: list[str] = [str(c) for c in model.classes_]
        thresholds = metadata.get("thresholds", {})
        per_class = thresholds.get("per_class_min_confidence")
        self.min_confidence: float | dict[str, float] = (
            {str(k): float(v) for k, v in per_class.items()} if per_class
            else float(thresholds.get("min_confidence", DEFAULT_MIN_CONFIDENCE)))
        self.subtopic_threshold = float(
            thresholds.get("subtopic_threshold", DEFAULT_SUBTOPIC_THRESHOLD))
        short_t = thresholds.get("short_input_temperature")
        self.short_temperature: float | None = float(short_t) if short_t else None

    @classmethod
    def load(cls, model_path: Path = MODEL_PATH) -> TopicClassifier:
        model, metadata = load_artifact(model_path)
        return cls(model, metadata)

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def predict_proba(self, texts: list[str], short: bool = False) -> np.ndarray:
        if short and self.short_temperature:
            return self.model.predict_proba(texts, temperature=float(self.short_temperature))
        return self.model.predict_proba(texts)

    def predict_proba_auto(self, texts: list[str]) -> np.ndarray:
        """Her metin için uzunluğuna uygun sıcaklıkla olasılık üretir."""
        short = np.array([len(content_tokens(t)) <= SHORT_INPUT_MAX_TOKENS for t in texts])
        out = np.zeros((len(texts), len(self.classes)))
        for flag in (False, True):
            idx = np.flatnonzero(short == flag)
            if idx.size:
                out[idx] = self.predict_proba([texts[i] for i in idx], short=flag)
        return out

    def _word_weights(self):
        """(sözlük, katsayı matrisi, sınıflar); model yapısı beklenenden farklıysa None."""
        if getattr(self, "_weights_cache", None) is not None:
            return self._weights_cache
        try:
            pipe = self.model.estimator_
            word_vec = pipe.named_steps["vec"].transformer_list[0][1]
            clf = pipe.named_steps["clf"]
            cache = (word_vec.vocabulary_, clf.coef_, [str(c) for c in clf.classes_])
        except (AttributeError, KeyError, IndexError, TypeError):
            logger.debug("Anahtar sözcük çıkarımı bu model yapısında desteklenmiyor")
            cache = None
        self._weights_cache = cache
        return cache

    def keywords(self, text: str, general: str, k: int = 4) -> list[str]:
        """Metindeki, verilen genel konu için en yüksek pozitif model ağırlıklı sözcükler."""
        weights = self._word_weights()
        if weights is None or not general:
            return []
        vocab, coef, classes = weights
        rows = [i for i, c in enumerate(classes) if split_joint(c)[0] == general]
        if not rows:
            return []
        scored: dict[str, float] = {}
        for token in normalize(text).split():
            if len(token) <= 2 or token in TURKISH_STOPWORDS:
                continue
            col = vocab.get(f5_stem(token))
            if col is not None:
                weight = float(coef[rows, col].max())
                if weight >= KEYWORD_MIN_WEIGHT and weight > scored.get(token, 0.0):
                    scored[token] = weight
        return [t for t, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:k]]

    def predict(self, text: str) -> Prediction:
        start = time.perf_counter()
        if not isinstance(text, str):
            raise TypeError("metin str olmalı")
        clean = text.strip()
        truncated = len(clean) > MAX_INPUT_CHARS
        if truncated:
            logger.warning("Girdi %d karakterden kırpıldı", len(clean))
            clean = clean[:MAX_INPUT_CHARS]
        n_tokens = len(content_tokens(clean))
        if n_tokens == 0:
            return Prediction(text=text, status=STATUS_EMPTY, general=None, confidence=0.0,
                              latency_ms=(time.perf_counter() - start) * 1000)
        short = n_tokens <= SHORT_INPUT_MAX_TOKENS
        proba = self.predict_proba([clean], short=short)[0]
        general_scores, conditional = aggregate(self.classes, proba)
        status, label, conf = decide(general_scores, self.min_confidence)
        top_guess = max(general_scores, key=lambda k: general_scores[k])
        subtopics = (select_subtopics(conditional.get(label, {}), self.subtopic_threshold)
                     if status == STATUS_OK else [])
        return Prediction(
            text=text,
            status=status,
            general=label,
            confidence=round(conf, 4),
            general_scores={k: round(v, 4) for k, v in
                            sorted(general_scores.items(), key=lambda kv: -kv[1])},
            subtopics=subtopics,
            top_guess=top_guess,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
            truncated=truncated,
            short_input=short,
            joint_subtopic_scores={
                g: {sub: round(p * general_scores[g], 4) for sub, p in subs.items()}
                for g, subs in conditional.items()},
        )
