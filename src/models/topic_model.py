"""
Dosya   : src/models/topic_model.py
Konu    : Konu Sınıflandırma Modeli
Açıklama: Eğitilmiş modeli temsil eder: metinden genel konu ve alt konu olasılıklarını
          hesaplar, güven eşiğine göre karar verir, anahtar sözcük çıkarır ve modeli
          .npz/.json dosyalarına kaydedip yükler.
Yazar   : Ebrar Cemre Çetin
Tarih   : 25.09.2026
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    KEYWORD_MIN_WEIGHT,
    MAX_INPUT_CHARS,
    MAX_SUBTOPICS,
    MODEL_PATH,
    OTHER_LABEL,
    RELATED_TOPIC_MIN_SCORE,
    SUBTOPIC_MIN_SCORE,
    UNCERTAIN_LABEL,
)
from src.ml.softmax_regression import softmax
from src.ml.vectorizer import CombinedVectorizer, TfidfVectorizer
from src.preprocessing.text import (
    TURKISH_STOPWORDS,
    content_tokens,
    f5_preprocess,
    f5_stem,
    normalize,
)

logger = logging.getLogger(__name__)

JOINT_SEPARATOR = " > "
STATUS_OK = "ok"
STATUS_UNCERTAIN = "uncertain"
STATUS_OUT_OF_SCOPE = "out_of_scope"
STATUS_EMPTY = "empty"
FORMAT_VERSION = 1


class ModelFileError(RuntimeError):
    """Model dosyası bulunamadı ya da bozuk."""


def split_joint(label: str) -> tuple[str, str | None]:
    if JOINT_SEPARATOR in label:
        general, subtopic = label.split(JOINT_SEPARATOR, 1)
        return general, subtopic
    return label, None


def joint_label(general: str, subtopic: str | None) -> str:
    return f"{general}{JOINT_SEPARATOR}{subtopic}" if subtopic else general


def build_vectorizer() -> CombinedVectorizer:
    """Sözcük (F5 kök, 1-2 gram) ve karakter (2-5 gram) TF-IDF özellikleri."""
    stop_words = {normalize(word)[:5] for word in TURKISH_STOPWORDS}
    return CombinedVectorizer([
        TfidfVectorizer("word", (1, 2), min_df=1, preprocessor=f5_preprocess,
                        stop_words=stop_words),
        TfidfVectorizer("char", (2, 5), min_df=2, preprocessor=normalize),
    ])


@dataclass
class Prediction:
    text: str
    status: str
    general: str | None
    confidence: float
    general_scores: dict[str, float] = field(default_factory=dict)
    subtopics: list[tuple[str, float]] = field(default_factory=list)
    related_topics: list[tuple[str, float]] = field(default_factory=list)
    top_guess: str | None = None
    joint_subtopic_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    latency_ms: float = 0.0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["subtopics"] = [{"name": n, "score": s} for n, s in self.subtopics]
        payload["related_topics"] = [{"name": n, "score": s} for n, s in self.related_topics]
        return payload


def aggregate(classes: list[str], probabilities: np.ndarray) -> tuple[dict, dict]:
    """"Genel > Alt" olasılıklarından genel konu olasılıklarını (toplam) ve her genel konu
    içindeki koşullu alt konu dağılımını üretir."""
    general: dict[str, float] = defaultdict(float)
    subtopics: dict[str, dict[str, float]] = defaultdict(dict)
    for label, probability in zip(classes, probabilities, strict=True):
        topic, subtopic = split_joint(label)
        general[topic] += float(probability)
        if subtopic:
            subtopics[topic][subtopic] = float(probability)
    conditional = {topic: {s: p / general[topic] for s, p in subs.items()}
                   for topic, subs in subtopics.items() if general[topic] > 0}
    return dict(general), conditional


def select_subtopics(conditional: dict[str, float], min_score: float = SUBTOPIC_MIN_SCORE,
                     limit: int = MAX_SUBTOPICS) -> list[tuple[str, float]]:
    """En olası alt konu her zaman, diğerleri yeterince olasıysa listelenir."""
    ranked = sorted(conditional.items(), key=lambda item: item[1], reverse=True)
    chosen = ranked[:1] + [item for item in ranked[1:] if item[1] >= min_score]
    return [(name, round(score, 4)) for name, score in chosen[:limit]]


def decide(general_scores: dict[str, float], min_confidence: float) -> tuple[str, str, float]:
    """(durum, gösterilecek etiket, güven) döndürür."""
    top, confidence = max(general_scores.items(), key=lambda item: item[1])
    if top == OTHER_LABEL:
        return STATUS_OUT_OF_SCOPE, OTHER_LABEL, confidence
    if confidence < min_confidence:
        return STATUS_UNCERTAIN, UNCERTAIN_LABEL, confidence
    return STATUS_OK, top, confidence


class TopicModel:
    """Vektörleştirici + doğrusal sınıflandırıcı (skor = X·W + b) + sıcaklık kalibrasyonu."""

    def __init__(self, vectorizer: CombinedVectorizer, classes: list[str], weights: np.ndarray,
                 bias: np.ndarray, temperature: float, min_confidence: float,
                 metadata: dict | None = None):
        if weights.shape != (vectorizer.n_features, len(classes)):
            raise ModelFileError("Ağırlık matrisi boyutu vektörleştiriciyle uyuşmuyor")
        self.vectorizer = vectorizer
        self.classes = classes
        self.weights = weights
        self.bias = bias
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.metadata = metadata or {}

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "bilinmiyor"))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        logits = np.asarray(self.vectorizer.transform(texts) @ self.weights) + self.bias
        return softmax(logits / self.temperature)

    def predict(self, text: str) -> Prediction:
        start = time.perf_counter()
        clean = text.strip()
        truncated = len(clean) > MAX_INPUT_CHARS
        clean = clean[:MAX_INPUT_CHARS]
        if not content_tokens(clean):
            return Prediction(text=text, status=STATUS_EMPTY, general=None, confidence=0.0)

        general, conditional = aggregate(self.classes, self.predict_proba([clean])[0])
        status, label, confidence = decide(general, self.min_confidence)
        top_guess = max(general, key=lambda topic: general[topic])
        subtopics = select_subtopics(conditional.get(label, {})) if status == STATUS_OK else []
        related = [(topic, round(score, 4)) for topic, score in
                   sorted(general.items(), key=lambda item: item[1], reverse=True)
                   if topic not in (top_guess, OTHER_LABEL) and score >= RELATED_TOPIC_MIN_SCORE]
        return Prediction(
            text=text, status=status, general=label, confidence=round(confidence, 4),
            general_scores={t: round(s, 4) for t, s in
                            sorted(general.items(), key=lambda item: item[1], reverse=True)},
            subtopics=subtopics, related_topics=related, top_guess=top_guess,
            joint_subtopic_scores={t: {s: round(p * general[t], 4) for s, p in subs.items()}
                                   for t, subs in conditional.items()},
            latency_ms=round((time.perf_counter() - start) * 1000, 3), truncated=truncated)

    def keywords(self, text: str, general: str | None, limit: int = 4) -> list[str]:
        """Metindeki, verilen genel konu için modelde en yüksek ağırlığa sahip sözcükler."""
        if not general:
            return []
        rows = [i for i, label in enumerate(self.classes) if split_joint(label)[0] == general]
        vocabulary = self.vectorizer.vectorizers[0].vocabulary  # sözcük (F5) özellikleri
        scored: dict[str, float] = {}
        for token in normalize(text).split():
            if len(token) <= 2 or token in TURKISH_STOPWORDS:
                continue
            column = vocabulary.get(f5_stem(token))
            if column is not None and rows:
                weight = float(self.weights[column, rows].max())
                if weight >= KEYWORD_MIN_WEIGHT and weight > scored.get(token, 0.0):
                    scored[token] = weight
        return [t for t, _ in sorted(scored.items(), key=lambda item: -item[1])[:limit]]

    def save(self, path: Path = MODEL_PATH) -> None:
        """Ağırlıklar .npz, sözlük ve ayarlar .json dosyasına yazılır (pickle kullanılmaz)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {"weights": self.weights, "bias": self.bias}
        for i, vectorizer in enumerate(self.vectorizer.vectorizers):
            arrays[f"idf_{i}"] = vectorizer.idf
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        config = {"format_version": FORMAT_VERSION, "classes": self.classes,
                  "temperature": self.temperature, "min_confidence": self.min_confidence,
                  "vectorizers": [v.state() for v in self.vectorizer.vectorizers],
                  "metadata": self.metadata}
        path.with_suffix(".json").write_text(json.dumps(config, ensure_ascii=False),
                                             encoding="utf-8")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> TopicModel:
        npz_path, json_path = path.with_suffix(".npz"), path.with_suffix(".json")
        if not npz_path.exists() or not json_path.exists():
            raise ModelFileError(f"Model dosyası bulunamadı: {npz_path}")
        try:
            config = json.loads(json_path.read_text(encoding="utf-8"))
            with np.load(npz_path, allow_pickle=False) as arrays:
                data = {key: arrays[key] for key in arrays.files}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ModelFileError(f"Model dosyası okunamadı: {exc}") from exc
        if config.get("format_version") != FORMAT_VERSION:
            raise ModelFileError("Model dosyası bu sürümle uyumsuz; yeniden eğitin.")

        vectorizer = build_vectorizer()
        for i, (target, state) in enumerate(zip(vectorizer.vectorizers, config["vectorizers"],
                                                strict=True)):
            target.vocabulary = {term: j for j, term in enumerate(state["terms"])}
            target.idf = data[f"idf_{i}"]
            if len(target.idf) != len(target.vocabulary):
                raise ModelFileError("Sözlük ile IDF boyutları uyuşmuyor")
        return cls(vectorizer, config["classes"], data["weights"], data["bias"],
                   float(config["temperature"]), float(config["min_confidence"]),
                   config.get("metadata"))
