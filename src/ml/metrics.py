"""
Dosya   : src/ml/metrics.py
Konu    : Değerlendirme Metrikleri
Açıklama: Doğruluk, sınıf bazında precision/recall/F1, macro/weighted F1, karmaşıklık matrisi
          ve beklenen kalibrasyon hatasını (ECE) hesaplar.
Yazar   : Ebrar Cemre Çetin
Tarih   : 24.09.2026
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class ClassificationReport:
    """Doğruluk, sınıf bazında precision/recall/F1 ve karmaşıklık matrisini hesaplar."""

    def __init__(self, y_true: Sequence[str], y_pred: Sequence[str],
                 labels: Sequence[str] | None = None):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true ve y_pred aynı uzunlukta olmalı")
        self.labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
        index = {label: i for i, label in enumerate(self.labels)}
        size = len(self.labels)
        self.confusion = np.zeros((size, size), dtype=int)
        for true, pred in zip(y_true, y_pred, strict=True):
            if true in index and pred in index:
                self.confusion[index[true], index[pred]] += 1
        self.n = len(y_true)
        correct = sum(t == p for t, p in zip(y_true, y_pred, strict=True))
        self.accuracy = correct / self.n if self.n else 0.0

        # Satırlar gerçek, sütunlar tahmin: köşegen doğru tahminlerdir.
        true_positive = np.diag(self.confusion).astype(float)
        predicted = self.confusion.sum(axis=0).astype(float)
        support = self.confusion.sum(axis=1).astype(float)
        self.precision = _safe_divide(true_positive, predicted)
        self.recall = _safe_divide(true_positive, support)
        self.f1 = _safe_divide(2 * self.precision * self.recall, self.precision + self.recall)
        self.support = support

    @property
    def macro_f1(self) -> float:
        return float(self.f1.mean())

    @property
    def weighted_f1(self) -> float:
        total = self.support.sum()
        return float((self.f1 * self.support).sum() / total) if total else 0.0

    def summary(self, digits: int = 4) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, digits),
            "precision_macro": round(float(self.precision.mean()), digits),
            "recall_macro": round(float(self.recall.mean()), digits),
            "macro_f1": round(self.macro_f1, digits),
            "weighted_f1": round(self.weighted_f1, digits),
        }

    def per_class(self, digits: int = 4) -> dict[str, dict]:
        return {
            label: {"precision": round(float(self.precision[i]), digits),
                    "recall": round(float(self.recall[i]), digits),
                    "f1": round(float(self.f1[i]), digits),
                    "support": int(self.support[i])}
            for i, label in enumerate(self.labels)
        }


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def expected_calibration_error(confidences, correct, n_bins: int = 15) -> float:
    """Güven aralıklarındaki |doğruluk - ortalama güven| farklarının ağırlıklı ortalaması."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidences.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)
