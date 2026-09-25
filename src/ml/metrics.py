"""
Dosya   : src/ml/metrics.py
Konu    : Değerlendirme Metrikleri
Açıklama: Bu dosyada amacım doğruluk, sınıf bazında precision/recall/F1, macro/weighted F1,
          karmaşıklık matrisi ve beklenen kalibrasyon hatasını (ECE) hesaplamak.
Yazar   : Ebrar Cemre Çetin
Tarih   : 24.09.2026
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class ClassificationReport:
    """Amacım doğruluğu, sınıf bazında precision/recall/F1'ı ve karmaşıklık matrisini
    hesaplamak.

    precision: modelin "X" dediklerinin ne kadarı gerçekten X
    recall   : gerçekten X olanların ne kadarını model X olarak buldu
    F1       : ikisinin harmonik ortalaması; biri düşükse F1 de düşer
    macro F1 : sınıfların F1 ortalaması; her sınıfı eşit önemde sayıyorum. Veride "Diğer"
               sınıfı çok büyük olduğundan başarıyı doğruluk yerine bununla ölçmeyi seçtim.
    """

    def __init__(self, y_true: Sequence[str], y_pred: Sequence[str],
                 labels: Sequence[str] | None = None):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true ve y_pred aynı uzunlukta olmalı")
        self.labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
        index = {label: i for i, label in enumerate(self.labels)}
        size = len(self.labels)
        # Karmaşıklık matrisinde satırı gerçek sınıf, sütunu tahmin olarak tutuyorum;
        # [i][j] = i olup j denenler.
        self.confusion = np.zeros((size, size), dtype=int)
        for true, pred in zip(y_true, y_pred, strict=True):
            if true in index and pred in index:
                self.confusion[index[true], index[pred]] += 1
        self.n = len(y_true)
        correct = sum(t == p for t, p in zip(y_true, y_pred, strict=True))
        self.accuracy = correct / self.n if self.n else 0.0

        # Köşegeni doğru tahminler (true positive) olarak alıyorum. Sütun toplamı o sınıfa
        # verilen tahmin sayısı, satır toplamı o sınıfın gerçek örnek sayısıdır (support).
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
        """F1'ı örnek sayısıyla ağırlıklandırıyorum; büyük sınıflar daha çok etki ediyor."""
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
    """Payda 0 olan yerlerde (ör. hiç tahmin edilmemiş sınıf) sonucu 0 kabul ediyorum."""
    result = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def expected_calibration_error(confidences, correct, n_bins: int = 15) -> float:
    """Amacım beklenen kalibrasyon hatasıyla (ECE) modelin güveninin gerçek doğruluğuyla ne
    kadar uyumlu olduğunu ölçmek.

    Tahminleri güven değerine göre 15 aralığa (0-0,067, 0,067-0,133, ...) ayırıyorum. Her
    aralıkta |doğruluk - ortalama güven| farkını alıp aralıktaki örnek oranıyla
    ağırlıklandırıyorum. 0 mükemmel uyum demektir.
    """
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
