"""
Dosya   : src/ml/calibration.py
Konu    : Güven Skoru Kalibrasyonu
Açıklama: Temperature scaling yöntemiyle model olasılıklarını gerçek doğruluğa yaklaştırır;
          en iyi sıcaklık altın oran aramasıyla bulunur.
Yazar   : Ebrar Cemre Çetin
Tarih   : 24.09.2026
"""

from __future__ import annotations

import math

import numpy as np

from src.ml.softmax_regression import softmax


def negative_log_likelihood(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    probabilities = softmax(logits / temperature)
    return float(-np.log(probabilities[np.arange(len(targets)), targets] + 1e-12).mean())


def fit_temperature(logits: np.ndarray, targets: np.ndarray, low: float = 0.05,
                    high: float = 10.0, iterations: int = 60) -> float:
    """Doğrulama verisinde negatif log-olabilirliği en küçükleyen sıcaklığı bulur.

    Temperature scaling (Guo vd., 2017): logitler T'ye bölünür. Sınıf sıralaması değişmez,
    yalnızca güven skorları gerçek doğruluğa yaklaştırılır. NLL, T'ye göre tek tepeli
    olduğundan altın oran araması yeterlidir.
    """
    ratio = (math.sqrt(5) - 1) / 2
    a, b = math.log(low), math.log(high)  # arama log uzayında daha dengeli
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    f_c = negative_log_likelihood(logits, targets, math.exp(c))
    f_d = negative_log_likelihood(logits, targets, math.exp(d))
    for _ in range(iterations):
        if f_c < f_d:
            b, d, f_d = d, c, f_c
            c = b - ratio * (b - a)
            f_c = negative_log_likelihood(logits, targets, math.exp(c))
        else:
            a, c, f_c = c, d, f_d
            d = a + ratio * (b - a)
            f_d = negative_log_likelihood(logits, targets, math.exp(d))
    return math.exp((a + b) / 2)
