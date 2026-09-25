"""
Dosya   : src/ml/calibration.py
Konu    : Güven Skoru Kalibrasyonu
Açıklama: Bu dosyada amacım temperature scaling yöntemiyle model olasılıklarını gerçek
          doğruluğa yaklaştırmak; en iyi sıcaklığı altın oran aramasıyla buluyorum.
Yazar   : Ebrar Cemre Çetin
Tarih   : 24.09.2026
"""

from __future__ import annotations

import math

import numpy as np

from src.ml.softmax_regression import softmax


def negative_log_likelihood(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    """Doğru sınıfa verilen olasılığın ortalama -log değerini hesaplıyorum
    (küçük = daha iyi kalibrasyon)."""
    probabilities = softmax(logits / temperature)
    # 1e-12 ekleyerek olasılık 0 olduğunda log(0) = -sonsuz hatasını önlüyorum.
    return float(-np.log(probabilities[np.arange(len(targets)), targets] + 1e-12).mean())


def fit_temperature(logits: np.ndarray, targets: np.ndarray, low: float = 0.05,
                    high: float = 10.0, iterations: int = 60) -> float:
    """Amacım doğrulama verisinde negatif log-olabilirliği en küçükleyen sıcaklığı (T) bulmak.

    Temperature scaling (Guo vd., 2017): logitleri T'ye bölüyorum. T > 1 olasılıkları
    yumuşatır (model aşırı eminse), T < 1 keskinleştirir (model fazla çekingense). Sınıf
    sıralaması değişmez, yani doğruluk aynı kalır; yalnızca "%80 eminim" dediğinde gerçekten
    yaklaşık %80 doğru olacak şekilde güven skorlarını düzeltiyorum. Bu, güven eşiğinin
    ("Belirsiz" kararı) anlamlı çalışması için gerekli.

    NLL, T'ye göre tek tepeli bir fonksiyon olduğundan en küçük noktasını altın oran
    aramasıyla türev almadan buluyorum: her adımda aralığı 0,618 oranında daraltıyorum.
    """
    ratio = (math.sqrt(5) - 1) / 2  # altın oran ≈ 0,618
    # Aramayı log(T) üzerinde yapıyorum: 0,05-1 ve 1-10 aralıkları eşit önem kazanıyor.
    a, b = math.log(low), math.log(high)
    # [a, b] aralığında iki iç nokta alıyorum; hangisinde kayıp küçükse en küçük nokta o
    # taraftadır.
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    f_c = negative_log_likelihood(logits, targets, math.exp(c))
    f_d = negative_log_likelihood(logits, targets, math.exp(d))
    for _ in range(iterations):
        if f_c < f_d:
            # En küçük nokta [a, d] içinde: sağ ucu d'ye çekiyorum, eski c yeni d oluyor.
            b, d, f_d = d, c, f_c
            c = b - ratio * (b - a)
            f_c = negative_log_likelihood(logits, targets, math.exp(c))
        else:
            # En küçük nokta [c, b] içinde: sol ucu c'ye çekiyorum, eski d yeni c oluyor.
            a, c, f_c = c, d, f_d
            d = a + ratio * (b - a)
            f_d = negative_log_likelihood(logits, targets, math.exp(d))
    # 60 adımda aralık çok küçülüyor; sonuç olarak orta noktayı alıyorum.
    return math.exp((a + b) / 2)
