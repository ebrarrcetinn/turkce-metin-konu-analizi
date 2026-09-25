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
    """Doğru sınıfa verilen olasılığın ortalama -log değeri (küçük = daha iyi kalibrasyon)."""
    probabilities = softmax(logits / temperature)
    # 1e-12: olasılık 0 olursa log(0) = -sonsuz hatasını önler.
    return float(-np.log(probabilities[np.arange(len(targets)), targets] + 1e-12).mean())


def fit_temperature(logits: np.ndarray, targets: np.ndarray, low: float = 0.05,
                    high: float = 10.0, iterations: int = 60) -> float:
    """Doğrulama verisinde negatif log-olabilirliği en küçükleyen sıcaklığı (T) bulur.

    Temperature scaling (Guo vd., 2017): logitler T'ye bölünür. T > 1 olasılıkları
    yumuşatır (model aşırı eminse), T < 1 keskinleştirir (model fazla çekingense). Sınıf
    sıralaması değişmez, yani doğruluk aynı kalır; yalnızca "%80 eminim" dediğinde gerçekten
    yaklaşık %80 doğru olacak şekilde güven skorları düzeltilir. Bu, güven eşiğinin
    ("Belirsiz" kararı) anlamlı çalışması için gereklidir.

    NLL, T'ye göre tek tepeli bir fonksiyon olduğundan altın oran araması ile en küçük
    noktası türev almadan bulunur: her adımda aralık 0,618 oranında daraltılır.
    """
    ratio = (math.sqrt(5) - 1) / 2  # altın oran ≈ 0,618
    # Arama log(T) üzerinde yapılır: 0,05-1 ve 1-10 aralıkları eşit önem kazanır.
    a, b = math.log(low), math.log(high)
    # [a, b] aralığında iki iç nokta; hangisinde kayıp küçükse en küçük nokta o taraftadır.
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    f_c = negative_log_likelihood(logits, targets, math.exp(c))
    f_d = negative_log_likelihood(logits, targets, math.exp(d))
    for _ in range(iterations):
        if f_c < f_d:
            # En küçük nokta [a, d] içinde: sağ uç d'ye çekilir, eski c yeni d olur.
            b, d, f_d = d, c, f_c
            c = b - ratio * (b - a)
            f_c = negative_log_likelihood(logits, targets, math.exp(c))
        else:
            # En küçük nokta [c, b] içinde: sol uç c'ye çekilir, eski d yeni c olur.
            a, c, f_c = c, d, f_d
            d = a + ratio * (b - a)
            f_d = negative_log_likelihood(logits, targets, math.exp(d))
    # 60 adımda aralık çok küçülür; orta nokta sonuç olarak alınır.
    return math.exp((a + b) / 2)
