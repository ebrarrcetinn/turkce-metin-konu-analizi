"""
Dosya   : src/ml/softmax_regression.py
Konu    : Softmax Regresyon Sınıflandırıcı
Açıklama: Bu dosyada amacım projenin ana modeli olan çok sınıflı lojistik (softmax)
          regresyonu mini-batch Adam, sınıf ağırlıkları, L2 düzenlileştirme ve erken durdurma
          ile eğitmek.
Yazar   : Ebrar Cemre Çetin
Tarih   : 24.09.2026
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Her satırdaki ham skorları (logit) toplamı 1 olan olasılıklara çeviriyorum.

    p_k = e^(z_k) / Σ_j e^(z_j). Büyük skor, büyük olasılık demektir.
    """
    # En büyük değeri çıkarıyorum: böylece e^1000 gibi taşmaları önlüyorum, sonuç
    # matematiksel olarak değişmiyor (pay ve payda aynı sabitle bölünmüş oluyor).
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    """Amacım çok sınıflı lojistik (softmax) regresyonu projenin ana sınıflandırıcısı olarak
    kurmak.

    Model: her sınıf için bir ağırlık sütunu tutuyorum. Metin vektörü x ile ağırlık matrisi
    W'yi çarpıp bias b ekliyorum (z = xW + b) ve softmax ile olasılığa çeviriyorum. Eğitimde
    doğru sınıfın olasılığını artıracak şekilde W ve b'yi küçük adımlarla güncelliyorum.

    Kayıp: sınıf ağırlıklı çapraz entropi (-log p_doğru_sınıf) + L2 düzenlileştirme
    (λ/2 · ||W||²). Sınıf ağırlıklarını n / (K · n_k) biçiminde seçtim; az örnekli sınıfların
    (ör. Bilim) hatasını daha büyük sayıyorum ve model bu sınıfları görmezden gelemiyor.
    Güncelleme: mini-batch Adam. Seyrek girdide yalnızca batch'te geçen özelliklerin
    ağırlıklarını güncelliyorum ("lazy" güncelleme); bu, 200 bin satırlık matrisin tamamını
    her adımda dolaşmaktan çok daha hızlı.
    """

    def __init__(self, learning_rate: float = 0.05, l2: float = 1e-6, epochs: int = 30,
                 batch_size: int = 256, patience: int = 3, seed: int = 42):
        self.learning_rate = learning_rate  # her adımın büyüklüğü
        self.l2 = l2  # ağırlıkların aşırı büyümesini (ezberlemeyi) cezalandırma katsayısı
        self.epochs = epochs  # eğitim verisini en fazla kaç kez baştan sona dolaşacağım
        self.batch_size = batch_size  # bir güncellemede kullanılan örnek sayısı
        self.patience = patience  # iyileşme olmadan kaç epoch bekleyeceğim (erken durdurma)
        self.seed = seed  # karıştırma sırasını sabitliyorum; sonuçlar tekrar üretilebilir
        self.classes: list[str] = []
        self.weights = np.zeros((0, 0), dtype=np.float32)  # W, boyut: özellik x sınıf
        self.bias = np.zeros(0, dtype=np.float32)  # b, boyut: sınıf
        self.best_epoch = 0
        self.history: list[dict] = []  # her epoch'un doğrulama skoru (rapor için)

    def fit(self, features: sparse.csr_matrix, labels: list[str],
            validation_score: Callable[[SoftmaxRegression], float] | None = None,
            ) -> SoftmaxRegression:
        """Modeli eğitiyorum. `validation_score` verilirse onu her epoch sonunda çağırıyor ve
        en iyi skorlu epoch'un ağırlıklarını saklıyorum (early stopping)."""
        self.classes = sorted(set(labels))
        index = {c: i for i, c in enumerate(self.classes)}
        y = np.array([index[label] for label in labels])  # etiketleri sınıf numarasına çeviriyorum
        n_samples, n_features = features.shape
        n_classes = len(self.classes)
        # Sınıf ağırlığı: n / (K · n_k). Örneğin "Diğer" 8.831, Bilim 182 örnekse Bilim
        # örneklerinin hatasını yaklaşık 48 kat daha ağır sayıyorum.
        counts = np.bincount(y, minlength=n_classes)
        sample_weight = (n_samples / (n_classes * counts))[y].astype(np.float32)

        rng = np.random.default_rng(self.seed)
        # Ağırlıkları sıfırdan başlatıyorum: başlangıçta her sınıf eşit olasılıklı.
        self.weights = np.zeros((n_features, n_classes), dtype=np.float32)
        self.bias = np.zeros(n_classes, dtype=np.float32)
        adam = _Adam(n_features, n_classes, self.learning_rate)

        best_score, best_state, stale = -np.inf, None, 0
        for epoch in range(1, self.epochs + 1):
            # Her epoch'ta örnek sırasını karıştırıyorum; böylece model veri sırasını öğrenmiyor.
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch = order[start:start + self.batch_size]
                self._step(features[batch], y[batch], sample_weight[batch], adam)
            if validation_score is None:
                self.best_epoch = epoch
                continue
            # Erken durdurma: model eğitim verisini ezberlemeye başladığında doğrulama
            # skoru düşer. En iyi epoch'un ağırlıklarını saklıyorum, `patience` epoch boyunca
            # iyileşme olmazsa eğitimi durduruyorum.
            score = validation_score(self)
            self.history.append({"epoch": epoch, "val_score": round(score, 4)})
            logger.info("epoch %d: doğrulama skoru %.4f", epoch, score)
            if score > best_score:
                best_score, stale, self.best_epoch = score, 0, epoch
                best_state = (self.weights.copy(), self.bias.copy())
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.weights, self.bias = best_state
        return self

    def _step(self, x: sparse.csr_matrix, y: np.ndarray, weight: np.ndarray,
              adam: _Adam) -> None:
        """Bir mini-batch için gradyanı hesaplıyor ve ağırlıkları güncelliyorum."""
        # Batch'te en az bir kez geçen özellik sütunlarını buluyorum. Geçmeyen özelliklerin
        # gradyanı (L2 hariç) sıfır olduğundan yalnızca bu satırları hesaplıyorum.
        columns = np.unique(x.indices)
        # Batch matrisini, yalnızca bu sütunlardan oluşan küçük bir matrise yeniden
        # numaralandırıyorum.
        local = sparse.csr_matrix((x.data, np.searchsorted(columns, x.indices), x.indptr),
                                  shape=(x.shape[0], len(columns)))
        w_local = self.weights[columns]
        # İleri geçiş: z = xW + b, p = softmax(z)
        probabilities = softmax(local @ w_local + self.bias)
        # Çapraz entropinin logitlere göre türevi p - y'dir (y: doğru sınıf için 1, diğerleri
        # 0). Yani doğru sınıfın olasılığından 1 çıkarıyorum.
        error = probabilities
        error[np.arange(len(y)), y] -= 1.0
        # Her örneğin hatasını sınıf ağırlığıyla ölçekliyorum; toplama bölerek batch'in
        # ortalamasını alıyorum, böylece adım büyüklüğü batch boyutuna bağlı olmuyor.
        error *= (weight / weight.sum())[:, None]
        # Zincir kuralı: dKayıp/dW = xᵀ · (p - y); L2 teriminin türevi λW'yi de ekliyorum.
        grad_w = np.asarray(local.T @ error) + self.l2 * w_local
        grad_b = error.sum(axis=0)
        adam.update(self, columns, grad_w.astype(np.float32), grad_b.astype(np.float32))

    def decision_function(self, features: sparse.csr_matrix) -> np.ndarray:
        """Ham sınıf skorlarını (logit) hesaplıyorum: z = xW + b."""
        return np.asarray(features @ self.weights) + self.bias

    def predict_proba(self, features: sparse.csr_matrix, temperature: float = 1.0) -> np.ndarray:
        return softmax(self.decision_function(features) / temperature)

    def predict(self, features: sparse.csr_matrix) -> list[str]:
        return [self.classes[i] for i in self.decision_function(features).argmax(axis=1)]


class _Adam:
    """Adam iyileştiricisi (Kingma & Ba, 2015); satır bazında güncelleme için uyarladım.

    Düz gradyan inişi her ağırlığı aynı adımla günceller. Adam'da ise her ağırlık için
    gradyanın hareketli ortalamasını (m, yön) ve karesinin hareketli ortalamasını
    (v, büyüklük) tutuyorum; sık ve büyük gradyan alan ağırlıklarda adımı küçültüp nadir
    özelliklerde büyütüyorum. Seyrek metin verisinde bu, düz gradyan inişinden çok daha
    hızlı yakınsıyor.
    """

    def __init__(self, n_features: int, n_classes: int, learning_rate: float,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        # beta1/beta2: ortalamaların geçmişi ne kadar hatırlayacağı (makaledeki varsayılanları
        # kullandım)
        # eps: sıfıra bölmeyi önlemek için eklediğim küçük sayı
        self.lr, self.beta1, self.beta2, self.eps = learning_rate, beta1, beta2, eps
        self.m_w = np.zeros((n_features, n_classes), dtype=np.float32)
        self.v_w = np.zeros((n_features, n_classes), dtype=np.float32)
        self.m_b: np.ndarray = np.zeros(n_classes, dtype=np.float32)
        self.v_b: np.ndarray = np.zeros(n_classes, dtype=np.float32)
        self.t = 0  # adım sayacı

    def update(self, model: SoftmaxRegression, rows: np.ndarray, grad_w: np.ndarray,
               grad_b: np.ndarray) -> None:
        self.t += 1
        # m ve v sıfırdan başladığı için ilk adımlarda küçük kalıyor; bu düzeltme
        # çarpanlarıyla yanlılığı gideriyorum.
        correction1 = 1 - self.beta1 ** self.t
        correction2 = 1 - self.beta2 ** self.t
        # Yalnızca batch'te geçen satırların (özelliklerin) momentlerini güncelliyorum.
        m = self.beta1 * self.m_w[rows] + (1 - self.beta1) * grad_w
        v = self.beta2 * self.v_w[rows] + (1 - self.beta2) * grad_w ** 2
        self.m_w[rows], self.v_w[rows] = m, v
        # Güncelleme: W ← W - lr · m̂ / (√v̂ + eps)
        model.weights[rows] -= self.lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)
        # Bias'ı her adımda tüm sınıflar için güncelliyorum.
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * grad_b ** 2
        model.bias -= self.lr * (self.m_b / correction1) / (np.sqrt(self.v_b / correction2)
                                                             + self.eps)
