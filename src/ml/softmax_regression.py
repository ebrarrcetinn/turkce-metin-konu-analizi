"""
Dosya   : src/ml/softmax_regression.py
Konu    : Softmax Regresyon Sınıflandırıcı
Açıklama: Çok sınıflı lojistik (softmax) regresyonu mini-batch Adam, sınıf ağırlıkları, L2
          düzenlileştirme ve erken durdurma ile eğitir. Projenin ana modelidir.
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
    """Her satırdaki ham skorları (logit) toplamı 1 olan olasılıklara çevirir.

    p_k = e^(z_k) / Σ_j e^(z_j). Büyük skor, büyük olasılık demektir.
    """
    # En büyük değer çıkarılır: e^1000 gibi taşmalar önlenir, sonuç matematiksel olarak
    # değişmez (pay ve payda aynı sabitle bölünmüş olur).
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    """Çok sınıflı lojistik (softmax) regresyon. Projenin ana sınıflandırıcısıdır.

    Model: her sınıf için bir ağırlık sütunu vardır. Metin vektörü x ile ağırlık matrisi W
    çarpılıp bias b eklenir (z = xW + b) ve softmax ile olasılığa çevrilir. Eğitim, doğru
    sınıfın olasılığını artıracak şekilde W ve b'yi küçük adımlarla günceller.

    Kayıp: sınıf ağırlıklı çapraz entropi (-log p_doğru_sınıf) + L2 düzenlileştirme
    (λ/2 · ||W||²). Sınıf ağırlıkları n / (K · n_k) biçimindedir; az örnekli sınıfların
    (ör. Bilim) hatası daha büyük sayılır ve model bu sınıfları görmezden gelemez.
    Güncelleme: mini-batch Adam. Seyrek girdide yalnızca batch'te geçen özelliklerin
    ağırlıkları güncellenir ("lazy" güncelleme); 200 bin satırlık matrisin tamamını her
    adımda dolaşmaktan çok daha hızlıdır.
    """

    def __init__(self, learning_rate: float = 0.05, l2: float = 1e-6, epochs: int = 30,
                 batch_size: int = 256, patience: int = 3, seed: int = 42):
        self.learning_rate = learning_rate  # her adımın büyüklüğü
        self.l2 = l2  # ağırlıkların aşırı büyümesini (ezberlemeyi) cezalandırma katsayısı
        self.epochs = epochs  # eğitim verisinin en fazla kaç kez baştan sona dolaşılacağı
        self.batch_size = batch_size  # bir güncellemede kullanılan örnek sayısı
        self.patience = patience  # iyileşme olmadan kaç epoch bekleneceği (erken durdurma)
        self.seed = seed  # karıştırma sırası sabit olsun diye; sonuçlar tekrar üretilebilir
        self.classes: list[str] = []
        self.weights = np.zeros((0, 0), dtype=np.float32)  # W, boyut: özellik x sınıf
        self.bias = np.zeros(0, dtype=np.float32)  # b, boyut: sınıf
        self.best_epoch = 0
        self.history: list[dict] = []  # her epoch'un doğrulama skoru (rapor için)

    def fit(self, features: sparse.csr_matrix, labels: list[str],
            validation_score: Callable[[SoftmaxRegression], float] | None = None,
            ) -> SoftmaxRegression:
        """Eğitir. `validation_score` verilirse her epoch sonunda çağrılır ve en iyi
        skorlu epoch'un ağırlıkları saklanır (early stopping)."""
        self.classes = sorted(set(labels))
        index = {c: i for i, c in enumerate(self.classes)}
        y = np.array([index[label] for label in labels])  # etiketler sınıf numarasına
        n_samples, n_features = features.shape
        n_classes = len(self.classes)
        # Sınıf ağırlığı: n / (K · n_k). Örneğin "Diğer" 8.831, Bilim 182 örnekse Bilim
        # örneklerinin hatası yaklaşık 48 kat daha ağır sayılır.
        counts = np.bincount(y, minlength=n_classes)
        sample_weight = (n_samples / (n_classes * counts))[y].astype(np.float32)

        rng = np.random.default_rng(self.seed)
        # Ağırlıklar sıfırdan başlar: başlangıçta her sınıf eşit olasılıklıdır.
        self.weights = np.zeros((n_features, n_classes), dtype=np.float32)
        self.bias = np.zeros(n_classes, dtype=np.float32)
        adam = _Adam(n_features, n_classes, self.learning_rate)

        best_score, best_state, stale = -np.inf, None, 0
        for epoch in range(1, self.epochs + 1):
            # Her epoch'ta örnek sırası karıştırılır; model veri sırasını öğrenmez.
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch = order[start:start + self.batch_size]
                self._step(features[batch], y[batch], sample_weight[batch], adam)
            if validation_score is None:
                self.best_epoch = epoch
                continue
            # Erken durdurma: model eğitim verisini ezberlemeye başladığında doğrulama
            # skoru düşer. En iyi epoch'un ağırlıkları saklanır, `patience` epoch boyunca
            # iyileşme olmazsa eğitim durur.
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
        """Bir mini-batch için gradyanı hesaplar ve ağırlıkları günceller."""
        # Batch'te en az bir kez geçen özellik sütunları. Geçmeyen özelliklerin gradyanı
        # (L2 hariç) sıfır olduğundan yalnızca bu satırlar hesaplanır.
        columns = np.unique(x.indices)
        # Batch matrisi, yalnızca bu sütunlardan oluşan küçük bir matrise yeniden numaralanır.
        local = sparse.csr_matrix((x.data, np.searchsorted(columns, x.indices), x.indptr),
                                  shape=(x.shape[0], len(columns)))
        w_local = self.weights[columns]
        # İleri geçiş: z = xW + b, p = softmax(z)
        probabilities = softmax(local @ w_local + self.bias)
        # Çapraz entropinin logitlere göre türevi p - y'dir (y: doğru sınıf için 1, diğerleri
        # 0). Yani doğru sınıfın olasılığından 1 çıkarılır.
        error = probabilities
        error[np.arange(len(y)), y] -= 1.0
        # Her örneğin hatası sınıf ağırlığıyla ölçeklenir; toplama bölmek batch'in ortalamasını
        # alır, böylece adım büyüklüğü batch boyutuna bağlı olmaz.
        error *= (weight / weight.sum())[:, None]
        # Zincir kuralı: dKayıp/dW = xᵀ · (p - y), L2 teriminin türevi λW eklenir.
        grad_w = np.asarray(local.T @ error) + self.l2 * w_local
        grad_b = error.sum(axis=0)
        adam.update(self, columns, grad_w.astype(np.float32), grad_b.astype(np.float32))

    def decision_function(self, features: sparse.csr_matrix) -> np.ndarray:
        """Ham sınıf skorları (logit): z = xW + b."""
        return np.asarray(features @ self.weights) + self.bias

    def predict_proba(self, features: sparse.csr_matrix, temperature: float = 1.0) -> np.ndarray:
        return softmax(self.decision_function(features) / temperature)

    def predict(self, features: sparse.csr_matrix) -> list[str]:
        return [self.classes[i] for i in self.decision_function(features).argmax(axis=1)]


class _Adam:
    """Adam iyileştiricisi (Kingma & Ba, 2015), satır bazında güncelleme için uyarlanmış.

    Düz gradyan inişi her ağırlığı aynı adımla günceller. Adam her ağırlık için gradyanın
    hareketli ortalamasını (m, yön) ve karesinin hareketli ortalamasını (v, büyüklük) tutar;
    sık ve büyük gradyan alan ağırlıklarda adımı küçültür, nadir özelliklerde büyütür. Seyrek
    metin verisinde bu, düz gradyan inişinden çok daha hızlı yakınsar.
    """

    def __init__(self, n_features: int, n_classes: int, learning_rate: float,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        # beta1/beta2: ortalamaların geçmişi ne kadar hatırlayacağı (makaledeki varsayılanlar)
        # eps: sıfıra bölmeyi önleyen küçük sayı
        self.lr, self.beta1, self.beta2, self.eps = learning_rate, beta1, beta2, eps
        self.m_w = np.zeros((n_features, n_classes), dtype=np.float32)
        self.v_w = np.zeros((n_features, n_classes), dtype=np.float32)
        self.m_b: np.ndarray = np.zeros(n_classes, dtype=np.float32)
        self.v_b: np.ndarray = np.zeros(n_classes, dtype=np.float32)
        self.t = 0  # adım sayacı

    def update(self, model: SoftmaxRegression, rows: np.ndarray, grad_w: np.ndarray,
               grad_b: np.ndarray) -> None:
        self.t += 1
        # m ve v sıfırdan başladığı için ilk adımlarda küçük kalır; bu düzeltme çarpanlarıyla
        # yanlılık giderilir.
        correction1 = 1 - self.beta1 ** self.t
        correction2 = 1 - self.beta2 ** self.t
        # Yalnızca batch'te geçen satırların (özelliklerin) momentleri güncellenir.
        m = self.beta1 * self.m_w[rows] + (1 - self.beta1) * grad_w
        v = self.beta2 * self.v_w[rows] + (1 - self.beta2) * grad_w ** 2
        self.m_w[rows], self.v_w[rows] = m, v
        # Güncelleme: W ← W - lr · m̂ / (√v̂ + eps)
        model.weights[rows] -= self.lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)
        # Bias her adımda tüm sınıflar için güncellenir.
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * grad_b ** 2
        model.bias -= self.lr * (self.m_b / correction1) / (np.sqrt(self.v_b / correction2)
                                                             + self.eps)
