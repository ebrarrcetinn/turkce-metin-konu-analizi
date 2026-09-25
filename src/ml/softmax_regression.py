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
    # En büyük değeri çıkarmak üstel fonksiyonda taşmayı önler; sonuç değişmez.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    """Çok sınıflı lojistik (softmax) regresyon, mini-batch Adam ile eğitilir.

    Kayıp: sınıf ağırlıklı çapraz entropi + L2 düzenlileştirme. Sınıf ağırlıkları
    n / (K · n_k) biçimindedir; az örnekli sınıflar (ör. Bilim) eğitimde bastırılmaz.
    Seyrek girdide yalnızca batch'te geçen özelliklerin ağırlıkları güncellenir ("lazy"
    Adam); bu, tüm ağırlık matrisini her adımda dolaşmaktan çok daha hızlıdır.
    """

    def __init__(self, learning_rate: float = 0.05, l2: float = 1e-6, epochs: int = 30,
                 batch_size: int = 256, patience: int = 3, seed: int = 42):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.classes: list[str] = []
        self.weights = np.zeros((0, 0), dtype=np.float32)
        self.bias = np.zeros(0, dtype=np.float32)
        self.best_epoch = 0
        self.history: list[dict] = []

    def fit(self, features: sparse.csr_matrix, labels: list[str],
            validation_score: Callable[[SoftmaxRegression], float] | None = None,
            ) -> SoftmaxRegression:
        """Eğitir. `validation_score` verilirse her epoch sonunda çağrılır ve en iyi
        skorlu epoch'un ağırlıkları saklanır (early stopping)."""
        self.classes = sorted(set(labels))
        index = {c: i for i, c in enumerate(self.classes)}
        y = np.array([index[label] for label in labels])
        n_samples, n_features = features.shape
        n_classes = len(self.classes)
        counts = np.bincount(y, minlength=n_classes)
        sample_weight = (n_samples / (n_classes * counts))[y].astype(np.float32)

        rng = np.random.default_rng(self.seed)
        self.weights = np.zeros((n_features, n_classes), dtype=np.float32)
        self.bias = np.zeros(n_classes, dtype=np.float32)
        adam = _Adam(n_features, n_classes, self.learning_rate)

        best_score, best_state, stale = -np.inf, None, 0
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch = order[start:start + self.batch_size]
                self._step(features[batch], y[batch], sample_weight[batch], adam)
            if validation_score is None:
                self.best_epoch = epoch
                continue
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
        # Batch'te geçen sütunlar; gradyan yalnızca bu satırlar için hesaplanır.
        columns = np.unique(x.indices)
        local = sparse.csr_matrix((x.data, np.searchsorted(columns, x.indices), x.indptr),
                                  shape=(x.shape[0], len(columns)))
        w_local = self.weights[columns]
        probabilities = softmax(local @ w_local + self.bias)
        # Çapraz entropinin logitlere göre türevi: (p - y), örnek ağırlığıyla ölçeklenir.
        error = probabilities
        error[np.arange(len(y)), y] -= 1.0
        error *= (weight / weight.sum())[:, None]
        grad_w = np.asarray(local.T @ error) + self.l2 * w_local
        grad_b = error.sum(axis=0)
        adam.update(self, columns, grad_w.astype(np.float32), grad_b.astype(np.float32))

    def decision_function(self, features: sparse.csr_matrix) -> np.ndarray:
        return np.asarray(features @ self.weights) + self.bias

    def predict_proba(self, features: sparse.csr_matrix, temperature: float = 1.0) -> np.ndarray:
        return softmax(self.decision_function(features) / temperature)

    def predict(self, features: sparse.csr_matrix) -> list[str]:
        return [self.classes[i] for i in self.decision_function(features).argmax(axis=1)]


class _Adam:
    """Satır bazında güncellenen Adam iyileştiricisi (Kingma & Ba, 2015)."""

    def __init__(self, n_features: int, n_classes: int, learning_rate: float,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = learning_rate, beta1, beta2, eps
        self.m_w = np.zeros((n_features, n_classes), dtype=np.float32)
        self.v_w = np.zeros((n_features, n_classes), dtype=np.float32)
        self.m_b: np.ndarray = np.zeros(n_classes, dtype=np.float32)
        self.v_b: np.ndarray = np.zeros(n_classes, dtype=np.float32)
        self.t = 0

    def update(self, model: SoftmaxRegression, rows: np.ndarray, grad_w: np.ndarray,
               grad_b: np.ndarray) -> None:
        self.t += 1
        # Birinci ve ikinci moment tahminleri; başlangıçtaki sıfır yanlılığı düzeltilir.
        correction1 = 1 - self.beta1 ** self.t
        correction2 = 1 - self.beta2 ** self.t
        m = self.beta1 * self.m_w[rows] + (1 - self.beta1) * grad_w
        v = self.beta2 * self.v_w[rows] + (1 - self.beta2) * grad_w ** 2
        self.m_w[rows], self.v_w[rows] = m, v
        model.weights[rows] -= self.lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * grad_b ** 2
        model.bias -= self.lr * (self.m_b / correction1) / (np.sqrt(self.v_b / correction2)
                                                             + self.eps)
