"""
Dosya   : src/ml/naive_bayes.py
Konu    : Çok Terimli Naive Bayes Sınıflandırıcı
Açıklama: Karşılaştırma (temel model) için Laplace düzeltmeli çok terimli Naive Bayes
          algoritmasını uygular.
Yazar   : Ebrar Cemre Çetin
Tarih   : 24.09.2026
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


class MultinomialNaiveBayes:
    """Çok terimli Naive Bayes; karşılaştırma için temel (baseline) model.

    P(sınıf | metin) ∝ P(sınıf) · Π P(özellik | sınıf)^değer. Hesap log uzayında yapılır.
    """

    def __init__(self, alpha: float = 0.3, uniform_prior: bool = True):
        self.alpha = alpha
        self.uniform_prior = uniform_prior
        self.classes: list[str] = []

    def fit(self, features: sparse.csr_matrix, labels: list[str]) -> MultinomialNaiveBayes:
        self.classes = sorted(set(labels))
        index = {c: i for i, c in enumerate(self.classes)}
        y = np.array([index[label] for label in labels])
        one_hot = sparse.csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))),
                                    shape=(len(self.classes), len(y)))
        feature_totals = np.asarray((one_hot @ features).todense())
        # Laplace (alpha) düzeltmesi: eğitimde hiç görülmeyen özellik olasılığı sıfırlamaz.
        smoothed = feature_totals + self.alpha
        self.feature_log_prob = np.log(smoothed / smoothed.sum(axis=1, keepdims=True)).T
        counts = np.bincount(y, minlength=len(self.classes))
        prior = np.full(len(self.classes), 1 / len(self.classes)) if self.uniform_prior \
            else counts / counts.sum()
        self.class_log_prior = np.log(prior)
        return self

    def decision_function(self, features: sparse.csr_matrix) -> np.ndarray:
        return np.asarray(features @ self.feature_log_prob) + self.class_log_prior

    def predict(self, features: sparse.csr_matrix) -> list[str]:
        best = self.decision_function(features).argmax(axis=1)
        return [self.classes[i] for i in best]
