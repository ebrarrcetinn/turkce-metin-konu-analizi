"""
Dosya   : src/ml/naive_bayes.py
Konu    : Çok Terimli Naive Bayes Sınıflandırıcı
Açıklama: Bu dosyada amacım karşılaştırma (temel model) için Laplace düzeltmeli çok
          terimli Naive Bayes algoritmasını uygulamak.
Yazar   : Ebrar Cemre Çetin
Tarih   : 24.09.2026
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


class MultinomialNaiveBayes:
    """Amacım softmax regresyonla karşılaştırmak için çok terimli Naive Bayes'i temel
    (baseline) model olarak kurmak.

    Bayes kuralı: P(sınıf | metin) ∝ P(sınıf) · Π P(özellik | sınıf)^değer.
    "Naive" (saf) varsayım: sınıf verildiğinde özellikleri birbirinden bağımsız kabul ediyorum.
    Çok sayıda küçük olasılığın çarpımı sayısal olarak sıfıra yuvarlanacağı için hesabı
    logaritma ile toplamaya çevirerek yapıyorum.
    """

    def __init__(self, alpha: float = 1.0, uniform_prior: bool = True):
        self.alpha = alpha  # Laplace düzeltme miktarı
        self.uniform_prior = uniform_prior  # tüm sınıflara eşit önsel olasılık verip vermeyeceğim
        self.classes: list[str] = []
        self.feature_log_prob = np.zeros((0, 0))  # log P(özellik | sınıf), boyut F x K
        self.class_log_prior = np.zeros(0)  # log P(sınıf), boyut K

    def fit(self, features: sparse.csr_matrix, labels: list[str]) -> MultinomialNaiveBayes:
        self.classes = sorted(set(labels))
        index = {c: i for i, c in enumerate(self.classes)}
        y = np.array([index[label] for label in labels])
        # K x N boyutlu "hangi örnek hangi sınıfta" matrisini kuruyorum. Bunu özellik
        # matrisiyle çarparak her sınıf için özellik değerlerinin toplamını tek matris
        # çarpımında alıyorum.
        one_hot = sparse.csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))),
                                    shape=(len(self.classes), len(y)))
        feature_totals = np.asarray((one_hot @ features).todense())
        # Laplace düzeltmesi: bir sınıfta hiç görülmemiş özelliğin olasılığı 0 olsaydı, o
        # özelliği içeren her metin için sınıf olasılığı da 0 olurdu. Bunu alpha ekleyerek
        # önlüyorum.
        smoothed = feature_totals + self.alpha
        # Her sınıf satırını kendi toplamına bölerek P(özellik | sınıf) değerini elde ediyorum.
        self.feature_log_prob = np.log(smoothed / smoothed.sum(axis=1, keepdims=True)).T
        counts = np.bincount(y, minlength=len(self.classes))
        # Veride "Diğer" sınıfı çok baskın olduğu için eşit önsel kullanmayı seçtim; bu,
        # az örnekli sınıfların (ör. Bilim) sürekli ezilmesini önlüyor.
        prior = np.full(len(self.classes), 1 / len(self.classes)) if self.uniform_prior \
            else counts / counts.sum()
        self.class_log_prior = np.log(prior)
        return self

    def decision_function(self, features: sparse.csr_matrix) -> np.ndarray:
        """Her sınıf için log P(sınıf) + Σ değer · log P(özellik | sınıf) skorlarını
        hesaplıyorum."""
        return np.asarray(features @ self.feature_log_prob) + self.class_log_prior

    def predict(self, features: sparse.csr_matrix) -> list[str]:
        best = self.decision_function(features).argmax(axis=1)
        return [self.classes[i] for i in best]
