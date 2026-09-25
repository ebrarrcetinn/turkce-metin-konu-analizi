"""
Dosya   : src/ml/vectorizer.py
Konu    : TF-IDF Vektörleştirici
Açıklama: Metinleri sözcük ve karakter n-gram'larından oluşan TF-IDF ağırlıklı, L2 normlu
          seyrek vektörlere dönüştürür. Hazır kütüphane kullanılmadan yazılmıştır.
Yazar   : Ebrar Cemre Çetin
Tarih   : 24.09.2026
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable

import numpy as np
from scipy import sparse


class TfidfVectorizer:
    """Metinleri TF-IDF ağırlıklı, L2 normlu seyrek vektörlere dönüştürür.

    analyzer="word": sözcük n-gram'ları (stopword'ler n-gram kurulmadan önce atılır).
    analyzer="char": her sözcüğün başına/sonuna boşluk eklenerek karakter n-gram'ları.
    Karakter n-gram'ları Türkçedeki ekleri ve yazım varyasyonlarını kökleme gerekmeden yakalar.
    """

    def __init__(self, analyzer: str = "word", ngram_range: tuple[int, int] = (1, 1),
                 min_df: int = 1, preprocessor: Callable[[str], str] | None = None,
                 stop_words: Iterable[str] = ()):
        if analyzer not in ("word", "char"):
            raise ValueError("analyzer 'word' veya 'char' olmalı")
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.preprocessor = preprocessor
        self.stop_words = frozenset(stop_words)
        self.vocabulary: dict[str, int] = {}
        self.idf = np.zeros(0, dtype=np.float32)

    def _features(self, text: str) -> list[str]:
        if self.preprocessor is not None:
            text = self.preprocessor(text)
        low, high = self.ngram_range
        if self.analyzer == "word":
            tokens = [t for t in text.split() if len(t) > 1 and t not in self.stop_words]
            return [" ".join(tokens[i:i + n]) for n in range(low, high + 1)
                    for i in range(len(tokens) - n + 1)]
        grams: list[str] = []
        for word in text.split():
            padded = f" {word} "
            for n in range(low, high + 1):
                grams.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
        return grams

    def fit(self, texts: list[str]) -> TfidfVectorizer:
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(self._features(text)))
        kept = sorted(term for term, df in document_frequency.items() if df >= self.min_df)
        self.vocabulary = {term: i for i, term in enumerate(kept)}
        # Yumuşatılmış IDF: hiçbir terim sıfır ağırlık almaz, sık terimler baskılanır.
        n_docs = len(texts)
        self.idf = np.array([math.log((1 + n_docs) / (1 + document_frequency[t])) + 1
                             for t in kept], dtype=np.float32)
        return self

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        indptr, indices, values = [0], [], []
        for text in texts:
            counts = Counter(self.vocabulary[f] for f in self._features(text)
                             if f in self.vocabulary)
            columns = sorted(counts)
            # Alt doğrusal TF (1 + log tf): tekrar eden terimlerin etkisi sınırlanır.
            weights = np.array([(1 + math.log(counts[c])) * self.idf[c] for c in columns],
                               dtype=np.float32)
            norm = float(np.sqrt((weights ** 2).sum()))
            if norm > 0:
                weights /= norm
            indices.extend(columns)
            values.extend(weights.tolist())
            indptr.append(len(indices))
        return sparse.csr_matrix(
            (np.asarray(values, dtype=np.float32), np.asarray(indices, dtype=np.int32),
             np.asarray(indptr, dtype=np.int64)),
            shape=(len(texts), len(self.vocabulary)))

    def fit_transform(self, texts: list[str]) -> sparse.csr_matrix:
        return self.fit(texts).transform(texts)

    def state(self) -> dict:
        return {"analyzer": self.analyzer, "ngram_range": list(self.ngram_range),
                "min_df": self.min_df, "terms": list(self.vocabulary)}


class CombinedVectorizer:
    """Birden fazla vektörleştiricinin çıktısını yan yana birleştirir."""

    def __init__(self, vectorizers: list[TfidfVectorizer]):
        self.vectorizers = vectorizers

    @property
    def n_features(self) -> int:
        return sum(len(v.vocabulary) for v in self.vectorizers)

    def fit(self, texts: list[str]) -> CombinedVectorizer:
        for vectorizer in self.vectorizers:
            vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        return sparse.hstack([v.transform(texts) for v in self.vectorizers], format="csr")

    def fit_transform(self, texts: list[str]) -> sparse.csr_matrix:
        return self.fit(texts).transform(texts)

    def feature_offset(self, index: int) -> int:
        return sum(len(v.vocabulary) for v in self.vectorizers[:index])
