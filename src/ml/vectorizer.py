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

    Model sayılarla çalıştığı için her metin, sözlükteki her terim için bir sütunu olan bir
    vektöre çevrilir. Bir metinde sözlüğün çok küçük bir kısmı geçtiğinden vektörün büyük
    çoğunluğu sıfırdır; bu yüzden yalnızca sıfır olmayan değerleri tutan seyrek (sparse)
    matris kullanılır.

    analyzer="word": sözcük n-gram'ları ("kuantum", "kuantum bilgisayar").
    analyzer="char": karakter n-gram'ları ("kua", "uan", "ant"). Türkçe eklemeli bir dil
    olduğu için "kitap", "kitaplar", "kitabın" gibi biçimler ortak karakter parçaları
    üzerinden birbirine bağlanır; kökleme hatalarına karşı dayanıklıdır.
    """

    def __init__(self, analyzer: str = "word", ngram_range: tuple[int, int] = (1, 1),
                 min_df: int = 1, preprocessor: Callable[[str], str] | None = None,
                 stop_words: Iterable[str] = ()):
        if analyzer not in ("word", "char"):
            raise ValueError("analyzer 'word' veya 'char' olmalı")
        self.analyzer = analyzer
        self.ngram_range = ngram_range  # (en kısa, en uzun) n-gram boyu
        self.min_df = min_df  # bir terimin sözlüğe girmesi için geçmesi gereken en az belge
        self.preprocessor = preprocessor  # ör. normalize veya F5 kökleme
        self.stop_words = frozenset(stop_words)
        self.vocabulary: dict[str, int] = {}  # terim -> sütun numarası
        self.idf = np.zeros(0, dtype=np.float32)  # her sütunun IDF ağırlığı

    def _features(self, text: str) -> list[str]:
        """Bir metindeki tüm n-gram'ları (tekrarlarıyla birlikte) çıkarır."""
        if self.preprocessor is not None:
            text = self.preprocessor(text)
        low, high = self.ngram_range
        if self.analyzer == "word":
            # Tek harfli parçalar ve stopword'ler anlam taşımadığı için atılır; n-gram'lar
            # temizlenmiş sözcük dizisi üzerinden kurulur.
            tokens = [t for t in text.split() if len(t) > 1 and t not in self.stop_words]
            return [" ".join(tokens[i:i + n]) for n in range(low, high + 1)
                    for i in range(len(tokens) - n + 1)]
        grams: list[str] = []
        for word in text.split():
            # Başa ve sona boşluk eklenir: " ki" ile başlayan ve "ar " ile biten parçalar
            # sözcük başı/sonu bilgisini taşır (ör. çoğul eki "-lar ").
            padded = f" {word} "
            for n in range(low, high + 1):
                grams.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
        return grams

    def fit(self, texts: list[str]) -> TfidfVectorizer:
        """Sözlüğü ve IDF değerlerini yalnızca verilen (eğitim) metinlerinden öğrenir."""
        # Belge frekansı (df): terimin kaç farklı belgede geçtiği. set() ile aynı belgedeki
        # tekrarlar bir kez sayılır.
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(self._features(text)))
        # Çok nadir terimler (min_df altı) genelde yazım hatası veya gürültüdür, atılır.
        # Sıralama, aynı veriyle her çalıştırmada aynı sütun numaralarını verir.
        kept = sorted(term for term, df in document_frequency.items() if df >= self.min_df)
        self.vocabulary = {term: i for i, term in enumerate(kept)}
        # IDF = ln((1 + n) / (1 + df)) + 1. Birçok belgede geçen terimin ağırlığı düşer,
        # az belgede geçen ayırt edici terimin ağırlığı artar. Paydaki ve paydadaki +1
        # sıfıra bölmeyi, sondaki +1 ise hiçbir terimin tamamen sıfırlanmamasını sağlar.
        n_docs = len(texts)
        self.idf = np.array([math.log((1 + n_docs) / (1 + document_frequency[t])) + 1
                             for t in kept], dtype=np.float32)
        return self

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        """Metinleri öğrenilmiş sözlükle vektöre çevirir; sözlükte olmayan terimler yok sayılır."""
        # CSR (compressed sparse row) biçimi üç diziyle kurulur: her satırın başladığı konum
        # (indptr), sıfır olmayan değerlerin sütunları (indices) ve değerleri (values).
        indptr, indices, values = [0], [], []
        for text in texts:
            # Terim frekansı (tf): terimin bu metinde kaç kez geçtiği.
            counts = Counter(self.vocabulary[f] for f in self._features(text)
                             if f in self.vocabulary)
            columns = sorted(counts)
            # Alt doğrusal TF: 1 + ln(tf). Bir sözcüğün 10 kez geçmesi, 1 kez geçmesinden
            # 10 kat değil yaklaşık 3,3 kat etki eder. Sonra IDF ile çarpılır.
            weights = np.array([(1 + math.log(counts[c])) * self.idf[c] for c in columns],
                               dtype=np.float32)
            # L2 normalizasyonu: vektör uzunluğu 1 yapılır, böylece uzun ve kısa metinler
            # aynı ölçekte karşılaştırılır.
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
        """Model dosyasına yazılacak ayarlar ve sözlük (IDF ayrıca .npz içinde saklanır)."""
        return {"analyzer": self.analyzer, "ngram_range": list(self.ngram_range),
                "min_df": self.min_df, "terms": list(self.vocabulary)}


class CombinedVectorizer:
    """Birden fazla vektörleştiricinin çıktısını yan yana birleştirir.

    Projede sözcük ve karakter vektörleri birleştirilir: sözcük özellikleri anlamı, karakter
    özellikleri ekleri ve yazım farklılıklarını yakalar.
    """

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
        # hstack: matrisler sütun yönünde uç uca eklenir; önce sözcük, sonra karakter sütunları.
        return sparse.hstack([v.transform(texts) for v in self.vectorizers], format="csr")

    def fit_transform(self, texts: list[str]) -> sparse.csr_matrix:
        return self.fit(texts).transform(texts)
