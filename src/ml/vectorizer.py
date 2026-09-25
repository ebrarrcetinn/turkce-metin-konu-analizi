"""
Dosya   : src/ml/vectorizer.py
Konu    : TF-IDF Vektörleştirici
Açıklama: Bu dosyada amacım metinleri sözcük ve karakter n-gram'larından oluşan TF-IDF
          ağırlıklı, L2 normlu seyrek vektörlere dönüştürmek. Hazır kütüphane kullanmadan
          kendim yazdım.
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
    """Amacım metinleri TF-IDF ağırlıklı, L2 normlu seyrek vektörlere dönüştürmek.

    Model sayılarla çalıştığı için her metni, sözlükteki her terim için bir sütunu olan bir
    vektöre çeviriyorum. Bir metinde sözlüğün çok küçük bir kısmı geçtiğinden vektörün büyük
    çoğunluğu sıfır oluyor; bu yüzden yalnızca sıfır olmayan değerleri tutan seyrek (sparse)
    matris kullanmayı seçtim.

    analyzer="word": sözcük n-gram'ları ("kuantum", "kuantum bilgisayar").
    analyzer="char": karakter n-gram'ları ("kua", "uan", "ant"). Türkçe eklemeli bir dil
    olduğu için "kitap", "kitaplar", "kitabın" gibi biçimleri ortak karakter parçaları
    üzerinden birbirine bağlıyorum; bu yöntem kökleme hatalarına karşı dayanıklı.
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
        """Bir metindeki tüm n-gram'ları (tekrarlarıyla birlikte) çıkarıyorum."""
        if self.preprocessor is not None:
            text = self.preprocessor(text)
        low, high = self.ngram_range
        if self.analyzer == "word":
            # Tek harfli parçalar ve stopword'ler anlam taşımadığı için onları atıyorum;
            # n-gram'ları temizlenmiş sözcük dizisi üzerinden kuruyorum.
            tokens = [t for t in text.split() if len(t) > 1 and t not in self.stop_words]
            return [" ".join(tokens[i:i + n]) for n in range(low, high + 1)
                    for i in range(len(tokens) - n + 1)]
        grams: list[str] = []
        for word in text.split():
            # Başa ve sona boşluk ekliyorum: " ki" ile başlayan ve "ar " ile biten parçalar
            # sözcük başı/sonu bilgisini taşıyor (ör. çoğul eki "-lar ").
            padded = f" {word} "
            for n in range(low, high + 1):
                grams.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
        return grams

    def fit(self, texts: list[str]) -> TfidfVectorizer:
        """Sözlüğü ve IDF değerlerini yalnızca verilen (eğitim) metinlerinden öğreniyorum."""
        # Belge frekansı (df): terimin kaç farklı belgede geçtiği. set() ile aynı belgedeki
        # tekrarları bir kez sayıyorum.
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(self._features(text)))
        # Çok nadir terimler (min_df altı) genelde yazım hatası veya gürültü, onları atıyorum.
        # Terimleri sıralıyorum; böylece aynı veriyle her çalıştırmada aynı sütun
        # numaralarını alıyorum.
        kept = sorted(term for term, df in document_frequency.items() if df >= self.min_df)
        self.vocabulary = {term: i for i, term in enumerate(kept)}
        # IDF = ln((1 + n) / (1 + df)) + 1. Birçok belgede geçen terimin ağırlığını düşürüp
        # az belgede geçen ayırt edici terimin ağırlığını artırıyorum. Paydaki ve paydadaki
        # +1 ile sıfıra bölmeyi önlüyorum, sondaki +1 ile de hiçbir terimin tamamen
        # sıfırlanmamasını sağlıyorum.
        n_docs = len(texts)
        self.idf = np.array([math.log((1 + n_docs) / (1 + document_frequency[t])) + 1
                             for t in kept], dtype=np.float32)
        return self

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        """Metinleri öğrendiğim sözlükle vektöre çeviriyorum; sözlükte olmayan terimleri yok
        sayıyorum."""
        # CSR (compressed sparse row) biçimini üç diziyle kuruyorum: her satırın başladığı
        # konum (indptr), sıfır olmayan değerlerin sütunları (indices) ve değerleri (values).
        indptr, indices, values = [0], [], []
        for text in texts:
            # Terim frekansı (tf): terimin bu metinde kaç kez geçtiği.
            counts = Counter(self.vocabulary[f] for f in self._features(text)
                             if f in self.vocabulary)
            columns = sorted(counts)
            # Alt doğrusal TF: 1 + ln(tf). Bir sözcüğün 10 kez geçmesi, 1 kez geçmesinden
            # 10 kat değil yaklaşık 3,3 kat etki eder. Sonra IDF ile çarpıyorum.
            weights = np.array([(1 + math.log(counts[c])) * self.idf[c] for c in columns],
                               dtype=np.float32)
            # L2 normalizasyonu: vektör uzunluğunu 1 yapıyorum, böylece uzun ve kısa
            # metinleri aynı ölçekte karşılaştırabiliyorum.
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
        """Model dosyasına yazacağım ayarları ve sözlüğü döndürüyorum (IDF'yi ayrıca .npz içinde
        saklıyorum)."""
        return {"analyzer": self.analyzer, "ngram_range": list(self.ngram_range),
                "min_df": self.min_df, "terms": list(self.vocabulary)}


class CombinedVectorizer:
    """Burada amacım birden fazla vektörleştiricinin çıktısını yan yana birleştirmek.

    Projede sözcük ve karakter vektörlerini birleştiriyorum: sözcük özellikleri anlamı,
    karakter özellikleri ekleri ve yazım farklılıklarını yakalıyor.
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
        # hstack: matrisleri sütun yönünde uç uca ekliyorum; önce sözcük, sonra karakter
        # sütunları.
        return sparse.hstack([v.transform(texts) for v in self.vectorizers], format="csr")

    def fit_transform(self, texts: list[str]) -> sparse.csr_matrix:
        return self.fit(texts).transform(texts)
