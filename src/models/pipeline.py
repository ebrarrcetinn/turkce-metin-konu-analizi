"""Deneylerde karşılaştırılan aday modeller ve kalibre final model kurucusu."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from src.config import RANDOM_SEED
from src.models.calibration import TemperatureScaledClassifier
from src.preprocessing.text import TURKISH_STOPWORDS, f5_preprocess, normalize

# Stopword listesi, vectorizer'ın göreceği biçime (normalize edilmiş) getirilir.
STOPWORDS = sorted({normalize(w) for w in TURKISH_STOPWORDS})
F5_STOPWORDS = sorted({w[:5] for w in STOPWORDS})


def word_tfidf(ngram_max: int = 2, stem: bool = False, min_df: int = 1) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=f5_preprocess if stem else normalize,
        ngram_range=(1, ngram_max),
        sublinear_tf=True,
        min_df=min_df,
        stop_words=F5_STOPWORDS if stem else STOPWORDS,
        token_pattern=r"(?u)\b\w\w+\b",
    )


def char_tfidf(ngram_range: tuple[int, int] = (2, 5), min_df: int = 2) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=normalize,
        analyzer="char_wb",
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=min_df,
    )


def word_char_union(stem: bool = False) -> FeatureUnion:
    return FeatureUnion([("word", word_tfidf(stem=stem)), ("char", char_tfidf())])


@dataclass(frozen=True)
class Candidate:
    exp_id: str
    name: str
    features: str
    build: Callable[[], Pipeline]


def _svc(c: float = 1.0) -> LinearSVC:
    return LinearSVC(C=c, class_weight="balanced", random_state=RANDOM_SEED)


def _lr(c: float = 10.0) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=3000, class_weight="balanced")


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("E01", "BoW + MultinomialNB", "word 1-2gram counts",
              lambda: Pipeline([("vec", CountVectorizer(preprocessor=normalize,
                                                        ngram_range=(1, 2),
                                                        stop_words=STOPWORDS)),
                                ("clf", MultinomialNB(alpha=0.3))])),
    Candidate("E02", "TF-IDF(word) + ComplementNB", "word 1-2gram tf-idf",
              lambda: Pipeline([("vec", word_tfidf()), ("clf", ComplementNB(alpha=0.3))])),
    Candidate("E03", "TF-IDF(word) + LogisticRegression", "word 1-2gram tf-idf",
              lambda: Pipeline([("vec", word_tfidf()), ("clf", _lr())])),
    Candidate("E04", "TF-IDF(word) + LinearSVM", "word 1-2gram tf-idf",
              lambda: Pipeline([("vec", word_tfidf()), ("clf", _svc())])),
    Candidate("E05", "TF-IDF(word, F5 stem) + LinearSVM", "F5-stemmed word 1-2gram tf-idf",
              lambda: Pipeline([("vec", word_tfidf(stem=True)), ("clf", _svc())])),
    Candidate("E06", "TF-IDF(char_wb 2-5) + LinearSVM", "char_wb 2-5gram tf-idf",
              lambda: Pipeline([("vec", char_tfidf()), ("clf", _svc())])),
    Candidate("E07", "TF-IDF(word+char) + LinearSVM", "word 1-2 + char_wb 2-5 tf-idf",
              lambda: Pipeline([("vec", word_char_union()), ("clf", _svc())])),
    Candidate("E08", "TF-IDF(word+char) + LogisticRegression", "word 1-2 + char_wb 2-5 tf-idf",
              lambda: Pipeline([("vec", word_char_union()), ("clf", _lr())])),
    Candidate("E09", "TF-IDF(word+char) + SGD(modified_huber)", "word 1-2 + char_wb 2-5 tf-idf",
              lambda: Pipeline([("vec", word_char_union()),
                                ("clf", SGDClassifier(loss="modified_huber", alpha=2e-5,
                                                      class_weight="balanced", max_iter=50,
                                                      random_state=RANDOM_SEED))])),
    Candidate("E10", "LSA(SVD-300) embedding + MLP", "char tf-idf -> 300d LSA dense embedding",
              lambda: Pipeline([("vec", char_tfidf()),
                                ("svd", TruncatedSVD(300, random_state=RANDOM_SEED)),
                                ("norm", Normalizer()),
                                # sklearn 1.8: early_stopping string etiketlerle hata veriyor;
                                # bu yüzden sabit epoch sayısı kullanılır.
                                ("clf", MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-4,
                                                      early_stopping=False, max_iter=30,
                                                      random_state=RANDOM_SEED))])),
    Candidate("E11", "TF-IDF(word F5 + char) + LinearSVM", "F5 word 1-2 + char_wb 2-5 tf-idf",
              lambda: Pipeline([("vec", word_char_union(stem=True)), ("clf", _svc())])),
)


def get_candidate(exp_id: str) -> Candidate:
    for candidate in CANDIDATES:
        if candidate.exp_id == exp_id:
            return candidate
    raise KeyError(f"Bilinmeyen deney: {exp_id}")


def build_estimator(exp_id: str, **clf_params: object) -> Pipeline:
    pipe = get_candidate(exp_id).build()
    if clf_params:
        pipe.set_params(**{f"clf__{k}": v for k, v in clf_params.items()})
    return pipe


CALIBRATION_METHODS = ("temperature", "sigmoid", "isotonic")


def build_calibrated(exp_id: str, calibration: str, **clf_params: object):
    """Seçilen adayı hiperparametreleriyle kurar ve olasılık kalibrasyonu ekler.

    Kalibrasyon yalnızca eğitim verisi içinde 3-katlı CV ile yapılır; validation/test
    kullanılmaz.
    """
    base = build_estimator(exp_id, **clf_params)
    if calibration == "temperature":
        return TemperatureScaledClassifier(base, cv=3, random_state=RANDOM_SEED)
    if calibration in ("sigmoid", "isotonic"):
        return CalibratedClassifierCV(base, method=calibration, cv=3, ensemble=True)
    raise ValueError(f"Bilinmeyen kalibrasyon yöntemi: {calibration}")
