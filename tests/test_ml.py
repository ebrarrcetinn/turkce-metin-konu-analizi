"""
Dosya   : tests/test_ml.py
Konu    : Makine Öğrenmesi Algoritma Testleri
Açıklama: TF-IDF, Naive Bayes, softmax regresyon, kalibrasyon, metrikler ve veri bölme
          fonksiyonlarını küçük örneklerle test eder.
Yazar   : Ebrar Cemre Çetin
Tarih   : 27.09.2026
"""

import math

import numpy as np
import pytest
from scipy import sparse

from src.ml.calibration import fit_temperature, negative_log_likelihood
from src.ml.metrics import ClassificationReport, expected_calibration_error
from src.ml.naive_bayes import MultinomialNaiveBayes
from src.ml.softmax_regression import SoftmaxRegression, softmax
from src.ml.splitting import grouped_stratified_split
from src.ml.vectorizer import CombinedVectorizer, TfidfVectorizer

DOCS = ["kedi süt içer", "köpek kemik yer", "kedi fare yakalar", "köpek havlar"]


def test_tfidf_word_vectors_are_l2_normalized_and_weight_rare_terms():
    vectorizer = TfidfVectorizer("word", (1, 1))
    matrix = vectorizer.fit_transform(DOCS).toarray()
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-6)
    row = matrix[0]
    vocab = vectorizer.vocabulary
    # "süt" tek belgede, "kedi" iki belgede geçer: nadir terimin ağırlığı daha büyük olmalı.
    assert row[vocab["süt"]] > row[vocab["kedi"]]


def test_tfidf_idf_formula():
    vectorizer = TfidfVectorizer("word", (1, 1)).fit(DOCS)
    # Düzeltilmiş IDF: ln((1 + n) / (1 + df)) + 1
    expected = math.log((1 + 4) / (1 + 2)) + 1
    assert vectorizer.idf[vectorizer.vocabulary["kedi"]] == pytest.approx(expected)


def test_tfidf_bigrams_char_ngrams_min_df_and_unknown_terms():
    words = TfidfVectorizer("word", (1, 2)).fit(DOCS)
    assert "kedi süt" in words.vocabulary
    chars = TfidfVectorizer("char", (2, 3), min_df=2).fit(DOCS)
    assert " k" in chars.vocabulary and all(len(term) in (2, 3) for term in chars.vocabulary)
    assert words.transform(["zürafa"]).nnz == 0


def test_combined_vectorizer_stacks_features():
    combined = CombinedVectorizer([TfidfVectorizer("word"), TfidfVectorizer("char", (2, 2))])
    matrix = combined.fit_transform(DOCS)
    assert matrix.shape == (4, combined.n_features)
    assert combined.feature_offset(1) == len(combined.vectorizers[0].vocabulary)


def _toy_problem():
    texts = ["kedi süt", "kedi fare", "kedi tüy", "köpek kemik", "köpek havlar", "köpek tasma"]
    labels = ["kedi", "kedi", "kedi", "köpek", "köpek", "köpek"]
    vectorizer = TfidfVectorizer("word")
    return vectorizer, vectorizer.fit_transform(texts), labels


def test_naive_bayes_learns_toy_problem():
    vectorizer, features, labels = _toy_problem()
    model = MultinomialNaiveBayes(alpha=0.1).fit(features, labels)
    assert model.predict(features) == labels
    assert model.predict(vectorizer.transform(["kedi"])) == ["kedi"]


def test_softmax_regression_learns_and_early_stops():
    vectorizer, features, labels = _toy_problem()
    scores = iter([0.5, 0.9, 0.8, 0.7, 0.6])
    model = SoftmaxRegression(learning_rate=0.1, epochs=5, batch_size=2, patience=2)
    model.fit(features, labels, validation_score=lambda _: next(scores))
    assert model.best_epoch == 2 and len(model.history) == 4
    assert model.predict(vectorizer.transform(["köpek"])) == ["köpek"]
    probabilities = model.predict_proba(features)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


def test_softmax_is_numerically_stable():
    result = softmax(np.array([[1000.0, 1000.0], [-1000.0, 0.0]]))
    np.testing.assert_allclose(result, [[0.5, 0.5], [0.0, 1.0]], atol=1e-12)


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 3, size=500)
    logits = rng.normal(size=(500, 3))
    logits[np.arange(500), targets] += 1.0
    overconfident = logits * 5  # doğru sıralama, abartılı güven
    temperature = fit_temperature(overconfident, targets)
    assert temperature > 1.5
    assert negative_log_likelihood(overconfident, targets, temperature) < \
        negative_log_likelihood(overconfident, targets, 1.0)


def test_classification_report_values():
    report = ClassificationReport(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert report.accuracy == 0.75
    assert report.confusion.tolist() == [[1, 1], [0, 2]]
    assert report.precision.tolist() == pytest.approx([1.0, 2 / 3])
    assert report.recall.tolist() == pytest.approx([0.5, 1.0])
    assert report.macro_f1 == pytest.approx((2 / 3 + 0.8) / 2)


def test_expected_calibration_error():
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([True, True])) == 0.0
    assert expected_calibration_error(np.array([0.9, 0.9]), np.array([False, False])) == \
        pytest.approx(0.9)


def test_grouped_split_keeps_groups_together_and_is_stratified():
    labels = ["a"] * 40 + ["b"] * 40
    groups = [f"g{i // 2}" for i in range(80)]  # her grupta 2 örnek
    kept, held = grouped_stratified_split(labels, groups, 0.25, seed=1)
    assert sorted(kept + held) == list(range(80))
    assert not {groups[i] for i in kept} & {groups[i] for i in held}
    held_labels = [labels[i] for i in held]
    assert 8 <= held_labels.count("a") <= 12 and 8 <= held_labels.count("b") <= 12


def test_sparse_input_type_is_supported():
    features = sparse.csr_matrix(np.eye(2, dtype=np.float32))
    model = MultinomialNaiveBayes().fit(features, ["x", "y"])
    assert model.predict(features) == ["x", "y"]
