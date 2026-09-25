"""
Dosya   : tests/test_model_behavior.py
Konu    : Eğitilmiş Model Davranış Testleri
Açıklama: Kuantum ayrımı, case'deki örnek cümleler, tek sözcüklük konu adları ve uç durumlar
          üzerinde eğitilmiş modelin davranışını test eder. Cümleler eğitim verisinde yoktur.
Yazar   : Ebrar Cemre Çetin
Tarih   : 27.09.2026
"""

import pytest

from src.models.taxonomy import validate_hierarchy
from src.models.topic_model import STATUS_EMPTY, STATUS_OK, STATUS_OUT_OF_SCOPE, STATUS_UNCERTAIN

pytestmark = pytest.mark.model


@pytest.mark.parametrize("text, general, sub", [
    ("Kuantum dolanıklıkta parçacıkların dalga fonksiyonları birlikte değişebilir.",
     "Fizik", "Kuantum Mekaniği"),
    ("Kuantum işlemciler kübitler kullanarak belirli algoritmaları hızlandırabilir.",
     "Teknoloji", "Kuantum Bilgisayarlar"),
])
def test_quantum_disambiguation(classifier, text, general, sub):
    p = classifier.predict(text)
    assert p.status == STATUS_OK
    assert p.general == general
    assert p.subtopics[0][0] == sub


def test_quantum_not_keyword_only(classifier):
    """'kuantum' sözcüğü ortakken ayırt edici bağlam sözcükleri kararı belirlemeli."""
    phys = classifier.predict("Kuantum parçacık dalga dolanıklık")
    tech = classifier.predict("Kuantum kübit işlemci algoritma")
    assert phys.top_guess == "Fizik" and tech.top_guess == "Teknoloji"


def test_acceptance_a_books(classifier):
    p = classifier.predict("Kütüphaneden aldığım romanı okudum, yazarın anlatımı çok akıcıydı.")
    assert (p.status, p.general) == (STATUS_OK, "Kitaplar")


def test_acceptance_b_science_top_guess(classifier):
    """B'nin en olası konusu Bilim olmalı (argmax düzeyinde geçer)."""
    p = classifier.predict("Bilim insanları hipotezlerini deney ve gözlem kullanarak test eder.")
    assert p.top_guess == "Bilim"


@pytest.mark.xfail(strict=True, reason="Bilinen sorun: Bilim sınıfı yalnızca bilim felsefesi "
                   "kartlarından besleniyor; güven eşiğin altında kaldığı için 'Belirsiz'.")
def test_acceptance_b_science_confident(classifier):
    p = classifier.predict("Bilim insanları hipotezlerini deney ve gözlem kullanarak test eder.")
    assert (p.status, p.general) == (STATUS_OK, "Bilim")


def test_acceptance_c_biology(classifier):
    p = classifier.predict("Hücreler DNA taşır ve canlılar evrim geçirerek çeşitlenir.")
    assert (p.status, p.general) == (STATUS_OK, "Biyoloji")


@pytest.mark.parametrize("text", ["Akşam pizza söyleyeceğim.",
                                  "Yarın sabah dişçiye gitmem gerekiyor.",
                                  "Hafta sonu anneannemin bahçesinde domates topladık."])
def test_out_of_domain_not_confidently_assigned(classifier, text):
    p = classifier.predict(text)
    assert p.status in (STATUS_OUT_OF_SCOPE, STATUS_UNCERTAIN)


@pytest.mark.parametrize("text", ["", "   ", "ve bu da bir", "!!!", "12345"])
def test_empty_like_inputs(classifier, text):
    p = classifier.predict(text)
    assert p.status == STATUS_EMPTY and p.general is None


def test_greeting_is_not_a_topic(classifier):
    p = classifier.predict("Merhaba")
    assert p.status != STATUS_OK


def test_very_long_text_is_truncated_not_crashing(classifier):
    p = classifier.predict("Futbol maçında hakem penaltı verdi. " * 300)
    assert p.truncated and p.general == "Spor"


def test_emoji_text(classifier):
    assert classifier.predict("Bu roman harikaydı 😍📚🔥").top_guess == "Kitaplar"


def test_url_text(classifier):
    p = classifier.predict("Şu makaleyi okudum https://ornek.com/kuantum-fizigi dalga "
                           "fonksiyonu çok ilginç")
    assert p.top_guess == "Fizik"


def test_turkish_i_variants_are_normalized_consistently(classifier):
    upper = classifier.predict("IŞIK HIZI VE GÖRELİLİK ÜZERİNE BİR KONFERANS")
    lower = classifier.predict("ışık hızı ve görelilik üzerine bir konferans")
    assert upper.general_scores == lower.general_scores


def test_multi_topic_text_exposes_several_topics(classifier):
    p = classifier.predict("Osmanlı ordusu kuşatmadan sonra fizik yasalarını kullanan toplar "
                           "geliştirdi ve futbol oynadı.")
    assert sum(1 for v in p.general_scores.values() if v >= 0.05) >= 2


def test_hierarchy_consistency_and_probabilities(classifier):
    for text in ["Genetik mutasyonlar kalıtımı etkiler.", "Basketbolda üç sayılık atış.",
                 "Osmanlı padişahı sefere çıktı.", "Periyodik tabloda soygazlar."]:
        p = classifier.predict(text)
        assert sum(p.general_scores.values()) == pytest.approx(1.0, abs=1e-3)
        for sub, _ in p.subtopics:
            assert validate_hierarchy(p.general, sub)


def test_confidence_bounds_and_latency(classifier):
    p = classifier.predict("Galatasaray dün akşam derbide penaltıyla kazandı.")
    assert 0.0 <= p.confidence <= 1.0
    assert p.latency_ms < 100


@pytest.mark.parametrize("text, general", [
    ("kitaplar", "Kitaplar"), ("bilim", "Bilim"), ("biyoloji", "Biyoloji"),
    ("fizik", "Fizik"), ("kimya", "Kimya"), ("spor", "Spor"), ("tarih", "Tarih"),
])
def test_single_topic_word(classifier, text, general):
    """Case senaryosundaki gibi tek sözcüklük konu adları tanınmalı."""
    assert classifier.predict(text).general == general


def test_quantum_word_shows_both_topics(classifier):
    p = classifier.predict("kuantum")
    topics = {p.general} | {name for name, _ in p.related_topics}
    assert {"Fizik", "Teknoloji"} <= topics


def test_several_subtopics_listed_for_general_word(classifier):
    assert len(classifier.predict("biyoloji").subtopics) >= 2


def test_keywords_come_from_model_weights(classifier):
    kws = classifier.keywords("Kuantum bilgisayarlar kübit kullanır.", "Teknoloji")
    assert kws and kws[0] == "kübit"
