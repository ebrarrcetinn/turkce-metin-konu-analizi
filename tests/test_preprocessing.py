"""
Dosya   : tests/test_preprocessing.py
Konu    : Metin Ön İşleme Testleri
Açıklama: Türkçe küçük harf, normalizasyon, tokenizasyon ve kökleme fonksiyonlarını test eder.
Yazar   : Ebrar Cemre Çetin
Tarih   : 27.09.2026
"""

import pytest

from src.preprocessing.text import (
    content_tokens,
    f5_preprocess,
    f5_stem,
    fix_mojibake,
    is_meaningful,
    normalize,
    tokenize,
    turkish_lower,
)


@pytest.mark.parametrize("raw, expected", [
    ("I", "ı"),
    ("İ", "i"),
    ("IĞDIR", "ığdır"),
    ("İSTANBUL", "istanbul"),
    ("ISPARTA ve İZMİR", "ısparta ve izmir"),
    ("ÇĞÖŞÜ", "çğöşü"),
])
def test_turkish_lower(raw, expected):
    assert turkish_lower(raw) == expected


def test_python_lower_is_wrong_for_turkish_but_ours_is_not():
    assert "İ".lower() != "i"  # Python varsayılanı birleşik nokta üretir
    assert normalize("İ") == "i"


def test_normalize_removes_url_mention_html_emoji_digits():
    text = "<b>Merhaba</b> @ali şuna bak https://ornek.com/a?b=1 www.x.org 😍 2024'te #Kuantum"
    assert normalize(text) == "merhaba şuna bak kuantum"


def test_normalize_apostrophe_suffix_and_punctuation():
    assert normalize("Ankara'da, Einstein’ın kuramı!") == "ankara einstein kuramı"


def test_normalize_nfc_equivalence():
    decomposed = "c\u0327ok s\u0327ekil"  # NFD: ç, ş
    assert normalize(decomposed) == "çok şekil"


def test_fix_mojibake():
    broken = "Ã§ok gÃ¼zel bir kitap"
    assert fix_mojibake(broken) == "çok güzel bir kitap"
    assert normalize(broken) == "çok güzel bir kitap"
    assert fix_mojibake("zaten düzgün") == "zaten düzgün"


def test_normalize_rejects_non_string():
    with pytest.raises(TypeError):
        normalize(None)  # type: ignore[arg-type]


def test_tokenize_and_stopwords():
    assert tokenize("Bu bir test ve deneme.") == ["bu", "bir", "test", "ve", "deneme"]
    assert tokenize("Bu bir test ve deneme.", remove_stopwords=True) == ["test", "deneme"]


@pytest.mark.parametrize("text", ["", "   ", "ve bu da bir", "!!!", "12345", "... 42 ...",
                                  "😍🔥", "@kullanici https://x.com"])
def test_not_meaningful(text):
    assert not is_meaningful(text)
    assert content_tokens(text) == []


def test_meaningful():
    assert is_meaningful("Merhaba")
    assert content_tokens("Kübitler ve işlemciler") == ["kübitler", "işlemciler"]


def test_f5():
    assert f5_stem("kütüphaneden") == "kütüp"
    assert f5_stem("dna") == "dna"
    assert f5_preprocess("Kübitleri kullanan işlemciler") == "kübit kulla işlem"
