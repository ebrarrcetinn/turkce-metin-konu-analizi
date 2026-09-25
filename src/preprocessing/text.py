"""Türkçe metin normalizasyonu ve tokenizasyonu.

Python'un str.lower() fonksiyonu Türkçe I/İ harflerini yanlış çevirdiği için
turkish_lower kullanılır. Aynı fonksiyonlar eğitimde ve çıkarımda ortaktır.
"""

from __future__ import annotations

import re
import unicodedata

_TR_UPPER_MAP = str.maketrans({"I": "ı", "İ": "i"})

URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
# Apostrof sonrası özel isim ekleri: "Ankara'da" -> "Ankara"
APOSTROPHE_SUFFIX_RE = re.compile(r"(\w)['’`´](\w+)")
DIGIT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
UNDERSCORE_RE = re.compile(r"_+")
MULTISPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-zçğıöşüâîû]+", re.UNICODE)

# Sık görülen mojibake (UTF-8 metnin latin-1/cp1252 olarak çözülmesi) işaretleri
_MOJIBAKE_MARKERS = ("Ã", "Ä", "Å")

# Küçük, elle derlenmiş Türkçe stopword listesi (bağlaç, zamir, edat, yardımcı sözcük).
TURKISH_STOPWORDS: frozenset[str] = frozenset(
    """
    acaba ama ancak artık aslında az bana bazı belki ben beni benim bile bir biraz birkaç
    birçok biri birşey biz bize bizi bizim bu buna bunda bundan bunlar bunları bunu bunun
    burada çok çünkü da daha dahi de defa değil diye diğer en gibi göre hem hep hepsi her
    herhangi hiç için ile ise işte kadar ki kim kime kimi mı mi mu mü nasıl ne neden nerede
    nereye niçin niye o olan olarak oldu olduğu olmak olup on ona onda ondan onlar onları
    onu onun öyle pek sanki şey siz size sizi sizin şu şuna şunda şundan şunu tüm ve veya
    ya yani yine yoksa zaten ayrıca yalnız yalnızca sadece önce sonra şimdi var yok olur
    olması ettiği eden etmek etti üzere karşı arasında başka böyle
    """.split()
)


def fix_mojibake(text: str) -> str:
    """UTF-8'in latin-1/cp1252 olarak çözülmesinden kaynaklanan bozuk karakterleri onarır."""
    if not any(marker in text for marker in _MOJIBAKE_MARKERS):
        return text
    for codec in ("cp1252", "latin-1"):
        try:
            return text.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def turkish_lower(text: str) -> str:
    """Türkçe kurallarına uygun küçük harfe çevirme (I->ı, İ->i)."""
    return text.translate(_TR_UPPER_MAP).lower()


def strip_emoji(text: str) -> str:
    """Emoji ve diğer sembol kategorisindeki karakterleri kaldırır."""
    return "".join(
        ch for ch in text if not unicodedata.category(ch).startswith(("So", "Sk", "Cs", "Co"))
    )


def normalize(text: str) -> str:
    """Model girdisi için metni normalize eder.

    Adımlar: mojibake onarımı -> NFC -> HTML/URL/mention temizliği -> hashtag kelimesi
    korunur -> apostrof ekleri atılır -> emoji temizliği -> Türkçe küçük harf ->
    rakam/noktalama temizliği.
    """
    if not isinstance(text, str):
        raise TypeError(f"metin str olmalı, gelen: {type(text).__name__}")
    text = fix_mojibake(text)
    text = unicodedata.normalize("NFC", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = APOSTROPHE_SUFFIX_RE.sub(r"\1", text)
    text = strip_emoji(text)
    text = turkish_lower(text)
    text = DIGIT_RE.sub(" ", text)
    text = NON_WORD_RE.sub(" ", text)
    text = UNDERSCORE_RE.sub(" ", text)
    text = text.replace("\u0307", "")  # birleşik nokta kalıntısı
    return MULTISPACE_RE.sub(" ", text).strip()


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """Normalize edilmiş metni sözcüklere ayırır."""
    tokens = TOKEN_RE.findall(normalize(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in TURKISH_STOPWORDS]
    return tokens


def f5_stem(token: str, length: int = 5) -> str:
    """Türkçe için basit ve etkili önek kırpma kökleyicisi (F5; Can vd., 2008)."""
    return token[:length]


def f5_preprocess(text: str) -> str:
    """Normalize + her sözcüğü ilk 5 harfe kırp."""
    return " ".join(f5_stem(tok) for tok in normalize(text).split())


def content_tokens(text: str) -> list[str]:
    """Stopword'ler ve tek harfli parçalar çıkarıldıktan sonraki anlamlı sözcükler."""
    return [t for t in tokenize(text, remove_stopwords=True) if len(t) > 1]


def is_meaningful(text: str) -> bool:
    """Metin sınıflandırılabilir içerik taşıyor mu (boş / noktalama / rakam / stopword değil)?"""
    return bool(content_tokens(text))
