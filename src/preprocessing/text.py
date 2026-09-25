"""
Dosya   : src/preprocessing/text.py
Konu    : Türkçe Metin Ön İşleme
Açıklama: Türkçeye uygun küçük harfe çevirme (I/İ sorunu), normalizasyon, tokenizasyon,
          stopword listesi ve F5 kökleme (sözcüğün ilk 5 harfi) işlemlerini yapar. Aynı
          fonksiyonlar eğitimde ve tahminde ortak kullanılır.
Yazar   : Ebrar Cemre Çetin
Tarih   : 23.09.2026
"""

from __future__ import annotations

import re
import unicodedata

# Python'un str.lower() fonksiyonu "I" harfini "i" yapar; Türkçede "ı" olmalıdır. "İ" ise
# "i" + birleşik nokta (i̇) olur. Bu iki harf küçültmeden önce elle çevrilir.
_TR_UPPER_MAP = str.maketrans({"I": "ı", "İ": "i"})

# Konudan bağımsız, modelin yanlış şeyler öğrenmesine yol açabilecek parçalar:
URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)  # bağlantılar
MENTION_RE = re.compile(r"(?<!\w)@\w+")  # @kullanıcı adları
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")  # #etiket: işaret atılır, sözcük korunur
HTML_TAG_RE = re.compile(r"<[^>]+>")  # <b>, <br> gibi etiketler
# Kesme işaretinden sonraki ekler atılır: "Ankara'da" ve "Ankara'nın" -> "Ankara"
APOSTROPHE_SUFFIX_RE = re.compile(r"(\w)['’`´](\w+)")
DIGIT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")  # 1299, 3,14 gibi sayılar
NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)  # noktalama ve semboller
UNDERSCORE_RE = re.compile(r"_+")  # \w alt çizgiyi de harf saydığı için ayrıca temizlenir
MULTISPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-zçğıöşüâîû]+", re.UNICODE)  # Türkçe harflerden oluşan sözcükler

# Sık görülen mojibake (UTF-8 metnin latin-1/cp1252 olarak çözülmesi) işaretleri.
# Örneğin "ş" harfi bu hatada "ÅŸ" olarak görünür.
_MOJIBAKE_MARKERS = ("Ã", "Ä", "Å")

# Elle derlenmiş Türkçe stopword listesi: bağlaç, zamir, edat ve yardımcı sözcükler.
# Her konuda geçtikleri için konu ayırt etmeye katkıları yoktur, özelliklerden çıkarılırlar.
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
    # Metin yanlış kodlamayla okunmuşsa aynı kodlamayla byte'a geri çevrilip UTF-8 olarak
    # yeniden okunur. Olmazsa metin olduğu gibi bırakılır.
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
    # Unicode kategorileri: So (sembol, emoji), Sk (değiştirici sembol), Cs (vekil
    # karakter), Co (özel kullanım alanı).
    return "".join(
        ch for ch in text if not unicodedata.category(ch).startswith(("So", "Sk", "Cs", "Co"))
    )


def normalize(text: str) -> str:
    """Model girdisi için metni normalize eder.

    Eğitimde ve tahminde aynı fonksiyon kullanılır; aksi halde model eğitimde gördüğünden
    farklı biçimde metin alır ve başarı düşer.
    """
    if not isinstance(text, str):
        raise TypeError(f"metin str olmalı, gelen: {type(text).__name__}")
    text = fix_mojibake(text)
    # NFC: "ş" harfinin tek karakter ya da "s + çengel" olarak yazılmış iki biçimi aynı
    # hale getirilir; aksi halde aynı sözcük iki farklı özellik olurdu.
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
    text = text.replace("\u0307", "")  # "İ" küçültmesinden kalabilecek birleşik nokta
    return MULTISPACE_RE.sub(" ", text).strip()


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """Normalize edilmiş metni sözcüklere ayırır."""
    tokens = TOKEN_RE.findall(normalize(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in TURKISH_STOPWORDS]
    return tokens


def f5_stem(token: str, length: int = 5) -> str:
    """F5 kökleme: sözcüğün ilk 5 harfi alınır.

    Türkçede ekler sözcüğün sonuna geldiği için ilk 5 harf çoğu zaman kökü korur:
    "kütüphaneden", "kütüphaneler" -> "kütüp". Türkçe bilgi erişimi çalışmalarında
    (Can vd., 2008) sözlük tabanlı kökleyicilere yakın sonuç verdiği gösterilmiştir.
    """
    return token[:length]


def f5_preprocess(text: str) -> str:
    """Normalize + her sözcüğü ilk 5 harfe kırp (sözcük özellikleri için)."""
    return " ".join(f5_stem(tok) for tok in normalize(text).split())


def content_tokens(text: str) -> list[str]:
    """Stopword'ler ve tek harfli parçalar çıkarıldıktan sonraki anlamlı sözcükler.

    Boş liste dönerse metin sınıflandırılmaz ("!!!", "12345", "ve bu da" gibi girdiler).
    """
    return [t for t in tokenize(text, remove_stopwords=True) if len(t) > 1]
