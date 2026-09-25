"""Hedef taksonomi ve kaynak veri kategorilerinden taksonomiye eşleme.

Ayrıntılı gerekçe: TAXONOMY.md
"""

from __future__ import annotations

import re
from typing import Final

from src.config import OTHER_LABEL

# kaynak kategori -> (genel konu, alt konu)
CATEGORY_TO_TAXONOMY: Final[dict[str, tuple[str, str]]] = {
    # Fizik
    "kuantum": ("Fizik", "Kuantum Mekaniği"),
    "termodinamik": ("Fizik", "Termodinamik"),
    "optik": ("Fizik", "Optik"),
    "akustik": ("Fizik", "Akustik"),
    "astronomi": ("Fizik", "Astrofizik ve Astronomi"),
    # Kimya
    "kimya": ("Kimya", "Genel Kimya"),
    "metalurji": ("Kimya", "Metalurji ve Malzeme"),
    # Biyoloji
    "genetik": ("Biyoloji", "Genetik"),
    "ekoloji": ("Biyoloji", "Ekoloji"),
    "mikrobiyoloji": ("Biyoloji", "Mikrobiyoloji"),
    "botanik": ("Biyoloji", "Botanik"),
    "zooloji": ("Biyoloji", "Zooloji"),
    "kuslar": ("Biyoloji", "Zooloji"),
    "denizcanlilari": ("Biyoloji", "Zooloji"),
    "paleontoloji": ("Biyoloji", "Evrim ve Paleontoloji"),
    # Teknoloji
    "yapayzeka": ("Teknoloji", "Yapay Zeka"),
    "donanim": ("Teknoloji", "Donanım"),
    "algoritmalar": ("Teknoloji", "Yazılım ve Algoritmalar"),
    "siber": ("Teknoloji", "Siber Güvenlik"),
    "robotik": ("Teknoloji", "Robotik"),
    "telekomunikasyon": ("Teknoloji", "Telekomünikasyon"),
    "nanoteknoloji": ("Teknoloji", "Nanoteknoloji"),
    # Bilim
    "epistemoloji": ("Bilim", "Bilim Felsefesi ve Yöntem"),
    # Kitaplar
    "edebiyat": ("Kitaplar", "Edebiyat, Roman ve Şiir"),
    "turkhalkedebiyati": ("Kitaplar", "Türk Halk Edebiyatı"),
    "masallar": ("Kitaplar", "Masallar"),
    "cizgiroman": ("Kitaplar", "Çizgi Roman"),
    # Spor
    "futbol": ("Spor", "Futbol"),
    "basketbol": ("Spor", "Basketbol"),
    "voleybol": ("Spor", "Voleybol"),
    "tenis": ("Spor", "Tenis"),
    "atletizm": ("Spor", "Atletizm ve Olimpiyatlar"),
    "motor_sporlari": ("Spor", "Motor Sporları"),
    "dovus": ("Spor", "Dövüş Sporları"),
    "binicilik": ("Spor", "Doğa ve Binicilik Sporları"),
    "dagcilik": ("Spor", "Doğa ve Binicilik Sporları"),
    # Tarih
    "osmanli": ("Tarih", "Osmanlı Tarihi"),
    "cumhuriyettarihi": ("Tarih", "Cumhuriyet Tarihi"),
    "savas": ("Tarih", "Savaşlar ve Askeri Tarih"),
    "arkeoloji": ("Tarih", "Antik Çağ ve Arkeoloji"),
    "epigrafi": ("Tarih", "Antik Çağ ve Arkeoloji"),
}

# Taksonomiyle anlamca örtüşen kategoriler; etiket gürültüsü yaratmamaları için dışlanır.
EXCLUDED_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "anatomi", "cerrahi", "dermatoloji", "farmakoloji", "immunoloji", "noroloji",
        "patoloji", "pediyatri", "epidemiyoloji", "psikiyatri", "saglikliyasam",
        "astronotik", "havacilik", "mekatronik", "jeoloji", "jeomorfoloji", "meteoroloji",
        "osinografi", "orman", "vahsidoga", "felsefe", "mantik", "mitoloji", "monarsi",
        "numizmatik", "tiyatro", "internethayati", "sosyalmedya", "videooyunlari", "kripto",
        "dalgiclik", "cleanup-2026-09-07",
    }
)

# "kuantum" kategorisindeki kuantum bilgisayar terimleri Teknoloji'ye taşınır.
QUANTUM_COMPUTING_TERM_RE = re.compile(
    r"kübit|qubit|kuantum (bilgisayar|işlemci|kapı|devre|algoritma|bellek|hata|hacmi|hızı|"
    r"üstünlüğü)|topolojik (kuantum )?bilgisayar|hadamard kapısı|cnot|pauli kapısı|"
    r"faz kapısı|yüzey kodu|transmon|hata düzeltme|bloch küresi|kuantum gürültüsünü",
    re.IGNORECASE,
)
QUANTUM_COMPUTING = ("Teknoloji", "Kuantum Bilgisayarlar")

GENERAL_TOPICS: Final[tuple[str, ...]] = (
    "Fizik", "Kimya", "Biyoloji", "Teknoloji", "Bilim", "Kitaplar", "Spor", "Tarih",
)
ALL_GENERAL_LABELS: Final[tuple[str, ...]] = (*GENERAL_TOPICS, OTHER_LABEL)


def map_card(category: str, term: str) -> tuple[str, str | None] | None:
    """Kaynak kartı (kategori, terim) -> (genel konu, alt konu) eşler.

    Dönüş: ``None`` -> kart dışlanır; ``(OTHER_LABEL, None)`` -> taksonomi dışı örnek.
    """
    if category in EXCLUDED_CATEGORIES:
        return None
    if category == "kuantum" and QUANTUM_COMPUTING_TERM_RE.search(term):
        return QUANTUM_COMPUTING
    if category in CATEGORY_TO_TAXONOMY:
        return CATEGORY_TO_TAXONOMY[category]
    return (OTHER_LABEL, None)


def subtopics_of(general: str) -> list[str]:
    subs = {sub for gen, sub in CATEGORY_TO_TAXONOMY.values() if gen == general}
    if general == QUANTUM_COMPUTING[0]:
        subs.add(QUANTUM_COMPUTING[1])
    return sorted(subs)


def validate_hierarchy(general: str, subtopic: str) -> bool:
    """Alt konunun gerçekten o genel konuya ait olup olmadığını denetler."""
    return subtopic in subtopics_of(general)
