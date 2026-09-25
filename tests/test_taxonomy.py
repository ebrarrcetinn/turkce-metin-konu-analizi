"""
Dosya   : tests/test_taxonomy.py
Konu    : Taksonomi Testleri
Açıklama: Kategori eşlemesini ve genel konu - alt konu hiyerarşisinin tutarlılığını test eder.
Yazar   : Ebrar Cemre Çetin
Tarih   : 27.09.2026
"""

from src.config import OTHER_LABEL
from src.models.taxonomy import (
    ALL_GENERAL_LABELS,
    CATEGORY_TO_TAXONOMY,
    EXCLUDED_CATEGORIES,
    GENERAL_TOPICS,
    map_card,
    subtopics_of,
    validate_hierarchy,
)


def test_quantum_cards_split_by_term():
    assert map_card("kuantum", "kübit") == ("Teknoloji", "Kuantum Bilgisayarlar")
    assert map_card("kuantum", "kuantum işlemcisi") == ("Teknoloji", "Kuantum Bilgisayarlar")
    assert map_card("kuantum", "kuantum dolanıklığı") == ("Fizik", "Kuantum Mekaniği")
    assert map_card("kuantum", "dalga fonksiyonu") == ("Fizik", "Kuantum Mekaniği")


def test_excluded_and_other():
    assert map_card("anatomi", "kalp") is None
    assert map_card("gastronomi", "sos") == (OTHER_LABEL, None)


def test_mapping_tables_are_disjoint_and_complete():
    assert not set(CATEGORY_TO_TAXONOMY) & EXCLUDED_CATEGORIES
    assert {g for g, _ in CATEGORY_TO_TAXONOMY.values()} == set(GENERAL_TOPICS)
    assert ALL_GENERAL_LABELS[-1] == OTHER_LABEL


def test_every_general_topic_has_subtopics_and_hierarchy_is_valid():
    for general in GENERAL_TOPICS:
        subs = subtopics_of(general)
        assert subs, general
        for sub in subs:
            assert validate_hierarchy(general, sub)
    assert "Kuantum Bilgisayarlar" in subtopics_of("Teknoloji")
    assert not validate_hierarchy("Fizik", "Futbol")
    assert subtopics_of(OTHER_LABEL) == []
