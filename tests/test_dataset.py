"""
Dosya   : tests/test_dataset.py
Konu    : Veri Seti Hazırlama Testleri
Açıklama: Veri temizleme, taksonomi örnekleri, hatalı veri kontrolleri ve yakın kopya
          ölçümünü küçük sahte dosyalarla test eder.
Yazar   : Ebrar Cemre Çetin
Tarih   : 27.09.2026
"""

import json

import pytest

from src.data.dataset import (
    DatasetBuilder,
    DatasetError,
    load_split,
    near_duplicate_rate,
    term_group,
)
from src.models.taxonomy import GENERAL_TOPICS, subtopics_of


def _write_cards(raw_dir, name, cards):
    path = raw_dir / "taboo" / "data" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cards, ensure_ascii=False), encoding="utf-8")


def test_term_group_merges_inflections():
    assert term_group("Fotosentez") == term_group("fotosentezin")


def test_taxonomy_seeds_cover_every_subtopic():
    seeds = DatasetBuilder.taxonomy_seeds()
    pairs = {(r["general"], r["subtopic"]) for r in seeds}
    assert pairs == {(g, s) for g in GENERAL_TOPICS for s in subtopics_of(g)}
    assert {"biyoloji", "Biyoloji"} & {str(r["text"]) for r in seeds}


def test_clean_maps_removes_duplicates_and_conflicts(tmp_path):
    builder = DatasetBuilder(raw_dir=tmp_path)
    cards = [
        {"category": "genetik", "term": "gen", "definition": "kalıtım birimi"},
        {"category": "genetik", "term": "gen", "definition": "kalıtım birimi"},   # tekrar
        {"category": "genetik", "term": "x", "definition": "aynı metin"},
        {"category": "futbol", "term": "x", "definition": "aynı metin"},          # çelişki
        {"category": "genetik", "term": "boş", "definition": ""},
    ]
    records = builder.clean(cards)
    assert [r["term"] for r in records] == ["gen"]
    assert records[0]["general"] == "Biyoloji"


def test_load_cards_validates_fields(tmp_path):
    _write_cards(tmp_path, "genetik", [{"kategori": "genetik", "kelime": "gen"}])
    with pytest.raises(DatasetError):
        DatasetBuilder(raw_dir=tmp_path).load_cards()


def test_missing_data_and_split_errors(tmp_path):
    with pytest.raises(DatasetError):
        DatasetBuilder(raw_dir=tmp_path).load_cards()
    with pytest.raises(DatasetError):
        load_split("train", tmp_path)


def test_near_duplicate_rate():
    reference = [{"text": "kuantum bilgisayarlar kübit kullanır"}]
    assert near_duplicate_rate(reference, [{"text": "kuantum bilgisayarlar kübit kullanır"},
                                           {"text": "futbol maçı berabere bitti"}]) == 0.5
