"""
Dosya   : tests/test_database.py
Konu    : Veritabanı Testleri
Açıklama: Tablo oluşturma, kayıt, şema geçişi ve önbellek işlemlerini test eder.
Yazar   : Ebrar Cemre Çetin
Tarih   : 27.09.2026
"""

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from src.database.db import MIGRATIONS, SCHEMA_VERSION, Database, DatabaseError

META = {"model_version": "1.0.0", "model_name": "m", "training_timestamp": "t",
        "dataset": {"version": "d"}, "weights_sha256": "abc", "metrics": {"f1": 0.9},
        "thresholds": {"min_confidence": 0.75}}
PRED = {"text": "Kuantum bilgisayarlar kübit kullanır.", "status": "ok",
        "general": "Teknoloji", "confidence": 0.99,
        "subtopics": [{"name": "Kuantum Bilgisayarlar", "score": 0.98}],
        "general_scores": {"Teknoloji": 0.99}, "latency_ms": 2.0}
THEME = {"label": "Teknoloji > Kuantum Bilgisayarlar", "phrase": "kuantum bilgisayarlar",
         "topics": [{"topic": "Teknoloji", "share": 1.0}]}


def _seed(db):
    db.register_model(META)
    db.start_session("s1", "1.0.0")
    metin = db.save_text("s1", 1, PRED, "1.0.0")
    theme = db.save_theme("s1", metin, THEME, "kuantum bilgisayarlar kübit")
    return metin, theme


def test_schema_created_with_version(db):
    assert db.schema_version() == SCHEMA_VERSION
    tables = {r[0] for r in db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"metinler", "sohbet_konulari", "arama_sonuclari", "sessions", "model_metadata",
            "search_cache"} <= tables
    indexes = {r[0] for r in db._conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert "idx_metinler_session" in indexes


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "x.db"
    Database(path).close()
    with Database(path) as again:
        assert again.schema_version() == SCHEMA_VERSION


def test_old_database_is_upgraded(tmp_path):
    """v1 şemasıyla oluşturulmuş eski bir dosya açılınca eksik adımlar uygulanır."""
    path = tmp_path / "eski.db"
    conn = sqlite3.connect(path)
    conn.executescript(MIGRATIONS[0] + "\nPRAGMA user_version = 1;")
    conn.close()
    with Database(path) as upgraded:
        assert upgraded.schema_version() == SCHEMA_VERSION
        columns = {r[1] for r in upgraded._conn.execute("PRAGMA table_info(model_metadata)")}
        assert "weights_sha256" in columns and "artifact_sha256" not in columns
        upgraded.register_model(META)


def test_newer_schema_is_rejected(tmp_path):
    path = tmp_path / "x.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    conn.close()
    with pytest.raises(DatabaseError, match="daha yeni"):
        Database(path)


def test_insert_and_read_history(db):
    _, theme_id = _seed(db)
    n = db.save_search_results("s1", theme_id, "kuantum", [
        {"title": "Kübit", "summary": "özet", "url": "https://tr.wikipedia.org/wiki/K",
         "source": "wikipedia"}], from_cache=False)
    assert n == 1
    history = db.session_history("s1")
    assert history[0]["general_topic"] == "Teknoloji"
    assert history[0]["subtopics"][0]["name"] == "Kuantum Bilgisayarlar"
    assert history[0]["search_query"] == "kuantum bilgisayarlar kübit"
    assert db.search_results_for_session("s1")[0]["title"] == "Kübit"
    assert db.count("arama_sonuclari") == 1


def test_foreign_keys_enforced(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.save_text("olmayan-oturum", 1, PRED, None)


def test_sql_injection_is_stored_literally(db):
    _seed(db)
    evil = dict(PRED, text="x'); DROP TABLE metinler; --")
    db.save_text("s1", 2, evil, "1.0.0")
    assert db.count("metinler") == 2
    assert db.session_history("s1")[1]["text"] == evil["text"]


def test_count_rejects_unknown_table(db):
    with pytest.raises(ValueError):
        db.count("sqlite_master; DROP TABLE x")


def test_cache_ttl(db):
    db.cache_put("kuantum", "wikipedia", [{"title": "a"}])
    assert db.cache_get("kuantum")["results"] == [{"title": "a"}]
    old = (datetime.now(UTC) - timedelta(hours=48)).isoformat(timespec="seconds")
    db._conn.execute("UPDATE search_cache SET created_at = ?", (old,))
    assert db.cache_get("kuantum", ttl_hours=24) is None
    db.cache_put("kuantum", "wikipedia", [{"title": "b"}])  # upsert
    assert db.cache_get("kuantum")["results"] == [{"title": "b"}]


def test_session_end_and_duplicate_message_index(db):
    _seed(db)
    db.end_session("s1")
    ended = db._conn.execute("SELECT ended_at FROM sessions WHERE session_id='s1'").fetchone()
    assert ended[0] is not None
    with pytest.raises(sqlite3.IntegrityError):
        db.save_text("s1", 1, PRED, "1.0.0")


def test_write_failure_raises_database_error(tmp_path):
    database = Database(tmp_path / "ro.db")
    database._conn.execute("PRAGMA query_only = ON")
    with pytest.raises(DatabaseError):
        database.start_session("s", None)
    database.close()


def test_unopenable_path_raises(tmp_path):
    blocker = tmp_path / "dosya"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(DatabaseError):
        Database(blocker / "alt" / "x.db")


def test_read_failure_raises_database_error(tmp_path):
    database = Database(tmp_path / "x.db")
    database._conn.execute("DROP TABLE search_cache")
    with pytest.raises(DatabaseError, match="okuma"):
        database.cache_get("kuantum")
    database.close()
