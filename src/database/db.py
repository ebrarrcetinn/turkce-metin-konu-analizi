"""
Dosya   : src/database/db.py
Konu    : SQLite Veritabanı Katmanı
Açıklama: Bu dosyada amacım metinleri, tahmin edilen konuları, sohbet konularını, arama
          sorgularını ve internet sonuçlarını SQLite veritabanına kaydetmek; şema sürümünü
          de burada yönetiyor ve arama sonuçlarını önbellekte tutuyorum.
Yazar   : Ebrar Cemre Çetin
Tarih   : 26.09.2026
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.config import CACHE_TTL_HOURS, DB_PATH

logger = logging.getLogger(__name__)

# Veritabanı başka bir işlem tarafından kilitliyse hata vermeden önce bekleyeceğim süre.
BUSY_TIMEOUT_MS = 5000

# Şema değişikliklerini sırayla uygulanan adımlar (migration) olarak tutuyorum. Veritabanının
# hangi adımda olduğunu SQLite'ın PRAGMA user_version değerinde saklıyorum; uygulama
# açılırken eksik adımları uyguluyorum. Böylece eski bir nlp_app.db dosyası silinmeden
# güncelleniyor.
#
# Tablolar:
#   model_metadata  : kayıtları üreten modelin sürümü, eğitim zamanı ve metrikleri
#   sessions        : her sohbet oturumu (sıfırla komutu yeni oturum açar)
#   metinler        : kullanıcının her mesajı, genel konu, güven, alt konular, tüm olasılıklar
#   sohbet_konulari : her mesajdan sonraki sohbet konusu ve üretilen arama sorgusu
#   arama_sonuclari : internetten gelen sonuçlar (başlık, özet, bağlantı)
#   search_cache    : aynı sorgu 24 saat içinde tekrar sorulursa ağa çıkmadan cevap vermek için
MIGRATIONS: tuple[str, ...] = (
    # v1 — temel şema
    """
    CREATE TABLE model_metadata (
        model_version      TEXT PRIMARY KEY,
        model_name         TEXT NOT NULL,
        training_timestamp TEXT NOT NULL,
        dataset_version    TEXT NOT NULL,
        artifact_sha256    TEXT NOT NULL,
        metrics_json       TEXT NOT NULL,
        thresholds_json    TEXT NOT NULL,
        registered_at      TEXT NOT NULL
    );
    CREATE TABLE sessions (
        session_id    TEXT PRIMARY KEY,
        started_at    TEXT NOT NULL,
        ended_at      TEXT,
        model_version TEXT REFERENCES model_metadata(model_version)
    );
    CREATE TABLE metinler (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        message_index   INTEGER NOT NULL,
        text            TEXT NOT NULL,
        status          TEXT NOT NULL,
        general_topic   TEXT,
        confidence      REAL NOT NULL,
        subtopics_json  TEXT NOT NULL,
        scores_json     TEXT NOT NULL,
        model_version   TEXT REFERENCES model_metadata(model_version),
        latency_ms      REAL,
        created_at      TEXT NOT NULL,
        UNIQUE (session_id, message_index)
    );
    CREATE TABLE sohbet_konulari (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        metin_id      INTEGER NOT NULL REFERENCES metinler(id) ON DELETE CASCADE,
        theme_label   TEXT NOT NULL,
        theme_phrase  TEXT NOT NULL,
        topics_json   TEXT NOT NULL,
        search_query  TEXT,
        created_at    TEXT NOT NULL
    );
    CREATE TABLE arama_sonuclari (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id       TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        sohbet_konusu_id INTEGER NOT NULL REFERENCES sohbet_konulari(id) ON DELETE CASCADE,
        search_query     TEXT NOT NULL,
        rank             INTEGER NOT NULL,
        title            TEXT NOT NULL,
        summary          TEXT NOT NULL,
        url              TEXT NOT NULL,
        source           TEXT NOT NULL,
        from_cache       INTEGER NOT NULL DEFAULT 0,
        created_at       TEXT NOT NULL
    );
    CREATE TABLE search_cache (
        query_key    TEXT PRIMARY KEY,
        source       TEXT NOT NULL,
        results_json TEXT NOT NULL,
        created_at   TEXT NOT NULL
    );
    CREATE INDEX idx_metinler_session ON metinler(session_id, message_index);
    CREATE INDEX idx_konular_session ON sohbet_konulari(session_id);
    CREATE INDEX idx_arama_session ON arama_sonuclari(session_id);
    CREATE INDEX idx_arama_query ON arama_sonuclari(search_query);
    CREATE INDEX idx_cache_created ON search_cache(created_at);
    """,
    # v2 — sütunu içeriğine uygun adla yeniden adlandırdım (model ağırlıklarının özeti)
    """
    ALTER TABLE model_metadata RENAME COLUMN artifact_sha256 TO weights_sha256;
    """,
)
SCHEMA_VERSION = len(MIGRATIONS)


class DatabaseError(RuntimeError):
    """Veritabanı açılamadı / yazılamadı."""


def utcnow() -> str:
    """Zaman damgalarını saat dilimi karışıklığı olmasın diye UTC ve ISO biçiminde tutuyorum."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dumps(value: Any) -> str:
    """Liste/sözlük alanlarını JSON metni olarak saklıyorum (SQLite'ta liste tipi yok)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Database:
    """Amacım SQLite bağlantısını ve tüm okuma/yazma işlemlerini bu sınıfta toplamak.

    Tüm sorguları parametreli (?) yazdım; kullanıcı metnini SQL'e hiçbir zaman doğrudan
    eklemiyorum, bu da SQL enjeksiyonunu önlüyor.
    """

    def __init__(self, path: Path | str = DB_PATH):
        self.path = Path(path) if str(path) != ":memory:" else path
        try:
            if isinstance(self.path, Path):
                self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), timeout=BUSY_TIMEOUT_MS / 1000)
        except (sqlite3.Error, OSError) as exc:
            raise DatabaseError(f"Veritabanı açılamadı ({path}): {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        try:
            # SQLite'ta yabancı anahtar denetimi varsayılan olarak kapalı; bu yüzden açıyorum.
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            self.migrate()
        except (sqlite3.Error, DatabaseError) as exc:
            self._conn.close()  # Windows'ta açık kalan bağlantı dosyayı kilitliyor
            if isinstance(exc, DatabaseError):
                raise
            raise DatabaseError(f"Veritabanı hazırlanamadı ({path}): {exc}") from exc

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """Yazma işlemlerini transaction içinde yapıyorum: hata olursa yarım kayıt kalmıyor
        (hepsini geri alıyorum), başarılıysa tek seferde kaydediyorum."""
        try:
            with self._conn:
                yield self._conn
        except sqlite3.IntegrityError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError(f"Veritabanı yazma hatası: {exc}") from exc

    def _read(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        try:
            return self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Veritabanı okuma hatası: {exc}") from exc

    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> None:
        """Eksik şema adımlarını sırayla uyguluyorum."""
        current = self.schema_version()
        if current > SCHEMA_VERSION:
            raise DatabaseError(
                f"Veritabanı şeması ({current}) bu uygulamadan ({SCHEMA_VERSION}) daha yeni.")
        for version in range(current, SCHEMA_VERSION):
            try:
                # Her adımı ve sürüm numarasını tek transaction'da yazıyorum: yarıda kalırsa
                # veritabanı eski sürümde kalıyor.
                self._conn.executescript("BEGIN;\n" + MIGRATIONS[version] +
                                         f"\nPRAGMA user_version = {version + 1};\nCOMMIT;")
            except sqlite3.Error as exc:
                self._conn.rollback()
                raise DatabaseError(f"Migration v{version + 1} başarısız: {exc}") from exc
            logger.info("Veritabanı şeması v%d'ye yükseltildi", version + 1)

    def register_model(self, metadata: dict) -> None:
        """Modeli bir kez kaydediyorum; aynı sürüm zaten varsa dokunmuyorum (INSERT OR IGNORE)."""
        with self._write() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO model_metadata
                   (model_version, model_name, training_timestamp, dataset_version,
                    weights_sha256, metrics_json, thresholds_json, registered_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (metadata["model_version"], metadata["model_name"],
                 metadata["training_timestamp"], metadata["dataset"]["version"],
                 metadata.get("weights_sha256", ""), _dumps(metadata["metrics"]),
                 _dumps(metadata["thresholds"]), utcnow()))

    def start_session(self, session_id: str, model_version: str | None) -> None:
        with self._write() as conn:
            conn.execute("INSERT INTO sessions (session_id, started_at, model_version) "
                         "VALUES (?, ?, ?)", (session_id, utcnow(), model_version))

    def end_session(self, session_id: str) -> None:
        with self._write() as conn:
            conn.execute("UPDATE sessions SET ended_at = ? WHERE session_id = ? AND ended_at "
                         "IS NULL", (utcnow(), session_id))

    def save_text(self, session_id: str, message_index: int, prediction: dict,
                  model_version: str | None) -> int:
        """Mesajı ve sınıflandırma sonucunu kaydediyorum.

        Sohbet konusu kaydı için satır id'sini döndürüyorum."""
        with self._write() as conn:
            cur = conn.execute(
                """INSERT INTO metinler (session_id, message_index, text, status, general_topic,
                   confidence, subtopics_json, scores_json, model_version, latency_ms,
                   created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, message_index, prediction["text"], prediction["status"],
                 prediction.get("general"), float(prediction.get("confidence", 0.0)),
                 _dumps(prediction.get("subtopics", [])),
                 _dumps(prediction.get("general_scores", {})), model_version,
                 prediction.get("latency_ms"), utcnow()))
            if cur.lastrowid is None:
                raise DatabaseError("Satır kimliği alınamadı")
            return int(cur.lastrowid)

    def save_theme(self, session_id: str, metin_id: int, theme: dict,
                   query: str | None) -> int:
        """Mesajdan sonraki sohbet konusunu ve arama sorgusunu kaydediyorum."""
        with self._write() as conn:
            cur = conn.execute(
                """INSERT INTO sohbet_konulari (session_id, metin_id, theme_label, theme_phrase,
                   topics_json, search_query, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, metin_id, theme["label"], theme["phrase"],
                 _dumps(theme["topics"]), query, utcnow()))
            if cur.lastrowid is None:
                raise DatabaseError("Satır kimliği alınamadı")
            return int(cur.lastrowid)

    def save_search_results(self, session_id: str, theme_id: int, query: str,
                            results: list[dict], from_cache: bool) -> int:
        """Arama sonuçlarını sırasıyla (rank 1 en alakalı) kaydediyorum."""
        rows = [(session_id, theme_id, query, rank, r["title"], r["summary"], r["url"],
                 r["source"], int(from_cache), utcnow())
                for rank, r in enumerate(results, start=1)]
        with self._write() as conn:
            conn.executemany(
                """INSERT INTO arama_sonuclari (session_id, sohbet_konusu_id, search_query,
                   rank, title, summary, url, source, from_cache, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)
        return len(rows)

    def cache_get(self, query_key: str, ttl_hours: float = CACHE_TTL_HOURS) -> dict | None:
        """Sorgu önbellekte ve süresi dolmamışsa kayıtlı sonuçları döndürüyorum."""
        rows = self._read(
            "SELECT source, results_json, created_at FROM search_cache WHERE query_key = ?",
            (query_key,))
        if not rows:
            return None
        row = rows[0]
        created = datetime.fromisoformat(row["created_at"])
        if datetime.now(UTC) - created > timedelta(hours=ttl_hours):
            return None
        return {"source": row["source"], "results": json.loads(row["results_json"])}

    def cache_put(self, query_key: str, source: str, results: list[dict]) -> None:
        """Sonucu önbelleğe yazıyorum; sorgu zaten varsa güncelliyorum (upsert)."""
        with self._write() as conn:
            conn.execute(
                """INSERT INTO search_cache (query_key, source, results_json, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(query_key) DO UPDATE SET source = excluded.source,
                   results_json = excluded.results_json, created_at = excluded.created_at""",
                (query_key, source, _dumps(results), utcnow()))

    def session_history(self, session_id: str) -> list[dict]:
        """Bir oturumdaki mesajları ve her birinden sonraki sohbet konusunu döndürüyorum."""
        rows = self._read(
            """SELECT m.message_index, m.text, m.status, m.general_topic, m.confidence,
                      m.subtopics_json, k.theme_label, k.search_query
               FROM metinler m LEFT JOIN sohbet_konulari k ON k.metin_id = m.id
               WHERE m.session_id = ? ORDER BY m.message_index""", (session_id,))
        return [{**dict(r), "subtopics": json.loads(r["subtopics_json"])} for r in rows]

    def count(self, table: str) -> int:
        # Tablo adı parametre olarak verilemediği için f-string kullandım; bu yüzden yalnızca
        # bilinen tablo adlarını kabul ediyorum.
        if table not in {"metinler", "sohbet_konulari", "arama_sonuclari", "sessions",
                         "search_cache", "model_metadata"}:
            raise ValueError(f"Bilinmeyen tablo: {table}")
        return int(self._read(f"SELECT COUNT(*) FROM {table}")[0][0])  # noqa: S608

    def search_results_for_session(self, session_id: str) -> list[dict]:
        rows = self._read(
            "SELECT search_query, rank, title, url, source, from_cache FROM arama_sonuclari "
            "WHERE session_id = ? ORDER BY id", (session_id,))
        return [dict(r) for r in rows]
