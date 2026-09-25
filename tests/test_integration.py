"""
Dosya   : tests/test_integration.py
Konu    : Uçtan Uca Testler
Açıklama: Sınıflandırmadan veritabanına ve arama katmanına kadar tüm akışı ve konsol
          uygulamasını test eder.
Yazar   : Ebrar Cemre Çetin
Tarih   : 27.09.2026
"""

import io
import os
import subprocess
import sys

import pytest

from src.config import PROJECT_ROOT
from src.database.db import Database
from src.services.app import ChatService
from src.services.web_search import (
    STATUS_OFFLINE,
    STATUS_OK,
    SearchOutcome,
    SearchResult,
    WebSearchClient,
)
from tests.conftest import FakeResponse, FakeSession, load_fixture

pytestmark = pytest.mark.model


class RecordingClient:
    def __init__(self, status=STATUS_OK):
        self.queries: list[str] = []
        self.status = status

    def search(self, query):
        self.queries.append(query)
        if self.status != STATUS_OK:
            return SearchOutcome(query=query, status=self.status, error="ağ yok")
        return SearchOutcome(query=query, status=STATUS_OK, source="wikipedia", results=[
            SearchResult("Kuantum bilgisayar", "özet", "https://tr.wikipedia.org/wiki/K",
                         "wikipedia")])


def test_full_chain_persists_everything(classifier, db):
    client = RecordingClient()
    service = ChatService(classifier, db, search_client=client, web_enabled=True)
    result = service.process("Kuantum işlemciler kübitler kullanarak algoritmaları hızlandırır.")

    assert result.prediction.top_guess == "Teknoloji"
    assert result.theme.topics[0][0] == "Teknoloji"
    assert result.query and "kuantum" in result.query
    assert client.queries == [result.query]
    assert result.search.status == STATUS_OK
    assert result.saved_to_db
    assert db.count("metinler") == 1 and db.count("sohbet_konulari") == 1
    assert db.count("arama_sonuclari") == 1 and db.count("model_metadata") == 1
    history = db.session_history(service.session_id)
    assert history[0]["search_query"] == result.query


def test_real_http_client_through_service_with_fake_transport(classifier, db):
    session = FakeSession({"wikipedia": [FakeResponse(200, load_fixture(
        "wikipedia_search.json"))]})
    client = WebSearchClient(session=session, sleep=lambda _: None)
    service = ChatService(classifier, db, search_client=client, web_enabled=True)
    result = service.process("Kuantum bilgisayarlar kübit kullanır.")
    assert result.search.status == STATUS_OK and result.search.source == "wikipedia"
    assert db.search_results_for_session(service.session_id)[0]["title"] == "Kuantum bilgisayar"


def test_search_cache_avoids_repeat_requests(classifier, db):
    client = RecordingClient()
    service = ChatService(classifier, db, search_client=client, web_enabled=True)
    first = service.process("Kuantum bilgisayarlar kübit kullanır.")
    second = service.process("Kuantum bilgisayarlar kübit kullanır.")
    assert first.query == second.query
    assert len(client.queries) == 1
    assert second.search.from_cache


def test_offline_does_not_break_classification_and_circuit_breaks(classifier, db):
    client = RecordingClient(status=STATUS_OFFLINE)
    now = [0.0]
    service = ChatService(classifier, db, search_client=client, web_enabled=True,
                          clock=lambda: now[0])
    r1 = service.process("Periyodik tabloda soygazlar kararlıdır.")
    r2 = service.process("Asitler ve bazlar tepkimeye girer.")
    assert r1.prediction.top_guess == "Kimya" and r1.saved_to_db
    assert r1.search.status == STATUS_OFFLINE and r2.search.status == STATUS_OFFLINE
    assert len(client.queries) == 1  # ikinci mesajda ağ denenmedi
    now[0] = 10_000.0
    service.process("Elementler periyodik tabloda sıralanır.")
    assert len(client.queries) == 2  # bekleme süresi dolunca tekrar denendi
    assert db.count("metinler") == 3


def test_database_failure_does_not_break_classification(classifier, tmp_path):
    database = Database(tmp_path / "x.db")
    service = ChatService(classifier, database, web_enabled=False)
    database._conn.execute("PRAGMA query_only = ON")  # yazma hatası simülasyonu
    result = service.process("Basketbolda üç sayılık atış çok önemlidir.")
    assert result.prediction.top_guess == "Spor"
    assert not result.saved_to_db and result.db_error
    assert result.theme is not None
    database.close()


def test_cache_read_failure_falls_back_to_web(classifier, tmp_path):
    database = Database(tmp_path / "x.db")
    database._conn.execute("DROP TABLE search_cache")
    client = RecordingClient()
    service = ChatService(classifier, database, search_client=client, web_enabled=True)
    result = service.process("Kuantum bilgisayarlar kübit kullanır.")
    assert result.search.status == STATUS_OK and len(client.queries) == 1
    database.close()


def test_no_database_mode(classifier):
    service = ChatService(classifier, None, web_enabled=False)
    result = service.process("Osmanlı padişahı sefere çıktı.")
    assert result.prediction.top_guess == "Tarih" and not result.saved_to_db


def test_empty_message_is_not_tracked_or_saved(classifier, db):
    service = ChatService(classifier, db, web_enabled=False)
    result = service.process("!!!")
    assert result.warnings and result.theme is None
    assert db.count("metinler") == 0 and service.tracker.turns == 0


def test_reset_keeps_db_rows_and_changes_session(classifier, db):
    service = ChatService(classifier, db, web_enabled=False)
    service.process("Futbol maçı berabere bitti.")
    old = service.session_id
    new = service.reset()
    assert new != old and service.tracker.turns == 0 and service.history == []
    assert db.count("metinler") == 1 and db.count("sessions") == 2
    service.process("Osmanlı padişahı sefere çıktı.")
    assert len(db.session_history(old)) == 1 and len(db.session_history(new)) == 1


def test_conversation_acceptance_books_science_biology(classifier, db):
    """Sohbet kabul testi: Kitaplar -> Bilim -> Biyoloji."""
    service = ChatService(classifier, db, web_enabled=False)
    steps = [service.process("Dün akşam sürükleyici bir roman okudum."),
             service.process("Bilimsel yöntem ve deneyler hakkında düşünüyorum."),
             service.process("Hücreler, DNA ve evrim konusu beni çok etkiliyor.")]
    assert [s.prediction.general for s in steps] == ["Kitaplar", "Bilim", "Biyoloji"]
    assert steps[1].theme.phrase == "bilimsel kitaplar"
    assert steps[2].theme.phrase == "biyoloji hakkında bilimsel kitaplar"


def test_conversation_with_spec_sentences_keeps_books_and_biology(classifier, db):
    """Şartnamedeki A, B, C cümleleriyle: B 'Belirsiz' olsa da yumuşak olasılıkla bağlama
    katılır."""
    service = ChatService(classifier, db, web_enabled=False)
    service.process("Kütüphaneden aldığım romanı okudum, yazarın anlatımı çok akıcıydı.")
    second = service.process("Bilim insanları hipotezlerini deney ve gözlem kullanarak test eder.")
    third = service.process("Hücreler DNA taşır ve canlılar evrim geçirerek çeşitlenir.")
    assert second.theme.phrase == "bilimsel kitaplar"
    assert {t for t, _ in third.theme.topics} >= {"Biyoloji", "Kitaplar"}


def _run_cli(args, stdin_text, tmp_path):
    return subprocess.run(
        [sys.executable, "proje.py", "--db", str(tmp_path / "cli.db"), *args],
        input=stdin_text, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "NLP_LOG_LEVEL": "ERROR",
             "NLP_WEB_SEARCH": "0"})


def test_cli_session_commands(tmp_path):
    script = ("Kuantum bilgisayarlar kübit kullanır.\n\n!!!\ngeçmiş\nsıfırla\ngeçmiş\n"
              "yardım\nq\n")
    proc = _run_cli(["--no-web"], script, tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "Model hazır" in out and "Genel Konu: Teknoloji" in out
    assert "Kuantum Bilgisayarlar" in out and 'Arama sorgusu: "kuantum bilgisayarlar' in out
    assert "Boş girdi" in out and "Anlamlı içerik bulunamadı" in out
    assert "1. Kuantum bilgisayarlar kübit kullanır." in out
    assert "Yeni sohbet başlatıldı" in out and "Bu oturumda henüz mesaj yok" in out
    assert "Güle güle" in out and "Traceback" not in proc.stderr
    with Database(tmp_path / "cli.db") as database:
        assert database.count("metinler") == 1 and database.count("sessions") == 2


def test_cli_handles_eof(tmp_path):
    proc = _run_cli(["--no-web"], "Futbol maçı\n", tmp_path)
    assert proc.returncode == 0 and "Girdi sonu" in proc.stdout


def test_model_preparer_trains_when_model_missing(tmp_path, monkeypatch):
    """Model yoksa proje.py veriyi indirip eğitir; burada adımlar sahte nesnelerle izlenir."""
    import proje

    calls = []
    processed = tmp_path / "processed"

    class FakeBuilder:
        def __init__(self, out_dir):
            calls.append("build")

        def build(self):
            processed.mkdir()

    class FakeTrainer:
        report = {"algorithm": "softmax_regression",
                  "metrics": {"val_thresholded": {"macro_f1": 0.8}}}

        def __init__(self, processed_dir, model_path):
            calls.append("train")

        def run(self):
            return "model"

    monkeypatch.setattr(proje, "fetch_all", lambda: calls.append("download"))
    monkeypatch.setattr(proje, "DatasetBuilder", FakeBuilder)
    monkeypatch.setattr(proje, "Trainer", FakeTrainer)
    preparer = proje.ModelPreparer(tmp_path / "model", processed, out=io.StringIO())
    assert preparer.prepare() == "model"
    assert calls == ["download", "build", "train"]
