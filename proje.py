"""
Dosya   : proje.py
Konu    : Türkçe Metin Konu Analizi - Konsol Uygulaması (tek giriş noktası)
Açıklama: Kullanıcıdan döngü içinde metin alır; metnin genel konusunu ve alt konularını,
          sohbetin genel konusunu, üretilen arama sorgusunu ve internet sonuçlarını
          konsola yazar, hepsini SQLite veritabanına kaydeder. Model dosyası yoksa veriyi
          indirip modeli kendisi eğitir.
Yazar   : Ebrar Cemre Çetin
Tarih   : 27.09.2026

Kullanım: python proje.py [--egit] [--degerlendir] [--no-web] [--debug] [--db YOL]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TextIO

from src.config import DB_PATH, MODEL_PATH, PROCESSED_DIR, WEB_SEARCH_ENABLED
from src.data.dataset import DatasetBuilder, DatasetError
from src.data.sources import DataAcquisitionError, fetch_all
from src.database.db import Database, DatabaseError
from src.logging_setup import setup_logging
from src.models.evaluator import Evaluator
from src.models.topic_model import (
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_OUT_OF_SCOPE,
    STATUS_UNCERTAIN,
    ModelFileError,
    TopicModel,
)
from src.models.trainer import Trainer
from src.services.app import ChatService, TurnResult
from src.services.web_search import STATUS_EMPTY as SEARCH_EMPTY
from src.services.web_search import STATUS_OFFLINE as SEARCH_OFFLINE
from src.services.web_search import STATUS_OK as SEARCH_OK
from src.services.web_search import SearchOutcome

logger = logging.getLogger("proje")

EXIT_COMMANDS = {"çıkış", "cikis", "q", "quit", "exit"}
HISTORY_COMMANDS = {"geçmiş", "gecmis"}
RESET_COMMANDS = {"sıfırla", "sifirla"}
HELP_COMMANDS = {"yardım", "yardim", "help", "?"}
HELP_TEXT = ("Komutlar: 'geçmiş' (bu oturumun mesajları), 'sıfırla' (yeni sohbet bağlamı; "
             "kayıtlar silinmez), 'yardım', 'çıkış' veya 'q'.")


def pct(value: float) -> str:
    return f"%{value * 100:.1f}".replace(".", ",")


class ModelPreparer:
    """Model yoksa (veya istenirse) veriyi indirir, veri setini kurar ve modeli eğitir."""

    def __init__(self, model_path: Path = MODEL_PATH, processed_dir: Path = PROCESSED_DIR,
                 out: TextIO = sys.stdout):
        self.model_path = model_path
        self.processed_dir = processed_dir
        self.out = out

    def model_exists(self) -> bool:
        return self.model_path.with_suffix(".npz").exists() and \
            self.model_path.with_suffix(".json").exists()

    def prepare(self, force: bool = False) -> TopicModel:
        if self.model_exists() and not force:
            return TopicModel.load(self.model_path)
        if not (self.processed_dir / "train.jsonl").exists() or force:
            print("Veri indiriliyor ve doğrulanıyor (ilk çalıştırmada birkaç dakika "
                  "sürebilir)...", file=self.out)
            fetch_all()
            print("Veri seti hazırlanıyor...", file=self.out)
            DatasetBuilder(out_dir=self.processed_dir).build()
        print("Model eğitiliyor (yaklaşık 1 dakika)...", file=self.out)
        trainer = Trainer(processed_dir=self.processed_dir, model_path=self.model_path)
        model = trainer.run()
        metrics = trainer.report["metrics"]
        print(f"Eğitim bitti: {trainer.report['algorithm']} seçildi (doğrulama macro F1 "
              f"{metrics['val_thresholded']['macro_f1']:.3f}).", file=self.out)
        return model


class ConsoleApp:
    """Konsol döngüsü: girdi okur, servisi çağırır, sonucu biçimlendirerek yazar."""

    def __init__(self, service: ChatService, stdin: TextIO = sys.stdin,
                 out: TextIO = sys.stdout):
        self.service = service
        self.stdin = stdin
        self.out = out

    def _print(self, text: str = "") -> None:
        print(text, file=self.out)

    def render_turn(self, result: TurnResult) -> None:
        p = result.prediction
        self._print("\n[Metin Analizi]")
        for warning in result.warnings:
            self._print(f"Uyarı: {warning}")
        if p.status == STATUS_EMPTY:
            return
        if p.status == STATUS_OUT_OF_SCOPE:
            self._print(f"Genel Konu: Taksonomi dışında olabilir ({pct(p.confidence)})")
        elif p.status == STATUS_UNCERTAIN:
            self._print(f"Genel Konu: Belirsiz — en olası: {p.top_guess} ({pct(p.confidence)})")
        else:
            self._print(f"Genel Konu: {p.general}")
            self._print(f"Güven: {pct(p.confidence)}")
            if p.subtopics:
                self._print("Alt Konular:")
                for name, score in p.subtopics:
                    self._print(f"  - {name} ({pct(score)})")
        if p.related_topics:
            self._print("İlişkili Konular: " + ", ".join(
                f"{name} ({pct(score)})" for name, score in p.related_topics))

        self._print("\n[Sohbet]")
        theme = result.theme
        if theme is None or theme.uncertain:
            self._print("Genel sohbet konusu: henüz belirsiz (taksonomi dışı mesajlar)")
        else:
            self._print(f"Genel sohbet konusu: {theme.label}")
            self._print(f"Tema: {theme.phrase}")
        if result.query:
            self._print(f'Arama sorgusu: "{result.query}"')

        if result.search is not None:
            self._render_search(result.search)
        if result.saved_to_db:
            self._print("\nSonuçlar veritabanına kaydedildi.")
        elif result.db_error:
            self._print("\nUyarı: Veritabanına yazılamadı; analiz yalnızca bu oturumda "
                        "görüntülendi.")

    def _render_search(self, search: SearchOutcome) -> None:
        self._print("\n[İnternet Sonuçları]")
        if search.status == SEARCH_OK:
            suffix = " (önbellekten)" if search.from_cache else ""
            self._print(f"Kaynak: {search.source}{suffix}")
            for i, item in enumerate(search.results, start=1):
                self._print(f"{i}. {item.title}\n   {item.summary}\n   {item.url}")
        elif search.status == SEARCH_OFFLINE:
            self._print("İnternete erişilemedi; sonuçlar gösterilemiyor (analiz kaydedildi).")
        elif search.status == SEARCH_EMPTY:
            self._print("Bu sorgu için sonuç bulunamadı.")
        else:
            self._print("Arama şu an yapılamadı; daha sonra tekrar deneyin.")

    def render_history(self) -> None:
        if not self.service.history:
            self._print("Bu oturumda henüz mesaj yok.")
            return
        self._print(f"Oturum: {self.service.session_id}")
        for item in self.service.history:
            p = item.prediction
            label = p.general if p.status == STATUS_OK else \
                f"{p.general} (en olası: {p.top_guess})"
            self._print(f"{item.index}. {item.text}\n   -> {label} {pct(p.confidence)} | "
                        f"sohbet: {item.theme_label}")

    def handle_command(self, command: str) -> bool:
        """Komutsa işler ve True döndürür."""
        if command in HISTORY_COMMANDS:
            self.render_history()
        elif command in RESET_COMMANDS:
            session_id = self.service.reset()
            self._print(f"Yeni sohbet başlatıldı (oturum {session_id[:8]}…). "
                        "Önceki kayıtlar korunuyor.")
        elif command in HELP_COMMANDS:
            self._print(HELP_TEXT)
        else:
            return False
        return True

    def run(self) -> int:
        self._print(HELP_TEXT)
        while True:
            self._print("\nMetin girin:")
            try:
                line = self.stdin.readline()
            except KeyboardInterrupt:
                self._print("\nÇıkılıyor.")
                break
            if line == "":  # EOF (Ctrl+D / Ctrl+Z ya da boru sonu)
                self._print("\nGirdi sonu; çıkılıyor.")
                break
            text = line.strip()
            command = text.casefold()
            if command in EXIT_COMMANDS:
                self._print("Güle güle!")
                break
            if self.handle_command(command):
                continue
            if not text:
                self._print("Boş girdi; lütfen bir metin yazın.")
                continue
            try:
                self.render_turn(self.service.process(text))
            except KeyboardInterrupt:
                self._print("\nİşlem iptal edildi.")
        self.service.close()
        return 0


def print_evaluation(report: dict) -> None:
    test = report["test"]["thresholded"]
    print(f"Test: doğruluk {test['accuracy']:.3f}, macro F1 {test['macro_f1']:.3f}")
    print(f"Görülmemiş konu (OOD) reddetme oranı: {report['ood_unseen']['rejected_rate']:.3f}")
    for name, item in report["external"].items():
        print(f"Harici set {name} ({item['target']}): {item['recall_argmax']:.3f}")
    for case in report["sentence_tests"] + report["conversation_scenarios"]:
        title = case.get("name") or case["text"]
        print(f"[{'GEÇTİ' if case['pass'] else 'KALDI'}] {title}")
    print("Ayrıntılar: reports/degerlendirme_raporu.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Türkçe Metin Konu Analizi")
    parser.add_argument("--egit", action="store_true",
                        help="Veriyi yeniden indirip modeli baştan eğit")
    parser.add_argument("--degerlendir", action="store_true",
                        help="Modeli test setlerinde değerlendir ve rapor yaz, sonra çık")
    parser.add_argument("--no-web", action="store_true", help="İnternet aramasını kapat")
    parser.add_argument("--debug", action="store_true", help="Ayrıntılı log")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite dosya yolu")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Model dosya yolu")
    args = parser.parse_args(argv)
    # Konsolda yalnızca hatalar; ayrıntılar logs/app.log dosyasında (--debug ile konsolda).
    setup_logging("DEBUG" if args.debug else "ERROR", to_file=True)

    print("Türkçe Metin Konu Analizi")
    try:
        model = ModelPreparer(args.model).prepare(force=args.egit)
    except (ModelFileError, DatasetError, DataAcquisitionError) as exc:
        print(f"HATA: {exc}\nİnternet bağlantısını kontrol edip 'python proje.py --egit' "
              "ile tekrar deneyin.", file=sys.stderr)
        return 2
    print(f"Model hazır (v{model.version}).")

    if args.degerlendir:
        print_evaluation(Evaluator(model).run())
        return 0

    try:
        db: Database | None = Database(args.db)
    except DatabaseError as exc:
        print(f"Uyarı: {exc}\nKayıt yapılmadan devam ediliyor.", file=sys.stderr)
        db = None
    web = WEB_SEARCH_ENABLED and not args.no_web
    if not web:
        print("Web araması kapalı (çevrimdışı mod).")
    service = ChatService(model, db, web_enabled=web)
    try:
        return ConsoleApp(service).run()
    finally:
        if db is not None:
            db.close()


def entrypoint() -> int:
    # Windows'ta çıktı boruya yönlendirildiğinde yerel kod sayfası Türkçe karakterleri
    # kodlayamazsa çökmek yerine yer tutucu yazılır.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    try:
        return main()
    except KeyboardInterrupt:
        print("\nİptal edildi.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(entrypoint())
