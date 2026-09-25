"""Türkçe konu analizi konsol uygulaması.

Kullanım: python proje.py [--no-web] [--debug] [--db YOL] [--model YOL]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import DB_PATH, MODEL_PATH, WEB_SEARCH_ENABLED
from src.database.db import Database, DatabaseError
from src.logging_setup import setup_logging
from src.models.artifact import ModelArtifactError
from src.models.predictor import (
    STATUS_OUT_OF_SCOPE,
    STATUS_UNCERTAIN,
    TopicClassifier,
)
from src.services.app import ChatService, TurnResult
from src.services.web_search import STATUS_EMPTY, STATUS_OFFLINE, STATUS_OK

logger = logging.getLogger("proje")

EXIT_COMMANDS = {"çıkış", "cikis", "q", "quit", "exit"}
HISTORY_COMMANDS = {"geçmiş", "gecmis"}
RESET_COMMANDS = {"sıfırla", "sifirla"}
HELP_COMMANDS = {"yardım", "yardim", "help", "?"}
HELP_TEXT = ("Komutlar: 'geçmiş' (bu oturumun mesajları), 'sıfırla' (yeni sohbet bağlamı; "
             "kayıtlar silinmez), 'yardım', 'çıkış' veya 'q'.")


def pct(value: float) -> str:
    return f"%{value * 100:.1f}".replace(".", ",")


def render_turn(result: TurnResult, out) -> None:
    p = result.prediction
    print("\n[Metin Analizi]", file=out)
    for warning in result.warnings:
        print(f"Uyarı: {warning}", file=out)
    if p.status == "empty":
        return
    if p.status == STATUS_OUT_OF_SCOPE:
        print(f"Genel Konu: Taksonomi dışında olabilir ({pct(p.confidence)})", file=out)
    elif p.status == STATUS_UNCERTAIN:
        print(f"Genel Konu: Belirsiz — en olası: {p.top_guess} ({pct(p.confidence)})", file=out)
    else:
        print(f"Genel Konu: {p.general}", file=out)
        print(f"Güven: {pct(p.confidence)}", file=out)
        if p.subtopics:
            print("Alt Konular:", file=out)
            for name, score in p.subtopics:
                print(f"  - {name} ({pct(score)})", file=out)
    if p.short_input:
        print("Not: Kısa girdi; güven skoru buna göre ayarlandı.", file=out)

    print("\n[Sohbet]", file=out)
    theme = result.theme
    if theme is None or theme.uncertain:
        print("Genel sohbet konusu: henüz belirsiz (taksonomi dışı mesajlar)", file=out)
    else:
        print(f"Genel sohbet konusu: {theme.label}", file=out)
        print(f"Tema: {theme.phrase}", file=out)
    if result.query:
        print(f'Arama sorgusu: "{result.query}"', file=out)

    if result.search is not None:
        print("\n[İnternet Sonuçları]", file=out)
        s = result.search
        if s.status == STATUS_OK:
            suffix = " (önbellekten)" if s.from_cache else ""
            print(f"Kaynak: {s.source}{suffix}", file=out)
            for i, r in enumerate(s.results, start=1):
                print(f"{i}. {r.title}\n   {r.summary}\n   {r.url}", file=out)
        elif s.status == STATUS_OFFLINE:
            print("İnternete erişilemedi; sonuçlar gösterilemiyor (analiz kaydedildi).", file=out)
        elif s.status == STATUS_EMPTY:
            print("Bu sorgu için sonuç bulunamadı.", file=out)
        else:
            print("Arama şu an yapılamadı; daha sonra tekrar deneyin.", file=out)
    if result.saved_to_db:
        print("\nSonuçlar veritabanına kaydedildi.", file=out)
    elif result.db_error:
        print("\nUyarı: Veritabanına yazılamadı; analiz yalnızca bu oturumda görüntülendi.",
              file=out)


def render_history(service: ChatService, out) -> None:
    if not service.history:
        print("Bu oturumda henüz mesaj yok.", file=out)
        return
    print(f"Oturum: {service.session_id}", file=out)
    for item in service.history:
        p = item.prediction
        label = p.general if p.status == "ok" else f"{p.general} (en olası: {p.top_guess})"
        print(f"{item.index}. {item.text}\n   -> {label} {pct(p.confidence)} | "
              f"sohbet: {item.theme_label}", file=out)


def run(service: ChatService, stdin=sys.stdin, out=sys.stdout) -> int:
    print(HELP_TEXT, file=out)
    while True:
        print("\nMetin girin:", file=out)
        try:
            line = stdin.readline()
        except KeyboardInterrupt:
            print("\nÇıkılıyor.", file=out)
            break
        if line == "":  # EOF (Ctrl+D / boru sonu)
            print("\nGirdi sonu; çıkılıyor.", file=out)
            break
        text = line.strip()
        command = text.casefold()
        if command in EXIT_COMMANDS:
            print("Güle güle!", file=out)
            break
        if command in HISTORY_COMMANDS:
            render_history(service, out)
            continue
        if command in RESET_COMMANDS:
            sid = service.reset()
            print(f"Yeni sohbet başlatıldı (oturum {sid[:8]}…). Önceki kayıtlar korunuyor.",
                  file=out)
            continue
        if command in HELP_COMMANDS:
            print(HELP_TEXT, file=out)
            continue
        if not text:
            print("Boş girdi; lütfen bir metin yazın.", file=out)
            continue
        try:
            render_turn(service.process(text), out)
        except KeyboardInterrupt:
            print("\nİşlem iptal edildi.", file=out)
    service.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Türkçe NLP Konu Analiz Sistemi")
    parser.add_argument("--no-web", action="store_true", help="İnternet aramasını kapat")
    parser.add_argument("--debug", action="store_true", help="Ayrıntılı log")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite dosya yolu")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Model artifact yolu")
    args = parser.parse_args(argv)
    # Konsolda yalnızca hatalar; ayrıntılar logs/app.log dosyasında (--debug ile konsolda).
    setup_logging("DEBUG" if args.debug else "ERROR", to_file=True)

    print("Türkçe NLP Konu Analiz Sistemi")
    print("Model yükleniyor...")
    try:
        classifier = TopicClassifier.load(args.model)
    except ModelArtifactError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 2
    print(f"Model hazır (v{classifier.version}).")

    try:
        db: Database | None = Database(args.db)
    except DatabaseError as exc:
        print(f"Uyarı: {exc}\nKayıt yapılmadan devam ediliyor.", file=sys.stderr)
        db = None
    web = WEB_SEARCH_ENABLED and not args.no_web
    if not web:
        print("Web araması kapalı (çevrimdışı mod).")
    service = ChatService(classifier, db, web_enabled=web)
    try:
        return run(service)
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
