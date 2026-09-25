"""
Dosya   : src/logging_setup.py
Konu    : Loglama Ayarları
Açıklama: Uygulama genelinde log biçimini ve seviyesini ayarlar; istenirse logları
          logs/app.log dosyasına da yazar.
Yazar   : Ebrar Cemre Çetin
Tarih   : 23.09.2026
"""

from __future__ import annotations

import logging
import sys

from src.config import LOG_DIR, LOG_LEVEL

# Örnek satır: 2026-09-27 10:15:02 INFO src.models.trainer: Özellik sayısı: 196785
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str | None = None, to_file: bool = False) -> None:
    """Konsola (varsayılan WARNING) ve isteğe bağlı olarak logs/app.log dosyasına (INFO) yazar."""
    root = logging.getLogger()
    if root.handlers:  # iki kez çağrılırsa aynı satırlar iki kez yazılmasın
        return
    # Kök logger her şeyi geçirir; asıl filtreleme her çıkışın (konsol, dosya) kendi
    # seviyesinde yapılır. Böylece konsol sade kalırken dosyada ayrıntı tutulur.
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, (level or LOG_LEVEL).upper(), logging.WARNING))
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)
    if to_file:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
        except OSError as exc:  # disk yazma hatası uygulamayı durdurmamalı
            root.warning("Log dosyası açılamadı, yalnızca konsol kullanılacak: %s", exc)
        else:
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(file_handler)
