"""
Dosya   : src/logging_setup.py
Konu    : Loglama Ayarları
Açıklama: Bu dosyada amacım uygulama genelinde log biçimini ve seviyesini ayarlamak;
          istenirse logları logs/app.log dosyasına da yazıyorum.
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
    """Logları konsola (varsayılan WARNING) ve istenirse logs/app.log dosyasına (INFO) yazıyorum."""
    root = logging.getLogger()
    if root.handlers:  # iki kez çağrılırsa aynı satırları iki kez yazmıyorum
        return
    # Kök logger'dan her şeyi geçiriyorum; asıl filtrelemeyi her çıkışın (konsol, dosya)
    # kendi seviyesinde yapıyorum. Böylece konsol sade kalırken dosyada ayrıntı tutuyorum.
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, (level or LOG_LEVEL).upper(), logging.WARNING))
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)
    if to_file:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
        except OSError as exc:  # disk yazma hatasının uygulamayı durdurmasını istemedim
            root.warning("Log dosyası açılamadı, yalnızca konsol kullanılacak: %s", exc)
        else:
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(file_handler)
