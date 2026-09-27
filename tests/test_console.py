"""
Dosya   : tests/test_console.py
Konu    : Konsol Komut Testleri
Açıklama: Komutların büyük-küçük harften (Türkçe I/İ dahil) bağımsız tanınmasını ve geçmiş
          çıktısını model gerektirmeden test eder.
İsim    : Ebrar Cemre Çetin
Tarih   : 27.09.2026
"""

import io
from types import SimpleNamespace

import pytest

from proje import (
    EXIT_COMMANDS,
    HELP_COMMANDS,
    HISTORY_COMMANDS,
    RESET_COMMANDS,
    ConsoleApp,
    is_command,
)
from src.models.topic_model import (
    STATUS_OK,
    STATUS_OUT_OF_SCOPE,
    STATUS_UNCERTAIN,
    Prediction,
)
from src.services.app import HistoryItem


@pytest.mark.parametrize("text, commands", [
    ("çıkış", EXIT_COMMANDS), ("Çıkış", EXIT_COMMANDS), ("ÇIKIŞ", EXIT_COMMANDS),
    ("CIKIS", EXIT_COMMANDS), ("Q", EXIT_COMMANDS), ("QUIT", EXIT_COMMANDS),
    ("GEÇMİŞ", HISTORY_COMMANDS), ("Geçmiş", HISTORY_COMMANDS), ("GECMIS", HISTORY_COMMANDS),
    ("SIFIRLA", RESET_COMMANDS), ("Sıfırla", RESET_COMMANDS),
    ("YARDIM", HELP_COMMANDS), ("HELP", HELP_COMMANDS),
])
def test_commands_are_case_insensitive_including_turkish_i(text, commands):
    assert is_command(text, commands)


@pytest.mark.parametrize("text", ["çıkışta", "Kitap", "", "geçmişte ne oldu"])
def test_regular_text_is_not_a_command(text):
    all_commands = EXIT_COMMANDS | HISTORY_COMMANDS | RESET_COMMANDS | HELP_COMMANDS
    assert not is_command(text, all_commands)


def _console(history=(), stdin_text=""):
    service = SimpleNamespace(history=list(history), session_id="abc123",
                              closed=False, process=None)
    service.close = lambda: setattr(service, "closed", True)
    out = io.StringIO()
    return ConsoleApp(service, stdin=io.StringIO(stdin_text), out=out), service, out


def test_uppercase_turkish_exit_command_ends_session():
    app, service, out = _console(stdin_text="ÇIKIŞ\n")
    assert app.run() == 0
    assert "Güle güle!" in out.getvalue() and service.closed


def test_history_does_not_repeat_same_label():
    items = [
        HistoryItem(1, "Hakem ofsayt verdi", Prediction(
            "Hakem ofsayt verdi", STATUS_OK, "Spor", 0.99, top_guess="Spor"), "Spor"),
        HistoryItem(2, "baklava tarifi", Prediction(
            "baklava tarifi", STATUS_OUT_OF_SCOPE, "Diğer", 0.9, top_guess="Diğer"), "Belirsiz"),
        HistoryItem(3, "hipotez", Prediction(
            "hipotez", STATUS_UNCERTAIN, "Belirsiz", 0.4, top_guess="Bilim"), "Bilim"),
    ]
    app, _, out = _console(items)
    app.render_history()
    text = out.getvalue()
    assert "-> Spor %99,0" in text
    assert "-> Diğer %90,0" in text and "en olası: Diğer" not in text
    assert "-> Belirsiz (en olası: Bilim) %40,0" in text
