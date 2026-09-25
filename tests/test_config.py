"""
Dosya   : tests/test_config.py
Konu    : Ayar Testleri
Açıklama: Geçersiz ortam değişkenlerinde varsayılan değerlere dönülmesini test eder.
Yazar   : Ebrar Cemre Çetin
Tarih   : 27.09.2026
"""

import importlib

import src.config as config


def test_invalid_env_float_falls_back(monkeypatch):
    monkeypatch.setenv("NLP_DECAY", "yedi")
    monkeypatch.setenv("NLP_WEB_TIMEOUT", "")
    reloaded = importlib.reload(config)
    assert reloaded.DECAY == 0.7 and reloaded.WEB_TIMEOUT == 6.0
    monkeypatch.delenv("NLP_DECAY")
    monkeypatch.delenv("NLP_WEB_TIMEOUT")
    importlib.reload(config)


def test_confidence_grid():
    assert config.CONFIDENCE_GRID[0] == 0.2 and config.CONFIDENCE_GRID[-1] == 0.9
