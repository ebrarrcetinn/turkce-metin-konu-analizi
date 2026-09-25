"""Ortak test yardımcıları."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import requests

from src.config import MODEL_PATH

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def classifier():
    """Eğitilmiş model gerekir; yoksa model testleri açık bir nedenle atlanır."""
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact yok: `python train.py` çalıştırın")
    from src.models.predictor import TopicClassifier

    return TopicClassifier.load()


@pytest.fixture
def db(tmp_path):
    from src.database.db import Database

    database = Database(tmp_path / "test.db")
    yield database
    database.close()


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str | None = None,
                 headers: dict | None = None):
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload, ensure_ascii=False)
        self.headers = headers or {}
        self.encoding = None


class FakeSession:
    """Sıralı yanıtlar döndüren sahte HTTP oturumu. Öğe bir istisna ise fırlatılır."""

    def __init__(self, responses_by_host: dict[str, list]):
        self.responses = {k: list(v) for k, v in responses_by_host.items()}
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None, **_):
        assert timeout is not None, "her istekte timeout zorunlu"
        self.calls.append((url, params or {}))
        host = next(h for h in self.responses if h in url)
        item = self.responses[host].pop(0) if self.responses[host] else FakeResponse(500, {})
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def offline_session():
    err = requests.ConnectionError("ağ yok")
    return FakeSession({"wikipedia": [err] * 10, "duckduckgo": [err] * 10})


def network_enabled() -> bool:
    return os.environ.get("NLP_NETWORK_TESTS") == "1"
