"""
Dosya   : tests/test_web_search.py
Konu    : İnternet Araması Testleri
Açıklama: Wikipedia/DuckDuckGo yanıt ayrıştırma, hata durumları, yeniden deneme ve URL
          filtresini sahte HTTP yanıtlarıyla test eder.
Yazar   : Ebrar Cemre Çetin
Tarih   : 27.09.2026
"""

import pytest
import requests

from src.services.web_search import (
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_OFFLINE,
    STATUS_OK,
    SearchError,
    WebSearchClient,
    clean_text,
    is_allowed_url,
    parse_duckduckgo,
    parse_wikipedia,
)
from tests.conftest import FakeResponse, FakeSession, load_fixture, network_enabled


def _client(session, **kw):
    slept = []
    client = WebSearchClient(session=session, sleep=slept.append, **kw)
    return client, slept


def test_parse_wikipedia_orders_cleans_and_filters():
    results = parse_wikipedia(load_fixture("wikipedia_search.json"), limit=5)
    assert [r.title for r in results] == ["Kuantum bilgisayar", "Kübit",
                                          "Shor algoritması & kübitler"]
    assert "<b>" not in results[1].summary and "Klasik" in results[1].summary
    assert all(r.url.startswith("https://tr.wikipedia.org/") for r in results)
    assert all(r.source == "wikipedia" for r in results)


def test_parse_wikipedia_limit_and_empty_and_error():
    assert len(parse_wikipedia(load_fixture("wikipedia_search.json"), limit=1)) == 1
    assert parse_wikipedia(load_fixture("wikipedia_empty.json")) == []
    with pytest.raises(SearchError, match="badvalue"):
        parse_wikipedia(load_fixture("wikipedia_error.json"))
    with pytest.raises(SearchError):
        parse_wikipedia(["liste"])


def test_parse_duckduckgo_nested_topics_and_https_only():
    results = parse_duckduckgo(load_fixture("duckduckgo.json"), limit=5)
    assert results[0].title == "Kuantum bilgisayar"
    assert [r.title for r in results[1:]] == ["Kübit", "Kuantum üstünlüğü"]
    assert all(r.url.startswith("https://") for r in results)


@pytest.mark.parametrize("url, ok", [
    ("https://tr.wikipedia.org/wiki/X", True),
    ("https://en.wikipedia.org/wiki/X", True),
    ("https://duckduckgo.com/X", True),
    ("http://tr.wikipedia.org/wiki/X", False),
    ("https://tr.wikipedia.org.evil.example/wiki/X", False),
    ("javascript:alert(1)", False),
    ("https://evilwikipedia.org/", False),
    (None, False),
])
def test_url_validation(url, ok):
    assert is_allowed_url(url) is ok


def test_clean_text_unescapes_strips_and_truncates():
    assert clean_text("<script>x</script> A &amp; B", 100) == "x A & B"
    assert len(clean_text("kelime " * 200, 50)) <= 50
    assert clean_text(None, 10) == ""


def test_client_success_uses_timeout_and_utf8():
    session = FakeSession({"wikipedia": [FakeResponse(200, load_fixture("wikipedia_search.json"))]})
    client, _ = _client(session)
    outcome = client.search("kuantum bilgisayar")
    assert outcome.status == STATUS_OK and outcome.source == "wikipedia"
    assert len(outcome.results) == 3
    assert session.calls[0][1]["gsrsearch"] == "kuantum bilgisayar"


def test_client_retries_on_503_then_succeeds():
    session = FakeSession({"wikipedia": [FakeResponse(503, {}), FakeResponse(
        200, load_fixture("wikipedia_search.json"))]})
    client, slept = _client(session)
    assert client.search("x").status == STATUS_OK
    assert len(slept) == 1


def test_client_honors_retry_after_with_cap():
    session = FakeSession({"wikipedia": [FakeResponse(429, {}, headers={"Retry-After": "60"}),
                                         FakeResponse(200, load_fixture("wikipedia_search.json"))]})
    client, slept = _client(session)
    client.search("x")
    assert slept == [5.0]


def test_client_malformed_json_falls_back_to_duckduckgo():
    session = FakeSession({
        "wikipedia": [FakeResponse(200, text="{bozuk json")],
        "duckduckgo": [FakeResponse(200, load_fixture("duckduckgo.json"))],
    })
    client, _ = _client(session)
    outcome = client.search("kuantum")
    assert outcome.status == STATUS_OK and outcome.source == "duckduckgo"


def test_client_offline(offline_session):
    client, slept = _client(offline_session, retries=1)
    outcome = client.search("kuantum")
    assert outcome.status == STATUS_OFFLINE
    assert "bağlantı yok" in outcome.error
    assert len(slept) == 2  # sağlayıcı başına bir bekleme


def test_client_timeout_is_offline():
    err = requests.Timeout("zaman aşımı")
    client, _ = _client(FakeSession({"wikipedia": [err] * 3, "duckduckgo": [err] * 3}))
    assert client.search("x").status == STATUS_OFFLINE


def test_client_http_404_is_error_not_crash():
    session = FakeSession({"wikipedia": [FakeResponse(404, {})],
                           "duckduckgo": [FakeResponse(404, {})]})
    client, _ = _client(session)
    outcome = client.search("x")
    assert outcome.status == STATUS_ERROR and "HTTP 404" in outcome.error


def test_client_no_results_is_empty():
    session = FakeSession({"wikipedia": [FakeResponse(200, load_fixture("wikipedia_empty.json"))],
                           "duckduckgo": [FakeResponse(200, {"RelatedTopics": []})]})
    client, _ = _client(session)
    assert client.search("x").status == STATUS_EMPTY


def test_client_other_request_errors_are_contained():
    err = requests.TooManyRedirects("döngü")
    client, _ = _client(FakeSession({"wikipedia": [err], "duckduckgo": [err]}))
    outcome = client.search("x")
    assert outcome.status == STATUS_ERROR and "TooManyRedirects" in outcome.error


def test_empty_query():
    client, _ = _client(FakeSession({}))
    assert client.search("   ").status == STATUS_EMPTY


@pytest.mark.network
@pytest.mark.skipif(not network_enabled(), reason="NLP_NETWORK_TESTS=1 değil")
def test_live_wikipedia():
    outcome = WebSearchClient().search("kuantum bilgisayar")
    assert outcome.status == STATUS_OK
    assert outcome.results and outcome.results[0].url.startswith("https://")
