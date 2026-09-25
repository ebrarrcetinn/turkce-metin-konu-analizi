"""
Dosya   : src/services/web_search.py
Konu    : İnternet Araması
Açıklama: Bu dosyada amacım Türkçe Wikipedia (birincil) ve DuckDuckGo (yedek) üzerinden
          arama yapmak; ağ ve API hatalarını yakalayıp sonuç durumuna çeviriyorum.
Yazar   : Ebrar Cemre Çetin
Tarih   : 26.09.2026
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

import requests

from src.config import (
    ALLOWED_RESULT_HOSTS,
    DUCKDUCKGO_API_URL,
    MAX_RESULTS,
    USER_AGENT,
    WEB_BACKOFF_SECONDS,
    WEB_RETRIES,
    WEB_TIMEOUT,
    WIKIPEDIA_API_URL,
)

logger = logging.getLogger(__name__)

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
MAX_SUMMARY_CHARS = 400  # konsolda ve veritabanında tuttuğum özet uzunluğu
MAX_TITLE_CHARS = 150
# Geçici hatalar: 429 (çok fazla istek), 5xx (sunucu hatası). Bunlarda tekrar deniyorum;
# 404 gibi kalıcı hatalarda denemiyorum.
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER_SECONDS = 5.0  # sunucu daha uzun beklememi istese de kullanıcıyı bekletmiyorum

# Arama sonucu durumları
STATUS_OK = "ok"  # sonuç bulundu
STATUS_EMPTY = "empty"  # aramayı yaptım ama sonuç yok
STATUS_OFFLINE = "offline"  # internete erişemedim
STATUS_ERROR = "error"  # diğer hatalar (bozuk yanıt, API hatası ...)


class SearchError(RuntimeError):
    """Kurtarılamayan arama hatası (bağlantı dışı)."""


class OfflineError(SearchError):
    """Ağ erişimi yok / bağlantı kurulamadı."""


@dataclass(frozen=True)
class SearchResult:
    """Tek bir arama sonucu; arama_sonuclari tablosuna bu alanlarla yazıyorum."""

    title: str
    summary: str
    url: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class SearchOutcome:
    """Bir aramanın bütün sonucu. Hata olsa bile istisna değil bu nesneyi döndürüyorum; böylece
    arama katmanındaki bir sorun sınıflandırma ve kayıt akışını durdurmuyor."""

    query: str
    status: str
    results: list[SearchResult] = field(default_factory=list)
    source: str | None = None
    error: str | None = None
    from_cache: bool = False
    elapsed_ms: float = 0.0


class HttpSession(Protocol):
    """requests.Session ile aynı `get` imzası; testlerde sahte oturum verebileyim diye ekledim."""

    def get(self, url: str, **kwargs: Any) -> requests.Response: ...


def clean_text(value: Any, limit: int) -> str:
    """Güvenilmeyen HTML/metni düz metne çevirip kırpıyorum."""
    if not isinstance(value, str):
        return ""
    # Wikipedia özetlerinde <b>, &amp; gibi HTML parçaları gelebilir; etiketleri atıyorum,
    # HTML karakter kodlarını normal karaktere çeviriyorum, fazla boşlukları da tek
    # boşluğa indiriyorum.
    text = SPACE_RE.sub(" ", html.unescape(TAG_RE.sub(" ", value))).strip()
    if len(text) > limit:
        text = text[: limit - 1].rsplit(" ", 1)[0] + "…"
    return text


def is_allowed_url(url: Any) -> bool:
    """Bağlantıyı yalnızca https ve izin verilen alan adlarından (wikipedia.org,
    duckduckgo.com ve alt alan adları) geliyorsa kabul ediyorum. Böylece yanıttaki
    beklenmedik bir bağlantının konsola yazılıp veritabanına kaydedilmesini önlüyorum."""
    if not isinstance(url, str) or len(url) > 2048:
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(
        host == allowed or host.endswith("." + allowed) for allowed in ALLOWED_RESULT_HOSTS)


def parse_wikipedia(payload: Any, limit: int = MAX_RESULTS) -> list[SearchResult]:
    """Wikipedia API yanıtını sonuç listesine çeviriyorum.

    İsteği, arama ile sayfa özetini tek çağrıda alacak şekilde kurdum: generator=search
    sorguya uyan sayfaları bulur, prop=extracts her sayfanın ilk cümlelerini, prop=info
    sayfa adresini ekler.
    """
    if not isinstance(payload, dict):
        raise SearchError("Wikipedia yanıtı nesne değil")
    if "error" in payload:
        err = payload["error"]
        code = err.get("code") if isinstance(err, dict) else err
        raise SearchError(f"Wikipedia API hatası: {code}")
    pages = (payload.get("query") or {}).get("pages") or []
    if isinstance(pages, dict):  # formatversion=1 ile uyumluluk için ekledim
        pages = list(pages.values())
    if not isinstance(pages, list):
        raise SearchError("Wikipedia yanıtında 'pages' beklenen biçimde değil")
    results = []
    # "index" arama sıralamasıdır (1 en alakalı). API sayfaları bu sırayla döndürmeyebildiği
    # için ben sıralıyorum.
    for page in sorted((p for p in pages if isinstance(p, dict)),
                       key=lambda p: p.get("index", 1_000)):
        title = clean_text(page.get("title"), MAX_TITLE_CHARS)
        url = page.get("fullurl") or page.get("canonicalurl")
        if not title or page.get("missing") or not is_allowed_url(url):
            continue
        results.append(SearchResult(title=title,
                                    summary=clean_text(page.get("extract"), MAX_SUMMARY_CHARS),
                                    url=str(url), source="wikipedia"))
        if len(results) >= limit:
            break
    return results


def parse_duckduckgo(payload: Any, limit: int = MAX_RESULTS) -> list[SearchResult]:
    """DuckDuckGo Instant Answer API yanıtını sonuç listesine çeviriyorum.

    Bu API tam bir arama sonucu listesi değil; konunun özeti (Abstract) ve ilgili konular
    (RelatedTopics) döner. Wikipedia sonuç vermezse bunu yedek olarak kullanıyorum.
    """
    if not isinstance(payload, dict):
        raise SearchError("DuckDuckGo yanıtı nesne değil")
    results: list[SearchResult] = []
    abstract_url = payload.get("AbstractURL")
    if payload.get("AbstractText") and is_allowed_url(abstract_url):
        results.append(SearchResult(
            title=clean_text(payload.get("Heading") or abstract_url, MAX_TITLE_CHARS),
            summary=clean_text(payload.get("AbstractText"), MAX_SUMMARY_CHARS),
            url=str(abstract_url), source="duckduckgo"))

    # İlgili konular iç içe gruplar halinde gelebilir; hepsini düz listeye açıyorum.
    def walk(topics: Any):
        for item in topics if isinstance(topics, list) else []:
            if isinstance(item, dict) and "Topics" in item:
                yield from walk(item["Topics"])
            elif isinstance(item, dict):
                yield item

    for item in walk(payload.get("RelatedTopics")):
        if len(results) >= limit:
            break
        text, url = item.get("Text"), item.get("FirstURL")
        if text and is_allowed_url(url):
            title = clean_text(text.split(" - ")[0], MAX_TITLE_CHARS)
            results.append(SearchResult(title=title, summary=clean_text(text, MAX_SUMMARY_CHARS),
                                        url=str(url), source="duckduckgo"))
    return results[:limit]


class WebSearchClient:
    """Amacım Wikipedia -> DuckDuckGo sırasıyla aramak ve ağ hatalarını SearchOutcome'a çevirmek.

    İki servis de API anahtarı istemiyor; bu yüzden projede gizli bilgi (.env) yok.
    """

    def __init__(self, session: HttpSession | None = None, timeout: float = WEB_TIMEOUT,
                 retries: int = WEB_RETRIES, backoff: float = WEB_BACKOFF_SECONDS,
                 sleep: Callable[[float], None] = time.sleep,
                 max_results: int = MAX_RESULTS):
        if session is None:
            default = requests.Session()
            default.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
            session = default  # type: ignore[assignment]
        self.session: Any = session
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep
        self.max_results = max_results

    def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        """GET isteği atıyorum, geçici hatalarda bekleyip tekrar deniyorum, JSON döndürüyorum."""
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = exc
                logger.info("Ağ hatası (deneme %d): %s", attempt + 1, exc)
            except requests.RequestException as exc:  # yönlendirme döngüsü, geçersiz URL vb.
                raise SearchError(f"İstek hatası: {exc.__class__.__name__}") from exc
            else:
                if resp.status_code in RETRY_STATUS:
                    last = SearchError(f"HTTP {resp.status_code}")
                    # Sunucu "Retry-After" başlığıyla ne kadar beklemem gerektiğini söyleyebilir.
                    retry_after = resp.headers.get("Retry-After", "")
                    wait = float(retry_after) if retry_after.isdigit() else None
                    logger.info("HTTP %s (deneme %d)", resp.status_code, attempt + 1)
                    if attempt < self.retries:
                        self.sleep(min(wait or self.backoff * 2 ** attempt,
                                       MAX_RETRY_AFTER_SECONDS))
                    continue
                if resp.status_code != 200:
                    raise SearchError(f"HTTP {resp.status_code}")
                resp.encoding = "utf-8"  # Türkçe karakterleri doğru çözmek için
                try:
                    return json.loads(resp.text)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise SearchError(f"Bozuk JSON: {exc}") from exc
            if attempt < self.retries:
                # Üstel bekleme: 0,8 sn, 1,6 sn ... sunucuya yük bindirmeden tekrar deniyorum.
                self.sleep(self.backoff * 2 ** attempt)
        # Tüm denemeler bağlantı hatasıyla bittiyse bunu "internet yok" durumu sayıyorum.
        if isinstance(last, requests.ConnectionError | requests.Timeout):
            raise OfflineError(str(last))
        raise SearchError(str(last))

    def search_wikipedia(self, query: str) -> list[SearchResult]:
        # Türkçe Wikipedia; ilk 3 sayfanın giriş bölümünden 2 cümleyi düz metin olarak istiyorum.
        payload = self._get_json(WIKIPEDIA_API_URL, {
            "action": "query", "format": "json", "formatversion": "2", "utf8": "1",
            "generator": "search", "gsrsearch": query, "gsrlimit": str(self.max_results),
            "gsrnamespace": "0", "prop": "extracts|info", "exintro": "1", "explaintext": "1",
            "exsentences": "2", "exlimit": str(self.max_results), "inprop": "url",
        })
        return parse_wikipedia(payload, self.max_results)

    def search_duckduckgo(self, query: str) -> list[SearchResult]:
        payload = self._get_json(DUCKDUCKGO_API_URL, {
            "q": query, "format": "json", "no_html": "1", "skip_disambig": "1", "kl": "tr-tr",
        })
        return parse_duckduckgo(payload, self.max_results)

    def search(self, query: str) -> SearchOutcome:
        """Önce Wikipedia'yı, sonuç yoksa DuckDuckGo'yu deniyorum; ilk sonuç vereni kullanıyorum."""
        start = time.perf_counter()
        query = (query or "").strip()
        if not query:
            return SearchOutcome(query=query, status=STATUS_EMPTY, error="boş sorgu")
        errors: list[str] = []
        offline = 0
        providers = (("wikipedia", self.search_wikipedia), ("duckduckgo", self.search_duckduckgo))
        for name, provider in providers:
            try:
                results = provider(query)
            except OfflineError as exc:
                offline += 1
                errors.append(f"{name}: bağlantı yok ({exc.__class__.__name__})")
                continue
            except SearchError as exc:
                errors.append(f"{name}: {exc}")
                continue
            if results:
                return SearchOutcome(query=query, status=STATUS_OK, results=results,
                                     source=name,
                                     elapsed_ms=round((time.perf_counter() - start) * 1000, 1))
            errors.append(f"{name}: sonuç yok")
        # İki servis de bağlantı hatası verdiyse durumu "çevrimdışı" sayıyorum; uygulama bir süre
        # aramayı dener.
        status = (STATUS_OFFLINE if offline == len(providers)
                  else STATUS_EMPTY if all("sonuç yok" in e for e in errors) else STATUS_ERROR)
        return SearchOutcome(query=query, status=status, error="; ".join(errors),
                             elapsed_ms=round((time.perf_counter() - start) * 1000, 1))
