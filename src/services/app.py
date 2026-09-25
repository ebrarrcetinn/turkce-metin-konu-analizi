"""
Dosya   : src/services/app.py
Konu    : Uygulama Servisi
Açıklama: Bir mesaj için sınıflandırma, sohbet takibi, sorgu üretimi, internet araması ve
          veritabanı kaydı adımlarını sırayla çalıştırır. Web veya veritabanı hataları
          sınıflandırmayı durdurmaz.
Yazar   : Ebrar Cemre Çetin
Tarih   : 27.09.2026
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.config import WEB_OFFLINE_COOLDOWN_SECONDS, WEB_SEARCH_ENABLED
from src.database.db import Database, DatabaseError
from src.models.topic_model import STATUS_EMPTY, Prediction, TopicModel
from src.preprocessing.text import normalize
from src.services.conversation import ConversationTracker, Theme
from src.services.query_builder import build_query
from src.services.web_search import (
    STATUS_ERROR,
    STATUS_OFFLINE,
    STATUS_OK,
    SearchOutcome,
    SearchResult,
    WebSearchClient,
)

logger = logging.getLogger(__name__)


@dataclass
class HistoryItem:
    """`geçmiş` komutunda gösterilen bir mesaj."""

    index: int
    text: str
    prediction: Prediction
    theme_label: str


@dataclass
class TurnResult:
    """Bir mesajın tüm işlenme sonucu; konsol bunu ekrana yazar."""

    prediction: Prediction
    theme: Theme | None = None
    query: str | None = None
    search: SearchOutcome | None = None
    saved_to_db: bool = False
    db_error: str | None = None
    warnings: list[str] = field(default_factory=list)


def new_session_id() -> str:
    return uuid.uuid4().hex  # rastgele, çakışmayan oturum kimliği


class ChatService:
    """Konsoldan bağımsız uygulama mantığı.

    Veritabanı ve arama istemcisi dışarıdan verilir (dependency injection): testlerde sahte
    istemci veya geçici veritabanı kullanılabilir, `db=None` ile kayıtsız da çalışır.
    """

    def __init__(self, classifier: TopicModel, db: Database | None,
                 search_client: WebSearchClient | None = None,
                 web_enabled: bool = WEB_SEARCH_ENABLED,
                 tracker: ConversationTracker | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.classifier = classifier
        self.db = db
        self.search_client = search_client or (WebSearchClient() if web_enabled else None)
        self.web_enabled = web_enabled and self.search_client is not None
        self.tracker = tracker or ConversationTracker()
        self.history: list[HistoryItem] = []
        self.session_id = new_session_id()
        self._clock = clock  # testlerde zamanı ileri almak için değiştirilebilir
        self._offline_until = 0.0  # bu zamana kadar internet denenmez
        if self.db:
            self._db_call(self.db.register_model, classifier.metadata)
        self._start_session()

    def _db_call(self, fn: Callable[..., Any], *args: Any) -> tuple[Any, str | None]:
        """DB işlemini çalıştırır; hata olursa loglar ve (None, hata) döndürür.

        Veritabanı hatası (disk dolu, dosya kilitli ...) kullanıcıya uyarı olarak gösterilir;
        sınıflandırma ve sohbet takibi çalışmaya devam eder.
        """
        try:
            return fn(*args), None
        except DatabaseError as exc:
            logger.error("Veritabanı hatası: %s", exc)
            return None, str(exc)

    def _start_session(self) -> None:
        if self.db:
            self._db_call(self.db.start_session, self.session_id, self.classifier.version)
        logger.info("Oturum başladı: %s", self.session_id)

    def reset(self) -> str:
        """Yeni sohbet bağlamı başlatır; eski kayıtlar DB'de kalır."""
        if self.db:
            self._db_call(self.db.end_session, self.session_id)
        self.tracker.reset()
        self.history = []
        self.session_id = new_session_id()
        self._start_session()
        return self.session_id

    def close(self) -> None:
        if self.db:
            self._db_call(self.db.end_session, self.session_id)

    def process(self, text: str) -> TurnResult:
        """Bir kullanıcı mesajını baştan sona işler."""
        # 1) Metnin genel konusu ve alt konuları
        prediction = self.classifier.predict(text)
        result = TurnResult(prediction=prediction)
        if prediction.status == STATUS_EMPTY:
            result.warnings.append("Anlamlı içerik bulunamadı; mesaj sınıflandırılmadı.")
            return result
        if prediction.truncated:
            result.warnings.append("Metin çok uzun olduğu için kırpılarak analiz edildi.")

        # 2) Sohbetin genel konusu: bu mesajın olasılıkları önceki mesajlarla birleştirilir.
        self.tracker.update(prediction.general_scores, prediction.joint_subtopic_scores)
        theme = self.tracker.theme()
        logger.debug("Sohbet skorları: %s | tema: %s", self.tracker.snapshot(), theme.label)
        result.theme = theme

        # 3) Arama sorgusu: sohbet konusu + (mesajın konusu sohbet konusuna dahilse)
        #    mesajdaki ayırt edici sözcükler. Konu dışı bir mesajın sözcükleri eklenmez.
        theme_topics = {t for t, _ in theme.topics}
        keywords = (self.classifier.keywords(text, prediction.top_guess)
                    if prediction.top_guess in theme_topics else [])
        result.query = build_query(theme, keywords)

        index = len(self.history) + 1
        self.history.append(HistoryItem(index, text, prediction, theme.label))

        # 4) Metin, sonuç ve sohbet konusu veritabanına yazılır.
        theme_id = self._save_turn(self.db, result, theme, index) if self.db else None

        # 5) İnternet araması ve sonuçların kaydı
        if self.web_enabled and result.query:
            result.search = self._safe_search(result.query)
            if result.search.status == STATUS_OK and self.db and theme_id is not None:
                _, err = self._db_call(self.db.save_search_results, self.session_id, theme_id,
                                       result.query,
                                       [r.to_dict() for r in result.search.results],
                                       result.search.from_cache)
                if err:
                    result.db_error = err
                    result.saved_to_db = False
        return result

    def _save_turn(self, db: Database, result: TurnResult, theme: Theme,
                   index: int) -> int | None:
        """metinler ve sohbet_konulari tablolarına yazar; sohbet konusu satırının id'sini döner."""
        theme_id = None
        metin_id, err = self._db_call(db.save_text, self.session_id, index,
                                      result.prediction.to_dict(), self.classifier.version)
        if metin_id is not None:
            # Sohbet konusu, onu oluşturan mesaja metin_id ile bağlanır.
            theme_id, err = self._db_call(db.save_theme, self.session_id, metin_id,
                                          theme.to_dict(), result.query)
        result.saved_to_db = err is None and theme_id is not None
        result.db_error = err
        return theme_id

    def _safe_search(self, query: str) -> SearchOutcome:
        try:
            return self._search(query)
        except Exception as exc:  # web katmanı hiçbir koşulda akışı kırmamalı
            logger.exception("Beklenmeyen arama hatası")
            return SearchOutcome(query=query, status=STATUS_ERROR, error=str(exc))

    def _search(self, query: str) -> SearchOutcome:
        """Önce önbellek, sonra (bağlantı varsa) internet."""
        key = normalize(query)  # "Kuantum Bilgisayarlar" ile "kuantum bilgisayarlar" aynı kayıt
        if self.db:
            cached, _ = self._db_call(self.db.cache_get, key)
            if cached:
                results = [SearchResult(**r) for r in cached["results"]]
                return SearchOutcome(query=query, status=STATUS_OK, results=results,
                                     source=cached["source"], from_cache=True)
        # Son denemede internet yoksa bekleme süresi dolana kadar tekrar denenmez; her mesajda
        # zaman aşımını beklemek konsolu yavaşlatırdı.
        if self._clock() < self._offline_until:
            return SearchOutcome(query=query, status=STATUS_OFFLINE,
                                 error="bağlantı yok (bekleme süresi dolmadı)")
        if self.search_client is None:
            return SearchOutcome(query=query, status=STATUS_ERROR, error="arama istemcisi yok")
        outcome = self.search_client.search(query)
        if outcome.status == STATUS_OFFLINE:
            self._offline_until = self._clock() + WEB_OFFLINE_COOLDOWN_SECONDS
            logger.warning("İnternete erişilemiyor; web araması %d sn askıya alındı",
                           WEB_OFFLINE_COOLDOWN_SECONDS)
        if outcome.status == STATUS_OK and self.db:
            self._db_call(self.db.cache_put, key, outcome.source or "",
                          [r.to_dict() for r in outcome.results])
        return outcome
