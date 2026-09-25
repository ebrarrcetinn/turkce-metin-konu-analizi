"""
Dosya   : src/services/app.py
Konu    : Uygulama Servisi
Açıklama: Bu dosyada amacım bir mesaj için sınıflandırma, sohbet takibi, sorgu üretimi,
          internet araması ve veritabanı kaydı adımlarını sırayla çalıştırmak. Web veya
          veritabanı hatalarının sınıflandırmayı durdurmasına izin vermiyorum.
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
    """`geçmiş` komutunda gösterdiğim bir mesaj."""

    index: int
    text: str
    prediction: Prediction
    theme_label: str


@dataclass
class TurnResult:
    """Bir mesajın tüm işlenme sonucu; konsolda bunu ekrana yazıyorum."""

    prediction: Prediction
    theme: Theme | None = None
    query: str | None = None
    search: SearchOutcome | None = None
    saved_to_db: bool = False
    db_error: str | None = None
    warnings: list[str] = field(default_factory=list)


def new_session_id() -> str:
    return uuid.uuid4().hex  # rastgele, çakışmayan bir oturum kimliği üretiyorum


class ChatService:
    """Burada amacım konsoldan bağımsız uygulama mantığını bir arada tutmak.

    Veritabanını ve arama istemcisini dışarıdan veriyorum (dependency injection): testlerde
    sahte istemci veya geçici veritabanı kullanabiliyorum, `db=None` ile kayıtsız da
    çalışıyor.
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
        self._clock = clock  # testlerde zamanı ileri alabilmek için bunu değiştiriyorum
        self._offline_until = 0.0  # bu zamana kadar interneti denemiyorum
        if self.db:
            self._db_call(self.db.register_model, classifier.metadata)
        self._start_session()

    def _db_call(self, fn: Callable[..., Any], *args: Any) -> tuple[Any, str | None]:
        """DB işlemini çalıştırıyorum; hata olursa logluyor ve (None, hata) döndürüyorum.

        Veritabanı hatasını (disk dolu, dosya kilitli ...) kullanıcıya uyarı olarak
        gösteriyorum; sınıflandırma ve sohbet takibi çalışmaya devam ediyor.
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
        """Yeni sohbet bağlamı başlatıyorum; eski kayıtlar DB'de kalıyor."""
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
        """Amacım bir kullanıcı mesajını baştan sona işlemek."""
        # 1) Önce metnin genel konusunu ve alt konularını buluyorum
        prediction = self.classifier.predict(text)
        result = TurnResult(prediction=prediction)
        if prediction.status == STATUS_EMPTY:
            result.warnings.append("Anlamlı içerik bulunamadı; mesaj sınıflandırılmadı.")
            return result
        if prediction.truncated:
            result.warnings.append("Metin çok uzun olduğu için kırpılarak analiz edildi.")

        # 2) Sohbetin genel konusu: bu mesajın olasılıklarını önceki mesajlarla birleştiriyorum.
        self.tracker.update(prediction.general_scores, prediction.joint_subtopic_scores)
        theme = self.tracker.theme()
        logger.debug("Sohbet skorları: %s | tema: %s", self.tracker.snapshot(), theme.label)
        result.theme = theme

        # 3) Arama sorgusu: sohbet konusu + (mesajın konusu sohbet konusuna dahilse)
        #    mesajdaki ayırt edici sözcükler. Konu dışı bir mesajın sözcüklerini eklemiyorum.
        theme_topics = {t for t, _ in theme.topics}
        keywords = (self.classifier.keywords(text, prediction.top_guess)
                    if prediction.top_guess in theme_topics else [])
        result.query = build_query(theme, keywords)

        index = len(self.history) + 1
        self.history.append(HistoryItem(index, text, prediction, theme.label))

        # 4) Metni, sonucu ve sohbet konusunu veritabanına yazıyorum.
        theme_id = self._save_turn(self.db, result, theme, index) if self.db else None

        # 5) İnternet araması yapıp sonuçları kaydediyorum
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
        """metinler ve sohbet_konulari tablolarına yazıp sohbet konusu satırının id'sini dönerim."""
        theme_id = None
        metin_id, err = self._db_call(db.save_text, self.session_id, index,
                                      result.prediction.to_dict(), self.classifier.version)
        if metin_id is not None:
            # Sohbet konusunu, onu oluşturan mesaja metin_id ile bağlıyorum.
            theme_id, err = self._db_call(db.save_theme, self.session_id, metin_id,
                                          theme.to_dict(), result.query)
        result.saved_to_db = err is None and theme_id is not None
        result.db_error = err
        return theme_id

    def _safe_search(self, query: str) -> SearchOutcome:
        try:
            return self._search(query)
        except Exception as exc:  # web katmanının hiçbir koşulda akışı kırmasını istemiyorum
            logger.exception("Beklenmeyen arama hatası")
            return SearchOutcome(query=query, status=STATUS_ERROR, error=str(exc))

    def _search(self, query: str) -> SearchOutcome:
        """Önce önbelleğe, sonra (bağlantı varsa) internete bakıyorum."""
        key = normalize(query)  # "Kuantum Bilgisayarlar" ile "kuantum bilgisayarlar" aynı kayıt
        if self.db:
            cached, _ = self._db_call(self.db.cache_get, key)
            if cached:
                results = [SearchResult(**r) for r in cached["results"]]
                return SearchOutcome(query=query, status=STATUS_OK, results=results,
                                     source=cached["source"], from_cache=True)
        # Son denemede internet yoksa bekleme süresi dolana kadar tekrar denemiyorum; her mesajda
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
