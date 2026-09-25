"""
Dosya   : src/services/conversation.py
Konu    : Sohbet Konusu Takibi
Açıklama: Her mesajın konu olasılıklarını azalan ağırlıkla (eski * decay + yeni) biriktirerek
          sohbetin genel konusunu bulur ve "bilimsel kitaplar", "biyoloji hakkında
          bilimsel kitaplar" gibi doğal bir tema ifadesi üretir.
Yazar   : Ebrar Cemre Çetin
Tarih   : 26.09.2026
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.config import DECAY, OTHER_LABEL, TOPIC_MIN_SHARE, UNCERTAIN_LABEL

MAX_THEME_TOPICS = 3  # sohbet konusu en fazla 3 genel konunun birleşimi olabilir
# Sohbet tek konudaysa ve bir alt konu o konunun en az yarısını oluşturuyorsa tema alt konuya
# daraltılır ("Teknoloji" yerine "kuantum bilgisayarlar").
SUBTOPIC_FOCUS_SHARE = 0.5


@dataclass(frozen=True)
class TopicForm:
    """Bir genel konunun tema cümlesinde kullanılan Türkçe biçimleri ve rolü.

    Roller:
      medium    : içeriğin türü (Kitaplar). Cümlenin başı olur: "... kitaplar"
      qualifier : niteleyici (Bilim). Sıfat olarak başa gelir: "bilimsel kitaplar"
      domain    : alan konusu (Biyoloji, Fizik ...). "biyoloji hakkında ...", "tarih kitapları"
    Ödevdeki "bilimsel kitaplar" ve "biyoloji hakkında bilimsel kitaplar" örnekleri bu rollerle
    kurulur; aynı kurallar diğer konu birleşimleri için de geçerlidir.
    """

    role: str        # "medium" | "qualifier" | "domain"
    noun: str        # yalın ad ("biyoloji", "kitaplar")
    compound: str    # ortamla birleşik ad kurarken kullanılan biçim ("tarih")
    adjective: str   # sıfat biçimi ("bilimsel")


TOPIC_FORMS: dict[str, TopicForm] = {
    "Kitaplar": TopicForm("medium", "kitaplar", "kitap", "edebi"),
    "Bilim": TopicForm("qualifier", "bilim", "bilim", "bilimsel"),
    "Fizik": TopicForm("domain", "fizik", "fizik", "fiziksel"),
    "Kimya": TopicForm("domain", "kimya", "kimya", "kimyasal"),
    "Biyoloji": TopicForm("domain", "biyoloji", "biyoloji", "biyolojik"),
    "Teknoloji": TopicForm("domain", "teknoloji", "teknoloji", "teknolojik"),
    "Spor": TopicForm("domain", "spor", "spor", "sportif"),
    "Tarih": TopicForm("domain", "tarih", "tarih", "tarihi"),
}
MEDIUM_PLURAL_HEAD = {"Kitaplar": "kitapları"}  # birleşik ad başı: "tarih kitapları"


@dataclass
class Theme:
    """Sohbetin o anki genel konusu."""

    topics: list[tuple[str, float]]          # (genel konu, pay) — azalan sırada
    phrase: str                              # insan okunur tema ("bilimsel kitaplar")
    focus_subtopic: str | None = None
    uncertain: bool = False

    @property
    def label(self) -> str:
        """Kısa gösterim: "Bilim + Biyoloji + Kitaplar" veya "Teknoloji > Kuantum Bilgisayarlar"."""
        if self.uncertain or not self.topics:
            return UNCERTAIN_LABEL
        names = " + ".join(t for t, _ in self.topics)
        return f"{names} > {self.focus_subtopic}" if self.focus_subtopic else names

    def to_dict(self) -> dict:
        return {"topics": [{"topic": t, "share": round(s, 4)} for t, s in self.topics],
                "phrase": self.phrase, "focus_subtopic": self.focus_subtopic,
                "uncertain": self.uncertain, "label": self.label}


def compose_phrase(topics: list[str]) -> str:
    """Konu listesini (önem sırasıyla) doğal bir Türkçe tema ifadesine çevirir."""
    if not topics:
        return ""
    # Konular rollerine göre üç gruba ayrılır.
    forms = {t: TOPIC_FORMS[t] for t in topics if t in TOPIC_FORMS}
    medium = next((t for t in topics if forms.get(t) and forms[t].role == "medium"), None)
    qualifiers = [t for t in topics if forms.get(t) and forms[t].role == "qualifier"]
    domains = [t for t in topics if forms.get(t) and forms[t].role == "domain"]

    if medium:
        if qualifiers:
            # Kitaplar + Bilim -> "bilimsel kitaplar"
            head = " ".join([forms[q].adjective for q in qualifiers] + [forms[medium].noun])
            if domains:
                # + Biyoloji -> "biyoloji hakkında bilimsel kitaplar"
                return f"{' ve '.join(forms[d].noun for d in domains)} hakkında {head}"
            return head
        if domains:
            # Kitaplar + Tarih -> "tarih kitapları" (Türkçe belirtisiz isim tamlaması)
            return f"{' ve '.join(forms[d].compound for d in domains)} " \
                   f"{MEDIUM_PLURAL_HEAD.get(medium, forms[medium].noun)}"
        return forms[medium].noun
    if qualifiers and domains:
        # Bilim + Biyoloji (kitap yok) -> "biyoloji alanında bilimsel araştırmalar"
        return f"{' ve '.join(forms[d].noun for d in domains)} alanında " \
               f"{forms[qualifiers[0]].adjective} araştırmalar"
    # Yalnızca alan konuları: "fizik ve teknoloji"
    return " ve ".join(forms[t].noun for t in topics if t in forms)


@dataclass
class ConversationTracker:
    """Sohbet boyunca her genel konunun birikmiş skorunu tutar.

    Her mesajda: yeni_skor = eski_skor · decay + mesajdaki olasılık. decay 0,7 olduğundan
    son mesaj en etkili, bir önceki 0,7, ondan önceki 0,49 ağırlıkla katkı verir. Böylece
    sohbet konusu yeni mesajlarla yavaşça değişir ama önceki konular hemen kaybolmaz.
    Skorlar kesin karar değil olasılıklardır: model bir mesajda "Belirsiz" dese bile o
    mesajın olasılıkları sohbet konusuna katkı verir.
    """

    decay: float = DECAY
    min_share: float = TOPIC_MIN_SHARE
    scores: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    sub_scores: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    other_score: float = 0.0
    turns: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay [0, 1) aralığında olmalı")

    def reset(self) -> None:
        self.scores = defaultdict(float)
        self.sub_scores = defaultdict(lambda: defaultdict(float))
        self.other_score = 0.0
        self.turns = 0

    def update(self, general_scores: dict[str, float],
               joint_subtopic_scores: dict[str, dict[str, float]] | None = None) -> None:
        """Bir mesajın olasılıklarıyla skorları günceller (boş mesajlar çağrılmamalı)."""
        # Önce eski skorlar azaltılır ...
        for topic in list(self.scores):
            self.scores[topic] *= self.decay
        for subs in self.sub_scores.values():
            for sub in subs:
                subs[sub] *= self.decay
        self.other_score *= self.decay
        # ... sonra yeni mesajın olasılıkları eklenir. "Diğer" ayrı tutulur; konu değil,
        # "taksonomi dışı konuşuluyor" bilgisidir.
        for topic, p in general_scores.items():
            if topic == OTHER_LABEL:
                self.other_score += p
            else:
                self.scores[topic] += p
        for topic, subs in (joint_subtopic_scores or {}).items():
            for sub, p in subs.items():
                self.sub_scores[topic][sub] += p
        self.turns += 1

    def theme(self) -> Theme:
        total = sum(self.scores.values())
        # Sohbet ağırlıklı olarak taksonomi dışıysa konu iddia edilmez (arama da yapılmaz).
        if self.turns == 0 or total <= 0 or self.other_score > total:
            return Theme(topics=[], phrase="", uncertain=True)
        shares = sorted(((t, s / total) for t, s in self.scores.items()),
                        key=lambda kv: kv[1], reverse=True)
        # Toplam içindeki payı eşiği (%15) aşan konular sohbet konusuna katılır.
        chosen = [(t, s) for t, s in shares if s >= self.min_share][:MAX_THEME_TOPICS]
        if not chosen:
            chosen = shares[:1]
        focus = None
        if len(chosen) == 1:
            subs = self.sub_scores.get(chosen[0][0], {})
            sub_total = sum(subs.values())
            if sub_total > 0:
                best_sub, best = max(subs.items(), key=lambda kv: kv[1])
                if best / sub_total >= SUBTOPIC_FOCUS_SHARE:
                    focus = best_sub
        phrase = focus.lower() if focus else compose_phrase([t for t, _ in chosen])
        return Theme(topics=chosen, phrase=phrase, focus_subtopic=focus)

    def snapshot(self) -> dict:
        """Hata ayıklama logu için anlık skorlar."""
        return {"scores": {k: round(v, 4) for k, v in self.scores.items()},
                "other_score": round(self.other_score, 4), "turns": self.turns}
