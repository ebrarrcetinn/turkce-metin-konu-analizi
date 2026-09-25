"""Sohbet temasından arama sorgusu üretir.

Sorgu, tema çekirdeğine son mesajdaki en ayırt edici sözcüklerin eklenmesiyle oluşur.
"""

from __future__ import annotations

import re

from src.preprocessing.text import f5_stem, normalize
from src.services.conversation import Theme

FILLER_WORDS = frozenset({"hakkında", "alanında", "ve"})
MAX_QUERY_KEYWORDS = 2
LONG_CORE_WORDS = 3  # uzun çekirdeğe yalnızca bir anahtar sözcük eklenir
MAX_QUERY_CHARS = 100
MIN_STEM_CHARS = 4

# Çekimli fiil sonları; bu sözcükler sorguya eklenmez.
FINITE_VERB_RE = re.compile(
    r"(?:[ıiuü]yor(?:um|sun|uz|lar)?|[dt][ıiuü](?:m|n|k|nız|niz|lar|ler)?|"
    r"[mn][ıiuü]ş(?:[ıiuü]m|lar|ler)?|[ae]c[ae]k(?:[ıi]m|lar|ler)?)$"
)
# Sorgu için kırpılan yaygın hal ekleri (araç, bulunma, ayrılma)
CASE_SUFFIX_RE = re.compile(r"(?:y?l[ae]|[dt][ae]n|[dt][ae])$")


def query_form(token: str) -> str | None:
    """Anahtar sözcüğü sorgu biçimine getirir; çekimli fiilse None döner."""
    # Kısa isimler ("kedi", "vadi") fiil kalıbına yanlışlıkla uyar.
    if len(token) >= 5 and FINITE_VERB_RE.search(token):
        return None
    stripped = CASE_SUFFIX_RE.sub("", token)
    return stripped if len(stripped) >= MIN_STEM_CHARS else token


def build_query(theme: Theme, keywords: list[str] | None = None) -> str | None:
    """Tema belirsizse ``None`` döner (arama yapılmaz)."""
    if theme.uncertain or not theme.topics:
        return None
    core = theme.focus_subtopic or theme.phrase
    words = [w for w in normalize(core).split() if w not in FILLER_WORDS]
    stems = {f5_stem(w) for w in words}
    budget = 1 if len(words) >= LONG_CORE_WORDS else MAX_QUERY_KEYWORDS
    added = 0
    for keyword in keywords or []:
        kw = query_form(normalize(keyword))
        if not kw or f5_stem(kw) in stems:
            continue
        words.append(kw)
        stems.add(f5_stem(kw))
        added += 1
        if added >= budget:
            break
    query = " ".join(words)
    if len(query) > MAX_QUERY_CHARS:
        query = query[:MAX_QUERY_CHARS].rsplit(" ", 1)[0]
    return query or None
