"""
Dosya   : src/services/query_builder.py
Konu    : Arama Sorgusu Oluşturma
Açıklama: Sohbet temasından ve mesajdaki ayırt edici sözcüklerden internet araması için
          kısa bir sorgu üretir.
Yazar   : Ebrar Cemre Çetin
Tarih   : 26.09.2026
"""

from __future__ import annotations

import re

from src.preprocessing.text import f5_stem, normalize
from src.services.conversation import Theme

# Tema cümlesindeki bağlaç ve dolgu sözcükleri arama motoruna katkı vermez, çıkarılır.
FILLER_WORDS = frozenset({"alanında", "ve"})
MAX_QUERY_KEYWORDS = 2  # temaya eklenecek en fazla anahtar sözcük
LONG_CORE_WORDS = 3  # bu uzunluktaki tema ifadesi zaten belirgin; sözcük eklenmez
MAX_QUERY_CHARS = 100  # çok uzun sorgular arama motorlarında sonuç vermez
MIN_STEM_CHARS = 4  # ek atıldıktan sonra bundan kısa kalan sözcük eski haliyle bırakılır

# Çekimli fiil sonları; bu sözcükler sorguya eklenmez. Geçmiş zaman eki ünsüz uyumuna
# göre yalnızca sert ünsüzden (ç f h k p s ş t) sonra "-tı" olur; böylece "kuantum"
# gibi isimler fiil sanılmaz.
FINITE_VERB_RE = re.compile(
    r"(?:[ıiuü]yor(?:um|sun|uz|lar)?|(?:d|(?<=[çfhkpsşt])t)[ıiuü](?:m|n|k|nız|niz|lar|ler)?|"
    r"[mn][ıiuü]ş(?:[ıiuü]m|lar|ler)?|[ae]c[ae]k(?:[ıi]m|lar|ler)?)$"
)
# Sorgu için kırpılan yaygın hal ekleri: -la/-le/-yla (araç), -da/-de/-ta/-te (bulunma),
# -dan/-den/-tan/-ten (ayrılma). "penaltıyla" -> "penaltı", "derbide" -> "derbi"
CASE_SUFFIX_RE = re.compile(r"(?:y?l[ae]|[dt][ae]n|[dt][ae])$")


def query_form(token: str) -> str | None:
    """Anahtar sözcüğü sorgu biçimine getirir; çekimli fiilse None döner."""
    # Kısa isimler ("kedi", "vadi") fiil kalıbına yanlışlıkla uyar.
    if len(token) >= 5 and FINITE_VERB_RE.search(token):
        return None
    stripped = CASE_SUFFIX_RE.sub("", token)
    return stripped if len(stripped) >= MIN_STEM_CHARS else token


def build_query(theme: Theme, keywords: list[str] | None = None) -> str | None:
    """Sohbet temasından arama sorgusu kurar. Tema belirsizse ``None`` döner (arama yapılmaz).

    Sorgu = tema ifadesi + (tema kısaysa) son mesajdaki en ayırt edici 1-2 sözcük.
    Örnek: tema "kuantum bilgisayarlar", mesajda "kübit" -> "kuantum bilgisayarlar kübit".
    Anahtar sözcükleri `TopicModel.keywords` modelin öğrendiği ağırlıklara göre seçer.
    """
    if theme.uncertain or not theme.topics:
        return None
    # Alt konuya odaklanmış sohbette alt konu adı, değilse tema cümlesi çekirdek olur.
    core = theme.focus_subtopic or theme.phrase
    words = [w for w in normalize(core).split() if w not in FILLER_WORDS]
    stems = {f5_stem(w) for w in words}
    budget = 0 if len(words) >= LONG_CORE_WORDS else MAX_QUERY_KEYWORDS
    for keyword in keywords or []:
        if budget == 0:
            break
        kw = query_form(normalize(keyword))
        # Temada zaten geçen (aynı köklü) sözcükler tekrar eklenmez.
        if not kw or f5_stem(kw) in stems:
            continue
        words.append(kw)
        stems.add(f5_stem(kw))
        budget -= 1
    query = " ".join(words)
    if len(query) > MAX_QUERY_CHARS:
        # Sözcük ortasından kesmemek için son boşluğa kadar kırpılır.
        query = query[:MAX_QUERY_CHARS].rsplit(" ", 1)[0]
    return query or None
