"""
Dosya   : src/ml/splitting.py
Konu    : Gruplu ve Tabakalı Veri Bölme
Açıklama: Veriyi, aynı terime ait örnekler farklı bölmelere düşmeyecek ve konu dağılımı
          korunacak şekilde eğitim/doğrulama/test olarak böler.
Yazar   : Ebrar Cemre Çetin
Tarih   : 24.09.2026
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence


def grouped_stratified_split(labels: Sequence[str], groups: Sequence[str], holdout: float,
                             seed: int) -> tuple[list[int], list[int]]:
    """Örnekleri (kalan, ayrılan) indekslerine böler.

    Aynı gruptaki örnekler hep aynı tarafa düşer (sızıntıyı önler). Her grup en sık
    görüldüğü etikete atanır ve her etiket içinde grupların yaklaşık `holdout` oranı
    ayrılır; böylece etiket dağılımı iki tarafta da korunur.
    """
    if not 0.0 < holdout < 1.0:
        raise ValueError("holdout 0 ile 1 arasında olmalı")
    members: dict[str, list[int]] = defaultdict(list)
    label_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for index, (label, group) in enumerate(zip(labels, groups, strict=True)):
        members[group].append(index)
        label_counts[group][label] += 1

    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for group in sorted(members):
        counts = label_counts[group]
        dominant = max(sorted(counts), key=lambda label: counts[label])
        groups_by_label[dominant].append(group)

    rng = random.Random(seed)
    kept, held = [], []
    for label in sorted(groups_by_label):
        label_groups = groups_by_label[label]
        rng.shuffle(label_groups)
        target = holdout * sum(len(members[g]) for g in label_groups)
        taken = 0
        for group in label_groups:
            if taken < target:
                held.extend(members[group])
                taken += len(members[group])
            else:
                kept.extend(members[group])
    return sorted(kept), sorted(held)
