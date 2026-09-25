"""
Dosya   : src/ml/splitting.py
Konu    : Gruplu ve Tabakalı Veri Bölme
Açıklama: Bu dosyada amacım veriyi, aynı terime ait örnekler farklı bölmelere düşmeyecek ve
          konu dağılımı korunacak şekilde eğitim/doğrulama/test olarak bölmek.
Yazar   : Ebrar Cemre Çetin
Tarih   : 24.09.2026
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence


def grouped_stratified_split(labels: Sequence[str], groups: Sequence[str], holdout: float,
                             seed: int) -> tuple[list[int], list[int]]:
    """Burada amacım örnekleri (kalan, ayrılan) indekslerine bölmek.

    Neden gruplu: Tabu verisinde aynı kavram birden fazla kartta geçebilir. Bu kartlar
    eğitim ve teste dağılırsa model testte "ezberlediği" kavramı görür ve başarı olduğundan
    yüksek ölçülür (veri sızıntısı). Bu yüzden aynı gruptaki örnekleri hep aynı tarafa
    koyuyorum.
    Neden tabakalı: her grubu en sık görüldüğü etikete atıyorum ve her etiket içinde
    grupların yaklaşık `holdout` oranını ayırıyorum; böylece konu dağılımı iki tarafta da
    korunuyor.
    """
    if not 0.0 < holdout < 1.0:
        raise ValueError("holdout 0 ile 1 arasında olmalı")
    # members: grup -> o gruptaki örneklerin indeksleri
    # label_counts: grup -> {etiket: o gruptaki sayısı}
    members: dict[str, list[int]] = defaultdict(list)
    label_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for index, (label, group) in enumerate(zip(labels, groups, strict=True)):
        members[group].append(index)
        label_counts[group][label] += 1

    # Her grubu, içinde en çok bulunan etiketin listesine koyuyorum. Eşitlikte alfabetik
    # ilk etiketi seçiyorum (sorted), böylece sonuç her çalıştırmada aynı oluyor.
    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for group in sorted(members):
        counts = label_counts[group]
        dominant = max(sorted(counts), key=lambda label: counts[label])
        groups_by_label[dominant].append(group)

    rng = random.Random(seed)
    kept, held = [], []
    for label in sorted(groups_by_label):
        label_groups = groups_by_label[label]
        rng.shuffle(label_groups)  # hangi grupların ayrılacağı rastgele ama tekrarlanabilir
        # Bu etiket için ayırmam gereken örnek sayısı; hedefe ulaşana kadar grup ekliyorum.
        target = holdout * sum(len(members[g]) for g in label_groups)
        taken = 0
        for group in label_groups:
            if taken < target:
                held.extend(members[group])
                taken += len(members[group])
            else:
                kept.extend(members[group])
    return sorted(kept), sorted(held)
