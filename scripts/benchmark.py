"""Gecikme, bellek ve donanım bilgisini reports/performance.json dosyasına yazar."""

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def rss_mb() -> float:
    """Anlık RSS (MB); /proc olmayan sistemlerde (Windows, macOS) NaN."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def system_info() -> dict:
    import joblib
    import pandas
    import requests
    import scipy
    import sklearn

    cpu = "bilinmiyor"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            cpu = next(line.split(":", 1)[1].strip() for line in handle
                       if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    mem_gb = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            mem_gb = round(int(handle.readline().split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    gpu = "yok"
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        gpu = out.stdout.strip() or "yok"
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "os": platform.platform(), "python": platform.python_version(), "cpu": cpu,
        "cpu_count": os.cpu_count(), "ram_gb": mem_gb, "gpu": gpu,
        "cuda": "yok (model CPU üzerinde çalışır; GPU gerekmez)",
        "frameworks": {"scikit-learn": sklearn.__version__, "numpy": np.__version__,
                       "scipy": scipy.__version__, "pandas": pandas.__version__,
                       "joblib": joblib.__version__, "requests": requests.__version__},
    }


def cold_start_seconds() -> float:
    code = ("import time;t=time.perf_counter();from src.models.predictor import TopicClassifier;"
            "c=TopicClassifier.load();c.predict('ısınma');print(time.perf_counter()-t)")
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root,
                         timeout=120, check=True)
    return float(out.stdout.strip().splitlines()[-1])


def main() -> int:
    from src.config import MODEL_PATH, REPORTS_DIR
    from src.data.build import load_split
    from src.database.db import Database
    from src.models.predictor import TopicClassifier
    from src.preprocessing.text import normalize
    from src.services.app import ChatService
    from src.services.web_search import WebSearchClient

    base_rss = rss_mb()
    t0 = time.perf_counter()
    clf = TopicClassifier.load()
    load_s = time.perf_counter() - t0
    model_rss = rss_mb() - base_rss

    texts = load_split("test")["text"].tolist()
    t0 = time.perf_counter()
    for text in texts:
        normalize(text)
    prep_ms = (time.perf_counter() - t0) / len(texts) * 1000

    clf.predict(texts[0])
    lat = []
    for text in texts[:500]:
        t = time.perf_counter()
        clf.predict(text)
        lat.append((time.perf_counter() - t) * 1000)
    t0 = time.perf_counter()
    clf.predict_proba_auto(texts)
    batch_ms = (time.perf_counter() - t0) / len(texts) * 1000

    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "b.db")
        service = ChatService(clf, db, web_enabled=False)
        turn = []
        for text in texts[:200]:
            t = time.perf_counter()
            service.process(text)
            turn.append((time.perf_counter() - t) * 1000)
        db.cache_put("kuantum", "wikipedia", [{"title": "x"}] * 3)
        t = time.perf_counter()
        for _ in range(200):
            db.cache_get("kuantum")
        cache_ms = (time.perf_counter() - t) / 200 * 1000

        offline_client = WebSearchClient(timeout=3)
        t = time.perf_counter()
        outcome = offline_client.search("kuantum bilgisayar")
        web_first_ms = (time.perf_counter() - t) * 1000
        web_service = ChatService(clf, db, search_client=offline_client, web_enabled=True)
        web_service.process("Kuantum bilgisayarlar kübit kullanır.")
        t = time.perf_counter()
        web_service.process("Kuantum işlemciler algoritmaları hızlandırır.")
        web_breaker_ms = (time.perf_counter() - t) * 1000
        db.close()

    report = {
        "system": system_info(),
        "cold_start_import_load_first_predict_s": round(cold_start_seconds(), 3),
        "model_load_s": round(load_s, 3),
        "model_memory_rss_mb": round(model_rss, 1),
        "process_rss_mb_after_benchmark": round(rss_mb(), 1),
        "artifact_size_mb": round(MODEL_PATH.stat().st_size / 1e6, 2),
        "preprocessing_ms_per_text": round(prep_ms, 4),
        "inference_ms_p50": round(float(np.percentile(lat, 50)), 3),
        "inference_ms_p95": round(float(np.percentile(lat, 95)), 3),
        "batch_inference_ms_per_text": round(batch_ms, 4),
        "full_turn_no_web_ms_p50": round(float(np.percentile(turn, 50)), 3),
        "full_turn_no_web_ms_p95": round(float(np.percentile(turn, 95)), 3),
        "search_cache_read_ms": round(cache_ms, 4),
        "web_request_status_in_this_environment": outcome.status,
        "web_first_request_ms_when_offline": round(web_first_ms, 1),
        "web_turn_ms_after_circuit_breaker": round(web_breaker_ms, 2),
        "note": "Canlı Wikipedia gecikmesi bu ortamda ölçülemedi (ağ politikası tr.wikipedia.org "
                "erişimini engelliyor).",
    }
    (REPORTS_DIR / "performance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
