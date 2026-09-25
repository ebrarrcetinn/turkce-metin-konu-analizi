"""Dokümanlardaki dosya yollarını, komutları ve anahtar metrikleri kodla karşılaştırır."""

import json
import re
import sys

from src.config import MODEL_METADATA_PATH, PROJECT_ROOT, REPORTS_DIR

PATH_RE = re.compile(r"`((?:scripts|src|reports|tests|models|data)/[^`\s*]+|[\w\-]+\.(?:py|md))`")
CMD_RE = re.compile(r"python ([\w/]+\.py)")
MODULE_CMD_RE = re.compile(r"python -m scripts\.(\w+)")
IGNORED_PATHS = {"data/raw/provenance.json", "logs/app.log"}  # çalışma zamanında üretilir


def tr(x: float, digits: int) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def main() -> int:
    problems: list[str] = []
    docs = sorted(PROJECT_ROOT.glob("*.md"))
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        for path in set(PATH_RE.findall(text)):
            clean = path.rstrip("/.,")
            if clean in IGNORED_PATHS or "*" in clean or "<" in clean:
                continue
            if clean.startswith(("data/raw", "data/processed", "models/topic_model")):
                continue  # git dışı, eğitimle üretilen dosyalar
            if not (PROJECT_ROOT / clean).exists():
                problems.append(f"{doc.name}: olmayan yol `{clean}`")
        if re.search(r"\b(TODO|FIXME)\b", text) and doc.name != "check_docs.py":
            problems.append(f"{doc.name}: TODO/FIXME içeriyor")

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    scripts = set(CMD_RE.findall(readme))
    scripts |= {f"scripts/{name}.py" for name in MODULE_CMD_RE.findall(readme)}
    for script in scripts:
        if not (PROJECT_ROOT / script).exists():
            problems.append(f"README.md: komut olmayan dosyaya işaret ediyor: {script}")

    evaluation = json.loads((REPORTS_DIR / "evaluation.json").read_text(encoding="utf-8"))
    meta = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8")) \
        if MODEL_METADATA_PATH.exists() else {}
    expected: dict[str, str | tuple[str, str]] = {
        "test macro F1 (eşikli)": tr(evaluation["test"]["with_threshold"]["macro_f1"], 3),
        "test accuracy (eşikli)": tr(evaluation["test"]["with_threshold"]["accuracy"], 3),
        "test ECE": tr(evaluation["test"]["ece_15bin"], 3),
        "OOD reddetme": (tr(evaluation["ood_unseen"]["rejected_rate"] * 100, 1),
                         tr(evaluation["ood_unseen"]["rejected_rate"], 3)),
    }
    if meta:
        expected["güven eşiği"] = tr(meta["thresholds"]["min_confidence"], 2)
    for doc_name in ("README.md", "MODEL_CARD.md", "FINAL_REPORT.md", "MODEL_REPORT.md"):
        path = PROJECT_ROOT / doc_name
        if not path.exists():
            problems.append(f"{doc_name} yok")
            continue
        text = path.read_text(encoding="utf-8")
        for label, value in expected.items():
            forms = value if isinstance(value, tuple) else (value,)
            variants = {f for v in forms for f in (v, v.rstrip("0").rstrip(","))}
            if not any(v in text for v in variants):
                problems.append(f"{doc_name}: {label} değeri ({forms[0]}) bulunamadı")

    report = PROJECT_ROOT / "TEST_REPORT.md"
    if report.exists():
        match = re.search(r"Sayım: PASS (\d+)", report.read_text(encoding="utf-8"))
        if match:
            passed = match.group(1)
            for doc_name in ("README.md", "FINAL_REPORT.md"):
                text = (PROJECT_ROOT / doc_name).read_text(encoding="utf-8")
                if f"{passed} geçti" not in text:
                    problems.append(f"{doc_name}: güncel test sayısı ({passed} geçti) yok")

    for p in problems:
        print("SORUN:", p)
    print(f"{len(docs)} doküman denetlendi, {len(problems)} sorun.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
