"""pytest, ruff ve mypy'yi çalıştırıp sonuçlardan TEST_REPORT.md üretir."""

import ast
import json
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def test_docs() -> dict[str, str]:
    """test fonksiyonu adı -> docstring'in ilk satırı."""
    docs = {}
    for path in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                doc = ast.get_docstring(node)
                docs[f"{path.stem}::{node.name}"] = doc.splitlines()[0] if doc else ""
    return docs


def expectation(name: str, doc: str) -> str:
    base = name.split("[")[0].removeprefix("test_").replace("_", " ")
    return doc or base


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "junit.xml"
        pytest_rc, pytest_out = run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
                                     "-rxs", f"--junitxml={xml_path}"])
        root = ET.parse(xml_path).getroot()
    ruff_rc, ruff_out = run([sys.executable, "-m", "ruff", "check", "."])
    mypy_rc, mypy_out = run([sys.executable, "-m", "mypy", "src", "scripts", "proje.py",
                             "train.py", "evaluate.py"])
    docs = test_docs()

    rows, counts = [], {"PASS": 0, "FAIL": 0, "SKIP": 0, "XFAIL (bilinen)": 0}
    for case in root.iter("testcase"):
        module = case.get("classname", "").split(".")[-1]
        name = case.get("name", "")
        failure = case.find("failure") or case.find("error")
        skipped = case.find("skipped")
        if failure is not None:
            result, actual = "FAIL", (failure.get("message") or "")[:160]
        elif skipped is not None and "xfail" in (skipped.get("type", "") +
                                                 (skipped.get("message") or "")).lower():
            result, actual = "XFAIL (bilinen)", (skipped.get("message") or "")[:160]
        elif skipped is not None:
            result, actual = "SKIP", (skipped.get("message") or "")[:160]
        else:
            result, actual = "PASS", "beklenen davranış gözlendi"
        counts[result] += 1
        doc = docs.get(f"{module}::{name.split('[')[0]}", "")
        rows.append((module, name, expectation(name, doc), actual.replace("|", "/")
                     .replace("\n", " "), result))

    evaluation = json.loads((ROOT / "reports" / "evaluation.json").read_text(encoding="utf-8"))
    summary_line = next((line for line in reversed(pytest_out.splitlines())
                         if " passed" in line or " failed" in line), pytest_out[-200:])
    now = datetime.now(UTC).isoformat(timespec="seconds")

    lines = [
        "# Test Raporu",
        "",
        f"Bu dosya `python -m scripts.make_test_report` tarafından **{now}** tarihinde, testler "
        "gerçekten çalıştırılarak üretildi; elle düzenlenmedi.",
        "",
        f"Ortam: Python {platform.python_version()}, {platform.platform()}.",
        "",
        "## Özet",
        "",
        "| Kontrol | Komut | Sonuç |",
        "|---|---|---|",
        f"| Birim + entegrasyon + kabul testleri | `python -m pytest` | {summary_line.strip('= ')} "
        f"(çıkış kodu {pytest_rc}) |",
        f"| Lint | `ruff check .` | {'temiz' if ruff_rc == 0 else 'HATA'} (çıkış kodu {ruff_rc}) |",
        f"| Tip denetimi | `mypy src scripts proje.py train.py evaluate.py` | "
        f"{mypy_out.splitlines()[-1] if mypy_out else ''} (çıkış kodu {mypy_rc}) |",
        "",
        f"Sayım: PASS {counts['PASS']}, FAIL {counts['FAIL']}, SKIP {counts['SKIP']}, "
        f"XFAIL (bilinen sorun) {counts['XFAIL (bilinen)']}.",
        "",
        "* SKIP: canlı ağ testi (`NLP_NETWORK_TESTS=1` gerekli) — bu ortamda Wikipedia erişimi "
        "engelli olduğu için **çalıştırılmadı** (KNOWN_ISSUES KI-002).",
        "* XFAIL: kabul testi B'nin güvenli sınıflandırması (KNOWN_ISSUES KI-001). `strict=True`: "
        "test beklenmedik şekilde geçerse paket başarısız olur.",
        "",
        "## Kritik ve kabul testleri (model çıktıları, `reports/evaluation.json`)",
        "",
        "| Test | Beklenen | Gerçekleşen | Güven | Pass/Fail |",
        "|---|---|---|---:|---|",
    ]
    for group, items in (("Kuantum", evaluation["quantum_tests"]),
                         ("Kabul", evaluation["acceptance_tests"])):
        for t in items:
            lines.append(f"| {group}: {t['text']} | {t['expected']} | {t['actual']} | "
                         f"{t['confidence']:.3f} | {'PASS' if t['pass'] else 'FAIL'} |")
    ood = evaluation["ood_unseen"]
    lines += [
        "",
        f"OOD (eğitimde görülmemiş 27 kategori, n={ood['n']}): reddetme oranı "
        f"{ood['rejected_rate']:.3f}, yanlış kabul {ood['false_accept_rate']:.3f}.",
        "",
        "## Tüm testler",
        "",
        "| Modül | Test | Beklenen | Gerçekleşen | Pass/Fail |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {m} | `{n}` | {e} | {a} | {r} |" for m, n, e, a, r in rows]
    if ruff_rc or mypy_rc:
        lines += ["", "## Lint / tip çıktısı", "", "```", ruff_out[-2000:], mypy_out[-2000:],
                  "```"]
    (ROOT / "TEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_line, "| ruff", ruff_rc, "| mypy", mypy_rc, "|", counts)
    return 0 if pytest_rc == 0 and ruff_rc == 0 and mypy_rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
