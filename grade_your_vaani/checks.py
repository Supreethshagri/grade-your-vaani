import argparse
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

from grade_your_vaani.dataset import PROJECT_ROOT
from grade_your_vaani.normalize import normalize

RESULTS_DIR = PROJECT_ROOT / "results"
EXPECTED_SCRIPT = {"en": "LATIN", "hi": "DEVANAGARI", "kn": "KANNADA"}
IGNORED_SCRIPTS = {"COMBINING", "MODIFIER"}  # generic marks that belong to no single script

# Heuristic thresholds: documented here so anyone can see and question them
MAX_LATIN_SHARE = 0.5   # hi/kn may contain English acronyms, but shouldn't be mostly English
MIN_LENGTH_RATIO = 0.5  # transcript far shorter than the reference: words were dropped
MAX_LENGTH_RATIO = 2.0  # transcript far longer: text was invented
PROMPT_ECHO_MIN = 2     # this many never-spoken vocab terms = the model echoed the prompt
REPEATED_LETTER = re.compile(r"([^\W\d_])\1{3,}")  # same letter 4+ times in a row

FLAGS = ("empty", "foreign_script", "romanized_or_translated", "repetition",
         "too_short", "too_long", "prompt_echo")
REQUIRED_COLUMNS = {"clip_id", "language", "condition", "reference", "hypothesis",
                    "kw_false_alarms", "error"}


def _script(ch: str) -> str | None:
    if unicodedata.category(ch)[0] not in ("L", "M"):  # letters and vowel signs only
        return None
    name = unicodedata.name(ch, "")
    script = name.split(" ")[0] if name else ""
    return None if not script or script in IGNORED_SCRIPTS else script


def check_transcript(norm_ref: str, norm_hyp: str, language: str,
                     false_alarms: int, prompt_used: bool) -> list[str]:
    if not norm_hyp:
        return ["empty"]
    flags: list[str] = []

    scripts = [s for s in map(_script, norm_hyp) if s]
    expected = EXPECTED_SCRIPT.get(language)
    if expected and scripts:
        if any(s not in (expected, "LATIN") for s in scripts):
            flags.append("foreign_script")
        if expected != "LATIN" and scripts.count("LATIN") / len(scripts) > MAX_LATIN_SHARE:
            flags.append("romanized_or_translated")

    words = norm_hyp.split()
    if REPEATED_LETTER.search(norm_hyp) or any(
        words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2)
    ):
        flags.append("repetition")

    ref_chars = len(norm_ref.replace(" ", ""))
    if ref_chars:
        ratio = len(norm_hyp.replace(" ", "")) / ref_chars
        if ratio < MIN_LENGTH_RATIO:
            flags.append("too_short")
        elif ratio > MAX_LENGTH_RATIO:
            flags.append("too_long")

    if prompt_used and false_alarms >= PROMPT_ECHO_MIN:
        flags.append("prompt_echo")
    return flags


def analyze_run(run_dir: Path) -> dict | None:
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        return None
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not REQUIRED_COLUMNS <= set(reader.fieldnames or []):
            return None  # runs from before keyword tracking don't have the columns we need
        rows = list(reader)

    counts: Counter[str] = Counter()
    out_rows: list[dict] = []
    checked = flagged = 0
    for row in rows:
        if row["error"]:
            continue
        checked += 1
        language = row["language"]
        if "vocab" in row:
            prompt_used = row["vocab"] != "none"
        else:  # first vocab run used an older column name
            prompt_used = row.get("prompt_used") == "yes"
        flags = check_transcript(
            normalize(row["reference"], language),
            normalize(row["hypothesis"], language),
            language,
            int(row["kw_false_alarms"] or 0),
            prompt_used,
        )
        counts.update(flags)
        flagged += bool(flags)
        out_rows.append({"clip_id": row["clip_id"], "language": language,
                         "condition": row["condition"], "flags": ";".join(flags),
                         "hypothesis": row["hypothesis"]})

    with open(run_dir / "flags.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "language", "condition",
                                               "flags", "hypothesis"])
        writer.writeheader()
        writer.writerows(out_rows)
    return {"run": run_dir.name, "checked": checked, "flagged": flagged, "counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Flag broken transcripts in saved runs")
    parser.add_argument("runs", nargs="*", help="run folder names inside results/ (default: all)")
    args = parser.parse_args()

    base = RESULTS_DIR.resolve()
    if args.runs:
        run_dirs = []
        for name in args.runs:
            path = (RESULTS_DIR / name).resolve()
            if not path.is_relative_to(base):
                raise SystemExit(f"not a folder inside results/: {name}")
            run_dirs.append(path)
    else:
        run_dirs = sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()) if RESULTS_DIR.is_dir() else []

    found = False
    print(f"{'run':<58} {'flagged':>9}  details")
    for run_dir in run_dirs:
        result = analyze_run(run_dir)
        if result is None:
            if args.runs:
                print(f"{run_dir.name}: no compatible results.csv, skipped")
            continue
        found = True
        pct = result["flagged"] / result["checked"] if result["checked"] else 0
        details = ", ".join(f"{flag} {result['counts'][flag]}"
                            for flag in FLAGS if result["counts"][flag])
        print(f"{result['run']:<58} {result['flagged']:>2}/{result['checked']:<2} {pct:4.0%}  "
              f"{details or '-'}")

    if not found:
        raise SystemExit("No compatible runs found. Run the benchmark first.")


if __name__ == "__main__":
    main()