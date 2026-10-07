import argparse
import shutil

from grade_your_vaani.dataset import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"
PUBLISHED_DIR = PROJECT_ROOT / "published" / "runs"
FILES = ("summary.json", "results.csv", "flags.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy chosen runs into published/ for the dashboard")
    parser.add_argument("runs", nargs="+", help="run folder names inside results/")
    args = parser.parse_args()

    base = RESULTS_DIR.resolve()
    sources = []
    for name in args.runs:  # validate everything first, so nothing is half-published
        src = (RESULTS_DIR / name).resolve()
        if not src.is_relative_to(base) or not src.is_dir():
            raise SystemExit(f"not a run folder inside results/: {name}")
        missing = [f for f in FILES if not (src / f).is_file()]
        if missing:
            raise SystemExit(f"{name} is missing {missing}. Run 'python -m grade_your_vaani.checks' first.")
        sources.append(src)

    for src in sources:
        dst = PUBLISHED_DIR / src.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            shutil.copy2(src / f, dst / f)
        print(f"published {src.name}")


if __name__ == "__main__":
    main()