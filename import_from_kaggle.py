"""Download the "Quebec Election 2026 - Search & Media Attention" dataset from Kaggle.

https://www.kaggle.com/datasets/jeanmidev/quebec-election-2026-search-attention
Requires the `kaggle` CLI, installed and authenticated -- see readme.md.

Usage:
    python import_from_kaggle.py                  # downloads + unzips into data/kaggle/
    python import_from_kaggle.py --out some/dir    # download elsewhere
    python import_from_kaggle.py --force           # re-download even if data/kaggle/ isn't empty
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASET = "jeanmidev/quebec-election-2026-search-attention"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "kaggle"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                     help=f"download directory (default: {DEFAULT_OUT})")
    ap.add_argument("--force", action="store_true",
                     help="re-download and overwrite even if --out already has files")
    args = ap.parse_args()

    if shutil.which("kaggle") is None:
        sys.exit("The `kaggle` CLI isn't installed -- run `pip install kaggle` first.")

    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()) and not args.force:
        sys.exit(f"{args.out} already has files -- pass --force to re-download and overwrite them.")

    print(f"Downloading {DATASET} -> {args.out}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", DATASET, "-p", str(args.out), "--unzip", "-o"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        if "401" in result.stderr or "403" in result.stderr:
            sys.exit("Kaggle authentication failed -- put your API token at ~/.kaggle/kaggle.json "
                      "(kaggle.com -> Account -> Settings -> Create New Token). Full error:\n" + result.stderr)
        sys.exit(f"kaggle CLI failed:\n{result.stderr}")

    csvs = sorted(args.out.glob("*.csv"))
    print(f"Done: {len(csvs)} tables downloaded to {args.out}. Read DATA_DICTIONARY.md there first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
