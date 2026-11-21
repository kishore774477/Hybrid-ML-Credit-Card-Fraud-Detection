"""Download the credit card fraud dataset into data/creditcard.csv.

Usage:
    python -m src.download_data
"""
import sys
import urllib.request

from src.config import DATA_DIR, DATA_PATH, DATASET_URL


def download_dataset(force: bool = False) -> None:
    if DATA_PATH.exists() and not force:
        print(f"Dataset already present at {DATA_PATH}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = DATA_PATH.with_suffix(".part")
    print(f"Downloading dataset (~150 MB) from {DATASET_URL} ...")

    def _progress(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            sys.stdout.write(f"\r  {pct:3d}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(DATASET_URL, tmp_path, _progress)
    tmp_path.replace(DATA_PATH)
    print(f"\nSaved to {DATA_PATH}")


if __name__ == "__main__":
    download_dataset(force="--force" in sys.argv)
