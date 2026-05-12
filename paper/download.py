"""Fetch test inputs into out_dir: human.fasta + test.mzML.

For paper-quality numbers swap these out for a real human reference
proteome (UniProt) and a real PRIDE mzML run. The defaults here are
enough to smoke-test the full pipeline end to end.

Usage: python -m paper.download <out_dir>
"""

import sys
import urllib.request
from pathlib import Path

MZML_URL = "https://raw.githubusercontent.com/tacular-omics/mzmlpy/main/tests/data/example.mzML"

# Insulin (P01308) — a single-protein placeholder so the pipeline runs without
# an internet-fetched FASTA. Replace with a real human proteome for real numbers.
TINY_FASTA = """>sp|P01308|INS_HUMAN
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN
"""


def _fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[download] {dest.name} already present, skipping")
        return
    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta = out_dir / "human.fasta"
    if not fasta.exists():
        fasta.write_text(TINY_FASTA)
        print(f"[download] wrote placeholder insulin FASTA -> {fasta}")
    _fetch(MZML_URL, out_dir / "test.mzML")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
