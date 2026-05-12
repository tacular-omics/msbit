"""mzML -> sagepy presearch candidates + per-spectrum timing.

Uses the same digest params as build_db.py so candidates are comparable.

Usage: python -m paper.run_sagepy <fasta> <mzml> <out.parquet>
"""

import sys
from pathlib import Path


def main(fasta: Path, mzml: Path, out_path: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
