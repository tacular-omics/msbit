"""recall@K + timing comparison between msbit and sagepy presearch outputs.

Usage: python -m paper.compare <msbit.parquet> <sagepy.parquet> <out.json>
"""

import sys
from pathlib import Path


def main(msbit_path: Path, sagepy_path: Path, out_path: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
