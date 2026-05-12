"""mzML -> msbit presearch candidates + per-spectrum timing.

Usage: python -m paper.run_msbit <db_dir> <mzml> <out.parquet> [--backend cpu|gpu]
"""

import sys
from pathlib import Path


def main(db_dir: Path, mzml: Path, out_path: Path, backend: str = "cpu") -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
