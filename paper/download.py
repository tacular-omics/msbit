"""Fetch human FASTA + test mzML from tacular-omics/mzmlpy.

Usage: python -m paper.download <out_dir>
"""

import sys
from pathlib import Path


def main(out_dir: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main(Path(sys.argv[1]))
