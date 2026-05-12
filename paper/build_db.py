"""FASTA -> BitmapIndex .npz + peptides.npy + masses.npy.

Usage: python -m paper.build_db <fasta> <out_dir>
"""

import sys
from pathlib import Path


def main(fasta: Path, out_dir: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
