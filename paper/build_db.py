"""FASTA -> BitmapIndex .npz + peptides.npy + mono_masses.npy.

Uses sagepy to digest the FASTA and emit b/y fragment ions with
Cam-C applied; msbit then encodes those into bitmaps.

Usage: python -m paper.build_db <fasta> <out_dir>
"""

import sys
from pathlib import Path


def main(fasta: Path, out_dir: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
