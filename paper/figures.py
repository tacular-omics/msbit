"""Manuscript figure: recall@K + timing.

Usage: python -m paper.figures <compare.json> <out.png>
"""

import sys
from pathlib import Path


def main(compare_path: Path, out_path: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
