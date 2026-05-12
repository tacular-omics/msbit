"""FASTA -> msbit BitmapIndex `.npz` + peptides + mono_masses + sage_params.

Uses paper.sage_db to digest the FASTA via sagepy (Cam-C applied), then
encodes the per-peptide fragment-mz lists into msbit bitmaps.

Usage: python -m paper.build_db <fasta> <out_dir>
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from msbit import BitmapIndex
from msbit.encoding import BitmapEncoder
from paper.sage_db import SageParams, build_indexed_database, per_peptide_fragments

BITMAP_MIN_MASS = 100.0
BITMAP_MAX_MASS = 2000.0
BITMAP_ARRAY_LENGTH = 100  # 6,400 log-spaced bins over [100, 2000] Da
BITMAP_TOLERANCE = 20.0  # ppm; matches typical Orbitrap fragment resolution


def _ion_matrix(frags: list[np.ndarray]) -> np.ndarray:
    """Pad per-peptide fragment arrays into a (n_peptides, max_ions) float64 matrix."""
    max_ions = max(len(f) for f in frags)
    mat = np.zeros((len(frags), max_ions), dtype=np.float64)
    for i, f in enumerate(frags):
        mat[i, : len(f)] = f
    return mat


def main(fasta: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    params = SageParams(fasta_path=fasta)

    t0 = time.perf_counter()
    db = build_indexed_database(params)
    t_digest = time.perf_counter() - t0

    peptides = db.peptides_as_string()
    mono = np.asarray(db.mono_masses(), dtype=np.float64)
    frags = per_peptide_fragments(db)
    print(f"[build_db] sagepy: {db.num_peptides} peptides, {db.num_fragments} fragments, {t_digest:.2f}s")

    encoder = BitmapEncoder(
        min_mass=BITMAP_MIN_MASS,
        max_mass=BITMAP_MAX_MASS,
        array_length=BITMAP_ARRAY_LENGTH,
        tolerance=BITMAP_TOLERANCE,
        is_ppm=True,
    )
    t0 = time.perf_counter()
    bitmaps = encoder.encode(_ion_matrix(frags))
    t_encode = time.perf_counter() - t0
    print(f"[build_db] msbit encode: {bitmaps.shape} {bitmaps.dtype}, {t_encode:.2f}s")

    index = BitmapIndex(bitmaps, backend="cpu", filter_array=mono)
    index.save(out_dir / "index.npz")
    np.save(out_dir / "peptides.npy", peptides)
    np.save(out_dir / "mono_masses.npy", mono)
    (out_dir / "params.json").write_text(
        json.dumps(
            {
                "sage": {**asdict(params), "fasta_path": str(params.fasta_path)},
                "bitmap": {
                    "min_mass": BITMAP_MIN_MASS,
                    "max_mass": BITMAP_MAX_MASS,
                    "array_length": BITMAP_ARRAY_LENGTH,
                    "tolerance_ppm": BITMAP_TOLERANCE,
                },
                "timing_s": {"digest": t_digest, "encode": t_encode},
            },
            indent=2,
        )
    )
    print(f"[build_db] wrote {out_dir}/index.npz peptides.npy mono_masses.npy params.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
