"""mzML -> sagepy presearch candidates + per-spectrum timing.

"Presearch" here = the precursor-mass window prefilter, i.e. all peptides
in `db.mono_masses` within +/- precursor_tolerance of the spectrum's
precursor mass. This is the same prefilter sage runs internally before
fragment scoring; isolating it gives a clean prefilter-vs-prefilter
comparison with msbit (which adds bitmap-overlap ranking on top).

We default to an OPEN search (±500 Da). Narrow ppm tolerance produces
windows of ~1-10 peptides on a real DB, where there's nothing for any
prefilter to do — open search expands the window to thousands and is
the regime where msbit's bitmap ranking actually pays for itself.

Reusing sagepy's IndexedDatabase guarantees the peptide universe is
byte-identical to build_db.py's msbit input.

Usage: python -m paper.run_sagepy <fasta> <mzml> <out.parquet> [--precursor-tol-da 500]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from paper.sage_db import SageParams, build_indexed_database

PROTON = 1.00728
DEFAULT_PRECURSOR_TOL_DA = 500.0


def main(fasta: Path, mzml: Path, out_path: Path, precursor_tol_da: float) -> None:
    import mzmlpy

    db = build_indexed_database(SageParams(fasta_path=fasta))
    mono = np.asarray(db.mono_masses(), dtype=np.float64)
    print(f"[run_sagepy] db: {db.num_peptides} peptides, {db.num_fragments} fragments")

    rows = []
    reader = mzmlpy.Mzml(mzml)
    for s in reader.spectra:
        if s.ms_level != 2 or not s.precursors:
            continue
        ions = s.precursors[0].selected_ions
        if not ions:
            continue
        prec = ions[0]
        charge = prec.charge_state or 2
        prec_mass = float(prec.selected_ion_mz) * charge - charge * PROTON

        t0 = time.perf_counter()
        lo = int(np.searchsorted(mono, prec_mass - precursor_tol_da, side="left"))
        hi = int(np.searchsorted(mono, prec_mass + precursor_tol_da, side="right"))
        elapsed = time.perf_counter() - t0

        rows.append(
            {
                "scan_id": s.id,
                "precursor_mass": prec_mass,
                "charge": charge,
                "candidates": list(range(lo, hi)),
                "window_size": hi - lo,
                "time_s": elapsed,
            }
        )
    reader.close()

    df = pl.DataFrame(rows)
    df.write_parquet(out_path)
    if len(df):
        print(
            f"[run_sagepy] {len(df)} MS2 spectra  "
            f"median window={int(df['window_size'].median())}  "
            f"median time={float(df['time_s'].median()) * 1e6:.1f}us"
        )
    else:
        print("[run_sagepy] no MS2 spectra found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta", type=Path)
    ap.add_argument("mzml", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--precursor-tol-da", type=float, default=DEFAULT_PRECURSOR_TOL_DA)
    args = ap.parse_args()
    main(args.fasta, args.mzml, args.out, args.precursor_tol_da)
