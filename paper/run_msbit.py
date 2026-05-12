"""mzML -> msbit presearch candidates + per-spectrum timing.

Loads the msbit BitmapIndex built by paper.build_db, reads every MS2
spectrum from the mzML, encodes its peak m/z list into a query bitmap
with the same encoder params, and runs `BitmapIndex.search(top_k)`
filtered to the precursor tolerance window.

Usage: python -m paper.run_msbit <db_dir> <mzml> <out.parquet> [--backend cpu|gpu] [--top-k 200]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from msbit import BitmapIndex
from msbit.encoding import BitmapEncoder

PROTON = 1.00728
DEFAULT_TOP_K = 200
PRECURSOR_TOL_PPM = 20.0


def _encoder_from_params(params: dict) -> BitmapEncoder:
    b = params["bitmap"]
    return BitmapEncoder(
        min_mass=b["min_mass"],
        max_mass=b["max_mass"],
        array_length=b["array_length"],
        tolerance=b["tolerance_ppm"],
        is_ppm=True,
    )


def _encode_query(encoder: BitmapEncoder, peaks_mz: np.ndarray) -> np.ndarray:
    peaks = np.unique(peaks_mz[peaks_mz > 0.0]).astype(np.float64)
    if peaks.size == 0:
        return np.zeros(encoder.array_length, dtype=np.uint64)
    mat = np.zeros((1, peaks.size), dtype=np.float64)
    mat[0, :] = peaks
    return encoder.encode(mat)[0]


def main(db_dir: Path, mzml: Path, out_path: Path, backend: str, top_k: int) -> None:
    import mzmlpy

    params = json.loads((db_dir / "params.json").read_text())
    encoder = _encoder_from_params(params)
    index = BitmapIndex.load(db_dir / "index.npz", backend=backend)

    rows = []
    reader = mzmlpy.Mzml(mzml)
    # Warmup JIT (and GPU upload, if applicable) before timing.
    warm_q = np.zeros(encoder.array_length, dtype=np.uint64)
    warm_q[0] = np.uint64(1)
    index.search(warm_q, k=top_k)

    for s in reader.spectra:
        if s.ms_level != 2 or not s.precursors:
            continue
        ions = s.precursors[0].selected_ions
        if not ions:
            continue
        prec = ions[0]
        charge = prec.charge_state or 2
        prec_mass = float(prec.selected_ion_mz) * charge - charge * PROTON
        tol = prec_mass * PRECURSOR_TOL_PPM / 1e6

        query = _encode_query(encoder, np.asarray(s.mz, dtype=np.float64))
        t0 = time.perf_counter()
        result = index.search(query, k=top_k, min_val=prec_mass - tol, max_val=prec_mass + tol)
        elapsed = time.perf_counter() - t0

        rows.append(
            {
                "scan_id": s.id,
                "precursor_mass": prec_mass,
                "charge": charge,
                "candidates": result.indices.tolist(),
                "matches": result.matches.tolist(),
                "pvalues": result.pvalues.tolist(),
                "time_s": elapsed,
            }
        )
    reader.close()

    df = pl.DataFrame(rows)
    df.write_parquet(out_path)
    if len(df):
        print(
            f"[run_msbit] {len(df)} MS2 spectra  "
            f"median={float(df['time_s'].median()) * 1000:.2f}ms  "
            f"total={float(df['time_s'].sum()):.2f}s"
        )
    else:
        print("[run_msbit] no MS2 spectra found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("db_dir", type=Path)
    ap.add_argument("mzml", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args()
    main(args.db_dir, args.mzml, args.out, args.backend, args.top_k)
