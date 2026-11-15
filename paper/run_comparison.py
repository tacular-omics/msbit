"""Single-script comparison: builds one DB, runs msbit and sagepy as presearches.

Builds the sagepy IndexedDatabase and msbit BitmapIndex from the same in-memory
database object so peptide indices are guaranteed identical across both presearches.

Both presearches produce a ranked list of candidate peptide indices over the same
precursor mass window:
  - msbit: bitmap fragment overlap, top-K
  - sagepy: hyperscore via sagepy.Scorer, top report_psms

Writes msbit and sagepy outputs as separate parquets for compare.py.

Usage:
    python -m paper.run_comparison <fasta> <mzml> <out_dir>
        [--backend cpu|gpu] [--top-k 200] [--precursor-tol-da 500]
        [--fragment-tol-ppm 20] [--report-psms 200] [--min-matched-peaks 1]
        [--max-spectra N]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from sagepy.core.mass import Tolerance
from sagepy.core.scoring import Scorer
from sagepy.core.spectrum import Peak, Precursor, ProcessedSpectrum
from spxtacular import MzmlReader

from msbit import BitmapIndex
from msbit.encoding import BitmapEncoder
from paper.sage_db import SageParams, build_indexed_database, per_peptide_fragments

PROTON = 1.00728
BITMAP_MIN_MASS = 200.0
BITMAP_MAX_MASS = 5000.0
BITMAP_ARRAY_LENGTH = 100
BITMAP_TOLERANCE_PPM = 20.0

DEFAULT_TOP_K = 200
DEFAULT_PRECURSOR_TOL_DA = 500.0
DEFAULT_FRAGMENT_TOL_PPM = 20.0
DEFAULT_REPORT_PSMS = 200
DEFAULT_MIN_MATCHED_PEAKS = 5
DEFAULT_MAX_SPECTRA = 1000


def _build_msbit_index(db, backend: str, frags: list[np.ndarray]) -> tuple[BitmapIndex, BitmapEncoder]:
    max_ions = max(len(f) for f in frags)
    mat = np.zeros((len(frags), max_ions), dtype=np.float64)
    for i, f in enumerate(frags):
        mat[i, : len(f)] = f
    encoder = BitmapEncoder(
        min_mass=BITMAP_MIN_MASS,
        max_mass=BITMAP_MAX_MASS,
        array_length=BITMAP_ARRAY_LENGTH,
        tolerance=BITMAP_TOLERANCE_PPM,
        is_ppm=True,
    )
    mono = np.asarray(db.mono_masses(), dtype=np.float64)
    bitmaps = encoder.encode(mat)
    index = BitmapIndex(bitmaps, backend=backend, filter_array=mono)
    return index, encoder


def _encode_query(
    encoder: BitmapEncoder,
    raw_mz: np.ndarray,
    max_charge: int = 3,
) -> np.ndarray:
    """Encode an observed peak list into a bitmap in DB neutral-mass space.

    sagepy's `db.fragment_mzs` is stored as NEUTRAL fragment mass (verified
    empirically: stored = expected_+1_m/z - PROTON), so the bitmap DB rows are
    in neutral-mass space. msbit's bitmap has no internal multi-charge support
    (sage handles this via `Scorer.max_fragment_charge`), so for each observed
    m/z we emit one neutral-mass hypothesis per assumed charge in 1..max_charge:
        neutral(z) = mz * z - z * PROTON
    """
    if raw_mz.size == 0:
        return np.zeros(encoder.array_length, dtype=np.uint64)
    charges = np.arange(1, max_charge + 1, dtype=np.float64).reshape(-1, 1)
    expanded = (raw_mz.reshape(1, -1) * charges - charges * PROTON).ravel()
    peaks = expanded[expanded > 0.0]
    if peaks.size == 0:
        return np.zeros(encoder.array_length, dtype=np.uint64)
    peaks.sort()
    mat = np.zeros((1, peaks.size), dtype=np.float64)
    mat[0, :] = peaks
    return encoder.encode(mat)[0]


def _build_processed_spectrum(
    raw_mz: np.ndarray,
    raw_intensity: np.ndarray,
    scan_id: str,
    prec_mz: float,
    charge: int,
) -> ProcessedSpectrum:
    """ProcessedSpectrum with peaks shifted to neutral mass for sage's Scorer.

    sagepy's `Peak.mass` is interpreted against the neutral-mass DB; passing
    +1 m/z shifts every peak by one proton and craters hyperscore matching
    (verified empirically: hyperscore 117 with neutral input vs 11.8 with +1 m/z
    on a perfect self-match query). Scorer's `max_fragment_charge>=2` then
    handles multi-charge fragments internally.
    """
    peaks = [
        Peak(mass=float(m - PROTON), intensity=float(i))
        for m, i in zip(raw_mz, raw_intensity, strict=True)
        if m > PROTON
    ]
    return ProcessedSpectrum(
        id=scan_id,
        level=2,
        file_id=0,
        scan_start_time=0.0,
        ion_injection_time=0.0,
        precursors=[Precursor(mz=prec_mz, charge=charge)],
        peaks=peaks,
        total_ion_current=float(raw_intensity.sum()),
    )


def main(
    fasta: Path,
    mzml: Path,
    out_dir: Path,
    backend: str,
    top_k: int,
    precursor_tol_da: float,
    fragment_tol_ppm: float,
    report_psms: int,
    min_matched_peaks: int,
    max_spectra: int | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    db = build_indexed_database(SageParams(fasta_path=fasta))
    print(f"[comparison] db: {db.num_peptides} peptides, {db.num_fragments} fragments")

    frags = per_peptide_fragments(db)
    mono = np.asarray(db.mono_masses(), dtype=np.float64)
    index, encoder = _build_msbit_index(db, backend, frags)
    scorer = Scorer(
        precursor_tolerance=Tolerance(da=(-precursor_tol_da, precursor_tol_da)),
        fragment_tolerance=Tolerance(ppm=(-fragment_tol_ppm, fragment_tol_ppm)),
        min_matched_peaks=min_matched_peaks,
        report_psms=report_psms,
        wide_window=False,
        max_fragment_charge=3,
    )

    # Warmup JIT / GPU before timing
    warm_q = np.zeros(encoder.array_length, dtype=np.uint64)
    warm_q[0] = np.uint64(1)
    index.search(warm_q, k=top_k)

    msbit_rows, sagepy_rows = [], []

    with MzmlReader(mzml) as reader:
        for s in reader.ms2:
            if max_spectra is not None and len(msbit_rows) >= max_spectra:
                break
            if not s.precursors:
                continue
            prec = s.precursors[0]
            charge = prec.charge or 2
            prec_mz = float(prec.mz)
            prec_mass = prec_mz * charge - charge * PROTON

            raw_mz = np.asarray(s.mz, dtype=np.float64)
            raw_intensity = np.asarray(s.intensity, dtype=np.float64)
            scan_id = s.native_id or str(s.scan_number)

            # --- msbit presearch ---
            t0 = time.perf_counter()
            query = _encode_query(encoder, raw_mz)
            result = index.search(
                query,
                k=top_k,
                min_val=prec_mass - precursor_tol_da,
                max_val=prec_mass + precursor_tol_da,
                min_hits=min_matched_peaks,
            )
            msbit_time = time.perf_counter() - t0

            msbit_rows.append(
                {
                    "scan_id": scan_id,
                    "precursor_mass": prec_mass,
                    "charge": charge,
                    "candidates": result.indices.tolist(),
                    "matches": result.matches.tolist(),
                    "pvalues": result.pvalues.tolist(),
                    "time_s": msbit_time,
                }
            )

            # --- sagepy presearch ---
            spectrum = _build_processed_spectrum(raw_mz, raw_intensity, scan_id, prec_mz, charge)
            t0 = time.perf_counter()
            features = scorer.score(db, spectrum)
            sagepy_time = time.perf_counter() - t0
            # Scorer.report_psms returns up to N PSMs total (across charge / isotope
            # states), often with the same peptide_idx repeated. Dedup preserving
            # hyperscore rank — the comparison metric is over unique peptides.
            sage_candidates = list(dict.fromkeys(f.peptide_idx.idx for f in features))

            lo = int(np.searchsorted(mono, prec_mass - precursor_tol_da, side="left"))
            hi = int(np.searchsorted(mono, prec_mass + precursor_tol_da, side="right"))
            window_size = hi - lo

            sagepy_rows.append(
                {
                    "scan_id": scan_id,
                    "precursor_mass": prec_mass,
                    "charge": charge,
                    "candidates": sage_candidates,
                    "window_size": window_size,
                    "time_s": sagepy_time,
                }
            )

    msbit_df = pl.DataFrame(msbit_rows)
    sagepy_df = pl.DataFrame(sagepy_rows)
    msbit_df.write_parquet(out_dir / "msbit.parquet")
    sagepy_df.write_parquet(out_dir / "sagepy.parquet")

    if len(msbit_df):
        m_t = msbit_df["time_s"].to_numpy()
        s_t = sagepy_df["time_s"].to_numpy()
        m_med = float(np.median(m_t))
        m_tot = float(m_t.sum())
        s_med = float(np.median(s_t))
        s_tot = float(s_t.sum())
        print(
            f"[comparison] {len(msbit_df)} MS2 spectra\n"
            f"  msbit  median={m_med * 1000:.2f}ms  total={m_tot:.1f}s\n"
            f"  sagepy median={s_med * 1000:.2f}ms  total={s_tot:.1f}s"
        )
    else:
        print("[comparison] no MS2 spectra found")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta", type=Path)
    ap.add_argument("mzml", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--precursor-tol-da", type=float, default=DEFAULT_PRECURSOR_TOL_DA)
    ap.add_argument("--fragment-tol-ppm", type=float, default=DEFAULT_FRAGMENT_TOL_PPM)
    ap.add_argument("--report-psms", type=int, default=DEFAULT_REPORT_PSMS)
    ap.add_argument("--min-matched-peaks", type=int, default=DEFAULT_MIN_MATCHED_PEAKS)
    ap.add_argument("--max-spectra", type=int, default=DEFAULT_MAX_SPECTRA)
    args = ap.parse_args()
    main(
        args.fasta,
        args.mzml,
        args.out_dir,
        args.backend,
        args.top_k,
        args.precursor_tol_da,
        args.fragment_tol_ppm,
        args.report_psms,
        args.min_matched_peaks,
        args.max_spectra,
    )
