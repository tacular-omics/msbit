"""Compute coverage@K + timing aggregates from msbit + sagepy outputs.

Both outputs are joined on `scan_id`. For each spectrum:
  window      = sagepy's precursor-window candidate set
  msbit_topK  = first K of msbit's bitmap-ranked candidates

We report:
  - window_size distribution
  - coverage@K = |msbit_topK ∩ window| / |window|, averaged over spectra
                 (saturates at min(K, |window|) / |window|)
  - timing: median + total per pipeline

Usage: python -m paper.compare <msbit.parquet> <sagepy.parquet> <out.json>
"""

import json
import sys
from pathlib import Path

import polars as pl

KS = (1, 10, 50, 100, 200)


def main(msbit_path: Path, sagepy_path: Path, out_path: Path) -> None:
    m = pl.read_parquet(msbit_path)
    s = pl.read_parquet(sagepy_path)
    joined = m.join(s, on="scan_id", how="inner", suffix="_sagepy")
    if not len(joined):
        raise SystemExit("[compare] no overlapping scans between msbit and sagepy outputs")

    coverage: dict[int, float] = {}
    for k in KS:
        per_row = []
        for row in joined.iter_rows(named=True):
            window = set(row["candidates_sagepy"])
            if not window:
                continue
            top_k = set(row["candidates"][:k])
            per_row.append(len(top_k & window) / len(window))
        coverage[k] = sum(per_row) / max(1, len(per_row))

    window_sizes = joined["window_size"].to_list()
    summary = {
        "n_spectra": len(joined),
        "window_size": {
            "mean": float(joined["window_size"].mean() or 0),
            "median": float(joined["window_size"].median() or 0),
            "max": int(joined["window_size"].max() or 0),
            "min": int(joined["window_size"].min() or 0),
            "p95": float(pl.Series(window_sizes).quantile(0.95) or 0),
        },
        "coverage_at_k": coverage,
        "msbit_time_s": {
            "median": float(joined["time_s"].median() or 0),
            "total": float(joined["time_s"].sum() or 0),
        },
        "sagepy_time_s": {
            "median": float(joined["time_s_sagepy"].median() or 0),
            "total": float(joined["time_s_sagepy"].sum() or 0),
        },
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
