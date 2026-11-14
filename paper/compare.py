"""Compute recall@K + timing aggregates from msbit + sagepy outputs.

Both outputs are joined on `scan_id`. For each spectrum:
  sagepy_topN = sagepy's top-N peptides ranked by hyperscore
  msbit_topK  = msbit's top-K peptides ranked by bitmap fragment overlap

We report:
  - window_size distribution (number of candidates sagepy scored)
  - recall@K = |msbit_topK ∩ sagepy_topN| / |sagepy_topN|, averaged over spectra
               i.e. what fraction of sagepy's hyperscore-ranked candidates does
               msbit's bitmap ranking recover in its top K?
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

    recall: dict[int, float] = {}
    for k in KS:
        per_row = []
        for row in joined.iter_rows(named=True):
            sagepy_top = row["candidates_sagepy"]
            if not sagepy_top:
                continue
            msbit_top_k = set(row["candidates"][:k])
            per_row.append(len(msbit_top_k & set(sagepy_top)) / len(sagepy_top))
        recall[k] = sum(per_row) / max(1, len(per_row))

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
        "recall_at_k": recall,
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
