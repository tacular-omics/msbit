"""Draft figure: window-size distribution, recall@K, timing bars.

Final manuscript figure will need a real-data run to be meaningful;
this just lays out the visual structure so the figure exists and
updates automatically when compare.py is rerun.

Usage: python -m paper.figures <compare.json> <out.png>
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main(compare_path: Path, out_path: Path) -> None:
    summary = json.loads(compare_path.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ws = summary["window_size"]
    axes[0].bar(["min", "median", "mean", "p95", "max"], [ws["min"], ws["median"], ws["mean"], ws["p95"], ws["max"]])
    axes[0].set_title("Precursor-window size")
    axes[0].set_ylabel("# candidate peptides")

    rec = summary["recall_at_k"]
    ks = sorted(int(k) for k in rec.keys())
    axes[1].plot(ks, [rec[str(k)] for k in ks], marker="o")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("K (msbit top-K)")
    axes[1].set_ylabel("recall@K (mean over spectra)")
    axes[1].set_title("msbit top-K recall of sagepy hyperscore ranking")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)

    msbit_t = summary["msbit_time_s"]["median"] * 1000
    sagepy_t = summary["sagepy_time_s"]["median"] * 1000
    axes[2].bar(["msbit\n(bitmap top-K)", "sagepy\n(full scoring)"], [msbit_t, sagepy_t])
    axes[2].set_ylabel("median time per spectrum (ms)")
    axes[2].set_title(f"Presearch latency  (n={summary['n_spectra']} spectra)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[figures] wrote {out_path}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
