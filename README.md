# msbit

A fragment-ion prefilter for peptide identification that was fast and did not
work well enough. This repo is kept as a record of the measurement, not as a
maintained package. It has never been on PyPI and there is no release.

## The idea

Encode each peptide's predicted b and y fragment ions as a bit vector: log-spaced
bins over [200, 5000] Da, 100 uint64 words per peptide (6,400 bins), one bit set
per fragment within 20 ppm. Encode a query spectrum the same way. Then the
fragment overlap between a spectrum and a peptide is the popcount of the AND of
the two vectors, which is one hardware instruction per 64 candidates. Rank by
that and hand the top K to a real scorer. Backends: Numba JIT on CPU, CUDA on
GPU.

## What was measured

`paper/` runs msbit and sagepy's own candidate generation over a single
in-memory sagepy database, so both see the same digest, the same b/y fragment
m/z arrays, and identical peptide indices. 1000 spectra of an in-house DDA run,
open search at ±500 Da precursor tolerance, mean candidate window 3,636
peptides. Aggregate numbers in `paper/data/compare_cpu.json`.

**It was 18× faster.** Median 0.45 ms per spectrum against sagepy presearch's
8.08 ms; 0.45 s against 10.29 s over the 1000 spectra.

**Its recall was too low to use.** Fraction of sagepy's own candidates recovered
in msbit's top K:

| K | 1 | 10 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| recall | 5.1% | 17.1% | 28.8% | 35.6% | 44.7% |

A prefilter is only useful if the answer survives it, so it has to keep almost
everything it is handed. This one loses more than half the right answers even at
K=200, out of a window of ~3,600. Spending 0.45 ms instead of 8 ms does not buy
anything if the subsequent scorer no longer has the right peptide to score.

**The GPU lost to the CPU.** Best throughput 609 M comparisons/s on the GPU
against 699 M/s on the CPU, both at the same configuration (10 uint64 words,
500k peptides) — `benchmark_results_gpu.csv` and `benchmark_results.csv`.
Popcount is close to free, so the kernel is bound by reading the bitmap matrix
out of memory, and the card's bandwidth advantage is spent on getting the data
there. There is also no batched-query API, so per-call launch overhead (~0.4 ms)
is charged to every spectrum.

## Why the idea has a narrow target

A prefilter only has work to do when the candidate window is large. At a normal
narrow precursor tolerance, the window is 1 to 10 peptides, and scoring 10
candidates exactly costs less than approximating them. So the only setting where
this could pay off is open search, where the window is thousands of peptides —
and that is the setting where the recall above was measured.

## Layout

| Path | What |
|---|---|
| `src/msbit/` | Bitmap encoding, CPU and CUDA search kernels, `BitmapIndex` |
| `paper/` | The sagepy comparison, with its parameters and result; see `paper/README.md` |
| `bench.py` | Throughput sweep over bitmap width and database size |
| `benchmark_results*.csv` | Recorded sweeps, CPU and GPU |

## Running it

```bash
git clone https://github.com/tacular-omics/msbit
cd msbit
just install        # uv sync
just test
```

`just lint`, `just ty`, `just test`. The comparison needs the `paper` extra and
your own FASTA and mzML; `paper/README.md` has the commands.

## License

MIT. See `LICENSE`.
