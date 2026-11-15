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
peptides. Aggregate numbers in `paper/data/compare_cpu.json`. The original run
used uncommitted code, so its parameters were re-established by rerunning all
four candidate configurations on 2026-11-14; exactly one reproduces the
published numbers exactly, and it is the shipped default. `paper/README.md` has
the detail.

**It was 18× faster.** Median 0.45 ms per spectrum against sagepy presearch's
8.08 ms; 0.45 s against 10.29 s over the 1000 spectra.

**Its recall was too low to use.** Fraction of sagepy's own candidates recovered
in msbit's top K:

| K | 1 | 10 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| recall | 5.1% | 17.1% | 28.8% | 35.6% | 44.7% |

Recall here is agreement with sagepy's candidate generation, not with confirmed
identifications: `compare.py` divides by sagepy's own candidate set. That does
not rescue the result. A prefilter is only useful if what the scorer needs
survives it, so it has to keep almost everything it is handed, and at K=200, out
of a window of ~3,600 peptides, this one recovers less than half of the
candidates sagepy would have gone on to score. Spending 0.45 ms instead of 8 ms
buys nothing if the scorer never receives the peptide it would have picked.

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

## Second attempt: one sketch per run, also negative

The same bit vectors were tried at run level: one fixed 8 kB sketch per LC-MS run,
compared by popcount, to tell runs apart without identification. Within one
experiment it worked. In-house replicate pairs separated from other preparations
of the same material at AUC 0.985, against 0.82 from sketch size alone. Across
instruments it did not. Of 351 PXD034932 pairs, compareMS2 found matches for 138;
on those the Spearman rho was -0.32, and +0.12 over all 351, where 0.90 was fixed
in advance. The fraction ordering of PXD001468 was not recovered either. The
measured cause is the 0.01 Da fragment bins, far finer than ion-trap accuracy, so
two ion-trap runs of one sample share almost no tokens. compareMS2 also scores a
pair in 2.4 s, leaving little speed to win.

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

Also `just lint` and `just ty`. The comparison needs the `paper` extra and your
own FASTA and mzML; `paper/README.md` has the commands. MIT licensed, see
`LICENSE`.
