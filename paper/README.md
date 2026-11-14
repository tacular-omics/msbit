# Paper: msbit vs sagepy presearch

End-to-end comparison of msbit's bitmap presearch against sagepy's candidate
generation, on the same FASTA and the same mzML, through the same peptide and
fragment universe.

## Reproduce

The real inputs used for the recorded numbers are not in this repo. Point the
justfile at your own FASTA and mzML:

```bash
export MSBIT_PAPER_FASTA=/path/to/proteome.fasta
export MSBIT_PAPER_MZML=/path/to/run.mzML.gz
```

```bash
just paper-setup        # install the [paper] extra (sagepy, spxtacular)
just paper-download     # smoke-test inputs into paper/data (skip if you set the vars above)
just paper-run          # mzML -> msbit + sagepy candidates, per-spectrum timing
just paper-compare      # recall@K + timing -> paper/data/compare_cpu.json
just paper-figures      # figure

# Or all of it:
just paper-all
```

## Layout

| File | Purpose |
|---|---|
| `sage_db.py` | Wraps `SageSearchConfiguration` -> `IndexedDatabase`; shared digest + fragment universe for both pipelines |
| `run_comparison.py` | Builds one DB, runs both presearches over it, writes `msbit.parquet` + `sagepy.parquet` with per-spectrum timing |
| `build_db.py` | sagepy DB -> standalone msbit `BitmapIndex` `.npz` + peptide/mass arrays |
| `compare.py` | recall@K curves + timing aggregate -> `compare_<backend>.json` |
| `figures.py` | Figure from that JSON |
| `download.py` | Fetch the smoke-test FASTA + mzML |
| `data/` | Inputs and intermediates (gitignored, except `compare_cpu.json`) |

Both presearches are built from one in-memory sagepy database object, so peptide
indices are identical on both sides and the recall comparison is index-safe.
Sagepy provides the digest and the b/y fragment m/z arrays — no hand-rolled mass
tables.

## Experimental parameters

Taken from the code, not from notes: `sage_db.py` (`SageParams`) and
`run_comparison.py` (module constants and CLI defaults).

- Digest: cleave C-terminally of R, not before P (ArgC), max 2 missed cleavages,
  peptide length 7–50, peptide mass 500–5000 Da, no decoys.
- Static modifications: none. `SageParams.static_mods` defaults to empty.
- Bitmap: log-spaced bins over [200, 5000] Da, `array_length=100` uint64 words
  (6,400 bins), 20 ppm fragment tolerance.
- **Precursor tolerance: ±500 Da (open search.)** At a narrow ppm tolerance the
  candidate window is 1–10 peptides, so there is nothing for any prefilter to
  do; the prefilter only has a job at open-search widths. Override with
  `--precursor-tol-da`.
- msbit top-K = 200; sagepy reports its top 200 by hyperscore; minimum 5 matched
  peaks; first 1000 spectra of the run.
- recall@K reported at K ∈ {1, 10, 50, 100, 200} (`compare.py`, `KS`).

## Result

`data/compare_cpu.json`, 1000 spectra, mean candidate window 3,636 peptides:

- msbit median 0.45 ms per spectrum against sagepy presearch 8.08 ms — 18× faster
  (totals 0.45 s against 10.29 s).
- Recall of sagepy's own candidates: 5.1% @1, 17.1% @10, 28.8% @50, 35.6% @100,
  44.7% @200.

The speed is real and the recall is not good enough: a prefilter has to keep
almost everything, and this one loses more than half the right answers even at
K=200.
