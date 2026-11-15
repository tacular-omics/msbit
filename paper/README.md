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

The published numbers were produced by `run_comparison.py`, and the parameters
below are read from it and from `sage_db.py` (`SageParams`), not from notes.

**Established by rerun on 2026-11-14, not by reading the repository.** The run
that produced `data/compare_cpu.json` on 2026-05-12 used code that was never
committed, so its bitmap range and static-modification setting were recorded
nowhere and the file timestamps actively mislead: a `data/db/params.json` left
behind that morning records a different path, with carbamidomethyl C and a
[100, 2000] Da bitmap, and an earlier pipeline of `run_msbit.py` and
`run_sagepy.py` existed at that point and was later deleted. To settle it, all
four combinations of bitmap range {[100, 2000], [200, 5000]} and static mods
{Cam-C, none} were rerun on the original inputs. Exactly one reproduces the
published result, exactly, at full float precision on every recall value and
every window statistic: **[200, 5000] Da with no static modifications**, which
are the current `run_comparison.py` defaults. So the code in this repository
today reproduces its own published numbers, and the parameters below are the
measured ones.

`build_db.py` remains a second, standalone path with different constants of its
own (log-spaced bins over [100, 2000] Da at `build_db.py:21-23`) and produced
nothing reported here.

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
- Produced with sagepy 0.4.5 and spxtacular 0.3.0, the versions the `[paper]`
  extra pins.

## Result

`data/compare_cpu.json`, 1000 spectra, mean candidate window 3,636 peptides:

- msbit median 0.45 ms per spectrum against sagepy presearch 8.08 ms — 18× faster
  (totals 0.45 s against 10.29 s).
- Recall of sagepy's own candidates: 5.1% @1, 17.1% @10, 28.8% @50, 35.6% @100,
  44.7% @200.

Recall here is agreement with sagepy's candidate generation, not with confirmed
identifications: `compare.py` divides by sagepy's own candidate set. The speed
is real and the recall is still not good enough — a prefilter has to keep almost
everything it is handed, and at K=200 this one recovers less than half of the
candidates sagepy would have gone on to score.
