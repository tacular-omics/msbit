# Paper: msbit vs sagepy presearch

End-to-end reproduction of the application-note experiment comparing
msbit's bitmap presearch against sagepy's candidate generation on the
same FASTA + mzML.

## Reproduce

```bash
just paper-setup        # install [paper] extra, fetch fasta + test mzml
just paper-build-db     # FASTA -> msbit BitmapIndex
just paper-run-msbit    # mzML -> msbit top-K candidates + timing
just paper-run-sagepy   # mzML -> sagepy top-K candidates + timing
just paper-compare      # recall@K + timing -> compare.json
just paper-figures      # final manuscript figure

# Or just:
just paper-all
```

## Layout

| File | Purpose |
|---|---|
| `digest.py` | Tryptic digest from FASTA (own implementation, no pyteomics) |
| `fragments.py` | Monoisotopic peptide + b/y ion masses |
| `build_db.py` | FASTA -> BitmapIndex `.npz` + peptide/mass arrays |
| `run_msbit.py` | mzML -> msbit candidates + per-spectrum timing |
| `run_sagepy.py` | mzML -> sagepy presearch candidates + timing |
| `compare.py` | recall@K curves + timing aggregate |
| `figures.py` | Final figure for the manuscript |
| `download.py` | Fetch FASTA + test mzML |
| `data/` | Inputs and intermediates (gitignored) |

## Experimental parameters

- Trypsin, max 2 missed cleavages, peptide length 7–50.
- Fixed mod: carbamidomethyl C (+57.02146 Da).
- Bitmap: log-spaced [100, 2000] Da, `array_length=100` (6,400 bins), 20 ppm.
- Precursor tolerance: 20 ppm.
- Top-K compared: K ∈ {1, 10, 50, 100, 200}.

## Status

Skeleton only — every module raises `NotImplementedError`.
