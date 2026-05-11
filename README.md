# msbit

Bitmap-based similarity search for peptide mass spectrometry. Encodes fragmentation ions (b/y) into bit vectors and uses hardware popcount for fast approximate matching against large peptide databases — with both CPU (Numba JIT) and GPU (CUDA) backends.

## Installation

```bash
pip install msbit
```

For development:

```bash
just install
```

## Usage

```python
import numpy as np
from msbit import MsBit
from msbit.bitmap import generate_bitmaps

# Generate bitmaps for a peptide database (n_peptides × array_size uint64s)
db = MsBit.random_bitmap_matrix(array_size=50, n_rows=100_000)

# Encode a query spectrum into a bitmap
query = db[42]  # or generate_bitmaps(ion_masses, array_size=50)

# Full search — returns scores for all database entries
scores = MsBit().search(query, db)

# Top-K search on CPU
indices, scores = MsBit.search_topk(query, db, k=10)

# Top-K search on GPU (requires CUDA)
indices, scores = MsBit.search_topk(query, db, k=10, gpu=True)
```

### GPU index for repeated queries

```python
from msbit import GpuBitmapIndex

index = GpuBitmapIndex(db)  # uploads db to GPU once
for query in queries:
    indices, scores = index.search_topk(query, top_k=10)
```

### Generating bitmaps from ion masses

```python
from msbit.bitmap import generate_bitmaps

# ion_masses: 1D array of m/z values (Da)
bitmaps = generate_bitmaps(ion_masses, array_size=50, tolerance=0.02, tolerance_mode="Da")
```

## Development

```bash
just lint      # ruff check
just format    # ruff format
just check     # type check (ty)
just test      # pytest
```

## Benchmarks

```bash
python bench.py        # runs CPU + GPU (if CUDA), writes benchmark_results.csv / .png
```

### Ballpark throughput

Two metrics matter and they're easy to confuse:

- **spectra/s** — how many query spectra you can score per second against a fixed DB. This is `1 / time_per_query_s`. It's what dictates "how long will my search take?".
- **M comparisons/s** — millions of peptide×spectrum scores per second, i.e. `mat_size / time_per_query_s / 1e6`. This is the raw bitmap-matching throughput and is what's stored as `throughput_M_per_s` in the CSVs.

They're related by DB size: `spectra/s = (M comparisons/s × 1e6) / mat_size`. So 555 M comparisons/s against a 100k DB = 5,550 underlying scores done per millisecond, which works out to ~1,600 whole spectra processed per second. Both numbers are real; they answer different questions.

Single-query top-10 search, bitmap width = 50 × uint64 (3,200 bits per spectrum, ~19 MB per 100k peptides), best of 3 trials × 10 queries:

| DB size | CPU spectra/s | CPU M comp/s | GPU spectra/s | GPU M comp/s |
|---:|---:|---:|---:|---:|
|  10,000  | ~6,200 |  ~62 | ~2,500 |  ~25 |
|  50,000  | ~3,200 | ~159 | ~1,300 |  ~64 |
| 100,000  | ~1,600 | ~158 |   ~900 |  ~91 |
| 250,000  |   ~550 | ~137 |   ~520 | ~129 |
| 500,000  |   ~300 | ~151 |   ~290 | ~147 |

A few caveats worth knowing before reading too much into these:

- `bench.py` calls `index.search(q)` in a plain Python `for` loop — per-call dispatch overhead is included. There is currently no batched-query API, so GPU launch overhead (~0.4 ms per call) dominates at this scale. A batched kernel would substantially raise GPU spectra/s.
- Queries are drawn from the database itself (`bitmaps[:N_QUERIES]`), so query popcount matches DB popcount exactly. Real spectra are typically sparser.
- Treat as order-of-magnitude — actual throughput depends on hardware, CUDA driver, and bitmap density.

Wider bitmaps (more ion bins) reduce spectra/s roughly linearly. The full sweep across `array_size ∈ {10, 25, 50, 100, 200, 500}` and `mat_size ∈ {10k … 500k}` is in `benchmark_results.csv` and visualised in `benchmark_results.png`.
