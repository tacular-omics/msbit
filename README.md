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

Spectra scored per second for a single query against a peptide database of `N` entries, top-10 search. Numbers are `1 / time_per_query_s` from `benchmark_results.csv` / `benchmark_results_gpu.csv` (best of 3 trials, 10 queries each). Treat as order-of-magnitude — actual throughput depends on hardware, CUDA driver, and bitmap density.

Bitmap width = 50 × uint64 (3,200 bits per spectrum, ~19 MB per 100k peptides):

| Database size | CPU (spectra/s) | GPU (spectra/s) |
|---:|---:|---:|
|  10,000  | ~6,200 | ~2,500 |
|  50,000  | ~3,200 | ~1,300 |
| 100,000  | ~1,600 | ~900   |
| 250,000  | ~550   | ~520   |
| 500,000  | ~300   | ~290   |

Wider bitmaps (more ion bins) reduce throughput roughly linearly; narrower bitmaps run faster. At this array size the CPU (Numba popcount) is competitive with the GPU because kernel-launch overhead dominates per-query cost — the GPU wins more clearly on wider bitmaps and large query batches via `GpuBitmapIndex`. The full sweep across `array_size ∈ {10, 25, 50, 100, 200, 500}` and `mat_size ∈ {10k … 500k}` is in `benchmark_results.csv` and visualised in `benchmark_results.png`.
