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
python run.py          # CPU benchmark
python gpu_bench.py    # GPU benchmark
python plot_results.py # Visualize results
```
