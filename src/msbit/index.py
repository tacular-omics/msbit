import logging
import math
from pathlib import Path
from typing import Literal

import numpy as np

from ._kernels import (
    _popcnt_array,
    _searchsorted_left,
    _searchsorted_right,
    bitmap_search_kernel,
    bitmap_search_kernel_topk,
)
from ._stats import _hypergeom_sf
from ._types import _EMPTY_SEARCH_RESULT, SearchResult

_log = logging.getLogger(__name__)

_THREADS_PER_BLOCK = 128


def _validate_filter(filter_array: np.ndarray, n: int) -> None:
    if len(filter_array) != n:
        raise ValueError(f"filter_array length {len(filter_array)} does not match db length {n}")
    if not np.all(np.diff(filter_array) >= 0):
        raise ValueError("filter_array must be sorted in ascending order")


def _get_range(
    filter_array: np.ndarray | None,
    n: int,
    min_val: float | None,
    max_val: float | None,
) -> tuple[int, int]:
    if min_val is None and max_val is None:
        return 0, n
    if filter_array is None:
        raise ValueError("min_val/max_val require a filter_array on the index")
    lo = int(_searchsorted_left(filter_array, float(min_val))) if min_val is not None else 0
    hi = int(_searchsorted_right(filter_array, float(max_val))) if max_val is not None else n
    return lo, hi


def _pvalues_for_topk(
    query: np.ndarray,
    db: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    n_bins = query.shape[0] * 64
    q = int(_popcnt_array(query))
    pvalues = np.empty(len(indices), dtype=np.float64)
    for j in range(len(indices)):
        d = int(_popcnt_array(db[indices[j]]))
        pvalues[j] = _hypergeom_sf(n_bins, d, q, int(scores[j]))
    return pvalues


def _pvalues_for_all(
    query: np.ndarray,
    db: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    n_bins = query.shape[0] * 64
    q = int(_popcnt_array(query))
    pvalues = np.empty(len(scores), dtype=np.float64)
    for j in range(len(scores)):
        d = int(_popcnt_array(db[j]))
        pvalues[j] = _hypergeom_sf(n_bins, d, q, int(scores[j]))
    return pvalues


def _apply_result_filters(
    indices: np.ndarray,
    scores: np.ndarray,
    pvalues: np.ndarray,
    min_hits: int | None,
    max_pvalue: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min_hits is None and max_pvalue is None:
        return indices, scores, pvalues
    mask = np.ones(len(indices), dtype=bool)
    if min_hits is not None:
        mask &= scores >= min_hits
    if max_pvalue is not None:
        mask &= pvalues <= max_pvalue
    return indices[mask], scores[mask], pvalues[mask]


class CpuBitmapIndex:
    def __init__(self, db: np.ndarray, filter_array: np.ndarray | None = None):
        """CPU bitmap index backed by Numba JIT kernels.

        filter_array, if provided, must be sorted ascending and the same length as db.
        Enables min_val/max_val range slicing via binary search.
        """
        if db.ndim != 2:
            raise ValueError(f"db must be 2-D, got shape {db.shape}")
        if db.dtype != np.uint64:
            raise ValueError(f"db must be uint64, got {db.dtype}")
        if db.shape[0] < 1:
            raise ValueError("db must have at least one row")
        if filter_array is not None:
            _validate_filter(filter_array, db.shape[0])
        self.db = db
        self.n = db.shape[0]
        self.filter = filter_array

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        min_val: float | None = None,
        max_val: float | None = None,
        min_hits: int | None = None,
        max_pvalue: float | None = None,
    ) -> SearchResult:
        """Return top-k results sorted descending by popcount overlap.

        Raises:
            ValueError: if min_val/max_val are passed without a filter_array.
        """
        if query.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {query.shape}")
        if query.shape[0] != self.db.shape[1]:
            raise ValueError(f"query width {query.shape[0]} does not match db width {self.db.shape[1]}")
        if query.dtype != np.uint64:
            raise ValueError(f"query must be uint64, got {query.dtype}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        start, end = _get_range(self.filter, self.n, min_val, max_val)
        if start >= end:
            return _EMPTY_SEARCH_RESULT
        indices, scores, pvalues = bitmap_search_kernel_topk(query, self.db[start:end], k)
        return SearchResult(
            *_apply_result_filters(
                (indices + start).astype(np.int32),
                scores,
                pvalues,
                min_hits,
                max_pvalue,
            )
        )

    def search_all(
        self,
        query: np.ndarray,
        min_val: float | None = None,
        max_val: float | None = None,
        min_hits: int | None = None,
        max_pvalue: float | None = None,
    ) -> SearchResult:
        """Return scores for all database entries in the optional mass range.

        Raises:
            ValueError: if min_val/max_val are passed without a filter_array.
        """
        if query.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {query.shape}")
        if query.shape[0] != self.db.shape[1]:
            raise ValueError(f"query width {query.shape[0]} does not match db width {self.db.shape[1]}")
        if query.dtype != np.uint64:
            raise ValueError(f"query must be uint64, got {query.dtype}")
        start, end = _get_range(self.filter, self.n, min_val, max_val)
        indices = np.arange(start, end, dtype=np.int32)
        if start >= end:
            return SearchResult(indices, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64))
        scores, pvalues = bitmap_search_kernel(query, self.db[start:end])
        return SearchResult(*_apply_result_filters(indices, scores, pvalues, min_hits, max_pvalue))

    def save(self, path: str | Path) -> None:
        """Save index to a compressed .npz file."""
        filter_data = self.filter if self.filter is not None else np.empty(0, dtype=np.float64)
        np.savez_compressed(path, db=self.db, filter=filter_data)

    @classmethod
    def load(cls, path: str | Path) -> "CpuBitmapIndex":
        """Load index from a .npz file saved by CpuBitmapIndex.save."""
        data = np.load(path)
        db = data["db"]
        filter_data = data["filter"]
        filter_array = filter_data if len(filter_data) > 0 else None
        return cls(db, filter_array)


class GpuBitmapIndex:
    def __init__(self, db: np.ndarray, filter_array: np.ndarray | None = None):
        """GPU bitmap index backed by CUDA kernels.

        Uploads db to GPU device memory on construction; kept resident across calls.
        Raises if CUDA is unavailable.
        """
        if db.ndim != 2:
            raise ValueError(f"db must be 2-D, got shape {db.shape}")
        if db.dtype != np.uint64:
            raise ValueError(f"db must be uint64, got {db.dtype}")
        if db.shape[0] < 1:
            raise ValueError("db must have at least one row")
        if filter_array is not None:
            _validate_filter(filter_array, db.shape[0])

        from numba import cuda  # lazy — avoids CUDA init in CPU-only environments

        from . import _cuda as _cuda_mod  # noqa: F401 — imported for side effects (kernel JIT)

        self._cuda_mod = _cuda_mod

        _log.debug("Uploading db (%d rows, %d words) to GPU device", db.shape[0], db.shape[1])
        self._db = db
        self.d_db = cuda.to_device(db)
        self.n = db.shape[0]
        self.filter = filter_array

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        min_val: float | None = None,
        max_val: float | None = None,
        min_hits: int | None = None,
        max_pvalue: float | None = None,
    ) -> SearchResult:
        """Return top-k results sorted descending by popcount overlap.

        Raises:
            ValueError: if min_val/max_val are passed without a filter_array.
        """
        if query.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {query.shape}")
        if query.shape[0] != self._db.shape[1]:
            raise ValueError(f"query width {query.shape[0]} does not match db width {self._db.shape[1]}")
        if query.dtype != np.uint64:
            raise ValueError(f"query must be uint64, got {query.dtype}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        from numba import cuda

        start, end = _get_range(self.filter, self.n, min_val, max_val)
        if start >= end:
            return _EMPTY_SEARCH_RESULT
        n_slice = end - start
        blocks = math.ceil(n_slice / _THREADS_PER_BLOCK)
        d_query = cuda.to_device(query)
        d_out_indices = cuda.device_array((blocks, k), dtype=np.int32)
        d_out_scores = cuda.device_array((blocks, k), dtype=np.int32)

        self._cuda_mod._bitmap_topk_kernel[blocks, _THREADS_PER_BLOCK](
            d_query, self.d_db[start:end], d_out_indices, d_out_scores, k
        )

        block_indices = d_out_indices.copy_to_host().ravel()
        block_scores = d_out_scores.copy_to_host().ravel()

        valid = block_indices >= 0
        block_indices = block_indices[valid]
        block_scores = block_scores[valid]

        n_candidates = len(block_scores)
        if n_candidates <= k:
            order = np.argsort(block_scores)[::-1]
            indices = block_indices[order].astype(np.int32)
            scores = block_scores[order].astype(np.int32)
        else:
            top_idx = np.argpartition(block_scores, -k)[-k:]
            top_idx = top_idx[np.argsort(block_scores[top_idx])[::-1]]
            indices = block_indices[top_idx].astype(np.int32)
            scores = block_scores[top_idx].astype(np.int32)

        indices = (indices + start).astype(np.int32)
        pvalues = _pvalues_for_topk(query, self._db, indices, scores)
        return SearchResult(*_apply_result_filters(indices, scores, pvalues, min_hits, max_pvalue))

    def search_all(
        self,
        query: np.ndarray,
        min_val: float | None = None,
        max_val: float | None = None,
        min_hits: int | None = None,
        max_pvalue: float | None = None,
    ) -> SearchResult:
        """Return scores for all database entries in the optional mass range.

        Raises:
            ValueError: if min_val/max_val are passed without a filter_array.
        """
        if query.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {query.shape}")
        if query.shape[0] != self._db.shape[1]:
            raise ValueError(f"query width {query.shape[0]} does not match db width {self._db.shape[1]}")
        if query.dtype != np.uint64:
            raise ValueError(f"query must be uint64, got {query.dtype}")

        from numba import cuda

        start, end = _get_range(self.filter, self.n, min_val, max_val)
        indices = np.arange(start, end, dtype=np.int32)
        n_slice = end - start
        if n_slice <= 0:
            return SearchResult(indices, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64))
        blocks = math.ceil(n_slice / _THREADS_PER_BLOCK)
        d_query = cuda.to_device(query)
        d_scores_slice = cuda.device_array(n_slice, dtype=np.int32)
        self._cuda_mod._bitmap_score_all_kernel[blocks, _THREADS_PER_BLOCK](
            d_query, self.d_db[start:end], d_scores_slice
        )
        scores = d_scores_slice.copy_to_host()
        pvalues = _pvalues_for_all(query, self._db[start:end], scores)
        return SearchResult(*_apply_result_filters(indices, scores, pvalues, min_hits, max_pvalue))

    def save(self, path: str | Path) -> None:
        """Save index to a compressed .npz file. GPU data is serialized from host memory."""
        filter_data = self.filter if self.filter is not None else np.empty(0, dtype=np.float64)
        np.savez_compressed(path, db=self._db, filter=filter_data)

    @classmethod
    def load(cls, path: str | Path) -> "GpuBitmapIndex":
        """Load index from a .npz file and upload to GPU."""
        data = np.load(path)
        db = data["db"]
        filter_data = data["filter"]
        filter_array = filter_data if len(filter_data) > 0 else None
        return cls(db, filter_array)


class BitmapIndex:
    def __init__(
        self,
        db: np.ndarray,
        backend: Literal["cpu", "gpu"] = "cpu",
        filter_array: np.ndarray | None = None,
    ):
        """Unified facade over CpuBitmapIndex and GpuBitmapIndex.

        backend must be 'cpu' or 'gpu'.

        Raises:
            ValueError: if backend is not 'cpu' or 'gpu'.
        """
        _log.debug("Using %s backend", backend)
        if backend == "gpu":
            self._impl: CpuBitmapIndex | GpuBitmapIndex = GpuBitmapIndex(db, filter_array)
        elif backend == "cpu":
            self._impl = CpuBitmapIndex(db, filter_array)
        else:
            raise ValueError(f"Unknown backend {backend!r}. Choose 'cpu' or 'gpu'.")

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        min_val: float | None = None,
        max_val: float | None = None,
        min_hits: int | None = None,
        max_pvalue: float | None = None,
    ) -> SearchResult:
        return self._impl.search(query, k, min_val, max_val, min_hits, max_pvalue)

    def search_all(
        self,
        query: np.ndarray,
        min_val: float | None = None,
        max_val: float | None = None,
        min_hits: int | None = None,
        max_pvalue: float | None = None,
    ) -> SearchResult:
        return self._impl.search_all(query, min_val, max_val, min_hits, max_pvalue)

    def save(self, path: str | Path) -> None:
        """Save the underlying index to a .npz file."""
        self._impl.save(path)

    @classmethod
    def load(cls, path: str | Path, backend: Literal["cpu", "gpu"] = "cpu") -> "BitmapIndex":
        """Load index from a .npz file with the specified backend."""
        if backend == "gpu":
            impl = GpuBitmapIndex.load(path)
        elif backend == "cpu":
            impl = CpuBitmapIndex.load(path)
        else:
            raise ValueError(f"Unknown backend {backend!r}. Choose 'cpu' or 'gpu'.")
        obj = object.__new__(cls)
        obj._impl = impl
        return obj
