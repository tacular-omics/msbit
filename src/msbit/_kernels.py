import llvmlite
import llvmlite.ir
import numba
import numpy as np
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

from ._stats import _hypergeom_sf


@intrinsic
def _hardware_popcnt(typingctx, x):
    """Calls the llvm.ctpop.i64 intrinsic — guaranteed hardware POPCNT."""

    def codegen(context, builder, signature, args):
        i64 = args[0].type
        fn = cgutils.get_or_insert_function(
            builder.module,
            llvmlite.ir.FunctionType(i64, [i64]),
            "llvm.ctpop.i64",
        )
        return builder.call(fn, args)

    sig = types.uint64(types.uint64)
    return sig, codegen


@numba.njit(cache=True)
def _popcnt_kernighan(x: np.uint64) -> np.uint64:
    """Kernighan bit-count — LLVM lowers to hardware POPCNT."""
    count = np.uint64(0)
    while x:
        x &= x - np.uint64(1)
        count += np.uint64(1)
    return count


@numba.njit(cache=True)
def _popcnt(x: np.uint64) -> np.uint64:
    return _hardware_popcnt(x)  # type: ignore


@numba.njit(cache=True)
def _bitmap_score(db_row: np.ndarray, query_row: np.ndarray) -> np.int32:
    """Popcount of AND between two bitmap rows."""
    count = np.uint64(0)
    for w in range(db_row.shape[0]):
        count += _popcnt(db_row[w] & query_row[w])
    return np.int32(count)


@numba.njit(cache=True)
def _popcnt_array(arr: np.ndarray) -> int:
    """Total number of set bits across all words in a bitmap row."""
    count = 0
    for w in range(arr.shape[0]):
        count += _popcnt(arr[w])
    return count


@numba.njit(cache=True)
def _searchsorted_left(arr: np.ndarray, val: float) -> int:
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.njit(cache=True)
def _searchsorted_right(arr: np.ndarray, val: float) -> int:
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] <= val:
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.njit(cache=True)
def _insert_topk(
    top_indices: np.ndarray,  # (top_k,) int32  — maintained sorted descending by score
    top_scores: np.ndarray,  # (top_k,) int32
    idx: int,
    score: np.int32,
    filled: int,  # how many slots are occupied so far
) -> int:
    """
    Insert (idx, score) into top-k buffers if score beats the current minimum.
    Returns updated filled count.
    Uses insertion sort — top_k is small (<=20) so this is fine.
    """
    if filled < top_scores.shape[0]:
        # buffer not full yet — insert and sort
        pos = filled
        while pos > 0 and top_scores[pos - 1] < score:
            top_scores[pos] = top_scores[pos - 1]
            top_indices[pos] = top_indices[pos - 1]
            pos -= 1
        top_scores[pos] = score
        top_indices[pos] = idx
        return filled + 1
    elif score > top_scores[filled - 1]:
        # beats current minimum — replace last slot and re-sort
        pos = filled - 1
        while pos > 0 and top_scores[pos - 1] < score:
            top_scores[pos] = top_scores[pos - 1]
            top_indices[pos] = top_indices[pos - 1]
            pos -= 1
        top_scores[pos] = score
        top_indices[pos] = np.int32(idx)
    return filled


@numba.njit(parallel=True, cache=True)
def bitmap_search_kernel(
    query_bitmap: np.ndarray,  # (array_length,) uint64
    db_bitmaps: np.ndarray,  # (N, array_length) uint64
) -> tuple[np.ndarray, np.ndarray]:  # (N,) int32 scores, (N,) float64 pvalues
    n = db_bitmaps.shape[0]
    n_bins = query_bitmap.shape[0] * 64
    q = _popcnt_array(query_bitmap)
    scores = np.zeros(n, dtype=np.int32)
    pvalues = np.ones(n, dtype=np.float64)
    for i in numba.prange(n):  # type: ignore
        score = _bitmap_score(db_bitmaps[i], query_bitmap)
        scores[i] = score
        d = _popcnt_array(db_bitmaps[i])
        pvalues[i] = _hypergeom_sf(n_bins, d, q, int(score))
    return scores, pvalues


@numba.njit(parallel=True, cache=True)
def bitmap_search_kernel_topk(
    query_bitmap: np.ndarray,  # (array_length,) uint64
    db_bitmaps: np.ndarray,  # (N, array_length) uint64
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = db_bitmaps.shape[0]
    n_threads = numba.get_num_threads()
    chunk_size = (n + n_threads - 1) // n_threads
    array_length = query_bitmap.shape[0]

    thread_indices = np.full((n_threads, top_k), -1, dtype=np.int32)
    thread_scores = np.zeros((n_threads, top_k), dtype=np.int32)
    thread_filled = np.zeros(n_threads, dtype=np.int32)

    for t in numba.prange(n_threads):  # type: ignore
        start = t * chunk_size
        end = min(start + chunk_size, n)

        # copy query into thread-local buffer — avoids false sharing and
        # repeated global memory fetch on every _bitmap_score call
        local_query = np.empty(array_length, dtype=np.uint64)
        for w in range(array_length):
            local_query[w] = query_bitmap[w]

        local_indices = thread_indices[t]
        local_scores = thread_scores[t]
        filled = 0
        min_score = np.int32(0)  # current worst score in top-k

        for i in range(start, end):
            # inline _bitmap_score to avoid function call overhead
            count = np.uint64(0)
            for w in range(array_length):
                count += _popcnt(db_bitmaps[i, w] & local_query[w])
            score = np.int32(count)

            # skip insert entirely if top-k is full and score can't beat minimum
            if filled == top_k and score <= min_score:
                continue

            filled = _insert_topk(local_indices, local_scores, i, score, filled)

            # update min_score threshold once buffer is full
            if filled == top_k:
                min_score = local_scores[top_k - 1]

        thread_filled[t] = filled

    # serial reduction
    global_indices = np.full(top_k, -1, dtype=np.int32)
    global_scores = np.zeros(top_k, dtype=np.int32)
    filled = 0
    for t in range(n_threads):
        for k in range(thread_filled[t]):
            filled = _insert_topk(global_indices, global_scores, thread_indices[t, k], thread_scores[t, k], filled)

    # compute p-values for the final top-k entries
    n_bins = query_bitmap.shape[0] * 64
    q = _popcnt_array(query_bitmap)
    pvalues = np.ones(top_k, dtype=np.float64)
    for j in range(top_k):
        idx = global_indices[j]
        if idx >= 0:
            d = _popcnt_array(db_bitmaps[idx])
            pvalues[j] = _hypergeom_sf(n_bins, d, q, int(global_scores[j]))

    return global_indices, global_scores, pvalues


@numba.njit(parallel=True, cache=True)
def windowed_bitmap_search_kernel(
    db_bitmaps: np.ndarray,  # (n_peptides, array_length) uint64, sorted by precursor mass
    db_masses: np.ndarray,  # (n_peptides,) float32, sorted ascending
    query_bitmaps: np.ndarray,  # (n_spectra, array_length) uint64
    query_windows: np.ndarray,  # (n_windows, 3) float32 (lower mass window, upper mass window, spx_index)
    spec_idxs: np.ndarray,  # (n_windows, 1) int32 — spectrum index for each window
    out_window_idx: np.ndarray,  # (n_windows * top_k,) int32
    out_idx: np.ndarray,  # (n_windows * top_k,) int32
    out_scores: np.ndarray,  # (n_windows * top_k,) int32
    top_k: int,
) -> None:
    for i in numba.prange(len(query_windows)):  # type: ignore
        base = i * top_k

        # pre-fill window_idx for this spectrum's slots
        for k in range(top_k):
            out_window_idx[base + k] = np.int32(i)

        # precursor mass window
        lo = _searchsorted_left(db_masses, query_windows[i, 0])
        hi = _searchsorted_right(db_masses, query_windows[i, 1])
        spec_idx = spec_idxs[i]

        if lo >= hi:
            continue

        # per-thread top-k buffers (stack allocated — top_k must be small)
        top_indices = np.full(top_k, -1, dtype=np.int32)
        top_scores = np.zeros(top_k, dtype=np.int32)
        filled = 0

        for j in range(lo, hi):
            score = _bitmap_score(db_bitmaps[j], query_bitmaps[spec_idx])
            filled = _insert_topk(top_indices, top_scores, j, score, filled)

        # write results into output arrays
        for k in range(filled):
            out_idx[base + k] = top_indices[k]
            out_scores[base + k] = top_scores[k]
