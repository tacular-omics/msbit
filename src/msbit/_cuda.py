import math

import numba
import numpy as np
from numba import cuda


@cuda.jit(device=True)
def _popcnt_cuda(x: np.uint64) -> np.uint64:
    # CUDA has __popc for 32-bit and __popcll for 64-bit
    return cuda.libdevice.popcll(x)  # ty: ignore


@cuda.jit(device=True)
def _bitmap_score_cuda(db_row, query_row, length: int) -> np.int32:
    count = np.uint64(0)
    for w in range(length):
        count += _popcnt_cuda(db_row[w] & query_row[w])
    return np.int32(count)


@cuda.jit
def _bitmap_topk_kernel(
    query_bitmap,  # (array_length,) uint64
    db_bitmaps,  # (N, array_length) uint64
    out_indices,  # (n_blocks, top_k) int32
    out_scores,  # (n_blocks, top_k) int32
    top_k: int,
):
    array_length = query_bitmap.shape[0]
    tid: int = cuda.threadIdx.x  # type: ignore[assignment]
    bid: int = cuda.blockIdx.x  # type: ignore[assignment]
    i: int = cuda.grid(1)  # type: ignore[assignment]

    # cache query in shared memory
    shared_query = cuda.shared.array(shape=(512,), dtype=numba.uint64)  # ty: ignore
    if tid < array_length:
        shared_query[tid] = query_bitmap[tid]

    # per-block top-k buffers in shared memory
    shared_indices = cuda.shared.array(shape=(1024,), dtype=numba.int32)  # ty: ignore
    shared_scores = cuda.shared.array(shape=(1024,), dtype=numba.int32)  # ty: ignore
    if tid < top_k:
        shared_indices[tid] = -1
        shared_scores[tid] = 0
    cuda.syncthreads()  # ty: ignore

    # score this thread's row
    score = numba.int32(0)
    if i < db_bitmaps.shape[0]:
        count = numba.uint64(0)
        for w in range(array_length):
            count += _popcnt_cuda(db_bitmaps[i, w] & shared_query[w])
        score = numba.int32(count)

    # warp-level reduction to find top scores within the warp
    # each warp elects its best candidate via shuffle
    warp_score = score
    warp_idx = i
    for offset in (16, 8, 4, 2, 1):
        other_score = cuda.shfl_down_sync(0xFFFFFFFF, warp_score, offset)  # ty: ignore
        other_idx = cuda.shfl_down_sync(0xFFFFFFFF, warp_idx, offset)  # ty: ignore
        if other_score > warp_score:
            warp_score = other_score
            warp_idx = other_idx

    # lane 0 of each warp writes its best into shared top-k
    lane = tid % 32
    if lane == 0 and i < db_bitmaps.shape[0]:
        # atomic insert into shared top-k — only top_k slots
        # simple: just write all warp winners, sort later
        warp_id = tid // 32
        if warp_id < top_k:
            shared_scores[warp_id] = warp_score
            shared_indices[warp_id] = warp_idx
    cuda.syncthreads()  # ty: ignore

    # thread 0 writes block's top-k to global output
    if tid < top_k:
        out_scores[bid, tid] = shared_scores[tid]
        out_indices[bid, tid] = shared_indices[tid]


@cuda.jit
def _bitmap_score_all_kernel(
    query_bitmap,  # (array_length,) uint64
    db_bitmaps,  # (N, array_length) uint64
    out_scores,  # (N,) int32
):
    array_length = query_bitmap.shape[0]
    tid: int = cuda.threadIdx.x  # type: ignore[assignment]
    i: int = cuda.grid(1)  # type: ignore[assignment]

    shared_query = cuda.shared.array(shape=(512,), dtype=numba.uint64)  # ty: ignore
    if tid < array_length:
        shared_query[tid] = query_bitmap[tid]
    cuda.syncthreads()  # ty: ignore

    if i < db_bitmaps.shape[0]:
        out_scores[i] = _bitmap_score_cuda(db_bitmaps[i], shared_query, array_length)


def bitmap_search_topk_gpu(
    query_bitmap: np.ndarray,
    db_bitmaps: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = db_bitmaps.shape[0]
    threads_per_block = 128
    blocks = math.ceil(n / threads_per_block)

    d_query = cuda.to_device(query_bitmap)
    d_out_indices = cuda.device_array((blocks, top_k), dtype=np.int32)
    d_out_scores = cuda.device_array((blocks, top_k), dtype=np.int32)

    _bitmap_topk_kernel[blocks, threads_per_block](d_query, db_bitmaps, d_out_indices, d_out_scores, top_k)

    # reduce block results on CPU — only (blocks * top_k) elements, tiny
    block_indices = d_out_indices.copy_to_host().ravel()
    block_scores = d_out_scores.copy_to_host().ravel()

    valid = block_indices >= 0
    block_indices = block_indices[valid]
    block_scores = block_scores[valid]

    top_idx = np.argpartition(block_scores, -min(top_k, len(block_scores)))[-top_k:]
    top_idx = top_idx[np.argsort(block_scores[top_idx])[::-1]]

    return block_indices[top_idx].astype(np.int32), block_scores[top_idx].astype(np.int32)
