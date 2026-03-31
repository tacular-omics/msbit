import math

import numpy as np
import pytest

from msbit._kernels import (
    _bitmap_score,
    _popcnt,
    bitmap_search_kernel,
    bitmap_search_kernel_topk,
)
from msbit._stats import _hypergeom_sf
from msbit.encoding import generate_bitmaps
from msbit.index import BitmapIndex, CpuBitmapIndex, GpuBitmapIndex

CUDA_AVAILABLE = False
try:
    from numba import cuda

    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LOG_MIN = math.log(100.0)
LOG_MAX = math.log(5000.0)
ARRAY_LENGTH = 16
N_BINS = ARRAY_LENGTH * 64


def _log_width(n_bins: int = N_BINS) -> float:
    return (math.log(5000.0) - math.log(100.0)) / n_bins


@pytest.fixture
def log_width():
    return _log_width()


@pytest.fixture
def simple_ion_matrix():
    """3 peptides, 5 residues each with known masses."""
    return np.array(
        [
            [200.0, 400.0, 600.0, 800.0, 1000.0],
            [150.0, 300.0, 450.0, 600.0, 750.0],
            [250.0, 500.0, 750.0, 1000.0, 1250.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def random_bitmaps():
    rng = np.random.default_rng(42)
    return rng.integers(0, np.iinfo(np.uint64).max, size=(1000, ARRAY_LENGTH), dtype=np.uint64)


# ---------------------------------------------------------------------------
# _popcnt
# ---------------------------------------------------------------------------


class TestPopcnt:
    def test_zero(self):
        assert _popcnt(np.uint64(0)) == 0

    def test_all_ones(self):
        assert _popcnt(np.uint64(0xFFFFFFFFFFFFFFFF)) == 64

    def test_single_bit(self):
        for i in range(64):
            assert _popcnt(np.uint64(1) << np.uint64(i)) == 1

    def test_known_values(self):
        assert _popcnt(np.uint64(0b10110101)) == 5
        assert _popcnt(np.uint64(0xFF00FF00FF00FF00)) == 32


# ---------------------------------------------------------------------------
# _bitmap_score
# ---------------------------------------------------------------------------


class TestBitmapScore:
    def test_identical_rows(self):
        row = np.array([0b1010, 0b1100], dtype=np.uint64)
        assert _bitmap_score(row, row) == 4  # popcount(0b1010) + popcount(0b1100)

    def test_disjoint_rows(self):
        a = np.array([0b1010], dtype=np.uint64)
        b = np.array([0b0101], dtype=np.uint64)
        assert _bitmap_score(a, b) == 0

    def test_zero_row(self):
        row = np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        zero = np.array([0], dtype=np.uint64)
        assert _bitmap_score(row, zero) == 0

    def test_all_ones(self):
        a = np.full(4, fill_value=np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        assert _bitmap_score(a, a) == 256

    def test_known_value(self):
        a = np.array([0b11001100], dtype=np.uint64)
        b = np.array([0b10101010], dtype=np.uint64)
        assert _bitmap_score(a, b) == 2  # 0b10001000


# ---------------------------------------------------------------------------
# generate_bitmaps
# ---------------------------------------------------------------------------


class TestGenerateBitmaps:
    def test_output_shape(self, simple_ion_matrix, log_width):
        bm = generate_bitmaps(
            simple_ion_matrix,
            ARRAY_LENGTH,
            LOG_MIN,
            log_width,
            N_BINS,
            True,
            0,
            False,
        )
        assert bm.shape == (3, ARRAY_LENGTH)
        assert bm.dtype == np.uint64

    def test_zero_padded_rows_ignored(self, log_width):
        mat = np.array([[500.0, 1000.0, 0.0, 0.0]], dtype=np.float64)
        bm = generate_bitmaps(mat, ARRAY_LENGTH, LOG_MIN, log_width, N_BINS, True, 0, False)
        assert bm.shape == (1, ARRAY_LENGTH)

    def test_more_ions_more_bits(self, log_width):
        few = np.array([[200.0, 400.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        many = np.array([[200.0, 400.0, 600.0, 800.0, 1000.0]], dtype=np.float64)
        bm_few = generate_bitmaps(few, ARRAY_LENGTH, LOG_MIN, log_width, N_BINS, True, 0, False)
        bm_many = generate_bitmaps(many, ARRAY_LENGTH, LOG_MIN, log_width, N_BINS, True, 0, False)
        bits_few = sum(bin(w).count("1") for w in bm_few[0])
        bits_many = sum(bin(w).count("1") for w in bm_many[0])
        assert bits_many >= bits_few

    def test_tolerance_widens_bits(self, log_width):
        mat = np.array([[500.0, 1000.0, 2000.0]], dtype=np.float64)
        bm_tight = generate_bitmaps(mat, ARRAY_LENGTH, LOG_MIN, log_width, N_BINS, True, 0, False)
        bm_wide = generate_bitmaps(mat, ARRAY_LENGTH, LOG_MIN, log_width, N_BINS, True, 20.0, False)
        bits_tight = sum(bin(w).count("1") for w in bm_tight[0])
        bits_wide = sum(bin(w).count("1") for w in bm_wide[0])
        assert bits_wide >= bits_tight

    def test_forward_vs_reverse_differ(self, simple_ion_matrix, log_width):
        bm_fwd = generate_bitmaps(simple_ion_matrix, ARRAY_LENGTH, LOG_MIN, log_width, N_BINS, True, 0, False)
        bm_rev = generate_bitmaps(simple_ion_matrix, ARRAY_LENGTH, LOG_MIN, log_width, N_BINS, False, 0, False)
        assert np.array_equal(bm_fwd, bm_rev)


# ---------------------------------------------------------------------------
# bitmap_search_kernel
# ---------------------------------------------------------------------------


class TestBitmapSearchKernel:
    def test_self_match_is_highest(self, random_bitmaps):
        query = random_bitmaps[0]
        scores, _ = bitmap_search_kernel(query, random_bitmaps)
        assert scores[0] == scores.max()

    def test_output_shape(self, random_bitmaps):
        scores, pvalues = bitmap_search_kernel(random_bitmaps[0], random_bitmaps)
        assert scores.shape == (len(random_bitmaps),)
        assert scores.dtype == np.int32
        assert pvalues.shape == (len(random_bitmaps),)
        assert pvalues.dtype == np.float64

    def test_scores_non_negative(self, random_bitmaps):
        scores, _ = bitmap_search_kernel(random_bitmaps[0], random_bitmaps)
        assert np.all(scores >= 0)

    def test_scores_bounded(self, random_bitmaps):
        scores, _ = bitmap_search_kernel(random_bitmaps[0], random_bitmaps)
        assert np.all(scores <= ARRAY_LENGTH * 64)

    def test_zero_query_all_zero(self, random_bitmaps):
        query = np.zeros(ARRAY_LENGTH, dtype=np.uint64)
        scores, _ = bitmap_search_kernel(query, random_bitmaps)
        assert np.all(scores == 0)

    def test_all_ones_query_max(self):
        db = np.full((10, ARRAY_LENGTH), fill_value=np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        query = np.full(ARRAY_LENGTH, fill_value=np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        scores, _ = bitmap_search_kernel(query, db)
        assert np.all(scores == ARRAY_LENGTH * 64)

    def test_consistent_with_bitmap_score(self, random_bitmaps):
        query = random_bitmaps[5]
        scores, _ = bitmap_search_kernel(query, random_bitmaps)
        for i in range(len(random_bitmaps)):
            expected = _bitmap_score(random_bitmaps[i], query)
            assert scores[i] == expected, f"Mismatch at index {i}"

    def test_pvalues_in_range(self, random_bitmaps):
        _, pvalues = bitmap_search_kernel(random_bitmaps[0], random_bitmaps)
        assert np.all(pvalues >= 0.0)
        assert np.all(pvalues <= 1.0)

    def test_pvalue_self_match_is_lowest(self, random_bitmaps):
        query = random_bitmaps[0]
        _, pvalues = bitmap_search_kernel(query, random_bitmaps)
        assert pvalues[0] == pvalues.min()

    def test_zero_query_pvalues_are_one(self, random_bitmaps):
        query = np.zeros(ARRAY_LENGTH, dtype=np.uint64)
        _, pvalues = bitmap_search_kernel(query, random_bitmaps)
        assert np.all(pvalues == 1.0)


# ---------------------------------------------------------------------------
# bitmap_search_kernel_topk
# ---------------------------------------------------------------------------


class TestBitmapSearchKernelTopk:
    def test_top1_matches_argmax(self, random_bitmaps):
        query = random_bitmaps[0]
        scores_full, _ = bitmap_search_kernel(query, random_bitmaps)
        indices, scores, _ = bitmap_search_kernel_topk(query, random_bitmaps, top_k=1)
        assert indices[0] == np.argmax(scores_full)
        assert scores[0] == scores_full.max()

    def test_topk_sorted_descending(self, random_bitmaps):
        _, scores, _ = bitmap_search_kernel_topk(random_bitmaps[0], random_bitmaps, top_k=10)
        assert np.all(scores[:-1] >= scores[1:])

    def test_topk_subset_of_full(self, random_bitmaps):
        query = random_bitmaps[0]
        scores_full, _ = bitmap_search_kernel(query, random_bitmaps)
        top_scores_full = np.sort(scores_full)[::-1][:10]
        _, scores_topk, _ = bitmap_search_kernel_topk(query, random_bitmaps, top_k=10)
        np.testing.assert_array_equal(scores_topk, top_scores_full)

    def test_topk_gt_n(self):
        """top_k larger than database should not crash."""
        db = np.random.randint(0, 2**32, size=(5, ARRAY_LENGTH), dtype=np.uint64)
        query = db[0]
        indices, scores, _ = bitmap_search_kernel_topk(query, db, top_k=20)
        assert len(indices) <= 20

    def test_indices_valid(self, random_bitmaps):
        indices, _, _ = bitmap_search_kernel_topk(random_bitmaps[0], random_bitmaps, top_k=10)
        assert np.all(indices >= 0)
        assert np.all(indices < len(random_bitmaps))

    def test_pvalues_in_range(self, random_bitmaps):
        _, _, pvalues = bitmap_search_kernel_topk(random_bitmaps[0], random_bitmaps, top_k=10)
        assert np.all(pvalues >= 0.0)
        assert np.all(pvalues <= 1.0)

    def test_pvalues_decrease_with_score_fixed_db_row(self):
        """For a fixed db row, higher score → lower p-value."""
        N_BINS_LOCAL = 1024
        n_words = N_BINS_LOCAL // 64
        # db row with exactly 100 bits set
        db_row = np.zeros(n_words, dtype=np.uint64)
        for i in range(100):
            db_row[i // 64] |= np.uint64(1) << np.uint64(i % 64)
        db = db_row.reshape(1, n_words)
        # queries with increasing overlap: 10, 20, 30 matching bits
        pvals = []
        for n_match in (10, 20, 30):
            query = np.zeros(n_words, dtype=np.uint64)
            for i in range(n_match):
                query[i // 64] |= np.uint64(1) << np.uint64(i % 64)
            _, pv = bitmap_search_kernel(query, db)
            pvals.append(float(pv[0]))
        assert pvals[0] > pvals[1] > pvals[2]


# ---------------------------------------------------------------------------
# _hypergeom_sf
# ---------------------------------------------------------------------------


class TestHypergeomSf:
    def test_zero_matches_returns_one(self):
        assert _hypergeom_sf(1000, 50, 50, 0) == 1.0

    def test_impossible_matches_returns_zero(self):
        # k > min(n, K) is impossible
        assert _hypergeom_sf(1000, 10, 10, 11) == 0.0

    def test_probability_in_range(self):
        p = _hypergeom_sf(1000, 100, 100, 10)
        assert 0.0 <= p <= 1.0

    def test_decreases_with_k(self):
        N, K, n = 1000, 100, 100
        p_low = _hypergeom_sf(N, K, n, 5)
        p_high = _hypergeom_sf(N, K, n, 15)
        assert p_low > p_high

    def test_perfect_overlap(self):
        # N bits, all K=n set, k=K matches — probability should be 1
        N, K = 100, 50
        p = _hypergeom_sf(N, K, K, K)
        assert p > 0.0


# ---------------------------------------------------------------------------
# CpuBitmapIndex
# ---------------------------------------------------------------------------


class TestCpuBitmapIndex:
    @pytest.fixture
    def index(self, random_bitmaps):
        return CpuBitmapIndex(random_bitmaps)

    def test_search_sorted_descending(self, index, random_bitmaps):
        _, scores, _ = index.search(random_bitmaps[0], k=10)
        assert np.all(scores[:-1] >= scores[1:])

    def test_search_top1_matches_argmax(self, index, random_bitmaps):
        query = random_bitmaps[0]
        indices, scores, _ = index.search(query, k=1)
        all_scores, _ = bitmap_search_kernel(query, random_bitmaps)
        assert scores[0] == all_scores.max()
        assert indices[0] == np.argmax(all_scores)

    def test_search_all_matches_kernel(self, index, random_bitmaps):
        query = random_bitmaps[3]
        _, scores_idx, _ = index.search_all(query)
        scores_kern, _ = bitmap_search_kernel(query, random_bitmaps)
        assert np.array_equal(scores_idx, scores_kern)

    def test_search_indices_valid(self, index, random_bitmaps):
        indices, _, _ = index.search(random_bitmaps[0], k=10)
        assert np.all(indices >= 0)
        assert np.all(indices < len(random_bitmaps))

    def test_search_pvalues_in_range(self, index, random_bitmaps):
        _, _, pvalues = index.search(random_bitmaps[0], k=10)
        assert np.all(pvalues >= 0.0)
        assert np.all(pvalues <= 1.0)

    def test_save_load_roundtrip(self, index, random_bitmaps, tmp_path):
        path = tmp_path / "index.npz"
        index.save(path)
        loaded = CpuBitmapIndex.load(path)
        assert np.array_equal(index.db, loaded.db)
        assert loaded.filter is None
        _, scores_orig, _ = index.search(random_bitmaps[0], k=5)
        _, scores_loaded, _ = loaded.search(random_bitmaps[0], k=5)
        np.testing.assert_array_equal(scores_orig, scores_loaded)

    def test_save_load_with_filter(self, random_bitmaps, tmp_path):
        n = len(random_bitmaps)
        filter_arr = np.sort(np.random.default_rng(1).uniform(100.0, 5000.0, n))
        index = CpuBitmapIndex(random_bitmaps, filter_arr)
        path = tmp_path / "index_filter.npz"
        index.save(path)
        loaded = CpuBitmapIndex.load(path)
        np.testing.assert_array_equal(index.filter, loaded.filter)


# ---------------------------------------------------------------------------
# GpuBitmapIndex
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGpuBitmapIndex:
    @pytest.fixture
    def gpu_index(self, random_bitmaps):
        return GpuBitmapIndex(random_bitmaps)

    def test_top1_matches_cpu(self, gpu_index, random_bitmaps):
        query = random_bitmaps[0]
        scores_cpu, _ = bitmap_search_kernel(query, random_bitmaps)
        _, scores_gpu, _ = gpu_index.search(query, k=1)
        assert scores_gpu[0] == scores_cpu.max()

    def test_topk_sorted_descending(self, gpu_index, random_bitmaps):
        _, scores, _ = gpu_index.search(random_bitmaps[0], k=10)
        assert np.all(scores[:-1] >= scores[1:])

    def test_topk_matches_cpu_scores(self, gpu_index, random_bitmaps):
        query = random_bitmaps[3]
        scores_cpu, _ = bitmap_search_kernel(query, random_bitmaps)
        top_scores_cpu = np.sort(scores_cpu)[::-1][:10]
        _, scores_gpu, _ = gpu_index.search(query, k=10)
        np.testing.assert_array_equal(scores_gpu, top_scores_cpu)

    def test_indices_valid(self, gpu_index, random_bitmaps):
        indices, _, _ = gpu_index.search(random_bitmaps[0], k=10)
        assert np.all(indices >= 0)
        assert np.all(indices < len(random_bitmaps))

    def test_multiple_queries_consistent(self, gpu_index, random_bitmaps):
        for i in range(5):
            query = random_bitmaps[i]
            scores_cpu, _ = bitmap_search_kernel(query, random_bitmaps)
            _, scores_gpu, _ = gpu_index.search(query, k=5)
            assert scores_gpu[0] == scores_cpu.max(), f"Mismatch on query {i}"

    def test_zero_query(self, gpu_index):
        query = np.zeros(ARRAY_LENGTH, dtype=np.uint64)
        _, scores, _ = gpu_index.search(query, k=5)
        assert np.all(scores == 0)

    def test_db_stays_on_gpu_across_calls(self, gpu_index, random_bitmaps):
        """Repeated searches should give consistent results without re-uploading db."""
        query = random_bitmaps[0]
        r1_idx, r1_scores, r1_pv = gpu_index.search(query, k=5)
        r2_idx, r2_scores, r2_pv = gpu_index.search(query, k=5)
        np.testing.assert_array_equal(r1_scores, r2_scores)
        np.testing.assert_array_equal(r1_idx, r2_idx)
        np.testing.assert_array_equal(r1_pv, r2_pv)

    def test_pvalues_in_range(self, gpu_index, random_bitmaps):
        _, _, pvalues = gpu_index.search(random_bitmaps[0], k=10)
        assert np.all(pvalues >= 0.0)
        assert np.all(pvalues <= 1.0)

    def test_save_load_roundtrip(self, gpu_index, random_bitmaps, tmp_path):
        path = tmp_path / "gpu_index.npz"
        gpu_index.save(path)
        loaded = GpuBitmapIndex.load(path)
        _, scores_orig, _ = gpu_index.search(random_bitmaps[0], k=5)
        _, scores_loaded, _ = loaded.search(random_bitmaps[0], k=5)
        np.testing.assert_array_equal(scores_orig, scores_loaded)


# ---------------------------------------------------------------------------
# GpuBitmapIndex.search_all
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGpuBitmapIndexSearchAll:
    @pytest.fixture
    def gpu_index(self, random_bitmaps):
        return GpuBitmapIndex(random_bitmaps)

    def test_output_shape(self, gpu_index, random_bitmaps):
        _, scores, pvalues = gpu_index.search_all(random_bitmaps[0])
        assert scores.shape == (len(random_bitmaps),)
        assert pvalues.shape == (len(random_bitmaps),)

    def test_matches_cpu_scores(self, gpu_index, random_bitmaps):
        query = random_bitmaps[0]
        _, scores_gpu, _ = gpu_index.search_all(query)
        scores_cpu, _ = bitmap_search_kernel(query, random_bitmaps)
        np.testing.assert_array_equal(scores_gpu, scores_cpu)

    def test_self_match_is_highest(self, gpu_index, random_bitmaps):
        _, scores, _ = gpu_index.search_all(random_bitmaps[0])
        assert scores[0] == scores.max()

    def test_pvalues_in_range(self, gpu_index, random_bitmaps):
        _, _, pvalues = gpu_index.search_all(random_bitmaps[0])
        assert np.all(pvalues >= 0.0)
        assert np.all(pvalues <= 1.0)


# ---------------------------------------------------------------------------
# BitmapIndex (unified)
# ---------------------------------------------------------------------------


class TestBitmapIndex:
    def test_cpu_backend_search(self, random_bitmaps):
        idx = BitmapIndex(random_bitmaps, backend="cpu")
        indices, scores, pvalues = idx.search(random_bitmaps[0], k=10)
        assert len(indices) == 10
        assert np.all(pvalues >= 0.0) and np.all(pvalues <= 1.0)

    def test_cpu_backend_search_all(self, random_bitmaps):
        idx = BitmapIndex(random_bitmaps, backend="cpu")
        _, scores, pvalues = idx.search_all(random_bitmaps[0])
        assert scores.shape == (len(random_bitmaps),)
        assert pvalues.shape == (len(random_bitmaps),)

    def test_invalid_backend(self, random_bitmaps):
        with pytest.raises(ValueError, match="Unknown backend"):
            BitmapIndex(random_bitmaps, backend="tpu")  # type: ignore[arg-type]

    def test_default_backend_is_cpu(self, random_bitmaps):
        idx = BitmapIndex(random_bitmaps)
        assert isinstance(idx._impl, CpuBitmapIndex)

    def test_save_load_roundtrip(self, random_bitmaps, tmp_path):
        idx = BitmapIndex(random_bitmaps, backend="cpu")
        path = tmp_path / "bitmap_index.npz"
        idx.save(path)
        loaded = BitmapIndex.load(path, backend="cpu")
        _, scores_orig, _ = idx.search(random_bitmaps[0], k=5)
        _, scores_loaded, _ = loaded.search(random_bitmaps[0], k=5)
        np.testing.assert_array_equal(scores_orig, scores_loaded)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_backend_matches_cpu(self, random_bitmaps):
        cpu_idx = BitmapIndex(random_bitmaps, backend="cpu")
        gpu_idx = BitmapIndex(random_bitmaps, backend="gpu")
        query = random_bitmaps[0]
        _, cpu_scores, _ = cpu_idx.search(query, k=10)
        _, gpu_scores, _ = gpu_idx.search(query, k=10)
        np.testing.assert_array_equal(cpu_scores, gpu_scores)


# ---------------------------------------------------------------------------
# filter_array / range search
# ---------------------------------------------------------------------------


class TestFilterArray:
    @pytest.fixture
    def db_with_filter(self, random_bitmaps):
        n = len(random_bitmaps)
        filter_arr = np.sort(np.random.default_rng(0).uniform(100.0, 5000.0, n))
        return random_bitmaps, filter_arr

    def test_unsorted_filter_raises(self, random_bitmaps):
        bad = np.array([3.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="sorted"):
            CpuBitmapIndex(random_bitmaps[:3], bad)

    def test_wrong_length_filter_raises(self, random_bitmaps):
        with pytest.raises(ValueError, match="length"):
            CpuBitmapIndex(random_bitmaps, np.ones(5))

    def test_min_max_without_filter_raises(self, random_bitmaps):
        idx = CpuBitmapIndex(random_bitmaps)
        with pytest.raises(ValueError, match="filter_array"):
            idx.search(random_bitmaps[0], min_val=200.0, max_val=1000.0)

    def test_search_range_subset_of_full(self, db_with_filter):
        db, filter_arr = db_with_filter
        idx = CpuBitmapIndex(db, filter_arr)
        query = db[0]
        min_val, max_val = float(np.percentile(filter_arr, 25)), float(np.percentile(filter_arr, 75))
        indices_range, _, _ = idx.search(query, k=10, min_val=min_val, max_val=max_val)
        # all returned indices must fall within the filter range
        assert np.all(filter_arr[indices_range] >= min_val)
        assert np.all(filter_arr[indices_range] <= max_val)

    def test_search_no_range_matches_full(self, db_with_filter):
        db, filter_arr = db_with_filter
        idx = CpuBitmapIndex(db, filter_arr)
        query = db[0]
        indices_full, scores_full, _ = idx.search(query, k=10)
        indices_no_range, scores_no_range, _ = idx.search(query, k=10, min_val=None, max_val=None)
        np.testing.assert_array_equal(indices_full, indices_no_range)
        np.testing.assert_array_equal(scores_full, scores_no_range)

    def test_search_all_range_returns_windowed(self, db_with_filter):
        db, filter_arr = db_with_filter
        idx = CpuBitmapIndex(db, filter_arr)
        query = db[0]
        min_val, max_val = float(np.percentile(filter_arr, 40)), float(np.percentile(filter_arr, 60))
        indices, scores, pvalues = idx.search_all(query, min_val=min_val, max_val=max_val)
        assert np.all(filter_arr[indices] >= min_val)
        assert np.all(filter_arr[indices] <= max_val)
        assert len(scores) == len(indices)
        assert len(pvalues) == len(indices)

    def test_empty_range_returns_empty(self, db_with_filter):
        db, filter_arr = db_with_filter
        idx = CpuBitmapIndex(db, filter_arr)
        # range below all values
        indices, scores, pvalues = idx.search(db[0], k=10, min_val=0.0, max_val=1.0)
        assert len(indices) == 0 and len(scores) == 0 and len(pvalues) == 0

    def test_bitmap_index_passes_filter_through(self, db_with_filter):
        db, filter_arr = db_with_filter
        idx = BitmapIndex(db, backend="cpu", filter_array=filter_arr)
        query = db[0]
        min_val, max_val = float(np.percentile(filter_arr, 25)), float(np.percentile(filter_arr, 75))
        indices, _, _ = idx.search(query, k=10, min_val=min_val, max_val=max_val)
        assert np.all(filter_arr[indices] >= min_val)
        assert np.all(filter_arr[indices] <= max_val)


if __name__ == "__main__":
    pytest.main([__file__])
