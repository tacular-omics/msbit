import math

import numba
import numpy as np


@numba.njit(parallel=True, cache=True)
def _encode_ion_matrix_kernel(
    ion_matrix: np.ndarray,  # (n_peptides, max_len) float64 masses, right-padded with zeros
    bitmaps: np.ndarray,  # (n_peptides, array_length) uint64 — written in place
    bin_min: float,
    bin_width: float,
    n_bins: int,
    forward: bool,  # True = b-ions (left to right), False = y-ions (right to left)
    tolerance: float,
    is_ppm: bool,
    use_log: bool,  # True = log-space binning, False = linear binning
) -> None:
    for i in numba.prange(ion_matrix.shape[0]):  # type: ignore
        if forward:
            prev = 0.0
            for j in range(ion_matrix.shape[1]):
                mass = ion_matrix[i, j]
                if mass <= 0.0 or mass == prev:
                    break  # hit padding or repeating cumsum — done
                prev = mass
                mass_offset = tolerance
                if is_ppm:
                    mass_offset = mass * tolerance / 1e6
                if use_log:
                    lower_bin_index = int((math.log(mass - mass_offset) - bin_min) / bin_width)
                    upper_bin_index = int((math.log(mass + mass_offset) - bin_min) / bin_width)
                else:
                    lower_bin_index = int((mass - mass_offset - bin_min) / bin_width)
                    upper_bin_index = int((mass + mass_offset - bin_min) / bin_width)
                for bin_idx in range(lower_bin_index, upper_bin_index + 1):
                    if 0 <= bin_idx < n_bins:
                        bitmaps[i, bin_idx >> 6] |= np.uint64(1) << np.uint64(bin_idx & 63)
        else:
            prev = 0.0
            for j in range(ion_matrix.shape[1] - 1, -1, -1):
                mass = ion_matrix[i, j]
                if mass <= 0.0 or mass == prev:
                    break  # hit padding or repeating cumsum — done
                prev = mass
                mass_offset = tolerance
                if is_ppm:
                    mass_offset = mass * tolerance / 1e6
                if use_log:
                    lower_bin_index = int((math.log(mass - mass_offset) - bin_min) / bin_width)
                    upper_bin_index = int((math.log(mass + mass_offset) - bin_min) / bin_width)
                else:
                    lower_bin_index = int((mass - mass_offset - bin_min) / bin_width)
                    upper_bin_index = int((mass + mass_offset - bin_min) / bin_width)
                for bin_idx in range(lower_bin_index, upper_bin_index + 1):
                    if 0 <= bin_idx < n_bins:
                        bitmaps[i, bin_idx >> 6] |= np.uint64(1) << np.uint64(bin_idx & 63)


def generate_bitmaps(
    ion_matrix: np.ndarray,
    array_length: int,  # number of uint64s per bitmap row
    bin_min: float,
    bin_width: float,
    n_bins: int,
    forward: bool,  # True = b-ions (left to right), False = y-ions (right to left)
    tolerance: float,  # 0 for theoretical masses; search tolerance in Da or ppm
    is_ppm: bool,
    use_log: bool = True,  # True = log-space binning (default), False = linear binning
) -> np.ndarray:
    """Encodes an ion mass matrix into a uint64 bitmap matrix.

    forward=True encodes b-ions (left to right); False encodes y-ions.
    tolerance=0 for theoretical masses; non-zero for search tolerances.
    Returns an (n_peptides, array_length) uint64 array.
    """
    bitmaps = np.zeros((ion_matrix.shape[0], array_length), dtype=np.uint64)
    _encode_ion_matrix_kernel(
        ion_matrix.astype(np.float64),
        bitmaps,
        bin_min,
        bin_width,
        n_bins,
        forward,
        tolerance,
        is_ppm,
        use_log,
    )
    return bitmaps


class BitmapEncoder:
    def __init__(
        self,
        min_mass: float,
        max_mass: float,
        array_length: int,
        tolerance: float = 0.0,
        is_ppm: bool = False,
        use_log: bool = True,
    ):
        """Encodes fragment ion masses into fixed-width bit vectors.

        Uses log-space or linear binning over [min_mass, max_mass].
        array_length uint64 words gives array_length * 64 bins total.
        """
        if min_mass <= 0:
            raise ValueError(f"min_mass must be > 0, got {min_mass}")
        if max_mass <= min_mass:
            raise ValueError(f"max_mass must be > min_mass, got {max_mass} <= {min_mass}")
        if array_length < 1:
            raise ValueError(f"array_length must be >= 1, got {array_length}")
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")

        self.array_length = array_length
        self.tolerance = tolerance
        self.is_ppm = is_ppm
        self.use_log = use_log
        self.n_bins = array_length * 64
        if use_log:
            self.bin_min = math.log(min_mass)
            self.bin_width = (math.log(max_mass) - math.log(min_mass)) / self.n_bins
        else:
            self.bin_min = min_mass
            self.bin_width = (max_mass - min_mass) / self.n_bins

    def encode(self, ion_matrix: np.ndarray) -> np.ndarray:
        """Encodes a (n_peptides, max_ions) ion mass matrix into bitmaps.

        Detects b/y ion direction from padding pattern (trailing zeros → b-ions).
        """
        if ion_matrix.ndim != 2:
            raise ValueError(f"ion_matrix must be 2-D, got shape {ion_matrix.shape}")
        forward = bool(np.any(ion_matrix[:, -1] == 0.0))
        return generate_bitmaps(
            ion_matrix,
            self.array_length,
            self.bin_min,
            self.bin_width,
            self.n_bins,
            forward,
            self.tolerance,
            self.is_ppm,
            self.use_log,
        )


def random_bitmap_matrix(array_size: int = 50, n_rows: int = 100) -> np.ndarray:
    """Generate a random bitmap matrix for testing and benchmarking."""
    return np.random.randint(0, 2**64, size=(n_rows, array_size), dtype=np.uint64)
