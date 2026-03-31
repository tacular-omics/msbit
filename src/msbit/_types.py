from typing import NamedTuple

import numpy as np


class SearchResult(NamedTuple):
    indices: np.ndarray  # (k,) int32 — indices into the database
    matches: np.ndarray  # (k,) int32 — popcount overlap
    pvalues: np.ndarray  # (k,) float64 — hypergeometric p-values


_EMPTY_SEARCH_RESULT = SearchResult(
    np.empty(0, dtype=np.int32),
    np.empty(0, dtype=np.int32),
    np.empty(0, dtype=np.float64),
)
