import math

import numba


@numba.njit(cache=True)
def _log_comb(n: int, k: int) -> float:
    """log C(n, k) via log-gamma."""
    if k < 0 or k > n:
        return -math.inf
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


@numba.njit(cache=True)
def _hypergeom_sf(N: int, K: int, n: int, k: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(N, K, n).

    N: total bins, K: bits set in db row, n: bits set in query, k: observed matches.
    Uses log C(N,n) seed + multiplicative recurrence for numerical stability.
    """
    k_min = max(0, n + K - N)  # minimum achievable matches
    k_max = min(n, K)  # maximum achievable matches
    if k <= k_min:
        return 1.0
    if k > k_max:
        return 0.0
    log_denom = _log_comb(N, n)
    cur = math.exp(_log_comb(K, k) + _log_comb(N - K, n - k) - log_denom)
    total = cur
    for i in range(k, k_max):
        den = (i + 1) * (N - K - n + i + 1)
        if den == 0:
            break
        cur = cur * (K - i) * (n - i) / den
        if cur <= 0.0:
            break
        total += cur
    return min(total, 1.0)
