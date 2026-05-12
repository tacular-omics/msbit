"""Monoisotopic mass + b/y fragment ion mass generation.

Hand-rolled AA table; carbamidomethyl C is the only fixed mod.
"""

import numpy as np

# Monoisotopic residue masses (Da), Unimod. C includes +57.02146 carbamidomethyl.
AA_MONO: dict[str, float] = {
    "A": 71.03711,
    "R": 156.10111,
    "N": 114.04293,
    "D": 115.02694,
    "C": 103.00919 + 57.02146,
    "E": 129.04259,
    "Q": 128.05858,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "L": 113.08406,
    "K": 128.09496,
    "M": 131.04049,
    "F": 147.06841,
    "P": 97.05276,
    "S": 87.03203,
    "T": 101.04768,
    "W": 186.07931,
    "Y": 163.06333,
    "V": 99.06841,
}
H2O = 18.01056
PROTON = 1.00728


def peptide_mass(seq: str) -> float:
    """Monoisotopic mass of a peptide (neutral, no proton)."""
    raise NotImplementedError


def b_ion_masses(seq: str) -> np.ndarray:
    """Singly-charged b-ion m/z for b1..b(n-1)."""
    raise NotImplementedError


def y_ion_masses(seq: str) -> np.ndarray:
    """Singly-charged y-ion m/z for y1..y(n-1)."""
    raise NotImplementedError
