"""Thin wrapper around sagepy's IndexedDatabase.

Centralises the SageSearchConfiguration so build_db.py (msbit side) and
run_sagepy.py (sagepy presearch) share the *exact same* peptide and
fragment universe — the comparison is only fair if both pipelines see
identical inputs.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SageParams:
    fasta_path: Path
    missed_cleavages: int = 2
    min_len: int = 7
    max_len: int = 50
    peptide_min_mass: float = 500.0
    peptide_max_mass: float = 5000.0
    static_mods: tuple[tuple[str, float], ...] = (("C", 57.0215),)
    generate_decoys: bool = False
    bucket_size: int = 8192


def build_indexed_database(params: SageParams):
    """Return a sagepy IndexedDatabase built from the given FASTA.

    Note: sagepy's `fasta` kwarg wants the FASTA *contents string*, not a
    path, despite what its docstring says.
    """
    raise NotImplementedError


def per_peptide_fragments(db) -> list[np.ndarray]:
    """Group db.fragment_mzs by db.fragment_indices into one array per peptide.

    Returned arrays are sorted ascending (msbit's encoder iterates left to right
    and short-circuits on repeats, so deduped+sorted is the safe input shape).
    """
    raise NotImplementedError
