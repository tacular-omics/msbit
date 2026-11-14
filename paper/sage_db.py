"""Thin wrapper around sagepy's IndexedDatabase.

Centralises the SageSearchConfiguration so build_db.py (msbit side) and
run_sagepy.py (sagepy presearch) share the *exact same* peptide and
fragment universe — the comparison is only fair if both pipelines see
identical inputs.
"""

from dataclasses import dataclass, field
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
    static_mods: tuple[tuple[str, float], ...] = field(default=())
    generate_decoys: bool = False
    bucket_size: int = 8192
    # Enzyme: ArgC cleaves C-terminally of R, not before P
    cleave_at: str = "R"
    restrict: str = "P"
    c_terminal: bool = True


def build_indexed_database(params: SageParams):
    """Return a sagepy IndexedDatabase built from the given FASTA.

    sagepy's `fasta` kwarg wants the FASTA contents string, not a path,
    despite what its docstring says.
    """
    from sagepy.core import EnzymeBuilder, SageSearchConfiguration

    eb = EnzymeBuilder(
        missed_cleavages=params.missed_cleavages,
        min_len=params.min_len,
        max_len=params.max_len,
        cleave_at=params.cleave_at,
        restrict=params.restrict,
        c_terminal=params.c_terminal,
    )
    cfg = SageSearchConfiguration(
        fasta=Path(params.fasta_path).read_text(),
        enzyme_builder=eb,
        peptide_min_mass=params.peptide_min_mass,
        peptide_max_mass=params.peptide_max_mass,
        static_mods=dict(params.static_mods),
        generate_decoys=params.generate_decoys,
        bucket_size=params.bucket_size,
    )
    return cfg.generate_indexed_database()


def per_peptide_fragments(db) -> list[np.ndarray]:
    """Group db.fragment_mzs by db.fragment_indices into one sorted array per peptide.

    msbit's encoder iterates left-to-right and short-circuits on repeated
    values, so the per-peptide arrays must be deduped + ascending.
    """
    mzs = np.asarray(db.fragment_mzs, dtype=np.float64)
    idx = np.asarray(db.fragment_indices, dtype=np.int64)
    order = np.argsort(idx, kind="stable")
    idx_sorted = idx[order]
    mzs_sorted = mzs[order]
    bounds = np.searchsorted(idx_sorted, np.arange(db.num_peptides + 1))
    return [np.unique(mzs_sorted[bounds[i] : bounds[i + 1]]) for i in range(db.num_peptides)]
