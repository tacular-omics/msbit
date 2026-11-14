"""Smoke test for paper.sage_db on insulin FASTA."""

from pathlib import Path

import numpy as np
import pytest

sagepy = pytest.importorskip("sagepy")

from paper.sage_db import SageParams, build_indexed_database, per_peptide_fragments  # noqa: E402

INSULIN_FASTA = (
    ">INS_HUMAN\n"
    "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPK"
    "TRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKR"
    "GIVEQCCTSICSLYQLENYCN\n"
)


def test_insulin_digest_yields_expected_peptide(tmp_path: Path) -> None:
    fasta = tmp_path / "insulin.fasta"
    fasta.write_text(INSULIN_FASTA)
    db = build_indexed_database(SageParams(fasta_path=fasta))

    peptides = list(db.peptides_as_string())
    mono = np.asarray(db.mono_masses())

    assert db.num_peptides == 11
    assert "GIVEQCCTSICSLYQLENYCN" in peptides
    assert np.all(np.diff(mono) >= 0), "peptides must be sorted by mono mass"


def test_cam_c_static_mod_applied(tmp_path: Path) -> None:
    fasta = tmp_path / "insulin.fasta"
    fasta.write_text(INSULIN_FASTA)
    db = build_indexed_database(SageParams(fasta_path=fasta))

    peptides = list(db.peptides_as_string())
    mono = np.asarray(db.mono_masses())
    idx = peptides.index("GIVEQCCTSICSLYQLENYCN")
    # Unmodified mass would be ~2382 Da; +57.021 x 4 cysteines = ~2610 Da.
    assert abs(mono[idx] - 2610.086) < 0.01


def test_per_peptide_fragments_partition(tmp_path: Path) -> None:
    fasta = tmp_path / "insulin.fasta"
    fasta.write_text(INSULIN_FASTA)
    db = build_indexed_database(SageParams(fasta_path=fasta))
    frags = per_peptide_fragments(db)

    assert len(frags) == db.num_peptides
    assert sum(len(f) for f in frags) == db.num_fragments
    for f in frags:
        assert np.all(np.diff(f) > 0), "per-peptide fragments must be strictly ascending"
