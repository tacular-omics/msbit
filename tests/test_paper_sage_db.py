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

    # 7, not 11: SageParams cleaves C-terminally of R only (ArgC). When these
    # tests were written at a7409e3 the default was cleave_at="KR" (trypsin),
    # which gives 11 from insulin; e5ef0b0 changed it to "R". The published
    # numbers in paper/data/compare_cpu.json come from the "R" configuration,
    # confirmed by rerun on 2026-11-14, so 7 is the correct expectation.
    assert db.num_peptides == 7
    assert "GIVEQCCTSICSLYQLENYCN" in peptides
    assert np.all(np.diff(mono) >= 0), "peptides must be sorted by mono mass"


def test_no_static_mods_by_default(tmp_path: Path) -> None:
    """SageParams applies no static modification, and the cysteines are bare.

    This test asserted the opposite until 2026-11-14. At a7409e3
    `SageParams.static_mods` defaulted to carbamidomethyl C, so the four
    cysteines of this peptide added 4 x 57.0215 Da and the expected mass was
    2610.086. e5ef0b0 changed the default to empty. The rerun on 2026-11-14
    showed that the empty setting is what produced the published numbers: with
    Cam-C the precursor window statistics come out at 3620.847 / 3431 / 4925
    against the published 3636.414 / 3458 / 4953, which the empty setting
    reproduces exactly. So the unmodified mass is the right expectation.
    """
    fasta = tmp_path / "insulin.fasta"
    fasta.write_text(INSULIN_FASTA)
    params = SageParams(fasta_path=fasta)
    assert params.static_mods == ()

    db = build_indexed_database(params)
    peptides = list(db.peptides_as_string())
    mono = np.asarray(db.mono_masses())
    idx = peptides.index("GIVEQCCTSICSLYQLENYCN")
    assert abs(mono[idx] - 2382.000) < 0.01

    # And the mod is applied when asked for, so this is a default and not a
    # missing feature: 4 cysteines x 57.0215 Da.
    with_mod = build_indexed_database(
        SageParams(fasta_path=fasta, static_mods=(("C", 57.0215),))
    )
    peps2 = list(with_mod.peptides_as_string())
    mono2 = np.asarray(with_mod.mono_masses())
    assert abs(mono2[peps2.index("GIVEQCCTSICSLYQLENYCN")] - 2610.086) < 0.01


def test_per_peptide_fragments_partition(tmp_path: Path) -> None:
    fasta = tmp_path / "insulin.fasta"
    fasta.write_text(INSULIN_FASTA)
    db = build_indexed_database(SageParams(fasta_path=fasta))
    frags = per_peptide_fragments(db)

    assert len(frags) == db.num_peptides
    assert sum(len(f) for f in frags) == db.num_fragments
    for f in frags:
        assert np.all(np.diff(f) > 0), "per-peptide fragments must be strictly ascending"
