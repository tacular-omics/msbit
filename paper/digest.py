"""Tryptic digestion of a FASTA file. No external proteomics deps."""

from collections.abc import Iterator
from pathlib import Path


def parse_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (header, sequence) pairs from a FASTA file."""
    raise NotImplementedError


def tryptic_peptides(
    sequence: str,
    missed_cleavages: int = 2,
    min_len: int = 7,
    max_len: int = 50,
) -> list[str]:
    """Trypsin cuts C-terminal to K/R unless followed by P."""
    raise NotImplementedError


def digest_fasta(
    path: Path,
    missed_cleavages: int = 2,
    min_len: int = 7,
    max_len: int = 50,
) -> list[tuple[str, str]]:
    """Return (protein_id, peptide_sequence) for every unique peptide."""
    raise NotImplementedError
