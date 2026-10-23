"""Definitions of the molecular systems used in the paper.

Each system is specified by a geometry, a basis set, and an active space
(N_e electrons in N_o spatial orbitals).  The active orbitals are the
canonical RHF orbitals selected by the PySCF CASCI default, i.e. the
frontier window of N_o orbitals around the HOMO--LUMO gap.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemSpec:
    name: str
    label: str
    geometry: tuple[tuple[str, tuple[float, float, float]], ...]
    basis: str
    n_orb_cas: int
    n_ele_cas: int
    symmetry: str

    @property
    def n_qubits(self) -> int:
        return 2 * self.n_orb_cas


def _hydrogen_chain(n_atoms: int, spacing: float = 1.0) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    return tuple(("H", (0.0, 0.0, spacing * i)) for i in range(n_atoms))


# Geometry of H2O: the HF/STO-3G optimized structure from the NIST CCCBDB.
_H2O_GEOMETRY = (
    ("O", (0.0, 0.0, 0.1272)),
    ("H", (0.0, 0.7581, -0.5086)),
    ("H", (0.0, -0.7581, -0.5086)),
)

_CR2_GEOMETRY = (("Cr", (0.0, 0.0, 0.0)), ("Cr", (0.0, 0.0, 1.6)))

SYSTEMS: dict[str, SystemSpec] = {
    "H10": SystemSpec("H10", "H10", _hydrogen_chain(10), "sto3g", 10, 10, "D2h"),
    "H12": SystemSpec("H12", "H12", _hydrogen_chain(12), "sto3g", 12, 12, "D2h"),
    "H2O_10o10e": SystemSpec("H2O_10o10e", "H2O", _H2O_GEOMETRY, "ccpvdz", 10, 10, "C2v"),
    "H2O_12o10e": SystemSpec("H2O_12o10e", "H2O", _H2O_GEOMETRY, "ccpvdz", 12, 10, "C2v"),
    "Cr2_10o10e": SystemSpec("Cr2_10o10e", "Cr2", _CR2_GEOMETRY, "ccpvdz", 10, 10, "D2h"),
    "Cr2_12o12e": SystemSpec("Cr2_12o12e", "Cr2", _CR2_GEOMETRY, "ccpvdz", 12, 12, "D2h"),
}

PAPER_SYSTEMS_20_QUBITS = ("H10", "H2O_10o10e", "Cr2_10o10e")
PAPER_SYSTEMS_24_QUBITS = ("H12", "H2O_12o10e", "Cr2_12o12e")
PAPER_SYSTEMS = PAPER_SYSTEMS_20_QUBITS + PAPER_SYSTEMS_24_QUBITS


def get_system(name: str) -> SystemSpec:
    try:
        return SYSTEMS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown system {name!r}. Available: {', '.join(SYSTEMS)}") from exc
