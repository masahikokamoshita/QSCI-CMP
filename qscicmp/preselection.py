"""Structural quantities of determinants and the chemically trivial subspace T(R).

Determinant indices follow the interleaved spin-orbital ordering (see
``qscicmp.chem.subspace``).  The reference is the Hartree--Fock determinant
with the lowest N_e spin orbitals occupied.
"""
from __future__ import annotations

from itertools import combinations


def hf_index(n_electrons: int) -> int:
    return (1 << n_electrons) - 1


def excitation_level(index: int, hf: int) -> int:
    """e(x): half the Hamming distance to the reference."""
    return (int(index) ^ int(hf)).bit_count() // 2


def seniority(index: int, n_orb: int) -> int:
    """Omega(x): number of singly occupied spatial orbitals."""
    return sum(((index >> (2 * p)) & 1) != ((index >> (2 * p + 1)) & 1) for p in range(n_orb))


def twice_sz(index: int, n_orb: int) -> int:
    return sum(((index >> (2 * p)) & 1) - ((index >> (2 * p + 1)) & 1) for p in range(n_orb))


def hierarchy_number(index: int, hf: int, n_orb: int) -> float:
    """h(x) = (e + Omega/2) / 2 of the hCI framework."""
    return 0.5 * (excitation_level(index, hf) + 0.5 * seniority(index, n_orb))


def determinants_up_to_level(n_qubits: int, n_electrons: int, max_level: int) -> set[int]:
    """All determinants with excitation level at most ``max_level`` (any spin)."""
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_qubits))
    hf = hf_index(n_electrons)
    indices = {hf}
    for level in range(1, max_level + 1):
        if level > len(occupied) or level > len(virtual):
            break
        for holes in combinations(occupied, level):
            for particles in combinations(virtual, level):
                index = hf
                for h in holes:
                    index &= ~(1 << h)
                for p in particles:
                    index |= 1 << p
                indices.add(index)
    return indices


def _sz_preserving(index: int, hf: int, n_orb: int) -> bool:
    return twice_sz(index, n_orb) == twice_sz(hf, n_orb)


def preselection_e2(n_qubits: int, n_electrons: int) -> set[int]:
    """R = {e <= 2}: all determinants up to double excitations."""
    return determinants_up_to_level(n_qubits, n_electrons, 2)


def preselection_h20(n_qubits: int, n_electrons: int) -> set[int]:
    """R = {h <= 2}: 2e + Omega <= 8, restricted to the S_z sector of the reference."""
    n_orb = n_qubits // 2
    hf = hf_index(n_electrons)
    result = set()
    for index in determinants_up_to_level(n_qubits, n_electrons, 4):
        e = excitation_level(index, hf)
        omega = seniority(index, n_orb)
        if e <= 2 or (e == 3 and omega == 2 and _sz_preserving(index, hf, n_orb)) or (e == 4 and omega == 0):
            result.add(index)
    return result


def preselection_h25(n_qubits: int, n_electrons: int) -> set[int]:
    """R = {h <= 2.5}: 2e + Omega <= 10, restricted to the S_z sector of the reference."""
    n_orb = n_qubits // 2
    hf = hf_index(n_electrons)
    result = set()
    for index in determinants_up_to_level(n_qubits, n_electrons, 4):
        e = excitation_level(index, hf)
        omega = seniority(index, n_orb)
        if e <= 2:
            result.add(index)
        elif e == 3 and omega in (2, 4) and _sz_preserving(index, hf, n_orb):
            result.add(index)
        elif e == 4 and (omega == 0 or (omega == 2 and _sz_preserving(index, hf, n_orb))):
            result.add(index)
    return result


PRESELECTIONS = {
    "e2": preselection_e2,
    "h20": preselection_h20,
    "h25": preselection_h25,
}
