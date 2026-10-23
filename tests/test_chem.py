"""Chemistry checks: energies of the cached molecules and the subspace solver."""
from pathlib import Path

import numpy as np
import pytest

from qscicmp.chem import SubspaceSolver, build_molecule, determinant_amplitudes, get_system
from qscicmp.chem.subspace import hf_index
from qscicmp.preselection import preselection_h25

# Reference energies (Hartree) from the runs reported in the paper.
REFERENCE = {
    "H10": {"hf": -5.214068803028285, "exact": -5.379954744061848},
    "H2O_12o10e": {"hf": -76.02312113913888, "exact": -76.12749664353626},
}


@pytest.fixture(scope="module")
def h10():
    return build_molecule(get_system("H10"))


def test_h10_energies(h10):
    assert abs(h10.hf_energy - REFERENCE["H10"]["hf"]) < 1e-8
    assert abs(h10.exact_energy - REFERENCE["H10"]["exact"]) < 1e-7


def test_hf_determinant_energy(h10):
    solver = SubspaceSolver(h10)
    assert abs(solver.ground_energy([hf_index(h10.n_ele_cas)]) - h10.hf_energy) < 1e-10


def test_full_space_energy(h10):
    solver = SubspaceSolver(h10)
    indices, _ = determinant_amplitudes(h10)
    assert abs(solver.ground_energy(indices) - h10.exact_energy) < 1e-7


def test_preselection_energy_is_variational(h10):
    solver = SubspaceSolver(h10)
    energy = solver.ground_energy(preselection_h25(h10.n_qubits, h10.n_ele_cas))
    assert h10.exact_energy - 1e-10 <= energy <= h10.hf_energy
