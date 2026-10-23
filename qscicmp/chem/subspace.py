"""Determinant indexing and subspace diagonalization.

A Slater determinant is labeled by an integer whose bits record the
occupations of the spin orbitals in the interleaved ordering of the paper:
bit 2p is the alpha spin orbital and bit 2p+1 the beta spin orbital of
spatial orbital p.  The Hartree--Fock determinant therefore has the lowest
N_e bits set.

The projected Hamiltonian on a set of determinants is applied with the
selected-CI contraction routines of PySCF, so no explicit matrix is
stored.  The set is embedded in the product space of its alpha and beta
strings and projected back after each application.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from pyscf.fci import direct_spin1, selected_ci
from scipy.sparse.linalg import LinearOperator, eigsh

from qscicmp.chem.hamiltonian import MoleculeData

DENSE_MAX_SIZE = 512
EIGSH_TOL = 1e-8


def hf_index(n_electrons: int) -> int:
    return (1 << n_electrons) - 1


def split_index(index: int, n_orb: int) -> tuple[int, int]:
    alpha = beta = 0
    for p in range(n_orb):
        if (index >> (2 * p)) & 1:
            alpha |= 1 << p
        if (index >> (2 * p + 1)) & 1:
            beta |= 1 << p
    return alpha, beta


def combine_index(alpha: int, beta: int, n_orb: int) -> int:
    index = 0
    for p in range(n_orb):
        if (alpha >> p) & 1:
            index |= 1 << (2 * p)
        if (beta >> p) & 1:
            index |= 1 << (2 * p + 1)
    return index


def _split_array(indices: np.ndarray, n_orb: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    alpha = np.zeros_like(indices)
    beta = np.zeros_like(indices)
    for p in range(n_orb):
        alpha |= ((indices >> (2 * p)) & 1) << p
        beta |= ((indices >> (2 * p + 1)) & 1) << p
    return alpha, beta


def determinant_amplitudes(data: MoleculeData) -> tuple[np.ndarray, np.ndarray]:
    """Nonzero amplitudes of the exact ground state, sorted by determinant index."""
    n_orb = data.n_orb_cas
    ia, ib = np.nonzero(data.ci)
    alpha = data.alpha_strings[ia]
    beta = data.beta_strings[ib]
    indices = np.zeros(len(ia), dtype=np.int64)
    for p in range(n_orb):
        indices |= ((alpha >> p) & 1) << (2 * p)
        indices |= ((beta >> p) & 1) << (2 * p + 1)
    order = np.argsort(indices, kind="stable")
    return indices[order], data.ci[ia, ib][order]


class SubspaceSolver:
    """Lowest eigenvalue of the Hamiltonian projected onto a set of determinants."""

    def __init__(self, data: MoleculeData) -> None:
        self.data = data
        self.n_orb = data.n_orb_cas
        self.nelec = (data.n_alpha, data.n_beta)
        self.h2e = direct_spin1.absorb_h1e(data.h1, data.eri, self.n_orb, self.nelec, 0.5)

    def _operator(self, indices: np.ndarray, nelec: tuple[int, int]):
        alpha, beta = _split_array(indices, self.n_orb)
        alpha_strings, ia = np.unique(alpha, return_inverse=True)
        beta_strings, ib = np.unique(beta, return_inverse=True)
        ci_strs = (alpha_strings.astype(np.int64), beta_strings.astype(np.int64))
        shape = (len(alpha_strings), len(beta_strings))
        h2e = self.h2e if nelec == self.nelec else direct_spin1.absorb_h1e(self.data.h1, self.data.eri, self.n_orb, nelec, 0.5)
        link_index = selected_ci._all_linkstr_index(ci_strs, self.n_orb, nelec)

        def matvec(vector: np.ndarray) -> np.ndarray:
            civec = np.zeros(shape, dtype=np.float64)
            civec[ia, ib] = np.asarray(vector, dtype=np.float64).ravel()
            hc = selected_ci.contract_2e(h2e, selected_ci._as_SCIvector(civec, ci_strs), self.n_orb, nelec, link_index)
            return np.asarray(hc)[ia, ib]

        return LinearOperator((len(indices), len(indices)), matvec=matvec, dtype=np.float64)

    def _lowest_eigenvalue(self, indices: np.ndarray, nelec: tuple[int, int]) -> float:
        size = len(indices)
        operator = self._operator(indices, nelec)
        if size <= DENSE_MAX_SIZE:
            matrix = np.column_stack([operator.matvec(np.eye(size)[:, j]) for j in range(size)])
            matrix = 0.5 * (matrix + matrix.T)
            return float(np.linalg.eigvalsh(matrix)[0])
        values = eigsh(operator, k=1, which="SA", return_eigenvectors=False, tol=EIGSH_TOL, maxiter=max(5 * size, 1000))
        return float(values[0])

    def ground_energy(self, indices: Sequence[int]) -> float:
        """Lowest eigenvalue of the Hamiltonian projected onto the given determinants.

        The Hamiltonian conserves the numbers of alpha and beta electrons, so
        the projected matrix is block diagonal in these numbers and the
        blocks are diagonalized separately.
        """
        indices = np.asarray(sorted({int(i) for i in indices}), dtype=np.int64)
        if indices.size == 0:
            raise ValueError("The subspace must contain at least one determinant.")
        alpha, beta = _split_array(indices, self.n_orb)
        n_alpha = np.array([int(a).bit_count() for a in alpha])
        n_beta = np.array([int(b).bit_count() for b in beta])
        energy = np.inf
        for na, nb in sorted({(int(a), int(b)) for a, b in zip(n_alpha, n_beta)}):
            block = indices[(n_alpha == na) & (n_beta == nb)]
            energy = min(energy, self._lowest_eigenvalue(block, (na, nb)))
        return energy + self.data.e_core
