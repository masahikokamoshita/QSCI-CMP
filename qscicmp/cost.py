"""T-gate cost models for the state-preparation circuit and the oracles (Sec. III D and IV A)."""
from __future__ import annotations

import numpy as np

TOFFOLI_TO_T = 4


def ucj_rotation_count(n_qubits: int, layers: int) -> int:
    """Number of single-qubit rotations in an L-layer UCJ circuit, L(5/2 n^2 - 1/2 n)."""
    half = n_qubits // 2
    return int(layers * (2 * (n_qubits + n_qubits * (half - 1)) + n_qubits + 3 * n_qubits * (n_qubits - 1) / 2))


def t_per_rotation(n_rotations: int, epsilon_total: float) -> float:
    """Repeat-until-success synthesis cost per rotation for a total error budget epsilon_total."""
    return 1.15 * np.log2(max(2.0, n_rotations / epsilon_total)) + 9.2


def state_preparation_t_count(n_qubits: int, layers: int, epsilon_total: float = 1e-4) -> float:
    """T gates of one query to U_GS."""
    n_rotations = ucj_rotation_count(n_qubits, layers)
    return float(n_rotations) * t_per_rotation(n_rotations, epsilon_total)


def enumerated_oracle_t_count(n_qubits: int, n_enumerated: int, grover_iterations: int) -> float:
    """T gates of the enumerated oracle O_{M \\ T} over ``grover_iterations`` applications.

    Each enumerated determinant costs one multi-controlled phase gate of
    4n - 6 Toffolis.
    """
    return float(grover_iterations) * float(n_enumerated) * float(4 * n_qubits - 6) * TOFFOLI_TO_T


def preselection_oracle_t_count(n_qubits: int) -> float:
    """T gates of the structural oracle O_T for an excitation or hierarchy threshold, 4n."""
    return float(4 * n_qubits)
