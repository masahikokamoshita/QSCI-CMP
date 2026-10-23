"""Sampling simulations of direct QSCI, SQD-AA, and QSCI-CMP.

All three methods sample determinants from the exact ground state of the
active space.  SQD-AA and QSCI-CMP amplify the unmeasured determinants
before each batch of measurements (Sec. II C and III A of the paper).  The
simulation records the cumulative number of queries to the state-preparation
unitary and the cumulative T-gate count, and evaluates the energy of the
collected subspace by classical diagonalization.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from qscicmp.amplification import amplitude_scales, choose_grover_iterations, sample_amplified
from qscicmp.chem.hamiltonian import MoleculeData
from qscicmp.chem.subspace import SubspaceSolver, determinant_amplitudes
from qscicmp.cost import enumerated_oracle_t_count, preselection_oracle_t_count, state_preparation_t_count
from qscicmp.preselection import PRESELECTIONS, hf_index

METHODS = ("direct", "sqdaa", "cmp-e2", "cmp-h20", "cmp-h25")
METHOD_LABELS = {
    "direct": "QSCI",
    "sqdaa": "SQD-AA",
    "cmp-e2": r"QSCI-CMP ($e \leq 2$)",
    "cmp-h20": r"QSCI-CMP ($h \leq 2$)",
    "cmp-h25": r"QSCI-CMP ($h \leq 2.5$)",
}
# Independent random streams per method, seeded as seed + offset + 104729 * trial.
_SEED_OFFSETS = {"direct": 0, "sqdaa": 1_000_000, "cmp-e2": 2_000_000, "cmp-h20": 4_000_000, "cmp-h25": 5_000_000}
NONZERO_PROBABILITY_EPS = 1e-15


@dataclass(frozen=True)
class Config:
    """Settings of a sampling run (defaults reproduce the paper)."""

    epsilon: float = 1.6e-3                 # chemical accuracy in Hartree
    seed: int = 7
    shots_per_iteration: int = 10           # N_s
    direct_shots_multiplier: int = 10       # direct QSCI draws 10 N_s samples per batch
    grover_iteration_policy: str = "maximize_unknown"
    epsilon_total: float = 1e-4             # rotation synthesis budget of the UCJ circuit
    energy_eval_interval_shots: int = 100   # exact energy every this many shots (plus bisection at the crossing)
    direct_energy_eval_interval_new_basis: int = 10
    stop_on_convergence: bool = True
    n_consecutive_failures: int = 3         # stagnation criterion of QSCI-CMP
    max_iterations: int | None = None
    direct_max_shots: int | None = 2_000_000


@dataclass
class Target:
    """The ground-state distribution of one system together with its subspace solver."""

    data: MoleculeData
    indices: np.ndarray            # nonzero determinants, sorted
    probabilities: np.ndarray      # their probabilities
    cumulative: np.ndarray
    solver: SubspaceSolver
    probability_of: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_molecule(cls, data: MoleculeData, amplitudes: tuple[np.ndarray, np.ndarray] | None = None) -> "Target":
        """Build the target from the exact ground state (or from explicitly given amplitudes).

        The weights of marked sets are summed over all nonzero determinants,
        while determinants of negligible probability are left out of the
        sampling arrays.
        """
        indices, amps = determinant_amplitudes(data) if amplitudes is None else amplitudes
        amps = np.asarray(amps)
        amps = amps / np.linalg.norm(amps)
        indices = np.asarray(indices, dtype=np.int64)
        probabilities = np.abs(amps) ** 2
        probability_of = dict(zip(indices.tolist(), probabilities.tolist()))
        keep = probabilities > NONZERO_PROBABILITY_EPS
        sampled_indices, sampled_probabilities = indices[keep], probabilities[keep]
        cumulative = np.cumsum(sampled_probabilities, dtype=np.float64)
        cumulative[-1] = 1.0
        return cls(data, sampled_indices, sampled_probabilities, cumulative, SubspaceSolver(data), probability_of)

    @property
    def n_qubits(self) -> int:
        return self.data.n_qubits

    @property
    def hf(self) -> int:
        return hf_index(self.data.n_ele_cas)

    @property
    def exact_energy(self) -> float:
        return self.data.exact_energy

    def weight(self, support) -> float:
        """Weight of a set of determinants in the ground state.

        The sum is evaluated with ``math.fsum``, which returns the correctly
        rounded value independently of the order of the terms, so the result
        does not depend on how the set is stored.
        """
        get = self.probability_of.get
        return math.fsum(get(int(i), 0.0) for i in support)

    def energy(self, support) -> float:
        return self.solver.ground_energy(support)


@dataclass
class TrialState:
    support: set[int]
    energy: float
    energy_exact: bool = True
    last_exact_total_shots: int = 0
    last_exact_support: set[int] = field(default_factory=set)
    total_shots: int = 0
    total_queries: int = 0
    total_t: float = 0.0
    total_stateprep_t: float = 0.0
    total_oracle_t: float = 0.0
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.last_exact_support:
            self.last_exact_support = set(self.support)


def trial_rng(config: Config, method: str, trial: int) -> np.random.Generator:
    return np.random.default_rng(config.seed + _SEED_OFFSETS[method] + 104729 * trial)


def initial_support(target: Target, method: str) -> set[int]:
    """M u T before any sampling: the HF determinant, or the preselected subspace T(R)."""
    if method in ("direct", "sqdaa"):
        return {target.hf}
    key = method.split("-", 1)[1]
    return PRESELECTIONS[key](target.n_qubits, target.data.n_ele_cas)


# --------------------------------------------------------------------------- energy bookkeeping

def _record(state: TrialState, entry: dict) -> None:
    entry.update({
        "energy": float(state.energy), "energy_exact": bool(state.energy_exact), "support_size": len(state.support),
        "total_shots": int(state.total_shots), "total_queries": int(state.total_queries),
        "total_stateprep_t": float(state.total_stateprep_t), "total_oracle_t": float(state.total_oracle_t),
        "total_t": float(state.total_t),
    })
    state.history.append(entry)


def _set_exact_energy(target: Target, state: TrialState, energy: float | None = None) -> None:
    state.energy = target.energy(state.support) if energy is None else float(energy)
    state.energy_exact = True
    state.last_exact_total_shots = int(state.total_shots)
    state.last_exact_support = set(state.support)
    if state.history:
        state.history[-1].update({"energy": state.energy, "energy_exact": True, "support_size": len(state.support),
                                  "total_shots": state.total_shots, "total_queries": state.total_queries,
                                  "total_stateprep_t": state.total_stateprep_t, "total_oracle_t": state.total_oracle_t,
                                  "total_t": state.total_t})


def _refresh_energy(target: Target, state: TrialState, config: Config) -> bool:
    """Evaluate the exact energy at the configured interval.

    When the energy drops below epsilon inside the window since the last
    exact evaluation, the first iteration inside the window that reaches
    epsilon is located by bisection and, with ``stop_on_convergence``, the
    state is truncated to that iteration.
    """
    interval = max(1, int(config.energy_eval_interval_shots))
    iteration = int(state.history[-1]["iteration"]) if state.history else 0
    if state.energy_exact or ((state.total_shots - state.last_exact_total_shots) < interval and iteration != 1):
        return False
    boundary_shots = int(state.last_exact_total_shots)
    boundary_support = set(state.last_exact_support)
    window_start = next((i for i, e in enumerate(state.history) if int(e["total_shots"]) > boundary_shots), len(state.history))
    window = state.history[window_start:]
    latest_energy = target.energy(state.support)
    previous_exact = next((e for e in reversed(state.history[:window_start]) if e.get("energy_exact")), None)
    previous_error = None if previous_exact is None else abs(float(previous_exact["energy"]) - target.exact_energy)
    reached = abs(latest_energy - target.exact_energy) <= config.epsilon
    best_index = best_support = best_energy = None
    if previous_error is not None and previous_error > config.epsilon and reached and window:
        low, high = 0, len(window) - 1
        while low <= high:
            mid = (low + high) // 2
            candidate_support = set(boundary_support)
            for e in window[: mid + 1]:
                candidate_support.update(e["new_indices"])
            candidate_energy = target.energy(candidate_support)
            window[mid].update({"energy": float(candidate_energy), "energy_exact": True})
            if abs(candidate_energy - target.exact_energy) <= config.epsilon:
                best_index, best_support, best_energy, high = mid, candidate_support, candidate_energy, mid - 1
            else:
                low = mid + 1
    if config.stop_on_convergence and reached and window:
        if best_index is None:
            best_index, best_support, best_energy = len(window) - 1, set(state.support), latest_energy
        cutoff = state.history[window_start + best_index]
        state.history = state.history[: window_start + best_index + 1]
        state.support = set(best_support)
        state.total_shots = int(cutoff["total_shots"])
        state.total_queries = int(cutoff["total_queries"])
        state.total_t = float(cutoff["total_t"])
        state.total_stateprep_t = float(cutoff["total_stateprep_t"])
        state.total_oracle_t = float(cutoff["total_oracle_t"])
        _set_exact_energy(target, state, best_energy)
        return True
    _set_exact_energy(target, state, latest_energy)
    return True


# --------------------------------------------------------------------------- one amplification round

def _amplification_round(
    target: Target, state: TrialState, config: Config, rng: np.random.Generator, iteration: int,
    *, oracle_t_per_round: Callable[[int], float], accept: Callable[[int], bool] | None = None,
) -> int:
    """Amplify the unmeasured determinants, measure N_s times, and add the new determinants to M."""
    marked = set(state.support)
    marked_probability = target.weight(marked)
    unmarked_probability = max(0.0, 1.0 - marked_probability)
    k = choose_grover_iterations(unmarked_probability, config.grover_iteration_policy)
    marked_scale, unmarked_scale = amplitude_scales(marked_probability, unmarked_probability, k)
    shots = config.shots_per_iteration
    sampled = sample_amplified(
        rng, shots, cumulative_probabilities=target.cumulative, indices=target.indices,
        marked=np.fromiter(marked, dtype=np.int64, count=len(marked)),
        marked_probability=marked_probability, unmarked_probability=unmarked_probability,
        marked_scale=marked_scale, unmarked_scale=unmarked_scale,
    )
    accepted = {i: c for i, c in sampled.items() if accept is None or accept(int(i))}
    new_indices = sorted(int(i) for i in accepted if int(i) not in marked)
    state.support.update(accepted)
    if new_indices:
        state.energy_exact = False
    query_increment = shots * (2 * k + 1)
    stateprep_increment = query_increment * state_preparation_t_count(target.n_qubits, target.data.n_orb_cas, config.epsilon_total)
    oracle_increment = shots * oracle_t_per_round(k)
    state.total_shots += shots
    state.total_queries += query_increment
    state.total_stateprep_t += stateprep_increment
    state.total_oracle_t += oracle_increment
    state.total_t += stateprep_increment + oracle_increment
    _record(state, {
        "iteration": int(iteration), "grover_iterations": int(k), "shots": int(shots),
        "marked_probability": float(marked_probability), "unmarked_probability": float(unmarked_probability),
        "sampled_counts": {int(i): int(c) for i, c in sampled.items()},
        "new_indices": new_indices, "new_unique_count": len(new_indices),
        "query_increment": int(query_increment), "stateprep_increment": float(stateprep_increment), "oracle_increment": float(oracle_increment),
    })
    _refresh_energy(target, state, config)
    return len(new_indices)


def _converged(target: Target, state: TrialState, config: Config) -> bool:
    return config.stop_on_convergence and abs(state.energy - target.exact_energy) <= config.epsilon


# --------------------------------------------------------------------------- the three methods


@dataclass
class TrialRun:
    """A resumable trial: one call to ``step`` performs one iteration.

    The random generator state is stored with the run, so a run that is
    pickled and resumed reproduces the uninterrupted run exactly.
    """

    target: Target
    config: Config
    method: str
    trial: int
    state: TrialState
    rng: np.random.Generator
    iteration: int = 0
    failures: int = 0       # consecutive iterations without a new determinant (QSCI-CMP)
    pending: int = 0        # new determinants since the last exact energy (direct QSCI)
    finished: bool = False
    stop_reason: str = ""

    @classmethod
    def start(cls, target: Target, config: Config, method: str, trial: int) -> "TrialRun":
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}. Available: {', '.join(METHODS)}")
        support = initial_support(target, method)
        state = TrialState(support=set(support), energy=target.energy(support))
        return cls(target, config, method, trial, state, trial_rng(config, method, trial))

    def _finish(self, reason: str) -> None:
        self.finished, self.stop_reason = True, reason

    def step(self) -> None:
        if self.finished:
            return
        if _converged(self.target, self.state, self.config):
            return self._finish("converged")
        self.iteration += 1
        if self.config.max_iterations is not None and self.iteration > self.config.max_iterations:
            self.iteration -= 1
            return self._finish("max_iterations")
        if self.method == "direct":
            return self._direct_step()
        if 1.0 - self.target.weight(self.state.support) <= 0.0:
            return self._finish("support_exhausted")
        n = self.target.n_qubits
        if self.method == "sqdaa":
            n_enumerated = len(self.state.support)
            new = _amplification_round(self.target, self.state, self.config, self.rng, self.iteration,
                                       oracle_t_per_round=lambda k, s=n_enumerated: enumerated_oracle_t_count(n, s, k))
            if new == 0:
                self._finish("stagnation")
        else:
            n_enumerated = len(self.state.support) - self._n_fixed
            fixed_oracle_t = preselection_oracle_t_count(n)
            new = _amplification_round(
                self.target, self.state, self.config, self.rng, self.iteration,
                oracle_t_per_round=lambda k, s=n_enumerated: k * fixed_oracle_t + enumerated_oracle_t_count(n, s, k),
            )
            self.failures = 0 if new > 0 else self.failures + 1
            if self.failures >= self.config.n_consecutive_failures:
                self._finish("stagnation")
        if not self.finished and _converged(self.target, self.state, self.config):
            self._finish("converged")

    @property
    def _n_fixed(self) -> int:
        return len(initial_support(self.target, self.method))

    def _direct_step(self) -> None:
        target, state, config = self.target, self.state, self.config
        shots = config.shots_per_iteration * config.direct_shots_multiplier
        max_batches = None if config.direct_max_shots is None else max(1, int(np.ceil(config.direct_max_shots / shots)))
        if max_batches is not None and self.iteration > max_batches:
            self.iteration -= 1
            return self._finish("max_shots")
        stateprep_t = state_preparation_t_count(target.n_qubits, target.data.n_orb_cas, config.epsilon_total)
        sampled = target.indices[np.searchsorted(target.cumulative, self.rng.random(shots), side="right")]
        new_indices = sorted(int(i) for i in np.unique(sampled) if int(i) not in state.support)
        state.support.update(new_indices)
        state.total_shots += shots
        state.total_queries += shots
        state.total_t += shots * stateprep_t
        state.total_stateprep_t = state.total_t
        if not new_indices:
            return
        state.energy_exact = False
        self.pending += len(new_indices)
        _record(state, {
            "iteration": int(self.iteration), "grover_iterations": 0, "shots": int(shots),
            "sampled_counts": {int(i): int(c) for i, c in Counter(int(i) for i in sampled).items()},
            "new_indices": new_indices, "new_unique_count": len(new_indices),
            "query_increment": int(shots), "stateprep_increment": float(shots * stateprep_t), "oracle_increment": 0.0,
        })
        if self.pending >= max(1, int(config.direct_energy_eval_interval_new_basis)):
            _set_exact_energy(target, state)
            self.pending = 0
        if _converged(target, state, config):
            self._finish("converged")

    def run(self, log=None, max_steps: int | None = None) -> TrialState:
        steps = 0
        while not self.finished and (max_steps is None or steps < max_steps):
            self.step()
            steps += 1
            if log and self.state.history:
                log(self.state)
        return self.state

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["rng"] = self.rng.bit_generator.state
        d["target"] = None
        return d

    def __setstate__(self, d: dict) -> None:
        rng = np.random.default_rng()
        rng.bit_generator.state = d["rng"]
        d["rng"] = rng
        self.__dict__.update(d)


def run_trial(target: Target, config: Config, method: str, trial: int, log=None) -> TrialState:
    """Run one trial of ``method`` to completion and return its final state."""
    return TrialRun.start(target, config, method, trial).run(log=log)


def trial_payload(target: Target, config: Config, method: str, trial: int, state: TrialState) -> dict:
    """Plain-dictionary record of one trial, suitable for pickling."""
    return {
        "system": target.data.name, "label": target.data.label, "n_qubits": target.n_qubits,
        "method": method, "trial": int(trial), "config": asdict(config),
        "exact_energy": float(target.exact_energy),
        "initial_support_size": len(initial_support(target, method)),
        "initial_energy": float(target.energy(initial_support(target, method))),
        "final": {
            "energy": float(state.energy), "energy_exact": bool(state.energy_exact),
            "energy_error": abs(float(state.energy) - target.exact_energy),
            "support_size": len(state.support), "total_shots": int(state.total_shots),
            "total_queries": int(state.total_queries), "total_t": float(state.total_t),
            "total_stateprep_t": float(state.total_stateprep_t), "total_oracle_t": float(state.total_oracle_t),
        },
        "history": state.history,
    }
