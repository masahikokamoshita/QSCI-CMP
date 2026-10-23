"""Reading and writing trial records.

A trial record is a plain dictionary (see ``simulation.trial_payload``)
pickled to ``results/checkpoints/<system>_<method>_trial<NNN>.pkl``.
"""
from __future__ import annotations

import pickle
from pathlib import Path


def checkpoint_path(root: Path, system: str, method: str, trial: int, policy: str = "maximize_unknown") -> Path:
    suffix = "" if policy == "maximize_unknown" else f"_{policy}"
    return Path(root) / f"{system}_{method}{suffix}_trial{trial:03d}.pkl"


def save_trial(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_trial(path: Path) -> dict:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_trials(root: Path, system: str, method: str, policy: str = "maximize_unknown") -> list[dict]:
    suffix = "" if policy == "maximize_unknown" else f"_{policy}"
    paths = sorted(Path(root).glob(f"{system}_{method}{suffix}_trial*.pkl"))
    return [load_trial(path) for path in paths]


def exact_points(payload: dict) -> list[dict]:
    """History entries whose energy was evaluated exactly, with the energy error attached."""
    exact_energy = float(payload["exact_energy"])
    points = []
    for entry in payload["history"]:
        if entry.get("energy_exact"):
            points.append({**entry, "energy_error": abs(float(entry["energy"]) - exact_energy)})
    return points


def reach_point(payload: dict, epsilon: float) -> dict | None:
    """First exact point whose energy error is at most ``epsilon`` (or the initial point, at zero cost)."""
    initial_energy = payload.get("initial_energy")
    initial_error = None if initial_energy is None else abs(float(initial_energy) - float(payload["exact_energy"]))
    if initial_error is not None and initial_error <= epsilon:
        return {"iteration": 0, "total_queries": 0, "total_t": 0.0, "total_shots": 0, "energy_error": initial_error,
                "total_stateprep_t": 0.0, "total_oracle_t": 0.0}
    for point in exact_points(payload):
        if point["energy_error"] <= epsilon:
            return point
    return None
