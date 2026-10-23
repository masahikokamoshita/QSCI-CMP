"""The simulation reproduces the archived runs of the paper iteration by iteration.

The archived records in ``results/checkpoints`` were produced with the same
random streams.  Rebuilding the molecule with PySCF and rerunning trial 0
of each method must reproduce the recorded samples, iteration counts,
costs, and exact energies.  The comparison covers the first iterations
only, to keep the test short.
"""
from pathlib import Path

import pytest

from qscicmp.checkpoint import load_trial
from qscicmp.chem import load_or_build_molecule
from qscicmp.simulation import Config, Target, TrialRun

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "checkpoints"
MOLECULES = ROOT / "data" / "molecules"
SYSTEM = "H2O_12o10e"


@pytest.fixture(scope="module")
def target():
    return Target.from_molecule(load_or_build_molecule(MOLECULES, SYSTEM))


@pytest.mark.parametrize("method, steps", [("sqdaa", 40), ("cmp-h25", 20)])
def test_trial_zero_matches_archive(target, method, steps):
    path = RESULTS / f"{SYSTEM}_{method}_trial000.pkl"
    if not path.exists():
        pytest.skip(f"archived record {path.name} is not available")
    archived = load_trial(path)["history"]
    run = TrialRun.start(target, Config(epsilon=1.6e-5), method, 0)
    run.run(max_steps=steps)
    assert len(run.state.history) == steps
    for recorded, entry in zip(archived, run.state.history):
        assert entry["grover_iterations"] == recorded["grover_iterations"]
        assert entry["sampled_counts"] == recorded["sampled_counts"]
        assert entry["support_size"] == recorded["support_size"]
        assert entry["total_queries"] == recorded["total_queries"]
        assert entry["total_t"] == pytest.approx(recorded["total_t"], rel=1e-12)
        if recorded["energy_exact"] and entry["energy_exact"]:
            assert entry["energy"] == pytest.approx(recorded["energy"], abs=1e-10)
