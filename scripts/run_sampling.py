"""Run the sampling simulation of one method on one system and store the trial records."""
from __future__ import annotations

import argparse
from pathlib import Path

from qscicmp.checkpoint import checkpoint_path, save_trial
from qscicmp.chem import load_or_build_molecule
from qscicmp.simulation import METHODS, Config, Target, run_trial, trial_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--first-trial", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1.6e-3, help="stopping accuracy in Hartree (the paper runs used 1.6e-5 for most systems; see README)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shots", type=int, default=10, help="N_s, measurements per iteration")
    parser.add_argument("--policy", default="maximize_unknown", choices=["maximize_unknown", "min_query_scaling"])
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--molecules", type=Path, default=Path("data/molecules"))
    parser.add_argument("--output", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data = load_or_build_molecule(args.molecules, args.system)
    target = Target.from_molecule(data)
    config = Config(epsilon=args.epsilon, seed=args.seed, shots_per_iteration=args.shots,
                    grover_iteration_policy=args.policy, max_iterations=args.max_iterations)

    def log(state):
        if not args.quiet and state.history and state.history[-1]["energy_exact"]:
            entry = state.history[-1]
            print(f"  iteration {entry['iteration']:5d}  support {entry['support_size']:7d}  queries {entry['total_queries']:10d}  "
                  f"T {entry['total_t']:.3e}  error {abs(entry['energy'] - target.exact_energy):.3e}", flush=True)

    for trial in range(args.first_trial, args.first_trial + args.trials):
        print(f"{args.system} {args.method} trial {trial}", flush=True)
        state = run_trial(target, config, args.method, trial, log=log)
        payload = trial_payload(target, config, args.method, trial, state)
        path = save_trial(payload, checkpoint_path(args.output, args.system, args.method, trial, args.policy))
        print(f"  final error {payload['final']['energy_error']:.3e} after {payload['final']['total_queries']} queries -> {path}", flush=True)


if __name__ == "__main__":
    main()
