"""Reproduce the numbers quoted in Sec. IV B of the paper from the trial records."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qscicmp.analysis import CHEMICAL_ACCURACY, bootstrap_reduction_interval, oracle_statistics, reach_statistics, reduction
from qscicmp.checkpoint import load_trials
from qscicmp.chem.molecules import PAPER_SYSTEMS
from qscicmp.simulation import METHOD_LABELS

PRESELECTION_METHODS = ("cmp-e2", "cmp-h20", "cmp-h25")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--epsilon", type=float, default=CHEMICAL_ACCURACY)
    parser.add_argument("--bootstrap", action="store_true", help="also print 95% bootstrap intervals of the median reductions")
    parser.add_argument("--oracle", action="store_true", help="also print the weight of the oracle in the T-gate count")
    args = parser.parse_args()

    print(f"Cost to reach {args.epsilon:.1e} Ha (median over trials; mean in parentheses)")
    for system in PAPER_SYSTEMS:
        stats = {m: reach_statistics(load_trials(args.results, system, m), args.epsilon) for m in ("sqdaa",) + PRESELECTION_METHODS}
        if stats["sqdaa"].n_trials == 0:
            continue
        print(f"\n{system}")
        for method, stat in stats.items():
            if stat.n_trials == 0:
                continue
            print(f"  {METHOD_LABELS[method]:24s} reached {stat.n_reached}/{stat.n_trials}  queries {stat.median_queries:10.0f} ({stat.mean_queries:10.0f})  T gates {stat.median_t:.3e} ({stat.mean_t:.3e})")
        sqdaa = stats["sqdaa"]
        for method in PRESELECTION_METHODS:
            stat = stats[method]
            if stat.n_reached == 0 or sqdaa.n_reached == 0:
                continue
            q_med, t_med = reduction(sqdaa, stat, "median")
            q_mean, t_mean = reduction(sqdaa, stat, "mean")
            line = f"  reduction vs SQD-AA, {METHOD_LABELS[method]:24s} queries {q_med:5.1f}% ({q_mean:5.1f}%)  T gates {t_med:5.1f}% ({t_mean:5.1f}%)"
            if args.bootstrap and stat.queries.size > 1:
                (ql, qh), (tl, th) = bootstrap_reduction_interval(sqdaa, stat)
                line += f"   95% CI queries [{ql:.1f}, {qh:.1f}]  T [{tl:.1f}, {th:.1f}]"
            print(line)
        if args.oracle:
            print("  oracle weight in the T-gate count at chemical accuracy (median over trials)")
            print("    method                    cumulative share   cumulative ratio   reach-iteration share   reach-iteration ratio")
            for method in ("sqdaa",) + PRESELECTION_METHODS:
                stat = oracle_statistics(load_trials(args.results, system, method), args.epsilon)
                if stat.n_reached == 0:
                    continue
                print(f"    {METHOD_LABELS[method]:24s}  {100 * np.median(stat.cumulative_share):6.1f}%           {np.median(stat.cumulative_ratio):6.2f}             "
                      f"{100 * np.median(stat.reach_share):6.1f}%                 {np.median(stat.reach_ratio):6.2f}")


if __name__ == "__main__":
    main()
