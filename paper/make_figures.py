"""Figures 2 and 3 of the paper: energy error versus cumulative T-gate count and versus cumulative queries."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qscicmp.analysis import CHEMICAL_ACCURACY, median_curve, trial_curve
from qscicmp.checkpoint import load_trials
from qscicmp.simulation import METHOD_LABELS

PANELS = [
    ("H10", r"$\mathrm{H}_{10}$"), ("H2O_10o10e", r"$\mathrm{H}_2\mathrm{O}$ (20 qubits)"), ("Cr2_10o10e", r"$\mathrm{Cr}_2$ (20 qubits)"),
    ("H12", r"$\mathrm{H}_{12}$"), ("H2O_12o10e", r"$\mathrm{H}_2\mathrm{O}$ (24 qubits)"), ("Cr2_12o12e", r"$\mathrm{Cr}_2$ (24 qubits)"),
]
METHOD_ORDER = ("direct", "sqdaa", "cmp-e2", "cmp-h20", "cmp-h25")
METHOD_COLORS = {"direct": "#4c72b0", "sqdaa": "#dd8452", "cmp-e2": "#55a868", "cmp-h20": "#17becf", "cmp-h25": "#bcbd22"}
PLOT_FLOOR = 1.6e-5   # the H2O and Cr2 runs were stopped at this accuracy
GRID_POINTS = 400


def plot(results: Path, output: Path, x_key: str, xlabel: str, representative: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 8.6))
    legend = {}
    for panel_index, (ax, (system, title)) in enumerate(zip(axes.flat, PANELS)):
        floor = PLOT_FLOOR if system.startswith(("H2O", "Cr2")) else None
        sqdaa_max_x = None
        seed_segments = []
        for method in METHOD_ORDER:
            payloads = load_trials(results, system, method)
            if not payloads:
                continue
            if representative == "median":
                xs, ys = median_curve(payloads, x_key, floor, n_grid=GRID_POINTS)
            else:
                xs, ys = trial_curve(payloads[0], x_key, floor)
            mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
            if method.startswith("cmp-"):
                initial = payloads[0].get("initial_energy")
                if initial is not None:
                    error = abs(float(initial) - float(payloads[0]["exact_energy"]))
                    seed_segments.append((error, METHOD_COLORS[method], float(xs[mask].min()) if np.any(mask) else None))
            if not np.any(mask):
                continue
            line, = ax.plot(xs[mask], ys[mask], linewidth=2.4, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            legend.setdefault(METHOD_LABELS[method], line)
            if method == "sqdaa":
                sqdaa_max_x = float(xs[mask].max())
        ax.axhline(CHEMICAL_ACCURACY, color="black", linestyle="--", linewidth=1.4, label="chemical accuracy")
        ax.set_title(title, fontsize=19)
        ax.set_xlabel(xlabel, fontsize=17)
        ax.set_ylabel("Energy error (Ha)" if panel_index % 3 == 0 else "", fontsize=17)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis="both", which="major", labelsize=13)
        ax.grid(True, which="both", linestyle=":", linewidth=0.9, alpha=0.7)
        if floor is not None:
            ax.set_ylim(bottom=floor)
        if sqdaa_max_x:
            left, _ = ax.get_xlim()
            ax.set_xlim(left=left, right=sqdaa_max_x * 1.2)
        xmin, xmax = ax.get_xlim()
        for error, color, first_x in seed_segments:
            ax.hlines(error, xmin=xmin, xmax=first_x if first_x else xmax, color=color, linestyle=":", linewidth=1.8, alpha=0.75)
    legend["chemical accuracy"] = axes.flat[0].lines[-1]
    fig.legend(legend.values(), legend.keys(), loc="center left", frameon=False, bbox_to_anchor=(0.92, 0.5), fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 0.90, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures"))
    parser.add_argument("--representative", choices=["median", "trial0"], default="median",
                        help="plot the median over trials (default) or the first trial")
    parser.add_argument("--format", default="pdf")
    args = parser.parse_args()
    plot(args.results, args.output / f"plot_tgate_QSCI-CMP_all.{args.format}", "total_t", "Total T gate count", args.representative)
    plot(args.results, args.output / f"plot_query_QSCI-CMP_all.{args.format}", "total_queries", r"Total $U_{\mathrm{GS}}$ queries", args.representative)


if __name__ == "__main__":
    main()
