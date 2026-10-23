"""Build and cache the active-space Hamiltonians and exact ground states of the paper's systems."""
from __future__ import annotations

import argparse
from pathlib import Path

from qscicmp.chem import PAPER_SYSTEMS, SYSTEMS, build_molecule, get_system, save_molecule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="*", default=list(PAPER_SYSTEMS), help=f"systems to build (default: all paper systems). Available: {', '.join(SYSTEMS)}")
    parser.add_argument("--output", type=Path, default=Path("data/molecules"))
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    for name in args.systems:
        data = build_molecule(get_system(name), verbose=args.verbose)
        directory = save_molecule(data, args.output)
        print(f"{name}: n_qubits={data.n_qubits} E_HF={data.hf_energy:.10f} E_exact={data.exact_energy:.10f} -> {directory}")


if __name__ == "__main__":
    main()
