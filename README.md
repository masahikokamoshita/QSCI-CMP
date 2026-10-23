# QSCI-CMP

Code and data for

> M. Kamoshita and K. Mitarai, *QSCI-CMP: Quantum-Selected Configuration Interaction with Chemically Motivated Preselection*, [arXiv:2608.05766](https://arxiv.org/abs/2608.05766).

QSCI-CMP is a quantum-classical hybrid algorithm for molecular ground-state calculations. It extends sample-based quantum diagonalization with amplitude amplification (SQD-AA) by fixing a chemically trivial subspace of determinants in advance, placing it in the diagonalization subspace from the start, and excluding it from the amplification target through a low-cost oracle that evaluates the excitation level and the seniority number in superposition.

This repository contains

- a simulator of the sampling procedure of direct QSCI, SQD-AA, and QSCI-CMP on the exact ground state of a molecular active space, including the query count and the T-gate cost model of the paper,
- the molecular systems of the paper (hydrogen chains, H2O, Cr2) built from scratch with PySCF,
- the records of the simulation runs behind Figures 2 and 3 and the numbers of Sec. IV B, and
- the scripts that produce the figures and the numbers from those records.

## Layout

```
qscicmp/                 library
  chem/molecules.py        geometries, basis sets, and active spaces of the six systems
  chem/hamiltonian.py      RHF -> CASCI with PySCF, active-space integrals and FCI ground state
  chem/subspace.py         determinant indexing and diagonalization in a determinant subspace
  preselection.py          excitation level, seniority number, hierarchy number, and T(R)
  amplification.py         exact simulation of amplitude amplification and the iteration count
  cost.py                  T-gate cost of the UCJ state preparation and of the oracles
  simulation.py            direct QSCI, SQD-AA, and QSCI-CMP sampling runs
  checkpoint.py            trial records
  analysis.py              costs to reach an accuracy, reductions, and curves
scripts/
  build_molecules.py       build and cache the six systems
  run_sampling.py          run one method on one system
paper/
  make_figures.py          Figures 2 and 3
  make_tables.py           the numbers of Sec. IV B
data/molecules/          cached systems (metadata.json and molecule.npz per system)
results/checkpoints/     trial records (one pickle per system, method, and trial)
tests/                   unit tests and a reproduction test against the archived records
```

## Installation

Python 3.10 or later. The only dependencies are NumPy, SciPy, PySCF, and Matplotlib.

```
pip install -r requirements.txt
pip install -e .
```

The code was verified with Python 3.12, NumPy 2.4, SciPy 1.17, PySCF 2.14, and Matplotlib 3.10.

The repository on GitHub contains the code only. The archive on Zenodo additionally contains `data/molecules/` and `results/checkpoints/`; with the archive in place, the figures and the numbers of the paper can be produced without rerunning anything.

## Reproducing the paper

Build the six systems (a few minutes; Cr2 in cc-pVDZ takes the longest):

```
python scripts/build_molecules.py
```

Run the sampling simulations. Each call runs one method on one system and writes one record per trial to `results/checkpoints/`:

```
python scripts/run_sampling.py --system H2O_12o10e --method sqdaa   --epsilon 1.6e-5
python scripts/run_sampling.py --system H2O_12o10e --method cmp-h25 --epsilon 1.6e-5
```

The methods are `direct` (QSCI), `sqdaa`, `cmp-e2`, `cmp-h20`, and `cmp-h25` (QSCI-CMP with the preselections e <= 2, h <= 2, and h <= 2.5). The systems are `H10`, `H12`, `H2O_10o10e`, `H2O_12o10e`, `Cr2_10o10e`, and `Cr2_12o12e`; the suffix gives the active space as (orbitals, electrons).

Produce the figures and the numbers from the records:

```
python paper/make_figures.py            # paper/figures/plot_tgate_QSCI-CMP_all.pdf and plot_query_QSCI-CMP_all.pdf
python paper/make_tables.py --bootstrap --oracle  # costs to reach chemical accuracy, reductions, bootstrap intervals, and the weight of the oracle
```

`make_figures.py` plots the median over the trials of the energy error as a function of the cumulative cost; `--representative trial0` plots the first trial instead.

### Settings of the archived runs

All runs use N_s = 10 measurements per iteration, the iteration count that maximizes the success probability (`--policy maximize_unknown`), seed 7, and ten trials. The runs were stopped at the accuracy below; the costs to reach chemical accuracy (1.6 mHa) are read off the records afterwards.

| System | Stopping accuracy (Ha) |
|---|---|
| H10 | 1.6e-5, or 2000 iterations for SQD-AA |
| H12 | 1.6e-3 |
| H2O (10e, 10o), H2O (10e, 12o) | 1.6e-5 |
| Cr2 (10e, 10o), Cr2 (12e, 12o) | 1.6e-5 |

Direct QSCI draws 100 samples per batch and was run up to 2 million samples. For Cr2 and H2O (10e, 10o), the preselected subspace alone reaches chemical accuracy, so the QSCI-CMP records of these systems start below chemical accuracy.

The random stream of a trial is fixed by the seed, the method, and the trial index, so a run can be repeated exactly. The cost of each iteration is derived from the recorded iteration counts with the cost model in `qscicmp/cost.py`.

The records with the suffix `_min_query_scaling` (H12 with all four amplification methods, and H2O (10e, 12o) with QSCI-CMP at h <= 2.5) use the query-optimal iteration count discussed in Sec. II C of the paper, which stops the amplification at (2k + 1) theta ~ 1.17 (`--policy min_query_scaling`). They are not used in the figures. Compared with the `maximize_unknown` records, they reach chemical accuracy with 12.1 to 12.4% fewer queries for H12 and 10.7% for H2O (10e, 12o), in line with the 12% expected from the analysis of quantum search. The H2O value is less precise because that run reaches chemical accuracy within about thirty iterations, so the one-iteration resolution of the reach point amounts to several percent of the total (95% bootstrap interval 7 to 17%). The comparison can be made with `qscicmp.analysis.reach_statistics` and `qscicmp.checkpoint.load_trials(root, system, method, "min_query_scaling")`.

## Record format

Each record is a pickled dictionary with the system, method, trial, settings, exact energy, initial subspace size and energy, and a `history` list with one entry per iteration: the number of amplification rounds, the sampled determinants, the newly added determinants, the subspace size, the cumulative numbers of measurements and of queries to the state-preparation unitary, the cumulative T-gate counts of the state preparation and of the oracle, and the subspace energy where it was evaluated exactly (`energy_exact`). Determinants are integers whose bits record the spin-orbital occupations in the interleaved ordering of the paper (bit 2p for the alpha and bit 2p + 1 for the beta spin orbital of spatial orbital p).

## Tests

```
pytest
```

The tests check the energies of the cached systems, the sizes of the preselected subspaces, the cost model, and that rerunning the first iterations of trial 0 of SQD-AA and QSCI-CMP on H2O (10e, 12o) reproduces the archived samples, costs, and energies.

## License

MIT, see `LICENSE`.
