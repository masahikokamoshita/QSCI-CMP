"""Active-space Hamiltonian and exact ground state from PySCF.

The workflow is RHF -> CASCI (frontier active space) -> FCI within the
active space.  The result is stored as the active-space integrals
(h1, eri, E_core) together with the FCI coefficient tensor, which is all
that the sampling simulations need.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf
from pyscf.fci.cistring import make_strings

from qscicmp.chem.molecules import SystemSpec, get_system


@dataclass
class MoleculeData:
    """Everything the simulations need about one molecular system."""

    name: str
    label: str
    n_orb_cas: int
    n_ele_cas: int
    h1: np.ndarray          # (N_o, N_o) one-electron integrals in the active space
    eri: np.ndarray         # (N_o, N_o, N_o, N_o) two-electron integrals, chemist notation
    e_core: float           # nuclear repulsion plus frozen-core energy
    ci: np.ndarray          # (n_alpha_strings, n_beta_strings) FCI coefficients
    alpha_strings: np.ndarray
    beta_strings: np.ndarray
    hf_energy: float
    exact_energy: float

    @property
    def n_qubits(self) -> int:
        return 2 * self.n_orb_cas

    @property
    def n_alpha(self) -> int:
        return self.n_ele_cas // 2

    @property
    def n_beta(self) -> int:
        return self.n_ele_cas - self.n_alpha


def build_molecule(spec: SystemSpec, verbose: int = 0) -> MoleculeData:
    mol = gto.Mole()
    mol.build(atom=list(spec.geometry), basis=spec.basis, spin=0, charge=0, symmetry=spec.symmetry, verbose=verbose)
    mf = scf.RHF(mol).run()
    cas = mcscf.CASCI(mf, spec.n_orb_cas, spec.n_ele_cas)
    cas.verbose = verbose
    cas.kernel()
    h1, e_core = cas.get_h1eff()
    eri = ao2mo.restore(1, cas.get_h2eff(), spec.n_orb_cas)
    n_alpha = spec.n_ele_cas // 2
    n_beta = spec.n_ele_cas - n_alpha
    orbitals = list(range(spec.n_orb_cas))
    return MoleculeData(
        name=spec.name, label=spec.label, n_orb_cas=spec.n_orb_cas, n_ele_cas=spec.n_ele_cas,
        h1=np.asarray(h1), eri=np.asarray(eri), e_core=float(e_core),
        ci=np.asarray(cas.ci, dtype=np.float64),
        alpha_strings=np.asarray(make_strings(orbitals, n_alpha), dtype=np.int64),
        beta_strings=np.asarray(make_strings(orbitals, n_beta), dtype=np.int64),
        hf_energy=float(mf.e_tot), exact_energy=float(cas.e_tot),
    )


def molecule_dir(root: Path, name: str) -> Path:
    return Path(root) / name


def save_molecule(data: MoleculeData, root: Path) -> Path:
    directory = molecule_dir(root, data.name)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "molecule.npz",
        h1=data.h1, eri=data.eri, ci=data.ci,
        alpha_strings=data.alpha_strings, beta_strings=data.beta_strings,
    )
    metadata = {
        "name": data.name, "label": data.label,
        "n_orb_cas": data.n_orb_cas, "n_ele_cas": data.n_ele_cas, "n_qubits": data.n_qubits,
        "e_core": data.e_core, "hf_energy": data.hf_energy, "exact_energy": data.exact_energy,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return directory


def load_molecule(root: Path, name: str) -> MoleculeData:
    directory = molecule_dir(root, name)
    metadata = json.loads((directory / "metadata.json").read_text())
    arrays = np.load(directory / "molecule.npz")
    return MoleculeData(
        name=metadata["name"], label=metadata["label"],
        n_orb_cas=int(metadata["n_orb_cas"]), n_ele_cas=int(metadata["n_ele_cas"]),
        h1=arrays["h1"], eri=arrays["eri"], e_core=float(metadata["e_core"]),
        ci=arrays["ci"], alpha_strings=arrays["alpha_strings"], beta_strings=arrays["beta_strings"],
        hf_energy=float(metadata["hf_energy"]), exact_energy=float(metadata["exact_energy"]),
    )


def load_or_build_molecule(root: Path, name: str, verbose: int = 0) -> MoleculeData:
    directory = molecule_dir(root, name)
    if (directory / "molecule.npz").exists():
        return load_molecule(root, name)
    data = build_molecule(get_system(name), verbose=verbose)
    save_molecule(data, root)
    return data
