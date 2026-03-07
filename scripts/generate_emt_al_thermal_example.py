#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a small periodic Al dataset labeled by ASE EMT for thermal-transport end-to-end tests."
    )
    parser.add_argument("--output", required=True, help="Output extxyz file")
    parser.add_argument("--n-structures", type=int, default=64, help="Number of distorted structures")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--a0", type=float, default=4.05, help="Reference cubic lattice constant in Angstrom")
    parser.add_argument("--max-strain", type=float, default=0.02, help="Maximum random strain magnitude")
    parser.add_argument("--max-rattle", type=float, default=0.04, help="Maximum random displacement magnitude in Angstrom")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    structures = []
    for idx in range(args.n_structures):
        atoms = bulk("Al", "fcc", a=args.a0, cubic=True)

        # Small symmetric strain keeps structures close to the harmonic region.
        raw = rng.normal(size=(3, 3))
        sym = 0.5 * (raw + raw.T)
        sym /= max(np.abs(sym).max(), 1e-12)
        strain = np.eye(3) + args.max_strain * sym
        atoms.set_cell(atoms.cell.array @ strain, scale_atoms=True)

        displacement_scale = args.max_rattle * (0.5 + 0.5 * np.cos(idx / max(args.n_structures - 1, 1) * np.pi))
        atoms.positions += rng.normal(scale=displacement_scale, size=atoms.positions.shape)

        atoms.calc = EMT()
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()
        structures.append(atoms)

    write(output, structures, format="extxyz")
    print(f"Wrote {len(structures)} structures to {output}")


if __name__ == "__main__":
    main()
