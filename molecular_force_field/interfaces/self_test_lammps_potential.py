"""Offline self-test for the LAMMPS Python potential wrapper.

This does NOT require LAMMPS. It validates the core contract:
- The wrapper can load a checkpoint and compute (E, F)
- Forces are consistent with energy gradients via finite-difference check

Run:
    python -m molecular_force_field.interfaces.self_test_lammps_potential
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass

import numpy as np
import torch

from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.interfaces.lammps_potential import LAMMPSPotential


@dataclass
class SelfTestResult:
    energy_kcalmol: float
    max_abs_force_kcalmol_per_ang: float
    max_abs_fd_err_kcalmol_per_ang: float


def _make_dummy_checkpoint(path: str, device: torch.device) -> ModelConfig:
    # Keep dtype explicit and stable
    config = ModelConfig(dtype=torch.float64)

    # Provide some atomic energies to avoid relying on fitted_E0.csv
    config.atomic_energy_keys = torch.tensor([1, 8], dtype=torch.long)
    config.atomic_energy_values = torch.tensor([-13.6, -75.0], dtype=config.dtype)

    model = E3_TransformerLayer_multi(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        irreps_input=config.get_irreps_output_conv(),
        irreps_query=config.get_irreps_query_main(),
        irreps_key=config.get_irreps_key_main(),
        irreps_value=config.get_irreps_value_main(),
        irreps_output=config.get_irreps_output_conv_2(),
        irreps_sh=config.get_irreps_sh_transformer(),
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        function_type_main=config.function_type,
        device=device,
    ).to(device)

    ckpt = {
        "e3trans_state_dict": model.state_dict(),
        "dtype": "float64",
    }
    torch.save(ckpt, path)
    return config


def run_self_test(seed: int = 0, eps: float = 1e-4) -> SelfTestResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = f"{td}/dummy_model.pth"
        _ = _make_dummy_checkpoint(ckpt_path, device=device)

        pot = LAMMPSPotential(
            checkpoint_path=ckpt_path,
            device="cpu",
            max_radius=3.0,
            atomic_energy_keys=[1, 8],
            atomic_energy_values=[-13.6, -75.0],
            # Here we intentionally test the common LAMMPS convention:
            # type 1 -> H (Z=1), type 2 -> O (Z=8)
            type_to_Z={1: 1, 2: 8},
        )

        # Build a tiny non-periodic system: H-O-H
        nlocal = 3
        nall = 3
        tag = np.arange(1, nall + 1, dtype=np.int32)
        type_array = np.array([1, 2, 1], dtype=np.int32)
        x = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ],
            dtype=np.float64,
        )
        boxlo = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        boxhi = np.array([50.0, 50.0, 50.0], dtype=np.float64)
        pbc = np.array([0, 0, 0], dtype=np.int32)

        E0, F0 = pot.compute(nlocal, nall, tag, type_array, x, boxlo, boxhi, pbc)
        max_abs_force = float(np.max(np.abs(F0)))

        # Finite difference check for a subset of coordinates
        # dE/dx ≈ (E(x+eps) - E(x-eps)) / (2eps), and F = -dE/dx
        coords_to_check = [(0, 0), (1, 1), (2, 2)]  # (atom_idx, dim)
        fd_errs = []
        for i, d in coords_to_check:
            x_p = x.copy()
            x_m = x.copy()
            x_p[i, d] += eps
            x_m[i, d] -= eps

            E_p, _ = pot.compute(nlocal, nall, tag, type_array, x_p, boxlo, boxhi, pbc)
            E_m, _ = pot.compute(nlocal, nall, tag, type_array, x_m, boxlo, boxhi, pbc)

            dE_dx = (E_p - E_m) / (2.0 * eps)  # kcal/mol/Ang
            fd_force = -dE_dx
            fd_errs.append(abs(fd_force - F0[i, d]))

        max_abs_fd_err = float(np.max(fd_errs)) if fd_errs else 0.0

        return SelfTestResult(
            energy_kcalmol=float(E0),
            max_abs_force_kcalmol_per_ang=max_abs_force,
            max_abs_fd_err_kcalmol_per_ang=max_abs_fd_err,
        )


def main():
    try:
        res = run_self_test()
    except ImportError as e:
        # Most common missing deps: torch_cluster / torch_scatter
        raise SystemExit(f"Self-test failed due to missing dependency: {e}")

    print("LAMMPS potential offline self-test:")
    print(f"  Energy (kcal/mol): {res.energy_kcalmol:.6f}")
    print(f"  Max |Force| (kcal/mol/Ang): {res.max_abs_force_kcalmol_per_ang:.6f}")
    print(f"  Max finite-diff error (kcal/mol/Ang): {res.max_abs_fd_err_kcalmol_per_ang:.6e}")

    # Heuristic threshold; random model is noisy, but FD should still broadly match.
    # If this is huge, something is wrong in sign/units/grad path.
    if res.max_abs_fd_err_kcalmol_per_ang > 1e-2:
        raise SystemExit(
            "Self-test FAILED: finite-difference error too large. "
            "This suggests a bug in energy/force sign, unit conversion, or graph construction."
        )

    print("Self-test PASSED.")


if __name__ == "__main__":
    main()

