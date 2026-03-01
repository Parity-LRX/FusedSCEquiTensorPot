#!/usr/bin/env python3
"""
测试 pure-cartesian-ictd 模型能否成功 TorchScript trace 并运行。

包含：
1. 直接 trace 测试（LAMMPS_MLIAP_MFF.from_checkpoint + torchscript=True）
2. 完整导出测试（export_core -> core.pt -> torch.jit.load 加载并 forward）

Usage:
  python -m molecular_force_field.test.test_torchscript_ictd
"""

from __future__ import annotations

import tempfile

import torch

from molecular_force_field.interfaces.self_test_lammps_potential import (
    _make_dummy_checkpoint_pure_cartesian_ictd,
)
from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF


def _run_forward(core: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """运行一次 forward，返回输出。"""
    N, E = 32, 256
    pos = torch.zeros(N, 3, device=device, dtype=dtype, requires_grad=True)
    A = torch.ones(N, device=device, dtype=torch.long)
    batch = torch.zeros(N, device=device, dtype=torch.long)
    edge_src = torch.randint(0, N, (E,), device=device, dtype=torch.long)
    edge_dst = torch.randint(0, N, (E,), device=device, dtype=torch.long)
    edge_shifts = torch.zeros(E, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * 100.0
    edge_vec = (pos[edge_dst] - pos[edge_src] + torch.randn(E, 3, device=device, dtype=dtype)).detach()

    with torch.no_grad():
        out = core(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec)
    return out


def test_trace_only():
    """测试：直接 trace 并运行 forward。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = f"{tmp}/dummy_ictd.pth"
        _make_dummy_checkpoint_pure_cartesian_ictd(ckpt_path, device=torch.device("cpu"))

        print("[test_trace_only] Loading model with TorchScript trace...")
        mliap = LAMMPS_MLIAP_MFF.from_checkpoint(
            checkpoint_path=ckpt_path,
            element_types=["H", "O"],
            max_radius=5.0,
            atomic_energy_keys=[1, 8],
            atomic_energy_values=[-13.6, -75.0],
            device=str(device),
            torchscript=True,
        )

        adapter = mliap.wrapper.model
        core = adapter.core
        assert isinstance(core, torch.jit.ScriptModule), "Expected ScriptModule"

        print("[test_trace_only] TorchScript trace OK. Running forward...")
        out = _run_forward(core, device, dtype)
        print(f"[test_trace_only] Forward OK. Output shape: {out.shape}")
        print("PASS: TorchScript ICTD trace works")


def test_export_core():
    """测试：完整 export_core 导出 -> 加载 core.pt -> forward。"""
    import os
    from molecular_force_field.cli.export_libtorch_core import export_core

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = f"{tmp}/dummy_ictd.pth"
        out_pt = f"{tmp}/core.pt"
        e0_csv = os.path.join(tmp, "e0.csv")

        _make_dummy_checkpoint_pure_cartesian_ictd(ckpt_path, device=torch.device("cpu"))

        # 创建最小 E0 CSV，供 export_core 使用
        with open(e0_csv, "w") as f:
            f.write("Atom,E0\n1,-13.6\n8,-75.0\n")

        print("[test_export_core] Running export_core...")
        export_core(
            checkpoint=ckpt_path,
            elements=["H", "O"],
            device=str(device),
            max_radius=5.0,
            num_interaction=2,
            out_pt=out_pt,
            tensor_product_mode="pure-cartesian-ictd",
            embed_e0=True,
            e0_csv=e0_csv,
        )

        print("[test_export_core] Loading core.pt...")
        core = torch.jit.load(out_pt, map_location=device)
        core.eval()

        print("[test_export_core] Running forward on loaded core...")
        out = _run_forward(core, device, dtype)
        print(f"[test_export_core] Forward OK. Output shape: {out.shape}")
        print("PASS: export_core -> core.pt -> load works")


def main():
    test_trace_only()
    print()
    test_export_core()
    print("\nAll TorchScript tests passed.")


if __name__ == "__main__":
    main()
