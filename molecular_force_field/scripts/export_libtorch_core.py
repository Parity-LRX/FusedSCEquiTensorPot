#!/usr/bin/env python3
"""
导出可被 LibTorch(C++) 直接加载的 TorchScript core 模型（torch.jit.save）。

与 `torch.save(obj)` 的区别：
- `torch.save()` 保存的是 Python pickle 对象（包含自定义类），C++ 侧不能直接加载。
- 纯 C++ 链路需要 `torch.jit.save(ScriptModule, path)` 导出的 TorchScript 文件。

本脚本导出的 `core.pt`：
- forward 签名：
    (pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec) -> atom_energies
- **可选内置 E0**（方案B）：把预处理/拟合得到的 per-element 常数能（E0）写进 TorchScript，
  导出后的 `core.pt` 输出的就是 “网络能量 + E0(Z)” 的 per-atom 能量。
  注意：E0 不影响力（常数对坐标梯度为 0）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_script_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def _pick_device(req: str) -> str:
    if req == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，改用 CPU")
        return "cpu"
    return req


def _parse_dtype(s: Optional[str]) -> Optional[torch.dtype]:
    if s is None:
        return None
    mapping = {
        "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
        "float64": torch.float64, "fp64": torch.float64, "double": torch.float64,
    }
    dt = mapping.get(s.lower().strip())
    if dt is None:
        raise ValueError(f"不支持的 dtype: {s!r}，可选: float32, float64")
    return dt


def _e0_lut_from_keys_values(
    keys: torch.Tensor, values: torch.Tensor, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Build a TorchScript-friendly lookup table lut[Z] = E0(Z)."""
    keys = keys.to(dtype=torch.long, device="cpu").contiguous()
    values = values.to(dtype=dtype, device="cpu").contiguous()
    max_z = int(keys.max().item()) if keys.numel() > 0 else 0
    size = max(119, max_z + 1)  # cover periodic table by default
    lut = torch.zeros(size, dtype=dtype)
    for k, v in zip(keys.tolist(), values.tolist()):
        if 0 <= int(k) < size:
            lut[int(k)] = float(v)
    return lut.to(device=device)


class _E0WrappedModel(torch.nn.Module):
    """Wrap an eager model to add E0(Z) into per-atom energies before tracing."""

    def __init__(self, model: torch.nn.Module, e0_lut: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("e0_lut", e0_lut)

    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        *,
        precomputed_edge_vec: Optional[torch.Tensor] = None,
        sync_after_scatter=None,
    ) -> torch.Tensor:
        # Keep the same signature the framework expects.
        out = self.model(
            pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
            precomputed_edge_vec=precomputed_edge_vec,
            sync_after_scatter=sync_after_scatter,
        )
        # E0 lookup: e0_lut[Z]
        e0 = self.e0_lut.index_select(0, A.to(torch.long))
        # Broadcast e0 to match out (usually (N,1)).
        if out.dim() == 2:
            e0 = e0.unsqueeze(1)
        return out + e0.to(dtype=out.dtype, device=out.device)


def export_core(
    *,
    checkpoint: str,
    elements: List[str],
    device: str,
    max_radius: float,
    num_interaction: int,
    out_pt: str,
    force_dtype: Optional[torch.dtype] = None,
    embed_e0: bool = True,
    e0_csv: Optional[str] = None,
) -> None:
    from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF, _maybe_torchscript_trace_model
    from molecular_force_field.utils.config import ModelConfig

    # Load atomic E0 from preprocessing output if requested.
    atomic_energy_keys = None
    atomic_energy_values = None
    if e0_csv:
        cfg = ModelConfig(dtype=torch.float64)
        cfg.load_atomic_energies_from_file(e0_csv)
        atomic_energy_keys = cfg.atomic_energy_keys.tolist()
        atomic_energy_values = [float(x) for x in cfg.atomic_energy_values.tolist()]

    # Build eager model first so we can (optionally) wrap E0 and/or cast dtype before tracing.
    obj = LAMMPS_MLIAP_MFF.from_checkpoint(
        checkpoint_path=checkpoint,
        element_types=elements,
        max_radius=max_radius,
        atomic_energy_keys=atomic_energy_keys,
        atomic_energy_values=atomic_energy_values,
        device=device,
        tensor_product_mode="pure-cartesian-ictd",
        num_interaction=num_interaction,
        torchscript=False,
    )

    actual_dtype = obj.dtype
    if force_dtype is not None:
        actual_dtype = force_dtype
        obj.wrapper = obj.wrapper.to(dtype=force_dtype)
        obj.wrapper.model = obj.wrapper.model.to(dtype=force_dtype)

    model_eager = obj.wrapper.model

    # Optional: embed E0(Z) into per-atom energies before tracing.
    if embed_e0:
        aek = obj.wrapper.atomic_energy_keys.detach().cpu()
        aev = obj.wrapper.atomic_energy_values.detach().cpu()
        lut = _e0_lut_from_keys_values(aek, aev, dtype=actual_dtype, device=torch.device(device))
        model_eager = _E0WrappedModel(model_eager, lut).to(device=torch.device(device))

    # Trace to TorchScript core (edge_vec positional arg) and export its ScriptModule.
    ts_model = _maybe_torchscript_trace_model(
        model_eager,
        device=torch.device(device),
        dtype=actual_dtype,
        enable=True,
    )
    core = getattr(ts_model, "core", None)
    if core is None or not isinstance(core, torch.jit.ScriptModule):
        raise RuntimeError("未拿到 TorchScript core 模块（trace 失败）")

    os.makedirs(os.path.dirname(os.path.abspath(out_pt)), exist_ok=True)
    core.eval()
    torch.jit.save(core, out_pt)
    print(f"已导出 LibTorch 可加载的 TorchScript core: {out_pt}")

    meta = {
        "elements": elements,
        "device_exported_from": device,
        "max_radius": float(max_radius),
        "num_interaction": int(num_interaction),
        "dtype": str(actual_dtype).replace("torch.", ""),
        "embed_e0": bool(embed_e0),
        "e0_source": (str(e0_csv) if e0_csv else "from_checkpoint_or_default"),
        "forward_signature": [
            "pos(N,3)",
            "A(N,) atomic number (int64)",
            "batch(N,) (int64)",
            "edge_src(E,) (int64)",
            "edge_dst(E,) (int64)",
            "edge_shifts(E,3)",
            "cell(1,3,3)",
            "edge_vec(E,3)",
        ],
        "notes": [
            "这是 core 模型：输出每原子能量。",
            "若 embed_e0=true：输出已包含 E0(Z) 常数偏置（来自预处理拟合或提供的 e0_csv）。",
            "力通过 dE/d(pos) 在 C++ 侧用 autograd 求得。",
            "本文件可用 C++: torch::jit::load(path) 直接加载。",
        ],
    }
    meta_path = out_pt + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"已写出元数据: {meta_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="导出 LibTorch 可加载的 TorchScript core 模型")
    p.add_argument("--checkpoint", type=str, required=True, help="checkpoint (.pth)")
    p.add_argument("--elements", nargs="+", default=["H", "O"], help="元素顺序（LAMMPS type 顺序）")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--max-radius", type=float, default=3.0)
    p.add_argument("--num-interaction", type=int, default=2)
    p.add_argument("--dtype", type=str, default=None,
                   help="强制导出精度: float32 或 float64。不指定则跟随 checkpoint。")
    p.add_argument("--embed-e0", action="store_true", help="将 per-element E0(Z) 常数偏置写入 TorchScript（方案B）")
    p.add_argument("--no-embed-e0", action="store_true", help="不把 E0 写入 TorchScript（仅导出网络能量）")
    p.add_argument("--e0-csv", type=str, default=None, help="E0 CSV 路径（包含 Atom,E0 列）。提供则优先使用。")
    p.add_argument("--out", type=str, default="core.pt", help="输出 TorchScript 文件路径")
    args = p.parse_args()

    device = _pick_device(args.device)
    force_dtype = _parse_dtype(args.dtype)
    embed_e0 = bool(args.embed_e0) and not bool(args.no_embed_e0)
    export_core(
        checkpoint=args.checkpoint,
        elements=list(args.elements),
        device=device,
        max_radius=float(args.max_radius),
        num_interaction=int(args.num_interaction),
        out_pt=str(args.out),
        force_dtype=force_dtype,
        embed_e0=embed_e0,
        e0_csv=args.e0_csv,
    )


if __name__ == "__main__":
    main()
