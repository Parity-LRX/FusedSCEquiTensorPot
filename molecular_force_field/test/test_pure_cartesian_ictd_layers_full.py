#!/usr/bin/env python3
"""
测试 pure_cartesian_ictd_layers_full 是否可用：导入、前向、反向、输出形状。
"""
from __future__ import annotations

import torch

from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer,
)


def make_dummy_graph(device: torch.device, dtype: torch.dtype, num_nodes: int = 64, avg_degree: int = 24, seed: int = 42):
    torch.manual_seed(seed)
    pos = torch.randn(num_nodes, 3, device=device, dtype=dtype) * 2.0
    A = torch.randint(1, 6, (num_nodes,), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    num_edges = num_nodes * avg_degree
    edge_dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_shifts = torch.zeros(num_edges, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_nodes, -1, -1)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    num_nodes = 64
    max_radius = 5.0

    print("Testing pure_cartesian_ictd_layers_full ...")
    print(f"  device={device}, dtype={dtype}, num_nodes={num_nodes}")

    model = PureCartesianICTDTransformerLayer(
        max_embed_radius=max_radius,
        main_max_radius=max_radius,
        main_number_of_basis=8,
        hidden_dim_conv=64,
        hidden_dim_sh=32,
        hidden_dim=64,
        lmax=2,
        num_interaction=2,
    ).to(device=device, dtype=dtype)

    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = make_dummy_graph(device, dtype, num_nodes=num_nodes)
    pos = pos.requires_grad_(True)

    # Forward
    out = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    assert out.shape == (num_nodes, 1), f"Expected (N, 1), got {out.shape}"
    print(f"  forward OK: out.shape = {out.shape}")

    # Backward (energy -> forces)
    E = out.sum()
    E.backward()
    assert pos.grad is not None and pos.grad.shape == pos.shape
    print(f"  backward OK: pos.grad.shape = {pos.grad.shape}")

    # 带 sync_after_scatter 调用（DDP 兼容：传入 identity 即等效无同步）
    out_sync = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, sync_after_scatter=lambda x: x)
    assert out_sync.shape == (num_nodes, 1)
    print(f"  forward with sync_after_scatter (identity): OK")

    print("  pure_cartesian_ictd_layers_full: OK (usable, DDP sync supported).")


if __name__ == "__main__":
    main()
