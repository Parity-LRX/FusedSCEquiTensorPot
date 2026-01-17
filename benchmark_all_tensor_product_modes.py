#!/usr/bin/env python3
"""
综合 benchmark：测试所有 tensor-product-mode 的速度、参数量和等变性。
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Tuple

from molecular_force_field.models import (
    E3_TransformerLayer_multi,
    CartesianTransformerLayer,
    CartesianTransformerLayerLoose,
    PureCartesianTransformerLayer,
    PureCartesianSparseTransformerLayer,
    PureCartesianICTDTransformerLayer,
)
from molecular_force_field.utils.config import ModelConfig


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_forward(
    model: nn.Module,
    pos: torch.Tensor,
    A: torch.Tensor,
    batch: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_shifts: torch.Tensor,
    cell: torch.Tensor,
    warmup: int = 5,
    iterations: int = 30,
) -> Tuple[float, torch.Tensor]:
    """测试前向传播速度"""
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)

    # Timing
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            output = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    return avg_time_ms, output


def test_equivariance(
    model: nn.Module,
    pos: torch.Tensor,
    A: torch.Tensor,
    batch: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_shifts: torch.Tensor,
    cell: torch.Tensor,
    n_tests: int = 10,
) -> float:
    """测试 O(3) 等变性（包括宇称）"""
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    max_err = 0.0

    with torch.no_grad():
        # 原始输出
        E_orig = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)

        for _ in range(n_tests):
            # 随机 O(3) 矩阵（可能包含反射）
            Q, _ = torch.linalg.qr(torch.randn(3, 3, device=device, dtype=dtype))
            if torch.rand(1).item() < 0.5:
                # 50% 概率包含反射
                Q = Q @ torch.diag(torch.tensor([1.0, 1.0, -1.0], device=device, dtype=dtype))

            # 旋转坐标
            pos_rot = pos @ Q.T
            cell_rot = cell @ Q.T
            edge_shifts_rot = edge_shifts @ Q.T

            # 旋转后的输出
            E_rot = model(pos_rot, A, batch, edge_src, edge_dst, edge_shifts_rot, cell_rot)

            # 能量应该是标量，不变
            err = torch.abs(E_orig - E_rot).max().item()
            max_err = max(max_err, err)

    return max_err


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # 固定随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 生成 dummy 数据
    N = 64  # 节点数
    E = 512  # 边数
    max_atomvalue = 10
    lmax = 2

    pos = torch.randn(N, 3, device=device, dtype=dtype)
    A = torch.randint(0, max_atomvalue, (N,), device=device)
    batch = torch.zeros(N, device=device, dtype=torch.long)
    edge_src = torch.randint(0, N, (E,), device=device)
    edge_dst = torch.randint(0, N, (E,), device=device)
    edge_shifts = torch.zeros(E, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)

    # 通用配置
    common_config = {
        "max_embed_radius": 5.0,
        "main_max_radius": 5.0,
        "main_number_of_basis": 32,
        "hidden_dim_conv": 64,
        "hidden_dim_sh": 64,
        "hidden_dim": 128,
        "embedding_dim": 16,
        "max_atomvalue": max_atomvalue,
        "output_size": 8,
        "function_type_main": "gaussian",
        "lmax": lmax,
    }

    # 所有模式
    modes = {
        "spherical": ("spherical", E3_TransformerLayer_multi),
        "partial-cartesian": ("partial-cartesian", CartesianTransformerLayer),
        "partial-cartesian-loose": ("partial-cartesian-loose", CartesianTransformerLayerLoose),
        "pure-cartesian": ("pure-cartesian", PureCartesianTransformerLayer),
        "pure-cartesian-sparse": ("pure-cartesian-sparse", PureCartesianSparseTransformerLayer),
        "pure-cartesian-ictd": ("pure-cartesian-ictd", PureCartesianICTDTransformerLayer),
    }

    results = []

    print("=" * 80)
    print("Benchmarking all tensor product modes")
    print("=" * 80)

    for mode_name, (mode_key, model_class) in modes.items():
        print(f"\n[{mode_name}]")
        try:
            # 初始化模型
            if mode_name == "spherical":
                # 使用 ModelConfig 来获取正确的 irreps 参数
                config = ModelConfig(
                    lmax=lmax,
                    irreps_output_conv_channels=64,
                    max_atomvalue=max_atomvalue,
                    embedding_dim=common_config["embedding_dim"],
                    number_of_basis_main=common_config["main_number_of_basis"],
                    max_radius_main=common_config["main_max_radius"],
                    function_type_main=common_config["function_type_main"],
                    dtype=dtype,
                )
                model = model_class(
                    max_embed_radius=common_config["max_embed_radius"],
                    main_max_radius=common_config["main_max_radius"],
                    main_number_of_basis=common_config["main_number_of_basis"],
                    irreps_input=config.get_irreps_output_conv(),
                    irreps_query=config.get_irreps_query_main(),
                    irreps_key=config.get_irreps_key_main(),
                    irreps_value=config.get_irreps_value_main(),
                    irreps_output=config.get_irreps_output_conv_2(),
                    irreps_sh=config.get_irreps_sh_transformer(),
                    hidden_dim_sh=config.get_hidden_dim_sh(),
                    hidden_dim=common_config["hidden_dim"],
                    channel_in2=config.channel_in2,
                    embedding_dim=common_config["embedding_dim"],
                    max_atomvalue=max_atomvalue,
                    output_size=common_config["output_size"],
                    embed_size=config.embed_size,
                    main_hidden_sizes3=config.main_hidden_sizes3,
                    num_layers=1,
                    device=device,
                    function_type_main=common_config["function_type_main"],
                ).to(device=device, dtype=dtype)
            else:
                model = model_class(**common_config).to(device=device, dtype=dtype)

            # 参数量
            n_params = count_parameters(model)
            print(f"  Parameters: {n_params:,}")

            # 速度
            avg_time_ms, _ = benchmark_forward(
                model, pos, A, batch, edge_src, edge_dst, edge_shifts, cell
            )
            print(f"  Forward time: {avg_time_ms:.2f} ms")

            # 等变性
            max_equiv_err = test_equivariance(
                model, pos, A, batch, edge_src, edge_dst, edge_shifts, cell
            )
            print(f"  Max equivariance error: {max_equiv_err:.2e}")

            results.append(
                {
                    "mode": mode_name,
                    "params": n_params,
                    "time_ms": avg_time_ms,
                    "equiv_err": max_equiv_err,
                }
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {
                    "mode": mode_name,
                    "params": None,
                    "time_ms": None,
                    "equiv_err": None,
                    "error": str(e),
                }
            )

    # 打印汇总表
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Mode':<30} {'Params':<15} {'Time (ms)':<15} {'Equiv Error':<15}")
    print("-" * 80)

    baseline_time = None
    baseline_params = None
    for r in results:
        if r["params"] is not None:
            if baseline_time is None:
                baseline_time = r["time_ms"]
                baseline_params = r["params"]
            time_ratio = r["time_ms"] / baseline_time if baseline_time > 0 else 1.0
            params_ratio = r["params"] / baseline_params if baseline_params > 0 else 1.0
            equiv_status = "✓" if r["equiv_err"] < 1e-6 else "✗"
            print(
                f"{r['mode']:<30} {r['params']:>14,} {r['time_ms']:>14.2f} ({time_ratio:>5.2f}x) {r['equiv_err']:>14.2e} {equiv_status}"
            )
        else:
            print(f"{r['mode']:<30} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15}")

    return results


if __name__ == "__main__":
    results = main()
