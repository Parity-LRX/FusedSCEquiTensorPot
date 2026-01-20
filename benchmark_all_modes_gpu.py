#!/usr/bin/env python3
"""
在 GPU 上全面测试所有张量积模块的性能
包括：前向传播、反向传播、参数量、等变性（包含宇称）
"""
import time
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
import argparse

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


def init_weighted_sum_weights(model):
    """初始化 weighted_sum 层的权重，确保输出非零以便测试等变性"""
    for name, module in model.named_modules():
        if hasattr(module, 'weights') and hasattr(module, 'num_features'):
            if module.weights is not None:
                with torch.no_grad():
                    module.weights.data = torch.randn(
                        module.num_features, 
                        device=module.weights.device, 
                        dtype=module.weights.dtype
                    ) * 0.1


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
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        
        # Benchmark
        if pos.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            output = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        if pos.device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time_ms = (end - start) / iterations * 1000
        return avg_time_ms, output


def benchmark_backward(
    model: nn.Module,
    pos: torch.Tensor,
    A: torch.Tensor,
    batch: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_shifts: torch.Tensor,
    cell: torch.Tensor,
    warmup: int = 3,
    iterations: int = 20,
) -> float:
    """测试反向传播速度"""
    model.train()
    pos.requires_grad_(True)
    
    # Warmup
    for _ in range(warmup):
        output = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        loss = output.sum()
        loss.backward()
        pos.grad = None
    
    # Benchmark
    if pos.device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        output = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        loss = output.sum()
        loss.backward()
        pos.grad = None
    if pos.device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    pos.requires_grad_(False)
    avg_time_ms = (end - start) / iterations * 1000
    return avg_time_ms


def test_equivariance(
    model: nn.Module,
    pos: torch.Tensor,
    A: torch.Tensor,
    batch: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_shifts: torch.Tensor,
    cell: torch.Tensor,
    n_tests: int = 20,
) -> float:
    """测试 O(3) 等变性（包括宇称/反射）"""
    model.eval()
    max_error = 0.0
    
    with torch.no_grad():
        # 确保输出非零 - 多次尝试初始化权重
        E_orig = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        max_abs = torch.abs(E_orig).max().item()
        
        # 如果输出接近0，尝试重新初始化权重
        if max_abs < 1e-10:
            for attempt in range(5):  # 最多尝试5次
                init_weighted_sum_weights(model)
                E_orig = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
                max_abs = torch.abs(E_orig).max().item()
                if max_abs >= 1e-10:
                    break
            
            # 如果仍然接近0，返回inf表示测试失败
            if max_abs < 1e-10:
                return float('inf')
        
        # 使用 E_orig.sum() 作为基准值（标量能量）
        E_orig_sum = E_orig.sum()
        
        for _ in range(n_tests):
            # 随机 O(3) 矩阵（可能包含反射）
            Q, _ = torch.linalg.qr(torch.randn(3, 3, device=pos.device, dtype=pos.dtype))
            if torch.rand(1, device=pos.device).item() < 0.5:
                # 50% 概率包含反射
                Q = Q @ torch.diag(torch.tensor([1.0, 1.0, -1.0], device=pos.device, dtype=pos.dtype))
            
            # 旋转/反射坐标
            pos_rot = pos @ Q.T
            cell_rot = cell @ Q.T if cell.numel() > 0 else cell
            edge_shifts_rot = edge_shifts @ Q.T
            
            # 旋转后的输出
            E_rot = model(pos_rot, A, batch, edge_src, edge_dst, edge_shifts_rot, cell_rot)
            E_rot_sum = E_rot.sum()
            
            # 能量应该是标量，在旋转/反射下不变
            # 使用相对误差，避免当能量很小时误差被放大
            if abs(E_orig_sum.item()) > 1e-10:
                error = abs((E_orig_sum - E_rot_sum) / E_orig_sum).item()
            else:
                error = abs(E_orig_sum - E_rot_sum).item()
            
            max_error = max(max_error, error)
    
    return max_error


def test_single_mode(
    mode_name: str,
    model_class,
    lmax: int,
    device: torch.device,
    dtype: torch.dtype,
    common_config: Dict,
    pos: torch.Tensor,
    A: torch.Tensor,
    batch: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_shifts: torch.Tensor,
    cell: torch.Tensor,
) -> Dict:
    """测试单个模式在指定 lmax 下的性能"""
    try:
        # 初始化模型
        if mode_name == "spherical":
            config = ModelConfig(
                lmax=lmax,
                irreps_output_conv_channels=64,
                max_atomvalue=common_config["max_atomvalue"],
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
                max_atomvalue=common_config["max_atomvalue"],
                output_size=common_config["output_size"],
                embed_size=config.embed_size,
                main_hidden_sizes3=config.main_hidden_sizes3,
                num_layers=1,
                device=device,
                function_type_main=common_config["function_type_main"],
            ).to(device=device, dtype=dtype)
        else:
            config_dict = common_config.copy()
            config_dict['lmax'] = lmax
            model = model_class(**config_dict).to(device=device, dtype=dtype)
        
        # 初始化权重（确保输出非零）
        init_weighted_sum_weights(model)
        
        # 验证输出非零（在测试前）
        model.eval()
        with torch.no_grad():
            test_output = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
            if torch.abs(test_output).max().item() < 1e-10:
                # 如果输出仍然接近0，多次尝试重新初始化
                for attempt in range(5):
                    init_weighted_sum_weights(model)
                    test_output = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
                    if torch.abs(test_output).max().item() >= 1e-10:
                        break
        
        # 参数量
        n_params = count_parameters(model)
        
        # 前向传播
        forward_time, _ = benchmark_forward(
            model, pos, A, batch, edge_src, edge_dst, edge_shifts, cell
        )
        
        # 反向传播
        backward_time = benchmark_backward(
            model, pos, A, batch, edge_src, edge_dst, edge_shifts, cell
        )
        
        # 等变性（此时输出应该已经非零）
        equiv_error = test_equivariance(
            model, pos, A, batch, edge_src, edge_dst, edge_shifts, cell
        )
        
        return {
            'params': n_params,
            'forward_ms': forward_time,
            'backward_ms': backward_time,
            'equiv_error': equiv_error,
            'success': True,
        }
    except Exception as e:
        print(f"    ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return {
            'params': 0,
            'forward_ms': float('inf'),
            'backward_ms': float('inf'),
            'equiv_error': float('inf'),
            'success': False,
        }


def main():
    parser = argparse.ArgumentParser(description='GPU 上全面测试所有张量积模块')
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device, choices=['cpu', 'cuda'])
    parser.add_argument('--dtype', type=str, default='float64', choices=['float32', 'float64'])
    parser.add_argument('--N', type=int, default=32, help='节点数')
    parser.add_argument('--E', type=int, default=256, help='边数')
    parser.add_argument('--lmax-max', type=int, default=6, help='最大 lmax 值')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，回退到 CPU")
        device = torch.device('cpu')
    elif device.type == 'cuda':
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    
    # 创建测试数据
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    
    pos = torch.randn(args.N, 3, device=device, dtype=dtype)
    A = torch.randint(1, 10, (args.N,), device=device, dtype=torch.long)
    batch = torch.zeros(args.N, dtype=torch.long, device=device)
    edge_src = torch.randint(0, args.N, (args.E,), device=device, dtype=torch.long)
    edge_dst = torch.randint(0, args.N, (args.E,), device=device, dtype=torch.long)
    edge_shifts = torch.zeros(args.E, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * 10.0
    
    # 通用配置
    common_config = {
        "max_embed_radius": 5.0,
        "main_max_radius": 5.0,
        "main_number_of_basis": 32,
        "hidden_dim_conv": 64,
        "hidden_dim_sh": 64,
        "hidden_dim": 128,
        "embedding_dim": 16,
        "max_atomvalue": 10,
        "output_size": 8,
        "function_type_main": "gaussian",
    }
    
    # 所有模式
    modes = {
        "spherical": E3_TransformerLayer_multi,
        "partial-cartesian": CartesianTransformerLayer,
        "partial-cartesian-loose": CartesianTransformerLayerLoose,
        "pure-cartesian": PureCartesianTransformerLayer,
        "pure-cartesian-sparse": PureCartesianSparseTransformerLayer,
        "pure-cartesian-ictd": PureCartesianICTDTransformerLayer,
    }
    
    lmax_list = list(range(args.lmax_max + 1))
    
    print("=" * 100)
    print("GPU 上全面测试所有张量积模块")
    print("=" * 100)
    print(f"设备: {device}, dtype: {dtype}")
    print(f"N={args.N}, E={args.E}, lmax: 0-{args.lmax_max}")
    print()
    
    # 存储所有结果
    all_results: Dict[str, Dict[int, Dict]] = {mode: {} for mode in modes.keys()}
    
    # 测试每个模式
    for mode_name, model_class in modes.items():
        print(f"\n{'='*100}")
        print(f"测试模式: {mode_name}")
        print(f"{'='*100}")
        
        # pure-cartesian 只测试到 lmax=3
        test_lmax_list = lmax_list if mode_name != 'pure-cartesian' else [l for l in lmax_list if l <= 3]
        
        for lmax in test_lmax_list:
            print(f"  lmax={lmax}...", end=' ', flush=True)
            result = test_single_mode(
                mode_name, model_class, lmax, device, dtype, common_config,
                pos, A, batch, edge_src, edge_dst, edge_shifts, cell
            )
            all_results[mode_name][lmax] = result
            if result['success']:
                print(f"✓ (forward: {result['forward_ms']:.2f}ms, backward: {result['backward_ms']:.2f}ms)")
            else:
                print("✗")
        
        # 对于 pure-cartesian，标记 lmax>3 为 FAILED
        if mode_name == 'pure-cartesian':
            for lmax in lmax_list:
                if lmax > 3 and lmax not in all_results[mode_name]:
                    all_results[mode_name][lmax] = {
                        'params': 0,
                        'forward_ms': float('inf'),
                        'backward_ms': float('inf'),
                        'equiv_error': float('inf'),
                        'success': False,
                    }
    
    # 打印汇总表格
    print("\n" + "=" * 100)
    print("汇总结果")
    print("=" * 100)
    
    # 按 lmax 分组打印
    for lmax in lmax_list:
        print(f"\n### lmax={lmax} ###")
        print(f"{'模式':<25} {'前向(ms)':<12} {'反向(ms)':<12} {'参数量':<15} {'等变误差':<15}")
        print("-" * 100)
        
        for mode_name in modes.keys():
            result = all_results[mode_name][lmax]
            if result['success']:
                print(f"{mode_name:<25} {result['forward_ms']:>10.2f} {result['backward_ms']:>10.2f} "
                      f"{result['params']:>13,} {result['equiv_error']:>13.2e}")
            else:
                print(f"{mode_name:<25} {'FAILED':<12} {'FAILED':<12} {'-':<15} {'-':<15}")
    
    # 打印速度对比表格（类似 spherical vs ictd 的格式）
    print("\n" + "=" * 100)
    print("速度对比汇总（前向传播，单位：ms）")
    print("=" * 100)
    print(f"{'lmax':<6} ", end='')
    for mode_name in modes.keys():
        print(f"{mode_name:<20} ", end='')
    print()
    print("-" * 100)
    
    for lmax in lmax_list:
        print(f"{lmax:<6} ", end='')
        for mode_name in modes.keys():
            result = all_results[mode_name][lmax]
            if result['success']:
                print(f"{result['forward_ms']:>18.2f} ", end='')
            else:
                print(f"{'FAILED':>18} ", end='')
        print()
    
    # 打印前向传播加速比（相对于 spherical）
    print("\n" + "=" * 100)
    print("前向传播加速比（相对于 spherical，spherical=1.00x）")
    print("=" * 100)
    print(f"{'lmax':<6} ", end='')
    for mode_name in modes.keys():
        if mode_name != 'spherical':
            print(f"{mode_name:<20} ", end='')
    print()
    print("-" * 100)
    
    for lmax in lmax_list:
        print(f"{lmax:<6} ", end='')
        spherical_result = all_results['spherical'][lmax]
        if not spherical_result['success']:
            for mode_name in modes.keys():
                if mode_name != 'spherical':
                    print(f"{'N/A':<20} ", end='')
        else:
            spherical_time = spherical_result['forward_ms']
            for mode_name in modes.keys():
                if mode_name != 'spherical':
                    result = all_results[mode_name][lmax]
                    if result['success']:
                        speedup = spherical_time / result['forward_ms']
                        print(f"{speedup:>18.2f}x ", end='')
                    else:
                        print(f"{'FAILED':<18} ", end='')
        print()
    
    # 打印反向传播速度对比
    print("\n" + "=" * 100)
    print("速度对比汇总（反向传播，单位：ms）")
    print("=" * 100)
    print(f"{'lmax':<6} ", end='')
    for mode_name in modes.keys():
        print(f"{mode_name:<20} ", end='')
    print()
    print("-" * 100)
    
    for lmax in lmax_list:
        print(f"{lmax:<6} ", end='')
        for mode_name in modes.keys():
            result = all_results[mode_name][lmax]
            if result['success']:
                print(f"{result['backward_ms']:>18.2f} ", end='')
            else:
                print(f"{'FAILED':<18} ", end='')
        print()
    
    # 打印反向传播加速比
    print("\n" + "=" * 100)
    print("反向传播加速比（相对于 spherical，spherical=1.00x）")
    print("=" * 100)
    print(f"{'lmax':<6} ", end='')
    for mode_name in modes.keys():
        if mode_name != 'spherical':
            print(f"{mode_name:<20} ", end='')
    print()
    print("-" * 100)
    
    for lmax in lmax_list:
        print(f"{lmax:<6} ", end='')
        spherical_result = all_results['spherical'][lmax]
        if not spherical_result['success']:
            for mode_name in modes.keys():
                if mode_name != 'spherical':
                    print(f"{'N/A':<20} ", end='')
        else:
            spherical_time = spherical_result['backward_ms']
            for mode_name in modes.keys():
                if mode_name != 'spherical':
                    result = all_results[mode_name][lmax]
                    if result['success']:
                        speedup = spherical_time / result['backward_ms']
                        print(f"{speedup:>18.2f}x ", end='')
                    else:
                        print(f"{'FAILED':<18} ", end='')
        print()
    
    # 打印总时间（前向+反向）对比
    print("\n" + "=" * 100)
    print("总训练时间对比（前向+反向，单位：ms）")
    print("=" * 100)
    print(f"{'lmax':<6} ", end='')
    for mode_name in modes.keys():
        print(f"{mode_name:<20} ", end='')
    print()
    print("-" * 100)
    
    for lmax in lmax_list:
        print(f"{lmax:<6} ", end='')
        for mode_name in modes.keys():
            result = all_results[mode_name][lmax]
            if result['success']:
                total_time = result['forward_ms'] + result['backward_ms']
                print(f"{total_time:>18.2f} ", end='')
            else:
                print(f"{'FAILED':<18} ", end='')
        print()
    
    # 打印总时间加速比
    print("\n" + "=" * 100)
    print("总训练时间加速比（相对于 spherical，spherical=1.00x）")
    print("=" * 100)
    print(f"{'lmax':<6} ", end='')
    for mode_name in modes.keys():
        if mode_name != 'spherical':
            print(f"{mode_name:<20} ", end='')
    print()
    print("-" * 100)
    
    for lmax in lmax_list:
        print(f"{lmax:<6} ", end='')
        spherical_result = all_results['spherical'][lmax]
        if not spherical_result['success']:
            for mode_name in modes.keys():
                if mode_name != 'spherical':
                    print(f"{'N/A':<20} ", end='')
        else:
            spherical_total = spherical_result['forward_ms'] + spherical_result['backward_ms']
            for mode_name in modes.keys():
                if mode_name != 'spherical':
                    result = all_results[mode_name][lmax]
                    if result['success']:
                        total_time = result['forward_ms'] + result['backward_ms']
                        speedup = spherical_total / total_time
                        print(f"{speedup:>18.2f}x ", end='')
                    else:
                        print(f"{'FAILED':<18} ", end='')
        print()
    
    # 打印参数量对比
    print("\n" + "=" * 100)
    print("参数量对比")
    print("=" * 100)
    print(f"{'lmax':<6} ", end='')
    for mode_name in modes.keys():
        print(f"{mode_name:<20} ", end='')
    print()
    print("-" * 100)
    
    for lmax in lmax_list:
        print(f"{lmax:<6} ", end='')
        for mode_name in modes.keys():
            result = all_results[mode_name][lmax]
            if result['success']:
                print(f"{result['params']:>18,} ", end='')
            else:
                print(f"{'FAILED':>18} ", end='')
        print()
    
    # 打印等变误差对比
    print("\n" + "=" * 100)
    print("等变误差对比（O(3)，包含宇称）")
    print("=" * 100)
    print(f"{'lmax':<6} ", end='')
    for mode_name in modes.keys():
        print(f"{mode_name:<20} ", end='')
    print()
    print("-" * 100)
    
    for lmax in lmax_list:
        print(f"{lmax:<6} ", end='')
        for mode_name in modes.keys():
            result = all_results[mode_name][lmax]
            if result['success']:
                print(f"{result['equiv_error']:>18.2e} ", end='')
            else:
                print(f"{'FAILED':>18} ", end='')
        print()


if __name__ == "__main__":
    main()
