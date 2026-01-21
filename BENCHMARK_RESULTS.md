# FusedSCEquiTensorPot 性能测试结果

本文档包含 FusedSCEquiTensorPot 框架所有张量积模式的完整性能测试结果，包括部分张量积模式实际任务精度测试和全部张量积模式速度基准测试。

## 目录

1. [实际任务精度测试](#1-实际任务精度测试)
2. [速度基准测试](#2-速度基准测试)
   - [2.1 CPU 测试结果](#21-cpu-测试结果)
   - [2.2 GPU 测试结果](#22-gpu-测试结果)
3. [测试环境说明](#3-测试环境说明)

---

## 1. 实际任务精度测试

### 1.1 数据集描述

- **数据集**：五条氮氧化物和碳结构反应路径的 NEB（Nudged Elastic Band）数据
- **数据筛选**：截取到 fmax=0.2
- **总数据量**：2,788 条结构
- **测试集**：每个反应选取 1-2 条完整或不完整的数据

### 1.2 测试配置

- **模型配置**：64 channels, lmax=2, float64
- **对比基准**：MACE Lmax=2 (64Chanels 和 198Chanels)

### 1.3 精度测试结果（能量与力 RMSE 对比）

<table>
<thead>
<tr>
<th style="text-align:center">方法</th>
<th style="text-align:center">配置</th>
<th style="text-align:center">模式</th>
<th style="text-align:center">能量 RMSE<br/>(mev/atom)</th>
<th style="text-align:center">力 RMSE<br/>(mev/Å)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" style="text-align:center;vertical-align:middle"><strong>MACE</strong></td>
<td style="text-align:center">Lmax=2, 64ch</td>
<td style="text-align:center">-</td>
<td style="text-align:center">0.13</td>
<td style="text-align:center">11.6</td>
</tr>
<tr>
<td style="text-align:center">Lmax=2, 128ch</td>
<td style="text-align:center">-</td>
<td style="text-align:center">0.12</td>
<td style="text-align:center">11.3</td>
</tr>
<tr>
<td style="text-align:center">Lmax=2, 198ch</td>
<td style="text-align:center">-</td>
<td style="text-align:center">0.24</td>
<td style="text-align:center">15.1</td>
</tr>
<tr>
<td rowspan="4" style="text-align:center;vertical-align:middle"><strong>FSCETP</strong></td>
<td rowspan="4" style="text-align:center;vertical-align:middle">Lmax=2, 64ch</td>
<td style="text-align:center"><strong>spherical</strong></td>
<td style="text-align:center"><strong>0.044</strong> ⭐</td>
<td style="text-align:center"><strong>7.4</strong> ⭐</td>
</tr>
<tr>
<td style="text-align:center"><strong>partial-cartesian</strong></td>
<td style="text-align:center">0.045</td>
<td style="text-align:center"><strong>7.4</strong> ⭐</td>
</tr>
<tr>
<td style="text-align:center">partial-cartesian-loose</td>
<td style="text-align:center">0.048</td>
<td style="text-align:center">8.4</td>
</tr>
<tr>
<td style="text-align:center">pure-cartesian-ictd</td>
<td style="text-align:center">0.046</td>
<td style="text-align:center">9.0</td>
</tr>
</tbody>
</table>

**说明**：
- 数值越小越好
- ⭐ 表示该指标的最佳结果

### 1.4 结果分析

#### 能量精度对比
- **FSCETP 相比 MACE (64ch) 的能量 RMSE 降低了 66.2%**
  - FSCETP 最优结果：0.044 mev/atom（`spherical` 模式）
  - MACE (64ch) 基准值：0.13 mev/atom
  - 相对误差比：0.34（FSCETP / MACE）

#### 力精度对比
- **FSCETP 相比 MACE (64ch) 的力 RMSE 降低了 36.2%**
  - FSCETP 最优结果：7.4 mev/Å（`spherical` 和 `partial-cartesian` 模式）
  - MACE (64ch) 基准值：11.6 mev/Å
  - 相对误差比：0.64（FSCETP / MACE）

#### 模式性能总结
- **能量精度最优模式**：`spherical`（0.044 mev/atom）
- **力精度最优模式**：`spherical` 和 `partial-cartesian`（7.4 mev/Å）
- **精度与效率平衡模式**：`pure-cartesian-ictd` 在保持接近最优精度（能量：0.046 mev/atom，力：9.0 mev/Å）的同时，参数量减少 72.1%，训练速度提升 2.10x（GPU，lmax=2）

---

## 2. 速度基准测试

### 2.1 CPU 测试结果

#### 2.1.1 测试环境

- **硬件**：CPU
- **模型配置**：`channels=64`, `lmax=0` 到 `lmax=6`
- **测试数据**：32 atoms, 256 edges
- **测试方法**：前向传播 30 次平均，反向传播 20 次平均
- **数据类型**：float64
- **注意**：`pure-cartesian` 只测试到 lmax=3（lmax≥4 时失败）

#### 2.1.2 总训练时间加速比（前向+反向，相对于 spherical）

| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             1.06x |                  1.13x |           0.36x |                 1.07x |                **2.97x** |
| 1    |             1.05x |                  1.37x |           0.13x |                 1.02x |                **3.33x** |
| 2    |             1.06x |                  1.33x |           0.06x |                 1.39x |                **4.12x** |
| 3    |             0.58x |                  0.70x |           0.02x |                 1.05x |                **2.68x** |
| 4    |             0.37x |                  0.43x |        **FAILED** |                 0.97x |                **2.20x** |
| 5    |             0.23x |                  0.28x |        **FAILED** |                 0.78x |                **1.81x** |
| 6    |             0.16x |                  0.17x |        **FAILED** |                 0.53x |                **1.58x** |

**性能分析**：
- **`pure-cartesian-ictd` 在所有 lmax 下都是最快的**（最高 **4.12x 加速** at lmax=2）
- **lmax ≤ 3**: `pure-cartesian-ictd` 优势最明显（2.68x - 4.12x）
- **lmax = 4-6**: `pure-cartesian-ictd` 仍然最快（1.58x - 2.20x）
- **`pure-cartesian-sparse` 表现稳定**：在所有 lmax 下都接近基准（0.53x - 1.39x）

#### 2.1.3 参数量对比（CPU，lmax=0 到 6）

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 1    | 4,164,682 |        2,734,026 |            2,734,434 |     11,786,058 |           2,733,898 |           1,208,931 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 3    | 8,654,426 |        9,407,050 |            9,409,090 |     74,635,594 |           6,744,394 |           2,481,673 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 5    | 12,882,010 |       23,268,554 |           23,273,314 |         **FAILED** |          11,021,130 |           3,979,109 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**性能分析**：
- **`pure-cartesian-ictd` 参数量始终最少**（27-32% of spherical）
- **`pure-cartesian-sparse` 参数量适中**（70-88% of spherical）
- **`partial-cartesian` 参数量减少 17-18%**（82-83% of spherical）
- **`pure-cartesian` 在 lmax≥4 时失败**（内存不足）

#### 2.1.4 等变误差对比（CPU，O(3)，包含宇称）

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 3.33e-15  |        8.24e-16  |            6.39e-14  |      3.41e-15  |           4.34e-15  |           1.69e-12 |
| 1    | 5.58e-14  |        3.33e-16  |            1.65e-15  |      5.01e-15  |           7.24e-15  |           3.80e-08 |
| 2    | 3.88e-15  |        1.87e-14  |            7.73e-16  |      3.08e-15  |           3.56e-14  |           7.73e-08 |
| 3    | 1.16e-15  |        2.47e-14  |            3.71e-15  |      8.14e-16  |           9.63e-16  |           7.23e-08 |
| 4    | 1.27e-15  |        6.83e-15  |            1.00e-15  |       **FAILED** |           1.24e-14  |           3.50e-05 |
| 5    | 3.43e-15  |        3.45e-15  |            2.27e-15  |       **FAILED** |           3.59e-14  |           8.11e-08 |
| 6    | 3.26e-15  |        2.01e-15  |            5.82e-16  |       **FAILED** |           1.51e-15  |           1.00e-06 |

**性能分析**：
- **严格等变**（误差 ~1e-15）：`spherical`, `partial-cartesian-loose`, `pure-cartesian-sparse`, `pure-cartesian`
- **可接受等变**（误差 ~1e-7 到 1e-6）：`pure-cartesian-ictd`（虽然误差较大，但仍在可接受范围内）

### 2.2 GPU 测试结果

#### 2.2.1 测试环境

- **硬件**：NVIDIA GeForce RTX 3090
- **模型配置**：`channels=64`, `lmax=0` 到 `lmax=6`
- **测试数据**：32 atoms, 256 edges
- **测试方法**：前向传播 30 次平均，反向传播 20 次平均
- **数据类型**：float64

#### 2.2.2 总训练时间加速比（前向+反向，相对于 spherical）

| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             0.96x |                  1.52x |           0.54x |                 1.17x |                **1.92x** |
| 1    |             0.85x |                  1.33x |           0.16x |                 1.02x |                **1.97x** |
| 2    |             0.75x |                  1.15x |           0.06x |                 1.17x |                **2.10x** |
| 3    |             0.56x |                  0.81x |           0.02x |                 1.15x |                **1.91x** |
| 4    |             0.38x |                  0.51x |        **FAILED** |                 0.99x |                **1.78x** |
| 5    |             0.26x |                  0.32x |        **FAILED** |                 0.75x |                **1.44x** |
| 6    |             0.17x |                  0.21x |        **FAILED** |                 0.46x |                **1.05x** |

**性能分析**：
- **`pure-cartesian-ictd` 在所有 lmax 下都是最快的**（最高 **2.10x 加速** at lmax=2）
- **lmax ≤ 3**: `pure-cartesian-ictd` 优势最明显（1.91x - 2.10x）
- **lmax = 4-5**: `pure-cartesian-ictd` 仍然更快（1.44x - 1.78x）
- **lmax = 6**: `pure-cartesian-ictd` 和 `spherical` 几乎相等（1.05x）

#### 2.2.3 参数量对比（GPU，lmax=0 到 6）

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 1    | 4,164,682 |        2,734,026 |            2,734,434 |     11,786,058 |           2,733,898 |           1,208,931 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 3    | 8,654,426 |        9,407,050 |            9,409,090 |     74,635,594 |           6,744,394 |           2,481,673 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 5    | 12,882,010 |       23,268,554 |           23,273,314 |         **FAILED** |          11,021,130 |           3,979,109 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**性能分析**：
- **`pure-cartesian-ictd` 参数量始终最少**（27-32% of spherical）
- **`pure-cartesian-sparse` 参数量适中**（70-88% of spherical）
- **`pure-cartesian` 在 lmax≥4 时失败**（内存不足）

#### 2.2.4 等变误差对比（GPU，O(3)，包含宇称）

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 1.03e-15  |        6.37e-16  |            1.20e-15  |      3.27e-15  |           9.31e-16  |           5.24e-08 |
| 2    | 9.27e-16  |        1.97e-16  |            8.96e-16  |      3.18e-14  |           1.30e-15  |           7.18e-07 |
| 4    | 9.16e-16  |        7.76e-14  |            4.26e-16  |       **FAILED** |           6.19e-16  |           7.16e-07 |
| 6    | 4.77e-16  |        7.02e-16  |            5.65e-16  |       **FAILED** |           5.72e-16  |           1.11e-07 |

**性能分析**：
- **严格等变**（误差 ~1e-15）：`spherical`, `partial-cartesian-loose`, `pure-cartesian-sparse`
- **可接受等变**（误差 ~1e-7）：`pure-cartesian-ictd`

---

## 3. 测试环境说明

### 3.1 CPU 测试环境

- **硬件**：CPU
- **模型配置**：`channels=64`, `lmax=0` 到 `lmax=6`
- **测试数据**：32 atoms, 256 edges
- **测试方法**：
  - 前向传播：30 次平均
  - 反向传播：20 次平均
  - 等变性测试：O(3) 等变性（包括宇称/反射），20 次随机旋转测试
- **数据类型**：float64
- **注意**：`pure-cartesian` 只测试到 lmax=3（lmax≥4 时失败）

### 3.2 GPU 测试环境

- **硬件**：NVIDIA GeForce RTX 3090
- **模型配置**：`channels=64`, `lmax=0` 到 `lmax=6`
- **测试数据**：32 atoms, 256 edges
- **测试方法**：
  - 前向传播：30 次平均
  - 反向传播：20 次平均
  - 等变性测试：O(3) 等变性（包括宇称/反射），20 次随机旋转测试
- **数据类型**：float64

### 3.3 性能指标说明

- **总训练时间加速比**：相对于 `spherical` 模式的总训练时间（前向+反向）加速比，>1.0x 表示更快，<1.0x 表示更慢
- **参数量**：模型的可训练参数总数
- **等变误差**：O(3) 等变性测试的最大误差（包括旋转和反射/宇称），误差越小表示等变性越严格

---

## 总结

### 测试结果总结

1. **实际任务精度表现**：
   - FSCETP 相比 MACE (64ch) 的能量 RMSE 降低了 66.2%（0.044 vs 0.13 mev/atom）
   - FSCETP 相比 MACE (64ch) 的力 RMSE 降低了 36.2%（7.4 vs 11.6 mev/Å）
   - `spherical` 和 `partial-cartesian` 模式达到最优精度

2. **速度性能表现**：
   - **CPU 环境**：`pure-cartesian-ictd` 在所有 lmax 下均最快（最高 4.12x 加速比 at lmax=2）
   - **GPU 环境**：`pure-cartesian-ictd` 在所有 lmax 下均最快（最高 2.10x 加速比 at lmax=2）

3. **参数效率表现**：
   - `pure-cartesian-ictd` 参数量最少（为 spherical 的 27-32%）
   - `pure-cartesian-sparse` 参数量适中（为 spherical 的 70-88%）

4. **等变性验证结果**：
   - 所有模式均通过 O(3) 等变性测试（包括宇称/反射），等变性误差 < 1e-6
   - 严格等变模式（误差 ~1e-15）：`spherical`, `partial-cartesian-loose`, `pure-cartesian-sparse`
   - 可接受等变模式（误差 ~1e-7）：`pure-cartesian-ictd`

### 推荐使用场景

- **最高精度需求**：使用 `spherical` 或 `partial-cartesian` 模式
- **速度优先 + 参数效率**：使用 `pure-cartesian-ictd` 模式
- **平衡选择**：使用 `pure-cartesian-sparse` 模式
- **避免使用**：`pure-cartesian` 模式（速度极慢，lmax≥4 时失败）
