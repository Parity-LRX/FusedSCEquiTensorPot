# 使用指南

## 安装

### 1. 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt

# 或者直接安装包（会自动安装依赖）
pip install -e .
```

### 2. 验证安装

安装完成后，可以通过以下命令验证CLI是否可用：

```bash
mff-preprocess --help
mff-train --help
mff-evaluate --help
```

## 完整使用流程

### 步骤 1: 数据预处理

首先，你需要将原始的XYZ文件预处理成库可以使用的格式。

```bash
mff-preprocess \
    --input-file 2000.xyz \
    --output-dir data \
    --max-atom 5 \
    --train-ratio 0.95 \
    --preprocess-h5 \
    --max-radius 5.0 \
    --num-workers 8
```

**参数说明：**
- `--input-file`: 输入的XYZ文件路径
- `--output-dir`: 输出目录（默认'data'），所有预处理文件会保存在这个目录下
- `--max-atom`: 每个结构最大原子数（用于填充）
- `--train-ratio`: 训练集比例（默认0.95）
- `--preprocess-h5`: 是否预处理H5文件（预计算邻居列表，加速训练）
- `--max-radius`: 邻居搜索的最大半径
- `--num-workers`: 并行处理的进程数

**输出文件（在 `data/` 目录下）：**
- `read_train.h5`, `read_val.h5` - 原子数据
- `raw_energy_train.h5`, `raw_energy_val.h5` - 原始能量
- `correction_energy_train.h5`, `correction_energy_val.h5` - 校正能量
- `cell_train.h5`, `cell_val.h5` - 晶胞信息
- `processed_train.h5`, `processed_val.h5` - 预处理后的数据（如果使用--preprocess-h5）
- `fitted_E0.csv` - 拟合的原子基准能量（默认：从训练集用最小二乘拟合得到）

### 步骤 2: 训练模型

**方式一：使用已预处理的数据**
```bash
mff-train \
    --data-dir data \
    --train-prefix train \
    --val-prefix val \
    --epochs 1000 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --min-learning-rate 2e-5 \
    --warmup-batches 1000 \
    --warmup-start-ratio 0.1 \
    --lr-decay-patience 1000 \
    --lr-decay-factor 0.98 \
    -a 1.0 \
    -b 10.0 \
    --update-param 1000 \
    --weight-a-growth 1.01 \
    --weight-b-decay 0.99 \
    --force-shift-value 1.0 \
    --patience 20 \
    --vhat-clamp-interval 2000 \
    --max-vhat-growth 5.0 \
    --max-grad-norm 0.5 \
    --grad-log-interval 500 \
    --dump-frequency 250 \
    --energy-log-frequency 10 \
    --device cuda \
    --dtype float64 \
    --num-workers 8 \
    --max-radius 5.0
```

**自定义原子能量（E0）**（可选）：
- **默认行为**：使用 `--data-dir` 下的 `fitted_E0.csv`（训练集最小二乘拟合得到）
- **方式 1：从 CSV 指定**：

```bash
mff-train \
    --data-dir data \
    --atomic-energy-file data/fitted_E0.csv
```

- **方式 2：直接在 CLI 指定**：

```bash
mff-train \
    --data-dir data \
    --atomic-energy-keys 1 6 7 8 \
    --atomic-energy-values -430.53299511 -821.03326787 -1488.18856918 -2044.3509823
```

**方式二：自动预处理并训练（一步完成）**
```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8 \
    --device cuda
```
如果 `data/` 目录下没有预处理文件，会自动从 `--input-file` 指定的XYZ文件进行预处理。

**快速开始示例（使用默认参数）：**
```bash
# 最简单的训练命令（使用所有默认值）
mff-train --input-file 2000.xyz --device cuda
```

**完整参数示例：**
```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --epochs 2000 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --min-learning-rate 1e-5 \
    --warmup-batches 1000 \
    --warmup-start-ratio 0.1 \
    --lr-decay-patience 1000 \
    --lr-decay-factor 0.98 \
    -a 1.0 \
    -b 10.0 \
    --update-param 1000 \
    --patience 20 \
    --dump-frequency 250 \
    --device cuda \
    --dtype float64
```

**参数说明：**

**数据参数：**
- `--data-dir`: 数据目录，包含预处理文件（默认'data'）
- `--input-file`: 输入XYZ文件路径，用于自动预处理（可选）
- `--train-prefix`: 训练数据文件前缀（默认'train'）
- `--val-prefix`: 验证数据文件前缀（默认'val'）

**基础参数：**
- `--epochs`: 训练轮数（默认1000）
- `--batch-size`: 批次大小（默认8）。如果GPU内存不足，可以减小此值（如4或2）
- `--checkpoint`: 模型保存路径（默认'combined_model.pth'）。训练完成后会保存最终模型，训练过程中也会定期保存
- `--device`: 设备（'cuda'或'cpu'，默认自动检测）。如果有GPU，强烈建议使用'cuda'
- `--dump-frequency`: 验证和保存模型的频率（每N个batch，默认250）。每N个batch会进行一次验证并保存模型检查点
- `--energy-log-frequency`: 记录能量预测的频率（每N个batch，默认10）。这些日志只写入文件，不输出到控制台
- `--dtype`: 张量数据类型（'float32'或'float64'，默认'float64'）。float64精度更高但速度较慢，float32速度更快但精度略低
- `--patience`: 早停耐心值（默认20）。如果连续N个epoch**加权验证损失**（`a × 能量损失 + b × 受力损失`）没有改善，训练会自动停止

**学习率参数：**
- `--learning-rate`: 目标学习率（默认2e-4）。这是预热后的学习率
- `--min-learning-rate`: 最小学习率（默认1e-5）。学习率不会低于此值
- `--warmup-batches`: 学习率预热的批次数（默认1000）。前N个batch学习率从`learning_rate × warmup_start_ratio`线性增长到`learning_rate`
- `--warmup-start-ratio`: 预热期间学习率的起始比例（默认0.1）。例如，如果`--learning-rate 2e-4`和`--warmup-start-ratio 0.1`，则学习率会从`2e-5`线性增长到`2e-4`
- `--lr-decay-patience`: 学习率衰减的间隔批次数（默认1000）。每N个batch后，如果验证指标没有改善，学习率会乘以`--lr-decay-factor`
- `--lr-decay-factor`: 学习率衰减因子（默认0.98）。每次衰减时学习率乘以该值（0.98表示减少2%）

**损失权重参数：**
- `--energy-weight` (或 `-a`): 能量损失的初始权重（默认1.0）。总损失 = `a × 能量损失 + b × 受力损失`
- `--force-weight` (或 `-b`): 受力损失的初始权重（默认10.0）。通常受力损失需要更大的权重，因为力的数量远多于能量
- `--update-param`: 自动调整权重 `a` 和 `b` 的频率（每N个batch，默认1000）。每N个batch会根据 `--weight-a-growth` 和 `--weight-b-decay` 调整权重
- `--weight-a-growth`: 能量权重 `a` 的增长率（默认1.01，即每次增长1%）。建议值：1.005（慢速，适合超长时间训练）、1.01（中速，推荐）、1.02（快速，适合短期训练）
- `--weight-b-decay`: 受力权重 `b` 的衰减率（默认0.99，即每次减少1%）。建议值：0.995（慢速）、0.99（中速，推荐）、0.98（快速）
- `--force-shift-value`: 受力标签的缩放系数（默认1.0）。如果力的单位需要转换，可以调整此值

**优化器参数：**
- `--vhat-clamp-interval`: 优化器 `v_hat` 钳位的频率（每N个batch，默认2000）。用于防止Adam优化器的二阶矩估计过大
- `--max-vhat-growth`: `v_hat` 的最大增长因子（默认5.0）。限制`v_hat`不超过历史最大值的N倍
- `--max-grad-norm`: 梯度裁剪阈值（默认0.5）。如果梯度范数超过此值，会被裁剪到此值，防止梯度爆炸
- `--grad-log-interval`: 记录梯度统计信息的频率（每N个batch，默认500）。用于监控训练稳定性

**数据处理参数：**
- `--num-workers`: 数据处理的并行进程数（默认8）。预处理时使用全部，训练时DataLoader会自动分配（训练使用一半，验证使用四分之一）
- `--max-radius`: 邻居搜索的最大半径（默认5.0 Å）。原子间距离超过此值不会被考虑为邻居

**模型架构超参数：**
- `--max-atomvalue`: 原子嵌入的最大原子序数（默认10）。如果数据集包含原子序数 > 10 的元素，需要增大此值
- `--embedding-dim`: 原子嵌入维度（默认16）。增大此值可以增强模型表达能力，但会增加计算量
- `--lmax`: 球谐函数的最高阶数（默认2）。控制不可约表示的最高阶
- `--irreps-output-conv-channels`: irreps_output_conv的通道数（默认64）。与`--lmax`共同决定irreps形式。例如：
  - `lmax=2, channels=64` → "64x0e + 64x1o + 64x2e"
  - `lmax=1, channels=64` → "64x0e + 64x1o"
  - `lmax=3, channels=64` → "64x0e + 64x1o + 64x2e + 64x3o"
  - 增大通道数可以提升模型容量，但会显著增加显存和计算量
- `--function-type`: 径向基函数类型（默认'gaussian'）。选项：
  - `gaussian`: 高斯基函数（默认，平滑且易于优化）
  - `bessel`: 贝塞尔基函数（适合周期性系统）
  - `fourier`: 傅里叶基函数（适合周期性边界条件）
  - `cosine`: 余弦基函数
  - `smooth_finite`: 平滑有限支撑基函数
- `--tensor-product-mode`: 张量积实现模式（默认'spherical'）。选项：
  - `spherical`: 使用 e3nn 球谐函数张量积（默认，精度高）
  - `cartesian`: 使用笛卡尔张量积（**速度快 2x+，参数少 45%**）

**SWA 和 EMA 参数：**
- `--swa-start-epoch`: 开始 SWA（Stochastic Weight Averaging）的 epoch。启用后，`a` 和 `b` 会在该 epoch 直接切换为 `--swa-a` 和 `--swa-b` 的值，并重置早停计数器
- `--swa-a`: SWA 阶段的能量权重 `a`（必须与 `--swa-start-epoch` 一起使用）
- `--swa-b`: SWA 阶段的力权重 `b`（必须与 `--swa-start-epoch` 一起使用）
- `--ema-start-epoch`: 开始 EMA（Exponential Moving Average）的 epoch。EMA 模型是主模型参数的指数滑动平均，通常在训练后期启用
- `--ema-decay`: EMA 衰减系数（默认 0.999）。越大越平滑，但响应越慢
- `--use-ema-for-validation`: 使用 EMA 模型进行验证（而非主模型）
- `--save-ema-model`: 在 checkpoint 中保存 EMA 模型权重

**分布式训练参数：**
- `--distributed`: 启用分布式训练（DDP模式）。需要配合`torchrun`或`torch.distributed.launch`使用
- `--local-rank`: 本地进程rank（通常由`torchrun`自动设置，无需手动指定）
- `--backend`: 分布式后端（默认'nccl'用于GPU，'gloo'用于CPU）

**训练输出：**
- `combined_model.pth` - 模型检查点
- `combined_model_epoch{epoch}_batch_count{batch_count}.pth` - 定期保存的模型（频率由`--dump-frequency`控制，默认每250个batch保存一次）
- `training_YYYYMMDD_HHMMSS.log` - 训练日志（包含详细的batch级别信息）
- `training_YYYYMMDD_HH_loss.csv` - 损失记录
- `val_energy_epoch{epoch}_batch{batch_count}.csv` - 验证集能量预测（频率与模型保存一致）
- `val_force_epoch{epoch}_batch{batch_count}.csv` - 验证集力预测（频率与模型保存一致）

### 步骤 3: 评估模型

#### 3.1 静态评估（计算RMSE和MAE）

**注意：评估时需要使用与训练时相同的模型超参数！**

```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --test-prefix test \
    --output-prefix test \
    --batch-size 1 \
    --max-atomvalue 10 \
    --embedding-dim 16 \
    --lmax 2 \
    --irreps-output-conv-channels 64 \
    --function-type gaussian \
    --tensor-product-mode spherical \
    --device cuda
```

**参数说明：**
- `--checkpoint`: 模型检查点路径
- `--test-prefix`: 测试数据文件前缀
- `--output-prefix`: 输出文件前缀
- `--use-h5`: 使用H5Dataset（如果已预处理），否则使用OnTheFlyDataset
- `--batch-size`: 批次大小（默认1）
- `--max-atomvalue`: 必须与训练时相同
- `--embedding-dim`: 必须与训练时相同
- `--lmax`: 必须与训练时相同
- `--irreps-output-conv-channels`: 必须与训练时相同（如果训练时设置了）
- `--function-type`: 必须与训练时相同
- `--tensor-product-mode`: 必须与训练时相同（`spherical` 或 `cartesian`）

**输出文件：**
- `test_loss.csv` - 测试集损失指标
- `test_energy.csv` - 测试集能量预测
- `test_force.csv` - 测试集力预测

#### 3.2 分子动力学模拟（MD）

使用 ASE 的 Langevin 恒温器进行分子动力学模拟。MD 模拟会自动跳过静态评估，直接进行动力学计算。

**基本用法：**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --md-sim \
    --md-input start.xyz \
    --device cuda
```

**完整参数示例：**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --md-sim \
    --md-input molecule.xyz \
    --md-temperature 300 \
    --md-timestep 1.0 \
    --md-steps 10000 \
    --md-friction 0.01 \
    --md-relax-fmax 0.05 \
    --md-log-interval 10 \
    --md-output md_traj.xyz \
    --device cuda
```

**MD 参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--md-input` | `start_structure.xyz` | 输入结构文件（XYZ 格式） |
| `--md-temperature` | 300.0 | 模拟温度 (K) |
| `--md-timestep` | 1.0 | 时间步长 (fs) |
| `--md-steps` | 10000 | 总步数 |
| `--md-friction` | 0.01 | Langevin 摩擦系数 |
| `--md-relax-fmax` | 0.05 | 预优化力收敛阈值 (eV/Å) |
| `--md-log-interval` | 10 | 日志和轨迹记录间隔 |
| `--md-output` | `md_traj.xyz` | 输出轨迹文件 |
| `--md-no-relax` | False | 跳过初始结构优化 |

**MD 工作流程：**
1. 加载初始结构（`--md-input`）
2. 可选：使用 BFGS 优化器进行结构优化（除非使用 `--md-no-relax`）
3. 初始化 Maxwell-Boltzmann 速度分布
4. 运行 Langevin 动力学模拟
5. 定期保存轨迹和日志

**输出文件：**
- `md_traj.xyz` - MD 轨迹（或 `--md-output` 指定的文件）
- `md_traj_log.txt` - 能量/温度日志（文件名基于 `--md-output`）
- `relaxed_structure.xyz` - 优化后的初始结构（如果进行了预优化）
- `md_relax.log` - 预优化过程的日志

**示例 1：标准 MD 模拟（300K，10 ps）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --md-sim \
    --md-input molecule.xyz \
    --md-temperature 300 \
    --md-timestep 1.0 \
    --md-steps 10000 \
    --md-output md_300K.xyz \
    --device cuda
```

**示例 2：高温 MD 模拟（500K，50 ps）**
```bash
# 500K 下运行 50 ps
mff-evaluate \
    --checkpoint best_model.pth \
    --md-sim \
    --md-input reactant.xyz \
    --md-temperature 500 \
    --md-timestep 0.5 \
    --md-steps 100000 \
    --md-friction 0.005 \
    --md-output md_500K.xyz \
    --device cuda
```

**示例 3：跳过预优化直接运行 MD**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --md-sim \
    --md-input pre_relaxed.xyz \
    --md-no-relax \
    --md-temperature 300 \
    --md-steps 50000 \
    --device cuda
```

**示例 4：长时间 MD 模拟（100 ps）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --md-sim \
    --md-input system.xyz \
    --md-temperature 300 \
    --md-timestep 1.0 \
    --md-steps 100000 \
    --md-log-interval 50 \
    --md-output md_long.xyz \
    --device cuda
```

**注意事项：**
1. **时间步长选择**：通常 0.5-2.0 fs 是安全的。对于轻原子（H），建议使用更小的时间步长（0.5 fs）
2. **摩擦系数**：0.01 是常用值。更小的值（0.001-0.005）适合长时间模拟，更大的值（0.05-0.1）适合快速平衡
3. **预优化**：建议对初始结构进行优化，除非结构已经优化过
4. **周期性边界条件**：如果输入结构包含晶胞信息，会自动使用 PBC
5. **能量单位**：所有能量以 eV 为单位，力以 eV/Å 为单位

#### 3.3 NEB（Nudged Elastic Band）计算

使用 ASE 的 NEB 方法寻找化学反应的过渡态和能垒。NEB 计算会自动跳过静态评估，直接进行路径优化。

**基本用法：**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --neb \
    --neb-initial reactant.xyz \
    --neb-final product.xyz \
    --device cuda
```

**完整参数示例：**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --neb \
    --neb-initial initial.xyz \
    --neb-final final.xyz \
    --neb-images 15 \
    --neb-fmax 0.03 \
    --neb-output neb.traj \
    --device cuda
```

**NEB 参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--neb-initial` | `initial.xyz` | 初始结构（反应物） |
| `--neb-final` | `final.xyz` | 最终结构（产物） |
| `--neb-images` | 10 | 中间图像数量（不包括端点） |
| `--neb-fmax` | 0.05 | 力收敛阈值 (eV/Å)，使用 NEB 投影力 |
| `--neb-output` | `neb.traj` | 输出轨迹文件（ASE Trajectory 格式） |

**NEB 工作流程：**
1. 加载初始和最终结构
2. 创建 NEB 图像链（初始 + N 个中间图像 + 最终）
3. 线性插值生成初始路径
4. 使用 FIRE 优化器优化路径（Climbing Image NEB）
5. 输出优化后的路径和能垒信息

**输出文件：**
- `neb.traj` - NEB 优化轨迹（ASE Trajectory 格式，包含所有图像）

**输出信息包括：**
- 正向能垒 (Forward barrier): E_saddle - E_initial
- 逆向能垒 (Reverse barrier): E_saddle - E_final
- 反应能 (Reaction energy): E_final - E_initial
- 每个优化步骤的详细日志

**准备输入文件示例：**
```bash
# 创建初始结构（反应物）
cat > initial.xyz << EOF
3

O  0.000  0.000  0.000
H  0.960  0.000  0.000
H -0.240  0.930  0.000
EOF

# 创建最终结构（产物）
cat > final.xyz << EOF
3

O  0.000  0.000  0.000
H  1.500  0.000  0.000
H -0.240  0.930  0.000
EOF
```

**示例 1：标准 NEB 计算**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --neb \
    --neb-initial reactant.xyz \
    --neb-final product.xyz \
    --neb-images 10 \
    --neb-fmax 0.05 \
    --device cuda
```

**示例 2：高精度 NEB 计算**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --neb \
    --neb-initial reactant.xyz \
    --neb-final product.xyz \
    --neb-images 20 \
    --neb-fmax 0.02 \
    --neb-output neb_high_res.traj \
    --dtype float64 \
    --device cuda
```

**示例 3：快速 NEB 计算（较少图像）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --neb \
    --neb-initial reactant.xyz \
    --neb-final product.xyz \
    --neb-images 5 \
    --neb-fmax 0.1 \
    --device cuda
```

**查看和分析 NEB 结果：**
```python
from ase.io import read
import matplotlib.pyplot as plt
import numpy as np

# 读取 NEB 轨迹
images = read('neb.traj', index=':')

# 提取能量
energies = [img.get_potential_energy() for img in images]

# 绘制能量曲线
plt.figure(figsize=(10, 6))
plt.plot(range(len(energies)), energies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Image Index', fontsize=12)
plt.ylabel('Energy (eV)', fontsize=12)
plt.title('NEB Energy Profile', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('neb_profile.png', dpi=300)
plt.close()

# 计算能垒和反应能
e_initial = energies[0]
e_saddle = max(energies)
e_final = energies[-1]

forward_barrier = e_saddle - e_initial
reverse_barrier = e_saddle - e_final
reaction_energy = e_final - e_initial

print(f"Initial energy:    {e_initial:.4f} eV")
print(f"Saddle point:      {e_saddle:.4f} eV")
print(f"Final energy:      {e_final:.4f} eV")
print(f"Forward barrier:   {forward_barrier:.4f} eV")
print(f"Reverse barrier:   {reverse_barrier:.4f} eV")
print(f"Reaction energy:   {reaction_energy:.4f} eV")

# 可视化结构（使用 ASE GUI）
from ase.visualize import view
view(images)
```

**使用 ASE 分析 NEB 结果：**
```python
from ase.io import read
from ase.neb import NEBTools

# 读取轨迹
images = read('neb.traj', index=':')

# 使用 NEBTools 分析
nebtools = NEBTools(images)

# 获取能垒
barrier = nebtools.get_barrier()[0]  # 正向能垒
print(f"Forward barrier: {barrier:.4f} eV")

# 获取鞍点索引
saddle_index = nebtools.get_fitted_pes()[1]
print(f"Saddle point at image: {saddle_index}")

# 绘制拟合的势能面
nebtools.plot_band()
```

**注意事项：**
1. **结构要求**：初始和最终结构必须有相同的原子数和原子类型
2. **预优化**：建议先分别优化初始和最终结构，确保它们是局部最小值
3. **图像数量**：更多图像（15-20）提供更高路径分辨率，但计算量更大。10 个图像通常是好的起点
4. **收敛标准**：`--neb-fmax` 使用 NEB 投影力（不是原始原子力），这是正确的收敛判据
5. **Climbing Image**：自动启用 Climbing Image NEB (CI-NEB) 来精确定位鞍点
6. **周期性边界条件**：如果输入结构包含晶胞信息，会自动使用 PBC
7. **插值方法**：使用改进的切线方法（improved tangent）进行路径插值
8. **优化器**：使用 FIRE 优化器，对 NEB 优化特别有效

## Python API 使用

### 示例 1: 数据预处理

```python
from molecular_force_field.data.preprocessing import (
    extract_data_blocks,
    fit_baseline_energies,
    compute_correction,
    save_set,
    save_to_h5_parallel
)

# 提取数据块（支持解析 energy / pbc / Lattice / Properties）
all_blocks, all_energy, all_raw_energy, all_cells, all_pbcs = extract_data_blocks('data.xyz')

# 拟合基准能量
keys = np.array([1, 6, 7, 8], dtype=np.int64)
fitted_values = fit_baseline_energies(
    train_blocks, train_raw_E, keys,
    initial_values=np.array([-0.01] * len(keys))
)

# 计算校正能量
train_correction = compute_correction(train_blocks, train_raw_E, keys, fitted_values)

# 保存数据（包含 pbc 信息）
save_set('train', train_indices, train_blocks, train_raw_E, train_correction, all_cells, pbc_list=all_pbcs)

# 预处理H5文件（预计算邻居列表）
save_to_h5_parallel('train', max_radius=5.0, num_workers=8)
```

### 示例 2: 训练模型

```python
import torch
from torch.utils.data import DataLoader
from molecular_force_field.models import E3_TransformerLayer_multi, MainNet
from molecular_force_field.data import H5Dataset
from molecular_force_field.data.collate import collate_fn_h5
from molecular_force_field.training.trainer import Trainer
from molecular_force_field.utils.config import ModelConfig

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
train_dataset = H5Dataset('train')
val_dataset = H5Dataset('val')

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn_h5,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn_h5,
    num_workers=2,
    pin_memory=True
)

# 模型配置
config = ModelConfig()

# 初始化模型
model = MainNet(
    input_size=config.input_dim_weight,
    hidden_sizes=config.main_hidden_sizes4,
    output_size=1
).to(device)

e3trans = E3_TransformerLayer_multi(
    max_embed_radius=config.max_radius,
    main_max_radius=config.max_radius_main,
    main_number_of_basis=config.number_of_basis_main,
    irreps_input=config.get_irreps_input_conv_main(),
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
    device=device
).to(device)

# 创建训练器
trainer = Trainer(
    model=model,
    e3trans=e3trans,
    train_loader=train_loader,
    val_loader=val_loader,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    device=device,
    config=config,
    learning_rate=2e-4,
    epoch_numbers=1000,
    checkpoint_path='combined_model.pth',
    atomic_energy_keys=config.atomic_energy_keys,
    atomic_energy_values=config.atomic_energy_values,
)

# 开始训练
trainer.run_training()
```

### 示例 3: 评估模型

```python
import torch
from torch.utils.data import DataLoader
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.data import OnTheFlyDataset
from molecular_force_field.data.collate import on_the_fly_collate
from molecular_force_field.evaluation.evaluator import Evaluator
from molecular_force_field.utils.config import ModelConfig

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('combined_model.pth', map_location=device)

config = ModelConfig()
e3trans = E3_TransformerLayer_multi(...).to(device)
e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
e3trans.eval()

# 加载测试数据集
test_dataset = OnTheFlyDataset(
    'read_test.h5',
    'raw_energy_test.h5',
    'cell_test.h5',
    max_radius=5.0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=on_the_fly_collate,
    num_workers=10
)

# 评估
evaluator = Evaluator(
    model=e3trans,
    dataset=test_dataset,
    device=device,
    atomic_energy_keys=config.atomic_energy_keys,
    atomic_energy_values=config.atomic_energy_values,
)

metrics = evaluator.evaluate(test_loader, output_prefix='test')
print(f"Energy RMSE: {metrics['energy_rmse']:.6f}")
print(f"Force RMSE: {metrics['force_rmse']:.6f}")
```

### 示例 4: 使用 ASE Calculator 进行 MD 模拟

```python
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase import units
from ase.md import MDLogger
from molecular_force_field.evaluation.calculator import MyE3NNCalculator
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.config import ModelConfig
import torch

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('combined_model.pth', map_location=device)

config = ModelConfig()
e3trans = E3_TransformerLayer_multi(
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
    device=device
).to(device)

e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
e3trans.eval()

# 创建计算器
atomic_energies_dict = {
    1: -430.53299511,
    6: -821.03326787,
    7: -1488.18856918,
    8: -2044.3509823
}
calc = MyE3NNCalculator(e3trans, atomic_energies_dict, device, max_radius=5.0)

# 读取结构
atoms = read('structure.xyz')
atoms.set_calculator(calc)

# 可选：预优化结构
print("Relaxing structure...")
opt = BFGS(atoms, logfile='relax.log')
opt.run(fmax=0.05)
write('relaxed_structure.xyz', atoms)

# 初始化速度分布
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# 设置 MD 参数
timestep = 1.0  # fs
temperature = 300.0  # K
friction = 0.01
n_steps = 10000
log_interval = 10

# 创建 Langevin 动力学
dyn = Langevin(
    atoms, 
    timestep * units.fs, 
    temperature_K=temperature, 
    friction=friction
)

# 添加日志和轨迹记录
dyn.attach(MDLogger(dyn, atoms, 'md_log.txt', header=True, mode="w"), interval=log_interval)
dyn.attach(lambda: write('md_traj.xyz', atoms, append=True), interval=log_interval)

# 运行 MD
print(f"Starting MD: {n_steps} steps ({n_steps * timestep / 1000:.2f} ps)")
dyn.run(n_steps)
print("MD simulation completed!")
```

### 示例 5: 使用 ASE NEB 计算反应能垒

```python
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE
from molecular_force_field.evaluation.calculator import MyE3NNCalculator
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.config import ModelConfig
import torch

# 加载模型（同上）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('combined_model.pth', map_location=device)

config = ModelConfig()
e3trans = E3_TransformerLayer_multi(...).to(device)
e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
e3trans.eval()

# 创建计算器
atomic_energies_dict = {
    1: -430.53299511,
    6: -821.03326787,
    7: -1488.18856918,
    8: -2044.3509823
}
calc = MyE3NNCalculator(e3trans, atomic_energies_dict, device, max_radius=5.0)

# 读取初始和最终结构
initial = read('reactant.xyz')
final = read('product.xyz')

# 可选：预优化端点
print("Optimizing initial structure...")
initial.set_calculator(calc)
opt_initial = BFGS(initial)
opt_initial.run(fmax=0.05)

print("Optimizing final structure...")
final.set_calculator(calc)
opt_final = BFGS(final)
opt_final.run(fmax=0.05)

# 创建 NEB 图像
n_images = 10  # 中间图像数量
images = [initial]
images += [initial.copy() for _ in range(n_images)]
images += [final]

# 设置计算器和晶胞
for image in images:
    if initial.cell.any():
        image.set_cell(initial.cell)
        image.set_pbc(initial.pbc)
    else:
        image.set_cell([100, 100, 100])
        image.set_pbc(False)
    image.set_calculator(calc)

# 创建 NEB 对象
neb = NEB(images, climb=True, method='improvedtangent', allow_shared_calculator=True)
neb.interpolate()

# 优化路径
optimizer = FIRE(neb, trajectory='neb.traj')

# 定义日志函数
def log_neb_status():
    energies = [img.get_potential_energy() for img in images]
    neb_forces = neb.get_forces()
    n_images_total = len(images)
    n_atoms = len(images[0])
    forces_reshaped = neb_forces.reshape(n_images_total, n_atoms, 3)
    intermediate_forces = forces_reshaped[1:-1]
    max_force = (intermediate_forces**2).sum(axis=2).max()**0.5
    print(
        f"NEB step {optimizer.nsteps:4d} | "
        f"E_min = {min(energies):.6f} eV | "
        f"E_max = {max(energies):.6f} eV | "
        f"Barrier = {max(energies) - energies[0]:.4f} eV | "
        f"F_max = {max_force:.4f} eV/Å"
    )

optimizer.attach(log_neb_status, interval=1)

# 运行优化
print("Starting NEB optimization...")
optimizer.run(fmax=0.05)

# 输出最终结果
energies = [img.get_potential_energy() for img in images]
e_initial = energies[0]
e_saddle = max(energies)
e_final = energies[-1]

print("\n" + "=" * 60)
print("NEB Results:")
print(f"  Initial energy:    {e_initial:.4f} eV")
print(f"  Saddle point:      {e_saddle:.4f} eV")
print(f"  Final energy:      {e_final:.4f} eV")
print(f"  Forward barrier:   {e_saddle - e_initial:.4f} eV")
print(f"  Reverse barrier:   {e_saddle - e_final:.4f} eV")
print(f"  Reaction energy:   {e_final - e_initial:.4f} eV")
print("=" * 60)
```

## 常见问题

### Q: 如何查看所有可用的命令行参数？

```bash
mff-preprocess --help
mff-train --help
mff-evaluate --help
```

### Q: 训练时如何恢复之前的检查点？

训练器会**自动检测并加载**检查点文件。如果 `--checkpoint` 指定的文件已存在，会自动恢复训练状态。

**自动恢复的内容：**
- ✅ 模型权重（`e3trans_state_dict`）
- ✅ Epoch 数（从中断的 epoch 继续）
- ✅ Batch count（累计的 batch 数）
- ✅ 损失权重 `a` 和 `b`
- ✅ 最佳验证损失（用于早停判断，基于 `a × 能量损失 + b × 受力损失`）
- ✅ 早停计数器（patience counter）

**使用示例：**

```bash
# 第一次训练（训练到 epoch 50 时中断）
mff-train \
    --data-dir data \
    --epochs 100 \
    --checkpoint my_model.pth \
    --device cuda

# 程序中断后，再次运行相同命令即可继续
# 会自动从 epoch 51 继续训练
mff-train \
    --data-dir data \
    --epochs 100 \
    --checkpoint my_model.pth \
    --device cuda

# 日志会显示：
# ================================================================================
# Checkpoint Loaded Successfully!
#   Resuming from epoch: 51
#   Batch count: 12500
#   Loss weights: a=1.0234, b=9.7845
#   Best validation loss: 0.012345
#   Early stopping patience counter: 3/20
# ================================================================================
```

**注意事项：**
1. 确保使用**相同的超参数**（`--max-atomvalue`, `--lmax`, `--function-type` 等）
2. 确保使用**相同的数据目录**和**相同的设备**
3. 可以调整 `--epochs` 来延长训练（例如从 100 改为 200）
4. **不能**在恢复时改变学习率调度器的初始设置（会重新初始化）

**如何从头重新训练（忽略已有检查点）：**
```bash
# 方法1: 使用不同的检查点名称
mff-train --checkpoint new_model.pth ...

# 方法2: 删除或重命名旧检查点
mv my_model.pth my_model_backup.pth
mff-train --checkpoint my_model.pth ...
```

### Q: 如何调整模型超参数？

可以通过修改 `molecular_force_field/utils/config.py` 中的 `ModelConfig` 类，或者在使用Python API时直接传入参数。

### Q: 如何调整模型超参数？

可以通过CLI参数调整模型架构：

**原子嵌入参数：**
```bash
# 默认设置（适用于H、C、N、O等常见元素）
mff-train --max-atomvalue 10 --embedding-dim 16

# 包含更多元素类型（如金属）
mff-train --max-atomvalue 50 --embedding-dim 32

# 增大嵌入维度以提升模型表达能力
mff-train --max-atomvalue 10 --embedding-dim 32
```

**模型容量参数（lmax 和通道数）：**
```bash
# 默认设置（lmax=2, 64通道, gaussian基函数）
# 生成: "64x0e + 64x1o + 64x2e"
mff-train --lmax 2 --irreps-output-conv-channels 64

# 降低最高阶数（更简单的体系，更快训练）
# 生成: "64x0e + 64x1o"
mff-train --lmax 1 --irreps-output-conv-channels 64

# 增加最高阶数（更复杂的体系，更强表达能力）
# 生成: "64x0e + 64x1o + 64x2e + 64x3o"
mff-train --lmax 3 --irreps-output-conv-channels 64

# 增大通道数（更大模型容量）
# 生成: "128x0e + 128x1o + 128x2e"
mff-train --lmax 2 --irreps-output-conv-channels 128

# 同时增大lmax和通道数（最大模型容量，适合复杂体系）
# 生成: "128x0e + 128x1o + 128x2e + 128x3o"
mff-train --lmax 3 --irreps-output-conv-channels 128

# 减小模型容量（适合简单体系或内存受限）
# 生成: "32x0e + 32x1o"
mff-train --lmax 1 --irreps-output-conv-channels 32
```

**径向基函数类型：**
```bash
# 默认：高斯基函数（推荐，平滑且易于优化）
mff-train --function-type gaussian

# 贝塞尔基函数（适合周期性系统）
mff-train --function-type bessel

# 傅里叶基函数（适合周期性边界条件）
mff-train --function-type fourier

# 余弦基函数
mff-train --function-type cosine

# 平滑有限支撑基函数
mff-train --function-type smooth_finite
```

**注意事项：**
- `--max-atomvalue` 必须大于等于数据集中最大的原子序数
- `--lmax` 控制球谐函数的最高阶数，影响模型对角度信息的表达能力
  - `lmax=0`: 仅标量 (scalar)
  - `lmax=1`: 标量 + 矢量 (scalar + vector)
  - `lmax=2`: 标量 + 矢量 + 二阶张量 (scalar + vector + rank-2 tensor)
  - `lmax=3`: 包含三阶张量
  - 更高的 lmax 可以捕捉更复杂的几何特征，但计算量显著增加
- `--function-type` 选择径向基函数类型：
  - `gaussian`: 最常用，适合大多数体系
  - `bessel`: 适合具有周期性的体系（如晶体）
  - `fourier`: 适合周期性边界条件
  - 通常默认的 `gaussian` 就足够了
- 增大 `--embedding-dim`、`--lmax` 和 `--irreps-output-conv-channels` 会显著增加显存占用和训练时间
- 评估时必须使用与训练时相同的超参数值
- 这些参数会影响模型性能，建议根据数据集规模和复杂度调整

**完整示例：大模型配置**
```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --max-atomvalue 20 \
    --embedding-dim 32 \
    --lmax 3 \
    --irreps-output-conv-channels 128 \
    --function-type gaussian \
    --batch-size 4 \
    --device cuda
```

### Q: 如何调整学习率？

训练时可以通过以下参数调整学习率策略：

**基础学习率设置：**
```bash
mff-train \
    --learning-rate 2e-4 \      # 目标学习率（预热后达到的值）
    --min-learning-rate 1e-5    # 最小学习率（防止过小）
```

**学习率预热（Warmup）：**
```bash
mff-train \
    --warmup-batches 1000 \     # 前1000个batch进行预热
    --warmup-start-ratio 0.1    # 从10%的目标学习率开始，线性增长到100%
```
预热期间，学习率从`learning_rate × warmup_start_ratio`线性增加到`learning_rate`。例如，如果`--learning-rate 2e-4`和`--warmup-start-ratio 0.1`，则学习率会从`2e-5`线性增长到`2e-4`。这有助于训练初期的稳定性。

**学习率衰减（Decay）：**
```bash
mff-train \
    --lr-decay-patience 1000 \  # 每1000个batch检查一次，如果没有改善则衰减
    --lr-decay-factor 0.98      # 每次衰减时学习率乘以0.98
```
例如：初始学习率2e-4，每1000个batch后如果验证指标没有改善，学习率变为2e-4 × 0.98 = 1.96e-4，再1000个batch后变为1.96e-4 × 0.98 = 1.92e-4，以此类推。

**完整示例：**
```bash
# 快速收敛设置（较大学习率，快速预热）
mff-train \
    --learning-rate 5e-4 \
    --min-learning-rate 1e-5 \
    --warmup-batches 500 \
    --warmup-start-ratio 0.1

# 稳定训练设置（较小学习率，更长预热）
mff-train \
    --learning-rate 1e-4 \
    --min-learning-rate 1e-6 \
    --warmup-batches 2000 \
    --warmup-start-ratio 0.05

# 精细调优设置（小学习率，慢衰减）
mff-train \
    --learning-rate 1e-5 \
    --lr-decay-factor 0.99 \
    --lr-decay-patience 2000 \
    --warmup-start-ratio 0.2
```

### Q: 如何调整损失权重？

损失函数为：`总损失 = a × 能量损失 + b × 受力损失`

**初始权重设置：**
```bash
# 默认设置（能量权重1.0，受力权重10.0）
mff-train -a 1.0 -b 10.0

# 更重视能量（如果能量预测较差）
mff-train -a 2.0 -b 5.0

# 更重视受力（如果受力预测较差）
mff-train -a 0.5 -b 20.0
```

**自动调整权重：**
```bash
# 每500个batch自动调整一次（更频繁）
mff-train -a 1.0 -b 10.0 --update-param 500

# 每2000个batch自动调整一次（更稳定）
mff-train -a 1.0 -b 10.0 --update-param 2000
```

**调整速率控制：**
```bash
# 慢速调整（适合超长时间训练，200k+ batches）
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.005 --weight-b-decay 0.995

# 中速调整（推荐，适合大多数情况）
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.01 --weight-b-decay 0.99

# 快速调整（适合短期训练，<50k batches）
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.02 --weight-b-decay 0.98
```

自动调整规则：每`--update-param`个batch，`a`会乘以`--weight-a-growth`（增加），`b`会乘以`--weight-b-decay`（减少），以逐渐平衡能量和受力的重要性。**注意：`a` 和 `b` 没有范围限制，会按照上述规则持续调整。**

**可选：为 a / b 设置范围（Clamp）**（适合长训练防止权重漂移过大）：
```bash
# 将 a 限制在 [0.5, 10]，b 限制在 [0.001, 100]
mff-train \
  -a 1.0 -b 10.0 \
  --a-min 0.5 --a-max 10 \
  --b-min 0.001 --b-max 100
```

**速率选择建议：**
- **慢速 (1.005/0.995)**: 适合超长时间训练（>200k batches），权重变化平缓，训练更稳定
- **中速 (1.01/0.99)**: **推荐默认值**，适合大多数训练场景（50k-200k batches），平衡稳定性和调整速度
- **快速 (1.02/0.98)**: 适合短期训练（<50k batches），快速调整权重，但可能导致训练后期不稳定

### Q: 如何调整优化器参数？

**梯度裁剪（防止梯度爆炸）：**
```bash
# 更严格的梯度裁剪（适合训练不稳定时）
mff-train --max-grad-norm 0.3

# 更宽松的梯度裁剪（适合训练稳定时）
mff-train --max-grad-norm 1.0
```

**优化器稳定性参数：**
```bash
# 更频繁地钳位v_hat（适合训练不稳定时）
mff-train --vhat-clamp-interval 1000 --max-vhat-growth 3.0

# 更宽松的v_hat控制（适合训练稳定时）
mff-train --vhat-clamp-interval 5000 --max-vhat-growth 10.0
```

**监控梯度：**
```bash
# 每100个batch记录一次梯度统计（用于调试）
mff-train --grad-log-interval 100

# 每1000个batch记录一次（减少日志量）
mff-train --grad-log-interval 1000
```

### Q: 如何选择合适的批次大小？

批次大小影响训练速度和内存使用：

```bash
# 小批次（适合GPU内存较小或数据集较小）
mff-train --batch-size 4

# 中等批次（默认，适合大多数情况）
mff-train --batch-size 8

# 大批次（适合GPU内存充足，可以加速训练）
mff-train --batch-size 16
```

**注意：** 批次大小会影响梯度估计的稳定性。如果批次太小（如2或4），可能需要降低学习率；如果批次太大，可能需要增加学习率。

### Q: 如何调整验证和保存频率？

```bash
# 更频繁的验证和保存（适合快速迭代）
mff-train --dump-frequency 100

# 默认频率（平衡性能和监控）
mff-train --dump-frequency 250

# 较少验证（适合长时间训练，减少I/O开销）
mff-train --dump-frequency 500
```

**注意：** `--dump-frequency`控制验证和模型保存的频率。更频繁的验证可以更早发现问题，但会增加训练时间。

### Q: 数据格式要求是什么？

输入的XYZ文件需要包含：
- 原子坐标 (x, y, z)
- 原子类型/原子序数
- 力 (Fx, Fy, Fz)
- 能量信息（在Properties行中）
- 晶胞信息（在Lattice属性中，可选）

### Q: 如何使用多卡并行训练？

多卡训练可以显著加速训练过程，特别是在大数据集上。使用步骤如下：

**1. 使用 torchrun（推荐，PyTorch 1.9+）：**
```bash
# 使用4张GPU
torchrun --nproc_per_node=4 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8
```

**2. 使用 torch.distributed.launch（旧版本兼容）：**
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8
```

**注意事项：**
- `--nproc_per_node` 指定使用的GPU数量
- 必须添加 `--distributed` 参数
- 每个GPU的有效batch size = `--batch-size`，总的有效batch size = `--batch-size × GPU数量`
- 建议根据总的有效batch size调整学习率（通常线性缩放）
- 所有日志、检查点和CSV文件只在主进程（rank 0）保存
- 验证结果会自动聚合所有进程的结果

**示例：4卡训练，每卡batch size=8，总有效batch size=32**
```bash
torchrun --nproc_per_node=4 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --batch-size 8 \
    --learning-rate 4e-3  # 可以适当增加学习率（4倍batch size）
```

### Q: 如何使用多卡并行训练？

多卡训练可以显著加速训练过程，特别是在大数据集上。使用步骤如下：

**1. 使用 torchrun（推荐，PyTorch 1.9+）：**
```bash
# 使用4张GPU
torchrun --nproc_per_node=4 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8
```

**2. 使用 torch.distributed.launch（旧版本兼容）：**
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8
```

**注意事项：**
- `--nproc_per_node` 指定使用的GPU数量
- 必须添加 `--distributed` 参数
- 每个GPU的有效batch size = `--batch-size`，总的有效batch size = `--batch-size × GPU数量`
- 建议根据总的有效batch size调整学习率（通常线性缩放）
- 所有日志、检查点和CSV文件只在主进程（rank 0）保存
- 验证结果会自动聚合所有进程的结果

**示例：4卡训练，每卡batch size=8，总有效batch size=32**
```bash
torchrun --nproc_per_node=4 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --batch-size 8 \
    --learning-rate 4e-3  # 可以适当增加学习率（4倍batch size）
```

### Q: 如何加速训练？

**1. 使用笛卡尔张量积模式（推荐，2.2x 加速）：**
```bash
# 笛卡尔模式比球谐模式快 2 倍以上，参数量减少 45%
mff-train --input-file data.xyz --tensor-product-mode cartesian
```

**2. 使用预处理数据：**
```bash
# 预处理时使用 --preprocess-h5 预计算邻居列表
mff-preprocess --input-file 2000.xyz --preprocess-h5

# 训练时使用预处理的数据（自动检测）
mff-train --data-dir data
```

**3. 使用GPU：**
```bash
mff-train --device cuda
```

**4. 增加并行处理：**
```bash
# 增加数据加载的并行进程数
mff-train --num-workers 16

# 注意：num_workers会根据CPU核心数自动调整，训练时DataLoader会使用一半线程
```

**5. 使用float32（如果精度足够）：**
```bash
# float32比float64快约2倍，但精度略低
mff-train --dtype float32
```

**6. 增加批次大小（如果GPU内存允许）：**
```bash
# 更大的批次可以更好地利用GPU
mff-train --batch-size 16
```

**7. 减少验证频率（长时间训练时）：**
```bash
# 减少I/O开销
mff-train --dump-frequency 500
```

**8. 使用多卡并行训练（最有效）：**
```bash
# 使用4张GPU并行训练，速度提升接近4倍
torchrun --nproc_per_node=4 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --batch-size 8
```

### Q: 应该选择 Spherical 还是 Cartesian 模式？

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| 发表论文/高精度需求 | `spherical` | 使用 e3nn 严格球谐，精度最高 |
| 大规模 MD 模拟 | `cartesian` | 推理速度快 2x+，适合长时间模拟 |
| GPU 内存受限 | `cartesian` | 参数量减少 45%，显存占用更低 |
| 生产环境部署 | `cartesian` | 速度优先 |
| 快速实验迭代 | `cartesian` | 训练更快 |
| 首次尝试/对比基准 | `spherical` | 验证模型正确性 |

**切换模式：**
```bash
# 球谐模式（默认）
mff-train --input-file data.xyz

# 笛卡尔模式
mff-train --input-file data.xyz --tensor-product-mode cartesian
```

**注意：** 训练和评估必须使用相同的模式！

### Q: 如何使用 SWA 和 EMA？

**SWA（Stochastic Weight Averaging）** 用于在训练后期固定损失权重，避免权重漂移：
```bash
# 从 epoch 100 开始，将 a 固定为 100，b 固定为 10
mff-train \
    --input-file data.xyz \
    -a 1.0 -b 10.0 \
    --swa-start-epoch 100 --swa-a 100 --swa-b 10
```

**EMA（Exponential Moving Average）** 用于维护模型参数的滑动平均，提高稳定性：
```bash
# 从 epoch 150 开始启用 EMA，衰减系数 0.999
mff-train \
    --input-file data.xyz \
    --ema-start-epoch 150 --ema-decay 0.999 \
    --use-ema-for-validation \
    --save-ema-model
```

**同时使用 SWA 和 EMA：**
```bash
mff-train \
    --input-file data.xyz \
    -a 1.0 -b 10.0 \
    --swa-start-epoch 100 --swa-a 100 --swa-b 10 \
    --ema-start-epoch 150 --ema-decay 0.999 --use-ema-for-validation --save-ema-model
```

**推荐设置：**
- SWA 开始时间：总 epoch 的 50%-70%
- EMA 开始时间：总 epoch 的 60%-80%
- EMA 衰减系数：0.999（默认，越大越平滑）

**注意：** SWA 启动时会重置早停计数器（patience_counter）和最佳验证损失（best_val_loss），以适应新的损失权重。

### Q: 训练时如何监控进度？

**控制台输出：**
- 验证结果会实时输出到控制台
- 每个epoch的摘要信息会输出到控制台

**日志文件：**
- `training_YYYYMMDD_HHMMSS.log` - 包含所有详细信息
  - Batch级别的训练指标（每N个batch，由`--energy-log-frequency`控制）
  - 验证时的详细结果
  - 梯度统计信息（每N个batch，由`--grad-log-interval`控制）

**CSV文件：**
- `training_YYYYMMDD_HH_loss.csv` - 训练损失记录
- `val_energy_epoch{epoch}_batch{batch_count}.csv` - 验证集能量预测
- `val_force_epoch{epoch}_batch{batch_count}.csv` - 验证集力预测

**实时监控：**
```bash
# 实时查看日志文件
tail -f training_*.log

# 查看最新的验证结果
tail -f training_*.log | grep "Validation"
```

## 完整训练示例

### 示例 1: 快速测试（小数据集，快速验证）

```bash
mff-train \
    --input-file small_test.xyz \
    --data-dir data \
    --epochs 100 \
    --batch-size 4 \
    --learning-rate 1e-3 \
    --warmup-batches 100 \
    --dump-frequency 50 \
    --device cuda \
    --dtype float32
```

**适用场景：** 快速验证代码和参数设置是否正确

### 示例 2: 标准训练（中等数据集，平衡设置）

```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --min-learning-rate 2e-5 \
    --warmup-batches 1000 \
    --warmup-start-ratio 0.1 \
    --lr-decay-patience 1000 \
    --lr-decay-factor 0.98 \
    -a 1.0 \
    -b 10.0 \
    --update-param 1000 \
    --patience 20 \
    --dump-frequency 250 \
    --device cuda \
    --dtype float64
```

**适用场景：** 大多数情况下的推荐配置

### 示例 3: 大规模训练（大数据集，长时间训练）

```bash
mff-train \
    --input-file large_dataset.xyz \
    --data-dir data \
    --epochs 5000 \
    --batch-size 16 \
    --learning-rate 1e-3 \
    --min-learning-rate 5e-6 \
    --warmup-batches 2000 \
    --warmup-start-ratio 0.05 \
    --lr-decay-patience 2000 \
    --lr-decay-factor 0.99 \
    -a 1.0 \
    -b 10.0 \
    --update-param 2000 \
    --patience 50 \
    --dump-frequency 500 \
    --max-grad-norm 0.5 \
    --vhat-clamp-interval 2000 \
    --device cuda \
    --dtype float64 \
    --num-workers 16
```

**适用场景：** 大规模数据集，需要长时间训练以获得最佳性能

### 示例 4: 精细调优（接近收敛，小步调整）

```bash
mff-train \
    --data-dir data \
    --checkpoint combined_model.pth \
    --epochs 500 \
    --batch-size 8 \
    --learning-rate 5e-5 \
    --min-learning-rate 1e-6 \
    --warmup-batches 500 \
    --warmup-start-ratio 0.2 \
    --lr-decay-patience 1000 \
    --lr-decay-factor 0.995 \
    -a 0.5 \
    -b 15.0 \
    --update-param 2000 \
    --patience 30 \
    --dump-frequency 100 \
    --device cuda \
    --dtype float64
```

**适用场景：** 从已有检查点继续训练，进行精细调优

### 示例 5: CPU训练（无GPU环境）

```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --epochs 500 \
    --batch-size 4 \
    --learning-rate 1e-3 \
    --warmup-batches 500 \
    --dump-frequency 100 \
    --device cpu \
    --dtype float32 \
    --num-workers 4
```

**适用场景：** 在没有GPU的机器上训练，使用float32和较小的批次大小以加速

## 注意事项

1. **内存使用**: 使用 `H5Dataset` 时，数据会预加载到内存，确保有足够的内存。如果内存不足，可以考虑使用 `OnTheFlyDataset`（但训练速度会较慢）。

2. **GPU内存**: 如果遇到GPU内存不足（OOM错误），可以：
   - 减小 `--batch-size`（如从8减到4）
   - 使用 `--dtype float32` 而不是 `float64`
   - 减小 `--max-radius`（减少邻居数量）

3. **数据格式**: 确保XYZ文件格式正确，特别是能量和力的单位。能量通常以eV为单位，力以eV/Å为单位。

4. **检查点**: 训练过程中会定期保存检查点（频率由`--dump-frequency`控制），建议定期备份。最终模型会保存在`--checkpoint`指定的路径。

5. **日志**: 训练日志保存在 `training_*.log` 文件中，包含详细的训练信息。控制台只显示验证结果和重要信息，详细日志请查看日志文件。

6. **早停**: 如果**加权验证损失**（`a × 能量损失 + b × 受力损失`）连续`--patience`个epoch没有改善，训练会自动停止。这可以防止过拟合并节省时间。早停依据与训练时的总损失保持一致。

7. **权重自动调整**: `a`和`b`会在训练过程中自动调整（每`--update-param`个batch）。`a`会乘以`--weight-a-growth`（默认1.01，即增长1%），`b`会乘以`--weight-b-decay`（默认0.99，即减少1%），以平衡能量和受力的重要性。如果训练不稳定，可以：
   - 增大`--update-param`（减少调整频率）
   - 使用更慢的调整速率（`--weight-a-growth 1.005 --weight-b-decay 0.995`）
   - 固定权重（设置`--update-param`为一个很大的值，如1000000）

8. **学习率策略**: 学习率会经历三个阶段：
   - **预热阶段**（前`--warmup-batches`个batch）：从`learning_rate × warmup_start_ratio`线性增长到`learning_rate`
   - **稳定阶段**：保持`learning_rate`
   - **衰减阶段**：如果验证指标不改善，每`--lr-decay-patience`个batch后乘以`--lr-decay-factor`

## 笛卡尔张量积模式

FusedEquiTensorPot 支持两种张量积实现模式：

### 模式对比

| 特性 | Spherical (e3nn) | Cartesian |
|------|------------------|-----------|
| 实现 | e3nn 球谐函数 | 笛卡尔坐标直接计算 |
| 速度 | 基准 | **2.2x 加速** |
| 参数量 | 基准 | **减少 45%** |
| 精度 | 最高 | 接近 |
| 等变性 | ✅ 严格等变 | ✅ 等变 |
| DDP 支持 | ✅ | ✅ |
| SWA/EMA 支持 | ✅ | ✅ |

### 使用方法

**训练时选择模式：**
```bash
# 球谐模式（默认，精度最高）
mff-train --input-file data.xyz --data-dir output

# 笛卡尔模式（速度快，参数少）
mff-train --input-file data.xyz --data-dir output --tensor-product-mode cartesian
```

**评估时必须使用相同模式：**
```bash
# 如果训练时用的是 cartesian
mff-evaluate --checkpoint model.pth --tensor-product-mode cartesian
```

### 性能对比（channels=64, lmax=2）

| 系统规模 | Spherical (ms) | Cartesian (ms) | 加速比 |
|----------|----------------|----------------|--------|
| 20 atoms | 35.2 | 15.5 | 2.28x |
| 50 atoms | 141.8 | 64.4 | 2.20x |
| 100 atoms | 404.3 | 182.9 | 2.21x |

**参数量：** Spherical=6,534,394 | Cartesian=3,609,780 (55.2%)

### 推荐使用场景

**使用 Spherical（默认）：**
- 需要最高精度
- 研究/发表论文场景
- 小规模数据集

**使用 Cartesian：**
- 需要更快的推理速度
- 大规模 MD 模拟
- GPU 内存受限
- 生产环境部署

### 完整示例：笛卡尔模式 + 多卡 + SWA + EMA

```bash
torchrun --nproc_per_node=4 -m molecular_force_field.cli.train \
    --input-file large_dataset.xyz \
    --data-dir data \
    --tensor-product-mode cartesian \
    --distributed \
    --epochs 2000 \
    --batch-size 8 \
    --lmax 2 \
    --irreps-output-conv-channels 64 \
    -a 1.0 -b 10.0 \
    --swa-start-epoch 1000 --swa-a 100 --swa-b 10 \
    --ema-start-epoch 1500 --ema-decay 0.999 --use-ema-for-validation --save-ema-model \
    --patience 50 \
    --device cuda
```

### Python API 使用

```python
from molecular_force_field.models import CartesianTransformerLayer
from molecular_force_field.utils.config import ModelConfig

config = ModelConfig(
    max_atomvalue=10,
    embedding_dim=16,
    lmax=2,
    channel_in=64,
)

# 创建笛卡尔模型
model = CartesianTransformerLayer(
    max_embed_radius=config.max_radius,
    main_max_radius=config.max_radius_main,
    main_number_of_basis=config.number_of_basis_main,
    hidden_dim_conv=config.channel_in,
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
    lmax=config.lmax,
    device='cuda'
).to('cuda')

# 前向传播
energy = model(pos, Z, batch, edge_src, edge_dst, edge_shifts, cell)
forces = -torch.autograd.grad(energy.sum(), pos)[0]
```

### 笛卡尔张量积 API

如果你需要单独使用笛卡尔张量积层：

```python
from molecular_force_field.models import CartesianFullyConnectedTensorProduct

# 创建张量积（类似 e3nn.o3.FullyConnectedTensorProduct）
tp = CartesianFullyConnectedTensorProduct(
    irreps_in1="64x0e + 64x1o + 64x2e",
    irreps_in2="1x0e + 1x1o + 1x2e",
    irreps_out="64x0e + 64x1o + 64x2e",
    shared_weights=True,
    internal_weights=True
)

# 前向传播
out = tp(x1, x2)

# 或者使用外部权重
tp_ext = CartesianFullyConnectedTensorProduct(
    irreps_in1="64x0e + 64x1o + 64x2e",
    irreps_in2="1x0e + 1x1o + 1x2e",
    irreps_out="64x0e",
    shared_weights=False,
    internal_weights=False
)
weights = weight_network(edge_features)  # shape: (..., tp_ext.weight_numel)
out = tp_ext(x1, x2, weights)
```