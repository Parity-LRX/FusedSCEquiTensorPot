# 使用指南

FusedSCEquiTensorPot 支持**八种等变张量积实现模式**，包括：
- `spherical`: 基于 e3nn 的球谐函数方法（默认）
- `spherical-save`: channelwise edge conv（e3nn 后端，参数量更少）
- `spherical-save-cue`: channelwise edge conv（cuEquivariance 后端，需可选依赖，GPU 加速）
- `partial-cartesian`: 笛卡尔坐标 + e3nn CG 系数（严格等变，部分使用 e3nn）
- `partial-cartesian-loose`: 近似等变（norm product 近似，部分使用 e3nn）
- `pure-cartesian`: 纯笛卡尔 \(3^L\) 表示（严格等变，速度较慢，完全自实现）
- `pure-cartesian-sparse`: 稀疏纯笛卡尔（严格等变，参数量优化，完全自实现）
- `pure-cartesian-ictd`: ICTD irreps 内部表示（严格等变，速度最快，参数量最少，完全自实现）

所有模式都保持 O(3) 等变性（包括旋转和反射）。详细性能对比见[张量积模式对比](#张量积模式对比)部分。

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
mff-export-core --help   # LAMMPS LibTorch 导出
mff-lammps --help        # LAMMPS fix external 接口
python -m molecular_force_field.cli.export_mliap --help  # LAMMPS ML-IAP 导出
```

### 3. LAMMPS 集成

本框架支持三种 LAMMPS 集成方式，详见 [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md)：

| 方式 | 速度 | 要求 | 适用场景 |
|------|------|------|----------|
| **USER-MFFTORCH（LibTorch 纯 C++）** | 最快，无 Python | LAMMPS 编译 KOKKOS + USER-MFFTORCH | HPC、超算、生产部署 |
| **ML-IAP unified** | 较快（约 1.7x fix external） | LAMMPS 编译 ML-IAP | 推荐，支持 GPU |
| **fix external / pair_style python** | 较慢 | 标准 LAMMPS + Python | 快速验证 |

Python API 示例见[示例 5a：LAMMPS LibTorch 接口](#示例-5a-使用-lammps-libtorch-接口usermfftorchhpc-推荐)和[示例 5b：LAMMPS ML-IAP 接口](#示例-5b-使用-lammps-mliap-unified-接口)。

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
- `stress_train.h5`, `stress_val.h5` - 应力张量（当 XYZ 含 stress/virial 时；否则为零矩阵）
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
    --weight-a-growth 1.05 \
    --weight-b-decay 0.98 \
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

**应力训练**（可选，用于周期性体系）：
若 XYZ 含 stress/virial 且为 PBC 体系，可启用应力损失：
```bash
mff-train \
    --data-dir data \
    --stress-weight 0.1
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
- `--data-dir`: 数据目录，包含预处理文件（默认'data'）。所有预处理后的数据文件（如processed_train.h5, processed_val.h5）应在此目录下
- `--input-file`: 输入XYZ文件路径，用于自动预处理（可选）。如果指定此参数且数据目录下没有预处理文件，会自动从XYZ文件进行预处理
- `--train-prefix`: 训练数据文件前缀（默认'train'）。用于查找`processed_{train-prefix}.h5`等文件
- `--val-prefix`: 验证数据文件前缀（默认'val'）。用于查找`processed_{val-prefix}.h5`等文件
- `--max-radius`: 邻居搜索的最大半径（默认5.0 Å）。原子间距离超过此值不会被考虑为邻居。此参数在预处理和训练时都需要使用

**基础训练参数：**
- `--epochs`: 训练轮数（默认1000）。训练会运行指定的epoch数，除非早停触发
- `--batch-size`: 批次大小（默认8）。如果GPU内存不足，可以减小此值（如4或2）。注意：验证时batch-size固定为1
- `--checkpoint`: 模型保存路径（默认'combined_model.pth'）。训练完成后会保存最终模型（最佳验证性能的模型）到`checkpoint/`目录下，文件名会包含张量积模式后缀（如`combined_model_pure_cartesian.pth`）。训练过程中也会定期保存中间checkpoint到`checkpoint/`目录
- `--reset-loss-weights`: 从checkpoint恢复训练时，忽略保存的损失权重（a, b），使用命令行参数指定的值（默认False，即使用checkpoint中的权重）
- `--device`: 设备（'cuda'或'cpu'，默认自动检测）。如果有GPU，强烈建议使用'cuda'。分布式训练时会自动为每个进程分配对应的GPU
- `--dtype`: 张量数据类型（'float32'或'float64'，默认'float64'）。选项：'float32'/'float'（32位浮点，速度快但精度略低）或'float64'/'double'（64位浮点，精度高但速度较慢）
- `--seed`: 随机种子（默认42）。用于确保实验可复现性。影响数据shuffle、train/val split等随机操作

**验证和日志参数：**
- `--dump-frequency`: 验证和保存模型的频率（每N个batch，默认250）。每N个batch会进行一次验证并保存模型检查点。注意：验证是基于batch_count触发的，不是固定在每个epoch结束时
- `--train-eval-sample-ratio`: 验证时对训练集采样的比例（0.0-1.0，默认0.2）。设置为1.0表示评估整个训练集，设置为0.2表示只评估20%的训练集（更快）。对于大型数据集，建议使用较小的值以加快验证速度
- `--energy-log-frequency`: 记录能量预测的频率（每N个batch，默认10）。这些日志只写入文件，不输出到控制台
- `--log-val-batch-energy`: 在控制台输出验证阶段每个batch的能量预测（默认False）。如果为False，验证batch能量信息只记录到日志文件，不输出到控制台；如果为True，会同时在控制台和日志文件中输出
- `--save-val-csv`: 保存验证能量和力预测到CSV文件（默认False）。如果启用，验证结果会保存到`validation/`目录下，文件名为`val_energy_epoch{epoch}_batch{batch_count}.csv`和`val_force_epoch{epoch}_batch{batch_count}.csv`
- `--no-save-val-csv`: 禁用保存验证CSV文件（与`--save-val-csv`互斥）。用于减少I/O开销

**早停参数：**
- `--patience`: 早停耐心值（默认20，单位：epoch）。如果连续N个epoch的验证损失（能量损失+力损失+应力损失，未加权）没有改善，训练会自动停止。注意：早停使用的是未加权的验证损失，而不是加权损失（`a × 能量损失 + b × 力损失 + c × 应力损失`）

**学习率参数：**
- `--learning-rate`: 目标学习率（默认1e-3）。这是预热后的学习率，也是训练过程中的主要学习率
- `--min-learning-rate`: 最小学习率（默认2e-5）。学习率不会低于此值，即使经过多次衰减
- `--warmup-batches`: 学习率预热的批次数（默认1000）。前N个batch学习率从`learning_rate × warmup_start_ratio`线性增长到`learning_rate`
- `--warmup-start-ratio`: 预热期间学习率的起始比例（默认0.1）。例如，如果`--learning-rate 1e-3`和`--warmup-start-ratio 0.1`，则学习率会从`1e-4`线性增长到`1e-3`
- `--lr-decay-patience`: 学习率衰减的间隔批次数（默认1000）。每N个batch后，如果验证指标没有改善，学习率会乘以`--lr-decay-factor`
- `--lr-decay-factor`: 学习率衰减因子（默认0.98）。每次衰减时学习率乘以该值（0.98表示减少2%）。学习率调度策略：如果验证损失在`lr-decay-patience`个batch内没有改善，学习率会乘以此因子

**损失权重参数：**
- `--energy-weight` (或 `-a`): 能量损失的初始权重（默认1.0）。总损失 = `a × 能量损失 + b × 受力损失 + c × 应力损失`。训练过程中，`a`会根据`--weight-a-growth`自动增长
- `--force-weight` (或 `-b`): 受力损失的初始权重（默认10.0）。通常受力损失需要更大的权重，因为力的数量远多于能量（每个原子有3个力分量）。训练过程中，`b`会根据`--weight-b-decay`自动衰减
- `--stress-weight` (或 `-c`): 应力损失的权重（默认0.0，即禁用）。设为 > 0 时启用应力训练，通过晶胞应变导数计算应力（σ = (1/V) × dE/dε）。**需要**训练 XYZ 文件中包含 stress 或 virial 数据，且为周期性体系（PBC）。应力单位：eV/Å³
- `--c-min`: 应力权重 `c` 的最小值（默认0.0）
- `--c-max`: 应力权重 `c` 的最大值（默认1000.0）
- `--update-param`: 自动调整权重 `a` 和 `b` 的频率（每N个batch，默认1000）。每N个batch会根据 `--weight-a-growth` 和 `--weight-b-decay` 调整权重。调整公式：`a = a × weight_a_growth`，`b = b × weight_b_decay`
- `--weight-a-growth`: 能量权重 `a` 的增长率（默认1.05，即每次增长5%）。建议值：1.005（慢速，适合超长时间训练）、1.01（中速，更稳定）、1.02（快速）、1.05（超快速）
- `--weight-b-decay`: 受力权重 `b` 的衰减率（默认0.98，即每次减少2%）。建议值：0.995（慢速）、0.99（中速，更稳定）、0.98（快速）
- `--a-min`: 能量权重 `a` 的最小值（默认1.0）。`a`在增长时不会低于此值
- `--a-max`: 能量权重 `a` 的最大值（默认1000.0）。`a`在增长时不会超过此值
- `--b-min`: 受力权重 `b` 的最小值（默认1.0）。`b`在衰减时不会低于此值
- `--b-max`: 受力权重 `b` 的最大值（默认1000.0）。`b`在衰减时不会超过此值
- `--force-shift-value`: 受力标签的缩放系数（默认1.0）。如果力的单位需要转换，可以调整此值。注意：此参数目前未在代码中使用，保留用于未来扩展

**优化器参数：**
- `--vhat-clamp-interval`: 优化器 `v_hat` 钳位的频率（每N个batch，默认2000）。用于防止Adam优化器的二阶矩估计（`v_hat`）过大。每N个batch会检查并限制`v_hat`的增长
- `--max-vhat-growth`: `v_hat` 的最大增长因子（默认5.0）。限制`v_hat`不超过历史最大值的N倍，防止优化器不稳定
- `--max-grad-norm`: 梯度裁剪阈值（默认0.5）。如果梯度范数超过此值，会被裁剪到此值，防止梯度爆炸。使用`torch.nn.utils.clip_grad_norm_`实现
- `--grad-log-interval`: 记录梯度统计信息的频率（每N个batch，默认500）。用于监控训练稳定性。梯度统计信息（范数、最大值、最小值、平均值）会记录到日志文件中

**数据处理参数：**
- `--num-workers`: 数据处理的并行进程数（默认8）。预处理时使用全部进程数，训练时DataLoader会自动分配：训练DataLoader使用`max(1, num_workers // 2)`，验证DataLoader使用`max(1, num_workers // 4)`

**模型架构超参数：**
- `--max-atomvalue`: 原子嵌入的最大原子序数（默认10）。如果数据集包含原子序数 > 10 的元素，需要增大此值。例如，如果数据集包含Cl（原子序数17），需要设置`--max-atomvalue 17`
- `--embedding-dim`: 原子嵌入维度（默认16）。增大此值可以增强模型表达能力，但会增加计算量和显存占用
- `--embed-size`: 读出MLP的隐藏层大小（默认[128, 128, 128]）。可以指定多个值，例如`--embed-size 128 256 128`表示三层隐藏层，大小分别为128、256、128
- `--output-size`: 原子读出MLP的输出大小（默认8）。这是每个原子的特征维度，用于后续的能量和力预测
- `--lmax`: 球谐函数的最高阶数（默认2）。控制不可约表示的最高阶。增大`lmax`可以捕获更高阶的几何信息，但会显著增加计算量。常见值：1（快速，适合简单系统）、2（推荐，平衡性能和精度）、3（高精度，但计算量大）
- `--irreps-output-conv-channels`: irreps_output_conv的通道数（可选，默认None，会使用config中的channel_in，通常为64）。与`--lmax`共同决定irreps形式。例如：
  - `lmax=2, channels=64` → "64x0e + 64x1o + 64x2e"
  - `lmax=1, channels=64` → "64x0e + 64x1o"
  - `lmax=3, channels=64` → "64x0e + 64x1o + 64x2e + 64x3o"
  - 增大通道数可以提升模型容量，但会显著增加显存和计算量
- `--function-type`: 径向基函数类型（默认'gaussian'）。选项：
  - `gaussian`: 高斯基函数（默认，平滑且易于优化，推荐用于大多数场景）
  - `bessel`: 贝塞尔基函数（适合周期性系统）
  - `fourier`: 傅里叶基函数（适合周期性边界条件）
  - `cosine`: 余弦基函数
  - `smooth_finite`: 平滑有限支撑基函数
- `--tensor-product-mode`: 等变张量积实现模式（默认'spherical'）。**本框架支持八种等变张量积模式**，选项：
  - `spherical`: 使用 e3nn 球谐函数张量积（默认，精度高，标准实现，推荐用于大多数场景）
  - `spherical-save`: channelwise edge conv（e3nn 后端，参数量更少）
  - `spherical-save-cue`: channelwise edge conv（cuEquivariance 后端，需 `pip install -e ".[cue]"`，GPU 加速）
  - `partial-cartesian`: 笛卡尔坐标 + e3nn CG 系数（严格等变，参数量减少 17.4%）
  - `partial-cartesian-loose`: 近似等变张量积（norm product 近似，速度较快，参数量减少 17.3%，非严格等变）
  - `pure-cartesian`: 纯笛卡尔张量积（\(3^L\) 表示，严格等变，速度极慢，lmax≥4 时失败，不推荐）
  - `pure-cartesian-sparse`: 稀疏纯笛卡尔张量积（严格等变，参数量减少 29.6%，需设置`--max-rank-other`和`--k-policy`）
  - `pure-cartesian-ictd`: ICTD irreps 内部表示（严格等变，参数量最少减少 72.1%，速度最快，推荐用于大规模训练）
  
  详细性能对比和推荐场景见[张量积模式对比](#张量积模式对比)部分。

**张量积模式特定参数（仅用于pure-cartesian-sparse和pure-cartesian-ictd）：**
- `--max-rank-other`: 稀疏张量积的最大rank（仅用于`pure-cartesian-sparse`模式，默认1）。只允许`min(L1, L2) <= max_rank_other`的相互作用。增大此值允许更多相互作用，但会增加参数量和计算量
- `--k-policy`: 稀疏张量积的K策略（仅用于`pure-cartesian-sparse`模式，默认'k0'）。选项：
  - `k0`: 只保留k=0（促进更高rank，推荐）
  - `k1`: 只保留k=1（收缩到更低rank）
  - `both`: 保留k=0和k=1（更多相互作用，但计算量更大）
- `--ictd-tp-path-policy`: ICTD张量积的路径策略（仅用于`pure-cartesian-ictd`模式，默认'full'）。选项：
  - `full`: 保留所有CG允许的(l1,l2->l3)路径（推荐，性能最好）
  - `max_rank_other`: 只保留`min(l1,l2) <= --ictd-tp-max-rank-other`的路径（减少参数量）
- `--ictd-tp-max-rank-other`: ICTD路径剪枝的最大rank（仅用于`pure-cartesian-ictd`模式，当`--ictd-tp-path-policy=max_rank_other`时使用，默认None）。例如，设置为1时只保留标量/向量耦合

**SWA 和 EMA 参数：**
- `--swa-start-epoch`: 开始 SWA（Stochastic Weight Averaging）的 epoch（可选，默认None）。启用后，`a` 和 `b` 会在该 epoch 直接切换为 `--swa-a` 和 `--swa-b` 的值，并重置早停计数器（`best_val_loss`重置为inf，`patience_counter`重置为0）。如果未设置，则使用连续的线性增长/衰减策略
- `--swa-a`: SWA 阶段的能量权重 `a`（必须与 `--swa-start-epoch` 一起使用）。当达到`swa-start-epoch`时，`a`会直接切换为此值，不再继续增长
- `--swa-b`: SWA 阶段的力权重 `b`（必须与 `--swa-start-epoch` 一起使用）。当达到`swa-start-epoch`时，`b`会直接切换为此值，不再继续衰减
- `--ema-start-epoch`: 开始 EMA（Exponential Moving Average）的 epoch（可选，默认None）。EMA 模型是主模型参数的指数滑动平均，通常在训练后期启用（建议在总epoch数的60%-80%时启用）。如果未设置，EMA功能被禁用
- `--ema-decay`: EMA 衰减系数（默认 0.999，范围0-1）。越大越平滑，但响应越慢。典型值：0.999（非常平滑，推荐）、0.99（较快响应）
- `--use-ema-for-validation`: 使用 EMA 模型进行验证（而非主模型）。如果启用，验证时会使用EMA权重进行前向传播，通常能获得更稳定的验证结果
- `--save-ema-model`: 在 checkpoint 中保存 EMA 模型权重。如果启用，checkpoint会包含`e3trans_ema_state_dict`，可以从checkpoint恢复EMA模型

**验证加速参数：**
- `--compile-val`: 验证时对 e3trans 使用 `torch.compile`，`none`（默认）或 `e3trans`
- `--compile-val-mode`: 编译模式，如 `reduce-overhead`、`max-autotune`
- `--compile-val-fullgraph`: 强制完整图编译
- `--compile-val-dynamic`: 动态形状

**分布式训练参数：**
- `--distributed`: 启用分布式训练（DDP模式）。需要配合`torchrun`或`torch.distributed.launch`使用。启用后，训练会自动使用多GPU并行
- `--local-rank`: 本地进程rank（默认-1，通常由`torchrun`自动设置，无需手动指定）。如果未设置，会从环境变量`LOCAL_RANK`读取
- `--backend`: 分布式后端（默认'nccl'，选项：'nccl'或'gloo'）。'nccl'用于GPU训练（推荐），'gloo'用于CPU训练
- `--init-method`: 分布式初始化方法（默认'env://'）。通常使用环境变量方式初始化，无需修改

**原子参考能量（E0）参数：**
- `--atomic-energy-file`: 包含原子参考能量的CSV文件路径（可选，默认None）。CSV文件应包含`Atom`和`E0`两列。如果未设置，会使用`{data-dir}/fitted_E0.csv`（由训练集最小二乘拟合得到）
- `--atomic-energy-keys`: 原子序数列表（可选，必须与`--atomic-energy-values`一起使用）。例如：`--atomic-energy-keys 1 6 7 8`表示H、C、N、O
- `--atomic-energy-values`: 对应的原子参考能量值（eV，可选，必须与`--atomic-energy-keys`一起使用）。例如：`--atomic-energy-values -430.53 -821.03 -1488.19 -2044.35`

**训练输出：**

所有输出文件会保存在当前工作目录下：

**模型检查点（保存在`checkpoint/`目录）：**
- `checkpoint/combined_model_{tensor_product_mode}.pth` - 最终模型（最佳验证性能的模型，文件名包含张量积模式后缀，如`_spherical`、`_pure_cartesian`等）
- `checkpoint/combined_model_epoch{epoch}_batch_count{batch_count}_{tensor_product_mode}.pth` - 定期保存的中间模型（频率由`--dump-frequency`控制，默认每250个batch保存一次）

**验证结果（保存在`validation/`目录，仅当启用`--save-val-csv`时）：**
- `validation/val_energy_epoch{epoch}_batch{batch_count}.csv` - 验证集能量预测（包含Target_Energy、Predicted_Energy、Delta三列）
- `validation/val_force_epoch{epoch}_batch{batch_count}.csv` - 验证集力预测（包含Fx_True、Fy_True、Fz_True、Fx_Pred、Fy_Pred、Fz_Pred六列）

**日志和记录文件（保存在当前目录）：**
- `training_YYYYMMDD_HHMMSS.log` - 训练日志（包含详细的batch级别信息，使用RotatingFileHandler，每个文件最大1GB，保留5个备份）
- `loss.csv` - 损失记录（包含每个验证点的训练和验证指标，如 loss、RMSE、MAE 等；启用应力训练时还包含 stress_loss、stress_rmse、stress_mae）

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
- `--compile`: 推理加速，`none`（默认）或 `e3trans`（对 e3trans 层使用 `torch.compile`）
- `--compile-mode`: 编译模式，如 `reduce-overhead`、`max-autotune`
- `--max-atomvalue`: 必须与训练时相同
- `--embedding-dim`: 必须与训练时相同
- `--lmax`: 必须与训练时相同
- `--irreps-output-conv-channels`: 必须与训练时相同（如果训练时设置了）
- `--function-type`: 必须与训练时相同
- `--tensor-product-mode`: 必须与训练时相同（支持六种模式：`spherical`、`partial-cartesian`、`partial-cartesian-loose`、`pure-cartesian`、`pure-cartesian-sparse`、`pure-cartesian-ictd`）

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
5. **能量单位**：所有能量以 eV 为单位，力以 eV/Å 为单位，应力以 eV/Å³ 为单位

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

#### 3.4 声子谱计算（Phonon Spectrum）

计算声子谱（Hessian 矩阵、振动频率）。支持非周期性和周期性体系，默认使用 ASE on-the-fly 邻居构建。

**基本用法：**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --phonon \
    --phonon-input structure.xyz \
    --device cuda
```

**完整参数示例：**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --phonon \
    --phonon-input structure.xyz \
    --phonon-relax-fmax 0.01 \
    --phonon-output my_phonon \
    --max-radius 5.0 \
    --atomic-energy-file fitted_E0.csv \
    --device cuda
```

**声子谱参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--phonon` | False | 启用声子谱计算（必需标志） |
| `--phonon-input` | `structure.xyz` | 输入结构文件（XYZ 格式） |
| `--phonon-relax-fmax` | 0.01 | 结构优化力收敛阈值 (eV/Å) |
| `--phonon-output` | `phonon` | 输出文件前缀（生成 `{prefix}_hessian.npy` 和 `{prefix}_frequencies.txt`） |
| `--phonon-no-relax` | False | 跳过结构优化（如果结构已优化） |
| `--max-radius` | 5.0 | 邻居搜索最大半径 (Å) |
| `--atomic-energy-file` | `fitted_E0.csv` | 原子参考能量文件（CSV 格式，包含 Atom 和 E0 列） |

**声子谱工作流程：**
1. 加载输入结构（`--phonon-input`）
2. 可选：使用 BFGS 优化器进行结构优化（除非使用 `--phonon-no-relax`）
3. 使用 ASE on-the-fly 邻居列表构建图结构（自动处理周期性边界条件）
4. 计算 Hessian 矩阵（能量对位置的二阶导数，形状：`[N_atoms * 3, N_atoms * 3]`）
5. 从 Hessian 矩阵计算声子频率（通过动力学矩阵对角化）
6. 保存结果到文件

**输出文件：**
- `{prefix}_hessian.npy` - Hessian 矩阵（NumPy 格式，单位：eV/Å²）
- `{prefix}_frequencies.txt` - 声子频率列表（单位：cm⁻¹，负值表示虚频/不稳定模式）

**输出文件格式：**

`{prefix}_frequencies.txt` 格式示例：
```
# Phonon frequencies (cm⁻¹)
# Negative values indicate imaginary frequencies (unstable modes)
# Index    Frequency (cm⁻¹)
     0       1234.567890
     1        987.654321
     2        456.789012
   ...
```

**示例 1：非周期性体系（分子）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --phonon \
    --phonon-input water.xyz \
    --phonon-relax-fmax 0.01 \
    --phonon-output water_phonon \
    --device cuda
```

**示例 2：周期性体系（晶体）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --phonon \
    --phonon-input crystal.xyz \
    --phonon-relax-fmax 0.01 \
    --phonon-output crystal_phonon \
    --max-radius 5.0 \
    --device cuda
```

**示例 3：跳过结构优化（已优化结构）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --phonon \
    --phonon-input pre_relaxed.xyz \
    --phonon-no-relax \
    --phonon-output phonon \
    --device cuda
```

**示例 4：高精度计算（更严格的优化）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --phonon \
    --phonon-input structure.xyz \
    --phonon-relax-fmax 0.005 \
    --phonon-output phonon_high_precision \
    --dtype float64 \
    --device cuda
```

**查看和分析结果：**
```python
import numpy as np
import matplotlib.pyplot as plt

# 读取 Hessian 矩阵
hessian = np.load('phonon_hessian.npy')
print(f"Hessian shape: {hessian.shape}")
print(f"Hessian range: [{hessian.min():.6f}, {hessian.max():.6f}] eV/Å²")

# 检查对称性
is_symmetric = np.allclose(hessian, hessian.T, atol=1e-4)
print(f"Hessian symmetric: {is_symmetric}")

# 读取频率
frequencies = []
with open('phonon_frequencies.txt', 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 2:
                freq = float(parts[1])
                frequencies.append(freq)

frequencies = np.array(frequencies)
print(f"\nTotal modes: {len(frequencies)}")
print(f"Real modes (≥0): {np.sum(frequencies >= 0)}")
print(f"Imaginary modes (<0): {np.sum(frequencies < 0)}")
print(f"Frequency range: [{frequencies.min():.2f}, {frequencies.max():.2f}] cm⁻¹")

# 绘制频率分布
plt.figure(figsize=(10, 6))
plt.hist(frequencies[frequencies >= 0], bins=50, alpha=0.7, label='Real frequencies')
if np.any(frequencies < 0):
    plt.hist(frequencies[frequencies < 0], bins=20, alpha=0.7, label='Imaginary frequencies', color='red')
plt.xlabel('Frequency (cm⁻¹)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Phonon Frequency Distribution', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('phonon_frequencies.png', dpi=300)
plt.close()
```

**注意事项：**
1. **结构优化**：计算前建议先优化结构到能量最小值（默认启用）。如果结构未优化，可能出现虚频（负频率）
2. **周期性边界条件**：如果输入结构包含晶胞信息（`Lattice` 属性），会自动使用 PBC 和 on-the-fly 邻居构建
3. **邻居构建**：默认使用 ASE on-the-fly 邻居列表，自动处理非周期性和周期性体系
4. **大系统**：对于 >100 原子的系统，Hessian 计算可能较慢（O(N²) 复杂度）
5. **原子能量**：需要提供正确的原子参考能量（通过 `--atomic-energy-file` 或训练时生成的 `fitted_E0.csv`）
6. **虚频**：如果出现虚频（负值），说明结构不稳定，需要进一步优化或检查结构
7. **Hessian 对称性**：理论上 Hessian 矩阵应该是对称的，数值误差通常 < 1e-4
8. **单位**：
   - Hessian 矩阵：eV/Å²
   - 频率：cm⁻¹
   - 实频：正值（稳定模式）
   - 虚频：负值（不稳定模式）

**依赖要求：**
```bash
pip install scipy  # 声子谱计算需要（用于矩阵对角化）
pip install ase    # 结构处理和邻居列表（通常已安装）
```

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

# 提取数据块（支持解析 energy / pbc / Lattice / Properties / stress / virial）
all_blocks, all_energy, all_raw_energy, all_cells, all_pbcs, all_stresses = extract_data_blocks('data.xyz')

# 拟合基准能量
keys = np.array([1, 6, 7, 8], dtype=np.int64)
fitted_values = fit_baseline_energies(
    train_blocks, train_raw_E, keys,
    initial_values=np.array([-0.01] * len(keys))
)

# 计算校正能量
train_correction = compute_correction(train_blocks, train_raw_E, keys, fitted_values)

# 保存数据（包含 pbc 和 stress 信息）
save_set('train', train_indices, train_blocks, train_raw_E, train_correction, all_cells, pbc_list=all_pbcs, stress_list=all_stresses)

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

# num_interaction 说明：
# - 含义：控制 E3_TransformerLayer_multi 以及各 cartesian/pure-cartesian 变体中的交互(卷积)层数
# - 取值：最小 2，默认 2；n 层时会串联 n 次卷积
# - 影响：n 层会拼接 f1..fn，并将 f_combine_product 输出通道扩展为 (n-1)*32x0e
#
# 用法示例：将交互层数设为 3
#   num_interaction=3

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
    num_interaction=2,
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
    num_interaction=2,
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

### 示例 5a: 使用 LAMMPS LibTorch 接口（USER-MFFTORCH，HPC 推荐）

**USER-MFFTORCH** 是自定义 LAMMPS 包，提供 `pair_style mff/torch`，在 C++ 侧用 LibTorch 加载 TorchScript 模型，**运行时完全不需要 Python**，适合超算与生产部署。

**模型限制**：目前支持 `pure-cartesian-ictd` 系列和 `spherical-save-cue` 模型。元素顺序、cutoff 需与导出时一致。

**步骤 1：导出 core.pt**（需 Python，一次性）：
```bash
mff-export-core \
  --checkpoint model.pth \
  --elements H O \
  --device cuda \
  --max-radius 5.0 \
  --dtype float32 \
  --embed-e0 \
  --e0-csv fitted_E0.csv \
  --out core.pt
```

**步骤 2：编译 LAMMPS**：启用 `PKG_KOKKOS`、`PKG_USER-MFFTORCH`，详见 [lammps_user_mfftorch/docs/BUILD_AND_RUN.md](lammps_user_mfftorch/docs/BUILD_AND_RUN.md)。

**步骤 3：运行 LAMMPS**（纯 LAMMPS，无 Python）：
```bash
# 设置 LibTorch 动态库路径
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"

# 使用 Kokkos GPU 运行
lmp -k on g 1 -sf kk -pk kokkos newton off neigh full -in in.mfftorch
```

**LAMMPS 输入示例**（`in.mfftorch`）：
```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data

neighbor 1.0 bin
pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/core.pt H O

velocity all create 300 42
fix 1 all nve
thermo 20
run 200
```

**spherical-save-cue 导出说明**：默认导出为纯 PyTorch 实现（`force_naive`），`core.pt` 不依赖 cuEquivariance 自定义 ops，可在任意 LibTorch 环境运行。

完整说明见 [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md)。

### 示例 5b: 使用 LAMMPS ML-IAP unified 接口

ML-IAP unified 是 LAMMPS 的机器学习势接口，相比 `fix external` 方式速度更快（约 1.7x），且支持 GPU 加速。使用前需将 checkpoint 导出为 ML-IAP 格式。

**模型限制**：仅以下五种模型支持 ML-IAP（因其支持 `precomputed_edge_vec`）：`e3nn_layers`、`e3nn_layers_channelwise`、`cue_layers_channelwise`（spherical-save-cue）、`pure_cartesian_ictd_layers`、`pure_cartesian_ictd_layers_full`。其他模型（如 pure-cartesian、pure-cartesian-sparse）暂不支持。

**步骤 1：导出模型**

```bash
python -m molecular_force_field.cli.export_mliap your_checkpoint.pth \
    --elements H O \
    --atomic-energy-keys 1 8 \
    --atomic-energy-values -13.6 -75.0 \
    --max-radius 5.0 \
    --output model-mliap.pt
```

**步骤 2：在 Python 中驱动 LAMMPS**

```python
import torch
import lammps
from lammps.mliap import activate_mliappy, load_unified

# 启动 LAMMPS 并激活 ML-IAP Python 模块
lmp = lammps.lammps()
activate_mliappy(lmp)

# 加载导出的模型
model = torch.load("model-mliap.pt", weights_only=False)
load_unified(model)

# 设置 LAMMPS 输入
lmp.commands_string("""
units metal
atom_style atomic
read_data your_system.data
pair_style mliap unified model-mliap.pt 0
pair_coeff * * H O
velocity all create 300 12345
fix 1 all nve
thermo 100
run 1000
""")
lmp.close()
```

**步骤 3：纯 LAMMPS 输入文件方式**

创建 `run.py`：

```python
import torch
import lammps
from lammps.mliap import activate_mliappy, load_unified

lmp = lammps.lammps()
activate_mliappy(lmp)
model = torch.load("model-mliap.pt", weights_only=False)
load_unified(model)
lmp.file("input.lammps")
lmp.close()
```

`input.lammps` 内容示例：

```
units metal
atom_style atomic
read_data system.data

pair_style mliap unified model-mliap.pt 0
pair_coeff * * H O

velocity all create 300 12345
fix 1 all nvt temp 300 300 0.1
thermo 100
dump 1 all xyz 100 traj.xyz
run 10000
```

运行：

```bash
export DYLD_LIBRARY_PATH="$HOME/.local/lib:/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib"
python run.py
```

**注意事项：**

- `pair_coeff * * H O` 中元素顺序必须与导出时 `--elements H O` 一致
- `pair_style mliap unified model.pt 0` 末尾的 `0` 表示不包含 ghost 邻居；若模型有多层消息传递，可改为 `1`
- 使用 `units metal`（eV, Angstrom），与模型内部单位一致
- macOS 上需设置 `DYLD_LIBRARY_PATH` 指向 LAMMPS 和 Python 共享库
- LAMMPS 需编译时开启 `PKG_ML-IAP=ON`、`MLIAP_ENABLE_PYTHON=ON`，详见 `molecular_force_field/docs/INSTALL_LAMMPS_PYTHON.md`

**Kokkos GPU 加速**：若 LAMMPS 已用 Kokkos+CUDA 编译，可直接运行 `lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in input.lammps`；Python 驱动时改用 `activate_mliappy_kokkos(lmp)`。详见 `LAMMPS_INTERFACE.md`。

### 示例 5c: 大体系多 GPU 推理（inference_ddp）

对于超大规模体系（如 10 万原子以上），可使用 DDP 推理进行多 GPU 并行计算。**仅支持 `pure-cartesian-ictd` 模式**。当前版本使用随机图进行测试；实际结构需在代码中接入。

```bash
torchrun --nproc_per_node=2 -m molecular_force_field.cli.inference_ddp \
  --checkpoint model.pth \
  --atoms 100000 \
  --forces
```

**参数说明**：
- `--atoms`: 原子数（用于生成随机测试图，默认 50000）
- `--checkpoint`: 模型检查点路径
- `--forces`: 同时计算并输出力
- `--partition`: 图分区策略，`modulo`（默认）或 `spatial`

### 示例 6: 声子谱计算（Hessian 和频率）

```python
from ase.io import read
from ase.optimize import BFGS
from ase.data import atomic_masses, atomic_numbers
from ase.neighborlist import neighbor_list
from molecular_force_field.evaluation.calculator import MyE3NNCalculator
from molecular_force_field.evaluation.evaluator import Evaluator
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.config import ModelConfig
import torch
import numpy as np

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('combined_model.pth', map_location=device)

config = ModelConfig()
config.load_atomic_energies_from_file('fitted_E0.csv')

e3trans = E3_TransformerLayer_multi(...).to(device)
e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
e3trans.eval()

# 创建计算器和评估器
ref_energies_dict = {
    k.item(): v.item()
    for k, v in zip(config.atomic_energy_keys, config.atomic_energy_values)
}
calc = MyE3NNCalculator(e3trans, ref_energies_dict, device, max_radius=5.0)

class SimpleDataset:
    def restore_force(self, x):
        return x
    def restore_energy(self, x):
        return x

dataset = SimpleDataset()
evaluator = Evaluator(
    model=e3trans,
    dataset=dataset,
    device=device,
    atomic_energy_keys=config.atomic_energy_keys,
    atomic_energy_values=config.atomic_energy_values,
)

# 读取结构
atoms = read('structure.xyz')
atoms.calc = calc

# 可选：优化结构
print("Relaxing structure...")
optimizer = BFGS(atoms, logfile=None)
optimizer.run(fmax=0.01)
print(f"Relaxation completed. Final forces: max={atoms.get_forces().max():.6f} eV/Å")

# 准备数据
pos = torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype(), device=device)
A = torch.tensor([atomic_numbers[symbol] for symbol in atoms.get_chemical_symbols()], 
                 dtype=torch.long, device=device)
batch_idx = torch.zeros(len(atoms), dtype=torch.long, device=device)

# 构建图（使用 ASE on-the-fly 邻居列表）
cell_array = atoms.cell.array
pbc_flags = atoms.pbc
if cell_array is None or np.abs(cell_array).sum() <= 1e-9:
    cell_array = np.eye(3) * 100.0
    pbc_flags = [False, False, False]

atoms_nl = atoms.copy()
atoms_nl.cell = cell_array
atoms_nl.pbc = pbc_flags

edge_src_np, edge_dst_np, edge_shifts_np = neighbor_list('ijS', atoms_nl, max_radius=5.0)
edge_src = torch.tensor(edge_src_np, dtype=torch.long, device=device)
edge_dst = torch.tensor(edge_dst_np, dtype=torch.long, device=device)
edge_shifts = torch.tensor(edge_shifts_np, dtype=torch.float64, device=device)
cell = torch.tensor(cell_array, dtype=torch.get_default_dtype(), device=device).unsqueeze(0)

# 计算 Hessian 矩阵
print("Computing Hessian matrix...")
hessian = evaluator.compute_hessian(
    pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell
)

print(f"Hessian shape: {hessian.shape}")
print(f"Hessian range: [{hessian.min():.6f}, {hessian.max():.6f}] eV/Å²")

# 检查对称性
is_symmetric = torch.allclose(hessian, hessian.T, atol=1e-4)
print(f"Hessian symmetric: {is_symmetric}")

# 获取原子质量
masses = torch.tensor([atomic_masses[atomic_numbers[symbol]] 
                      for symbol in atoms.get_chemical_symbols()],
                     dtype=torch.get_default_dtype())

# 计算声子谱
print("Computing phonon spectrum...")
frequencies = evaluator.compute_phonon_spectrum(
    hessian, masses, output_prefix='phonon'
)

print(f"\nPhonon calculation completed!")
print(f"Total modes: {len(frequencies)}")
print(f"Real modes: {np.sum(frequencies >= 0)}")
print(f"Imaginary modes: {np.sum(frequencies < 0)}")
print(f"Frequency range: [{frequencies.min():.2f}, {frequencies.max():.2f}] cm⁻¹")

# 分析结果
if np.any(frequencies < 0):
    print(f"\n⚠️  Warning: {np.sum(frequencies < 0)} imaginary frequencies detected!")
    print("   This indicates the structure may not be at a local minimum.")
    print("   Consider further optimization or checking the structure.")
else:
    print("\n✅ All frequencies are real (structure is stable).")
```

## 常见问题

### Q: 如何查看所有可用的命令行参数？

```bash
mff-preprocess --help
mff-train --help
mff-evaluate --help
mff-export-core --help   # LAMMPS LibTorch 导出
mff-lammps --help        # LAMMPS fix external 接口
python -m molecular_force_field.cli.export_mliap --help  # LAMMPS ML-IAP 导出
```

### Q: 如何使用 LAMMPS LibTorch 接口？

LAMMPS LibTorch 接口（USER-MFFTORCH）通过 `pair_style mff/torch` 在 C++ 侧用 LibTorch 加载 TorchScript 模型，**运行时无需 Python**，适合 HPC 与生产部署。

**快速步骤**：
1. 导出：`mff-export-core --checkpoint model.pth --elements H O --max-radius 5.0 --embed-e0 --e0-csv fitted_E0.csv --out core.pt`
2. 编译 LAMMPS：启用 `PKG_KOKKOS`、`PKG_USER-MFFTORCH`，见 [lammps_user_mfftorch/docs/BUILD_AND_RUN.md](lammps_user_mfftorch/docs/BUILD_AND_RUN.md)
3. 运行：`lmp -k on g 1 -sf kk -pk kokkos newton off neigh full -in in.mfftorch`

**支持模型**：`pure-cartesian-ictd` 系列、`spherical-save-cue`。完整说明见 [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md)。

### Q: 如何导出 ML-IAP 格式？

ML-IAP 用于 LAMMPS 的 `pair_style mliap unified`，比 fix external 更快且支持 Kokkos GPU：

```bash
python -m molecular_force_field.cli.export_mliap checkpoint.pth \
  --elements H O --atomic-energy-keys 1 8 --atomic-energy-values -13.6 -75.0 \
  --max-radius 5.0 --output model-mliap.pt
```

支持模型：`spherical`、`spherical-save`、`spherical-save-cue`、`pure-cartesian-ictd`、`pure-cartesian-ictd-save`。详见 [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md)。

### Q: 训练时如何恢复之前的检查点？

训练器会**自动检测并加载**检查点文件。如果 `--checkpoint` 指定的文件已存在，会自动恢复训练状态。

**自动恢复的内容：**
- ✅ 模型权重（`e3trans_state_dict`）
- ✅ Epoch 数（从中断的 epoch 继续）
- ✅ Batch count（累计的 batch 数）
- ✅ 损失权重 `a` 和 `b`
- ✅ 最佳验证损失（用于早停判断，基于 `a × 能量损失 + b × 受力损失 + c × 应力损失`）
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

损失函数为：`总损失 = a × 能量损失 + b × 受力损失 + c × 应力损失`（当 `-c > 0` 时包含应力项）

**初始权重设置：**
```bash
# 默认设置（能量权重1.0，受力权重10.0，应力权重0即禁用）
mff-train -a 1.0 -b 10.0

# 更重视能量（如果能量预测较差）
mff-train -a 2.0 -b 5.0

# 更重视受力（如果受力预测较差）
mff-train -a 0.5 -b 20.0

# 启用应力训练（需要 XYZ 含 stress/virial，且为 PBC 体系）
mff-train -a 1.0 -b 10.0 -c 0.1
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

# 中速调整（更稳定，适合大多数情况）
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.01 --weight-b-decay 0.99

# 快速调整（适合短期训练，<50k batches）
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.02 --weight-b-decay 0.98

# 超快速调整（默认）
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.05 --weight-b-decay 0.98
```

自动调整规则：每`--update-param`个batch，`a`会乘以`--weight-a-growth`（增加），`b`会乘以`--weight-b-decay`（减少），以逐渐平衡能量和受力的重要性。**注意：`a` 和 `b` 默认会被限制在 [1, 1000]，以防训练过长导致权重漂移过大。** 应力权重 `c` 可通过 `--c-min` 和 `--c-max` 限制范围。

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
- **中速 (1.01/0.99)**: 适合大多数训练场景（50k-200k batches），平衡稳定性和调整速度
- **快速 (1.02/0.98)**: 适合短期训练（<50k batches），快速调整权重，但可能导致训练后期不稳定
- **默认 (1.05/0.98)**: 调整更激进，收敛更快但更容易不稳定；如果出现震荡，建议改用中速或慢速

### Q: 如何进行应力训练？

应力训练用于周期性体系（PBC），通过晶胞应变导数计算应力 σ = (1/V) × dE/dε，与参考应力做 MSE 作为 stress_loss。

**前提条件：**
1. XYZ 文件为周期性体系（含 `pbc="T T T"` 和 `Lattice="..."`）
2. XYZ comment 行含 stress 或 virial，例如：`stress="0.01 0 0 0 0.01 0 0 0 0.01"`（9 个 3×3 分量，eV/Å³）

**启用应力训练：**
```bash
mff-train -a 1.0 -b 10.0 -c 0.1 --input-file pbc_with_stress.xyz
```

**权重建议：** 应力分量少（每结构 9 个），通常 `-c` 取 0.01～0.5。若应力预测偏差大，可适当增大 `-c`。

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
- **应力/维里张量**（可选，用于周期性体系的应力训练）：在 comment 行中支持以下格式：
  - `stress="xx yy zz yz xz xy"` — 6 个 Voigt 分量 (eV/Å³)
  - `stress="xx xy xz yx yy yz zx zy zz"` — 9 个 3×3 分量，行优先 (eV/Å³)
  - `virial="xx xy xz yx yy yz zx zy zz"` — 9 个维里分量 (eV)，会自动转换为 stress = -virial/V

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

**1. 使用优化的张量积模式：**
```bash
# Pure-Cartesian-ICTD: 参数量最少（减少 72.1%），速度最快（CPU: 最高 4.12x，GPU: 最高 2.10x）
mff-train --input-file data.xyz --tensor-product-mode pure-cartesian-ictd

# Pure-Cartesian-Sparse: 参数量减少 29.6%，速度稳定（CPU: 0.53x-1.39x，GPU: 0.46x-1.17x）
mff-train --input-file data.xyz --tensor-product-mode pure-cartesian-sparse

# Partial-Cartesian-Loose: 非严格等变（norm product 近似），速度较快（CPU: 0.17x-1.37x，GPU: 0.21x-1.52x），参数量减少 17.3%
mff-train --input-file data.xyz --tensor-product-mode partial-cartesian-loose

# Partial-Cartesian: 严格等变，参数量减少 17.4%
mff-train --input-file data.xyz --tensor-product-mode partial-cartesian
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

**9. 验证阶段使用 torch.compile（可选）：**
```bash
mff-train --compile-val e3trans --data-dir data
```

### Q: 应该选择哪种张量积模式？

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| 发表论文/高精度需求 | `spherical` | 使用 e3nn 严格球谐，精度最高，标准实现 |
| 参数量优化（最多） | `pure-cartesian-ictd` | 参数量减少 72.1%，速度最快（CPU: 最高 4.12x，GPU: 最高 2.10x），严格等变 |
| 参数量优化（平衡） | `pure-cartesian-sparse` | 参数量减少 29.6%，速度稳定（CPU: 0.53x-1.39x，GPU: 0.46x-1.17x），严格等变 |
| GPU 内存受限 | `pure-cartesian-ictd` 或 `pure-cartesian-sparse` | 参数量减少 72.1% 或 29.6%，显存占用更低 |
| 严格等变性要求 | `spherical`、`partial-cartesian`、`pure-cartesian-sparse` 或 `pure-cartesian-ictd` | 这些模式都严格等变 |
| 快速实验迭代 | `pure-cartesian-ictd` | 速度最快（CPU: 最高 4.12x，GPU: 最高 2.10x），严格等变 |
| 首次尝试/对比基准 | `spherical` | 验证模型正确性，性能基准 |
| 大规模模型部署 | `pure-cartesian-ictd` | 参数量最少（减少 72.1%），速度最快（CPU: 最高 4.12x，GPU: 最高 2.10x） |

**切换模式：**
```bash
# 球谐模式（默认）
mff-train --input-file data.xyz

# 其他模式
mff-train --input-file data.xyz --tensor-product-mode partial-cartesian
mff-train --input-file data.xyz --tensor-product-mode partial-cartesian-loose
mff-train --input-file data.xyz --tensor-product-mode pure-cartesian-sparse
mff-train --input-file data.xyz --tensor-product-mode pure-cartesian-ictd
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
- 验证阶段每个batch的能量预测（由`--log-val-batch-energy`控制，默认False，只记录到日志文件）

**日志文件：**
- `training_YYYYMMDD_HHMMSS.log` - 包含所有详细信息
  - Batch级别的训练指标（每N个batch，由`--energy-log-frequency`控制；启用应力训练时还会输出 Stress Loss）
  - 验证时的详细结果（包括每个batch的能量预测，无论`--log-val-batch-energy`设置如何；启用应力训练时包含 Val Stress Loss/RMSE/MAE）
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

3. **数据格式**: 确保XYZ文件格式正确，特别是能量、力和应力的单位。能量以 eV 为单位，力以 eV/Å 为单位，应力以 eV/Å³ 为单位。

4. **检查点**: 训练过程中会定期保存检查点（频率由`--dump-frequency`控制），建议定期备份。最终模型会保存在`--checkpoint`指定的路径。

5. **日志**: 训练日志保存在 `training_*.log` 文件中，包含详细的训练信息。控制台只显示验证结果和重要信息，详细日志请查看日志文件。验证阶段每个batch的能量预测默认只记录到日志文件，如需在控制台显示，请使用 `--log-val-batch-energy` 参数。

6. **早停**: 如果**加权验证损失**（`a × 能量损失 + b × 受力损失 + c × 应力损失`）连续`--patience`个epoch没有改善，训练会自动停止。这可以防止过拟合并节省时间。早停依据与训练时的总损失保持一致。

7. **权重自动调整**: `a`和`b`会在训练过程中自动调整（每`--update-param`个batch）。`a`会乘以`--weight-a-growth`（默认1.05，即增长5%），`b`会乘以`--weight-b-decay`（默认0.98，即减少2%），以平衡能量和受力的重要性。若启用应力训练（`-c > 0`），`c`参与总损失计算。如果训练不稳定，可以：
   - 增大`--update-param`（减少调整频率）
   - 使用更慢的调整速率（`--weight-a-growth 1.005 --weight-b-decay 0.995`）
   - 固定权重（设置`--update-param`为一个很大的值，如1000000）

8. **学习率策略**: 学习率会经历三个阶段：
   - **预热阶段**（前`--warmup-batches`个batch）：从`learning_rate × warmup_start_ratio`线性增长到`learning_rate`
   - **稳定阶段**：保持`learning_rate`
   - **衰减阶段**：如果验证指标不改善，每`--lr-decay-patience`个batch后乘以`--lr-decay-factor`

## 张量积模式对比

**FusedSCEquiTensorPot 支持八种等变张量积实现模式**，每种模式在速度、参数量和等变性方面有不同的特点：

1. **`spherical`**: 基于 e3nn 的球谐函数方法（默认，标准实现）
2. **`spherical-save`**: channelwise edge conv（e3nn 后端，参数量更少）
3. **`spherical-save-cue`**: channelwise edge conv（cuEquivariance 后端，需可选依赖，GPU 加速）
4. **`partial-cartesian`**: 笛卡尔坐标 + e3nn CG 系数（严格等变）
5. **`partial-cartesian-loose`**: 近似等变（norm product 近似）
6. **`pure-cartesian`**: 纯笛卡尔 \(3^L\) 表示（严格等变，速度极慢，不推荐）
7. **`pure-cartesian-sparse`**: 稀疏纯笛卡尔（严格等变，参数量优化）
8. **`pure-cartesian-ictd`**: ICTD irreps 内部表示（严格等变，速度最快，参数量最少）

所有模式都保持 O(3) 等变性（包括旋转和反射）。以下是对比数据：

### 模式对比总览

| 特性 | Spherical | Partial-Cartesian | Partial-Cartesian-Loose | Pure-Cartesian | Pure-Cartesian-Sparse | Pure-Cartesian-ICTD |
|------|-----------|-------------------|------------------------|----------------|----------------------|---------------------|
| 实现 | e3nn 球谐函数 | 笛卡尔坐标 + e3nn CG 系数 | 非严格等变（norm product 近似，使用 e3nn Irreps） | 纯笛卡尔（3^L，δ/ε，完全自实现） | 稀疏纯笛卡尔（δ/ε，完全自实现） | ICTD irreps 内部表示（完全自实现） |
| 等变性 | ✅ 严格等变 | ✅ 严格等变 | ⚠️ 近似等变 | ✅ 严格等变 | ✅ 严格等变 | ✅ 严格等变 |
| 速度 (CPU, lmax=2) | 1.00x (基准) | 1.06x | 1.33x | 0.06x (极慢) | 1.39x | **4.12x (最快)** |
| 速度 (GPU, lmax=2)* | 1.00x (基准) | 0.75x | 1.15x | 0.06x (极慢) | 1.17x | **2.10x (最快)** |
| 参数量 (lmax=2) | 6,540,634 (100%) | 5,404,938 (82.6%) | 5,406,026 (82.7%) | 33,626,186 (514.0%) | 4,606,026 (70.4%) | 1,824,497 (27.9%) |
| 参数量变化 | - | **减少 17.4%** | **减少 17.3%** | +414% | **减少 29.6%** | **减少 72.1%** |
| 等变性误差 (O(3), lmax=2) | ~1e-15 | ~1e-14 | ~1e-15 | ~1e-14 | ~1e-15 | ~1e-7 |
| 推荐场景 | 默认，最高精度 | 严格等变，平衡性能 | 快速迭代（CPU），非严格等变可接受 | 不推荐（速度慢） | 参数量优化，严格等变 | **参数量最少，GPU 最快，严格等变** |

*GPU 速度：总训练时间（前向+反向）加速比，相对于 spherical。测试环境：RTX 3090, float64, N=32, E=256。

### 使用方法

**训练时选择模式：**
```bash
# 球谐模式（默认，精度最高）
mff-train --input-file data.xyz --data-dir output

# 笛卡尔模式（速度快，参数少）
mff-train --input-file data.xyz --data-dir output --tensor-product-mode partial-cartesian
```

**评估时必须使用相同模式：**
```bash
# 如果训练时用的是 cartesian
mff-evaluate --checkpoint model.pth --tensor-product-mode partial-cartesian
```

### 详细性能对比

#### CPU 测试结果（channels=64, lmax=0 到 6, 32 atoms, 256 edges, float64）

**总训练时间加速比（前向+反向，相对于 spherical）：**

| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             1.06x |                  1.13x |           0.36x |                 1.07x |                **2.97x** |
| 1    |             1.05x |                  1.37x |           0.13x |                 1.02x |                **3.33x** |
| 2    |             1.06x |                  1.33x |           0.06x |                 1.39x |                **4.12x** |
| 3    |             0.58x |                  0.70x |           0.02x |                 1.05x |                **2.68x** |
| 4    |             0.37x |                  0.43x |        **FAILED** |                 0.97x |                **2.20x** |
| 5    |             0.23x |                  0.28x |        **FAILED** |                 0.78x |                **1.81x** |
| 6    |             0.16x |                  0.17x |        **FAILED** |                 0.53x |                **1.58x** |

**参数量对比（lmax=0 到 6）：**

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**等变误差对比（O(3)，包含宇称）：**

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 3.33e-15  |        8.24e-16  |            6.39e-14  |      3.41e-15  |           4.34e-15  |           1.69e-12 |
| 2    | 3.88e-15  |        1.87e-14  |            7.73e-16  |      3.08e-15  |           3.56e-14  |           7.73e-08 |
| 4    | 1.27e-15  |        6.83e-15  |            1.00e-15  |       **FAILED** |           1.24e-14  |           3.50e-05 |
| 6    | 3.26e-15  |        2.01e-15  |            5.82e-16  |       **FAILED** |           1.51e-15  |           1.00e-06 |

#### GPU 测试结果（channels=64, lmax=0 到 6, RTX 3090, float64, N=32, E=256）

**总训练时间加速比（前向+反向，相对于 spherical）：**

| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             0.96x |                  1.52x |           0.54x |                 1.17x |                **1.92x** |
| 1    |             0.85x |                  1.33x |           0.16x |                 1.02x |                **1.97x** |
| 2    |             0.75x |                  1.15x |           0.06x |                 1.17x |                **2.10x** |
| 3    |             0.56x |                  0.81x |           0.02x |                 1.15x |                **1.91x** |
| 4    |             0.38x |                  0.51x |        **FAILED** |                 0.99x |                **1.78x** |
| 5    |             0.26x |                  0.32x |        **FAILED** |                 0.75x |                **1.44x** |
| 6    |             0.17x |                  0.21x |        **FAILED** |                 0.46x |                **1.05x** |

**参数量对比（lmax=0 到 6）：**

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**等变误差对比（O(3)，包含宇称）：**

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 1.03e-15  |        6.37e-16  |            1.20e-15  |      3.27e-15  |           9.31e-16  |           5.24e-08 |
| 2    | 9.27e-16  |        1.97e-16  |            8.96e-16  |      3.18e-14  |           1.30e-15  |           7.18e-07 |
| 4    | 9.16e-16  |        7.76e-14  |            4.26e-16  |       **FAILED** |           6.19e-16  |           7.16e-07 |
| 6    | 4.77e-16  |        7.02e-16  |            5.65e-16  |       **FAILED** |           5.72e-16  |           1.11e-07 |

**性能分析：**
- ✅ **所有模式均通过 O(3) 等变性测试**（包括宇称/反射），等变性误差 < 1e-6
- 🚀 **CPU 上 `pure-cartesian-ictd` 在所有 lmax 下都是最快的**（最高 **4.12x 加速** at lmax=2）
- 🚀 **GPU 上 `pure-cartesian-ictd` 在所有 lmax 下都是最快的**（最高 **2.10x 加速** at lmax=2）
- 🚀 **CPU/GPU 上 `pure-cartesian-sparse` 表现稳定**（0.53x - 1.39x，接近基准）
- 💾 **`pure-cartesian-ictd` 参数量始终最少**（27-32% of spherical）
- 💾 **`pure-cartesian-sparse` 参数量适中**（70-88% of spherical）
- ⚠️ **`pure-cartesian` 极慢且 lmax≥4 时失败**（不推荐）
- 📊 **CPU 测试环境**：channels=64, lmax=0-6, 32 atoms, 256 edges, float64
- 📊 **GPU 测试环境**：channels=64, lmax=0-6, RTX 3090, float64, 32 atoms, 256 edges

### 推荐使用场景

**使用 Spherical（默认）：**
- ✅ 需要最高精度和兼容性
- ✅ 研究/发表论文场景（标准 e3nn 实现）
- ✅ 首次尝试/对比基准
- ✅ 小规模数据集

**使用 Partial-Cartesian：**
- ✅ 需要严格等变性但参数量更少（减少 17.4%）
- ✅ GPU 内存受限场景
- ✅ 需要平衡性能和等变性

**使用 Partial-Cartesian-Loose：**
- ✅ 快速实验迭代（速度较快，CPU: 0.17x-1.37x，GPU: 0.21x-1.52x）
- ⚠️ 对严格等变性要求不高的场景（使用 norm product 近似，非严格等变）
- ⚠️ 注意：虽然等变性误差 < 1e-6，但理论上非严格等变

**使用 Pure-Cartesian-Sparse（推荐用于 CPU/GPU 训练）：**
- ✅ **CPU 上表现稳定**（0.53x - 1.39x，接近基准）
- ✅ **GPU 上速度较快**（**1.17x 加速** at lmax=2）
- ✅ 参数量适中（减少 29.6%，70-88% of spherical）
- ✅ 严格等变（误差 ~1e-15，最高精度）
- ✅ 平衡参数量和性能的最佳选择

**使用 Pure-Cartesian-ICTD（推荐用于 CPU/GPU 训练）：**
- ✅ **CPU 上速度最快**（最高 **4.12x 加速** at lmax=2，所有 lmax 下 1.58x - 4.12x）
- ✅ **GPU 上速度最快**（最高 **2.10x 加速** at lmax=2，所有 lmax 下 1.05x - 2.10x）
- ✅ 参数量最少（减少 72.1%，27-32% of spherical）
- ✅ 严格等变（误差 ~1e-7 到 1e-6，可接受）
- ✅ 内存极度受限场景
- ✅ 大规模模型部署
- 📊 **CPU 最佳性能**：lmax ≤ 3 时优势最明显（2.68x - 4.12x 加速）
- 📊 **GPU 最佳性能**：lmax ≤ 3 时优势最明显（1.91x - 2.10x 加速）

### 实际任务测试结果

**数据集**：五条氮氧化物和碳结构反应路径的 NEB（Nudged Elastic Band）数据，截取到 fmax=0.2，总共 2,788 条数据。测试集：每个反应选取 1-2 条完整或不完整的数据。

**测试配置**：64 channels, lmax=2, float64

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

**结果分析**：
- **能量精度对比**：FSCETP 相比 MACE (64ch) 的能量 RMSE 降低了 66.2%（0.044 vs 0.13 mev/atom）
- **力精度对比**：FSCETP 相比 MACE (64ch) 的力 RMSE 降低了 36.2%（7.4 vs 11.6 mev/Å）
- **最佳性能模式**：`spherical` 和 `partial-cartesian` 模式达到最优精度（能量：0.044-0.045 mev/atom，力：7.4 mev/Å）
- **精度与效率平衡**：`pure-cartesian-ictd` 在保持接近最优精度（能量：0.046 mev/atom，力：9.0 mev/Å）的同时，参数量减少 72.1%，训练速度提升 2.10x（GPU，lmax=2）

**不推荐使用 Pure-Cartesian：**
- ❌ 速度极慢（CPU: 0.02x-0.36x，GPU: 0.02x-0.54x），参数量最大（+414%）
- ❌ lmax≥4 时失败（内存不足）
- ❌ 仅用于研究目的，不推荐实际使用

### 完整示例：笛卡尔模式 + 多卡 + SWA + EMA

```bash
torchrun --nproc_per_node=4 -m molecular_force_field.cli.train \
    --input-file large_dataset.xyz \
    --data-dir data \
    --tensor-product-mode partial-cartesian \
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