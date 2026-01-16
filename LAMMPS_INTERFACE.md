# LAMMPS 接口使用指南

本文档介绍如何将 **FusedEquiTensorPot** 模型集成到 LAMMPS 中进行分子动力学模拟。

## 概述

FusedEquiTensorPot 提供了 Python 接口，可以通过 LAMMPS 的 `fix python/invoke` 或 `pair_style python` 命令调用。接口会自动处理：
- 模型加载和初始化
- 邻居列表计算（支持周期性边界条件）
- 能量和力的计算
- 单位转换（模型使用 eV，LAMMPS 使用 kcal/mol）

## 快速开始

### 1. 生成 LAMMPS 接口文件

使用 `mff-lammps` 命令生成所需的文件：

```bash
mff-lammps path/to/your/model.pth \
    --output-dir ./lammps_setup \
    --max-radius 5.0 \
    --atomic-energy-file fitted_E0.csv \
    --device cuda
```

这会生成：
- `lammps_potential.py`: Python 初始化脚本
- `lammps_input.in`: 示例 LAMMPS 输入脚本

### 2. 在 LAMMPS 中使用

#### 方法 A: 使用 `fix python/invoke`（推荐）

```lammps
# 初始化 Python 接口
python lammps_potential.py input

# 设置 Python potential
fix 1 all python/invoke 1 1 1 lammps_potential.py lammps_potential_init lammps_potential \
    "path/to/model.pth" device cuda max_radius 5.0 atomic_energy_file "fitted_E0.csv"

# 设置 pair style
pair_style python 1
pair_coeff * *

# 设置系统（示例）
read_data your_system.data
# 或
create_atoms 1 random 100 12345 NULL

# 运行模拟
run 1000
```

## 单卡运行建议（你当前选择的方式）

- **建议先用单进程/单卡**跑通接口与数值验证：`mpirun -np 1 lmp ...` 或直接 `lmp ...`
- 单卡下不会碰到复杂的 MPI 域分解/ghost 交互问题，定位更快。

## triclinic 盒子支持（重要）

你的模型/邻居构图已经支持 triclinic，但 **LAMMPS 侧需要把 tilt 因子传给 Python**，否则你只能得到正交 cell（会算错）。

本接口支持在 `lammps_potential(...)` 额外传入：
- `xy`, `xz`, `yz`（LAMMPS triclinic tilt factors）

示例（在 LAMMPS 输入脚本里）：

```lammps
variable XY equal xy
variable XZ equal xz
variable YZ equal yz

python lammps_potential.py input

# 注意：下面这一行的 python/invoke 传参格式，取决于你使用的 LAMMPS 版本。
# 关键点是：把 ${XY} ${XZ} ${YZ} 作为额外参数传进 Python 的 lammps_potential(xy=..., xz=..., yz=...)
fix 1 all python/invoke 1 1 1 lammps_potential.py lammps_potential_init lammps_potential \
    "path/to/model.pth" device cuda max_radius 5.0 \
    xy ${XY} xz ${XZ} yz ${YZ}
```

接口内部会按 LAMMPS 常用约定构造 3×3 cell（行向量）：
\[
a=(L_x,0,0),\quad b=(xy,L_y,0),\quad c=(xz,yz,L_z)
\]

如果你用的是不同的 triclinic 约定（比如直接给 h 矩阵），告诉我你 LAMMPS 的版本和调用方式，我可以把“取 box/h 矩阵”的部分改成完全自动。

#### 方法 B: 使用 `pair_style python`（简化版）

```lammps
# 初始化（在 Python 脚本中完成）
python lammps_potential.py input

# 直接使用 pair_style
pair_style python 1 lammps_potential.py lammps_potential_init lammps_potential
pair_coeff * *
```

### 3. 完整示例

#### 示例 1: 简单 MD 模拟

```lammps
# LAMMPS input script for FusedEquiTensorPot

# 初始化 Python 接口
python lammps_potential.py input

# 设置 Python potential
fix 1 all python/invoke 1 1 1 lammps_potential.py lammps_potential_init lammps_potential \
    "combined_model.pth" device cuda max_radius 5.0

# 设置 pair style
pair_style python 1
pair_coeff * *

# 读取系统
read_data system.data

# 设置初始速度
velocity all create 300.0 12345

# 设置积分器
fix 1 all nve

# 输出设置
thermo_style custom step temp pe ke etotal
thermo 100
dump 1 all custom 100 traj.xyz id type x y z fx fy fz

# 运行
run 10000
```

#### 示例 2: NPT 系综（恒温恒压）

```lammps
# 初始化（同上）
python lammps_potential.py input
fix 1 all python/invoke 1 1 1 lammps_potential.py lammps_potential_init lammps_potential \
    "combined_model.pth" device cuda max_radius 5.0
pair_style python 1
pair_coeff * *

# 读取系统
read_data system.data

# NPT 系综
fix 1 all npt temp 300.0 300.0 0.1 iso 1.0 1.0 1.0

# 运行
run 10000
```

## 参数说明

### `lammps_potential_init` 参数

- `checkpoint_path` (必需): 模型检查点文件路径（.pth）
- `device` (可选): 计算设备，`'cuda'` 或 `'cpu'`（默认: `'cuda'`）
- `max_radius` (可选): 邻居搜索的最大半径，单位 Angstrom（默认: 5.0）
- `atomic_energy_file` (可选): 原子基准能量 CSV 文件路径（默认: `fitted_E0.csv`）
- `atomic_energy_keys` (可选): 自定义原子序数列表
- `atomic_energy_values` (可选): 对应的原子能量列表（eV）

### 单位转换

- **能量**: 模型使用 eV，LAMMPS 使用 kcal/mol
  - 转换系数: 1 eV = 23.06035 kcal/mol
- **力**: 模型使用 eV/Å，LAMMPS 使用 kcal/(mol·Å)
  - 转换系数: 1 eV/Å = 23.06035 kcal/(mol·Å)
- **距离**: 两者都使用 Angstrom，无需转换

## 性能优化建议

1. **使用 GPU**: 如果可用，设置 `device cuda` 可以显著加速计算
2. **邻居列表**: `max_radius` 应该与训练时使用的值一致
3. **批量处理**: LAMMPS 会自动处理多个时间步，模型在 `eval()` 模式下运行
4. **内存管理**: 对于大系统，考虑使用 CPU 模式或减少 `max_num_neighbors`

## 故障排除

### 问题 1: Python 模块导入错误

**错误信息**:
```
ImportError: No module named 'molecular_force_field'
```

**解决方案**:
- 确保已安装 `molecular_force_field` 包：`pip install -e .`
- 或者在 `lammps_potential.py` 中修改 `sys.path` 指向正确的包路径

### 问题 2: CUDA 设备不可用

**错误信息**:
```
RuntimeError: CUDA error: no kernel image is available
```

**解决方案**:
- 使用 CPU 模式：`device cpu`
- 或检查 CUDA 版本兼容性

### 问题 3: 邻居列表计算失败

**错误信息**:
```
RuntimeError: radius_graph failed
```

**解决方案**:
- 检查 `max_radius` 是否合理（通常 3-10 Å）
- 确保系统尺寸足够大（非周期性系统需要足够大的盒子）

### 问题 4: 原子类型不匹配

**错误信息**:
```
KeyError: atomic number not found in energy mapping
```

**解决方案**:
- 确保 `atomic_energy_file` 包含所有出现的原子类型
- 或使用 `atomic_energy_keys` 和 `atomic_energy_values` 手动指定

## 测试接口

你可以直接测试 Python 接口：

```bash
python -m molecular_force_field.interfaces.lammps_potential \
    path/to/model.pth \
    --device cuda \
    --max-radius 5.0 \
    --atomic-energy-file fitted_E0.csv
```

这会使用随机数据测试接口，输出能量和力的示例值。

## 高级用法

### 自定义原子能量

如果不想使用 `fitted_E0.csv`，可以直接指定：

```lammps
fix 1 all python/invoke 1 1 1 lammps_potential.py lammps_potential_init lammps_potential \
    "model.pth" device cuda max_radius 5.0 \
    atomic_energy_keys "[1, 6, 7, 8]" \
    atomic_energy_values "[-430.53, -821.03, -1488.19, -2044.35]"
```

### 非周期性系统

对于非周期性系统，接口会自动使用大盒子（100×100×100 Å）来避免计算错误。你仍然需要在 LAMMPS 中正确设置盒子尺寸。

## 注意事项

1. **模型超参数**: 确保 LAMMPS 中使用的 `max_radius` 与训练时一致
2. **原子类型**: LAMMPS 的原子类型（type）应该对应模型的原子序数（Z）
   - 更推荐：显式提供 `type_to_Z`（例如 type 1->Si=14），避免静默错误。
3. **周期性边界条件**: 接口会自动检测 PBC，但确保 LAMMPS 中的 PBC 设置正确
4. **性能**: Python 接口比原生 C++ 插件慢，但对于大多数应用仍然足够快

## 参考

- [LAMMPS Python 接口文档](https://docs.lammps.org/Python.html)
- [FusedEquiTensorPot 主文档](README.md)
- [使用指南](USAGE.md)
