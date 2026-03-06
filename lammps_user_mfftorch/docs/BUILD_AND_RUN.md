# LAMMPS + Kokkos + LibTorch 完整编译与运行指南

本文档描述如何从零编译带 **Kokkos GPU** 与 **USER-MFFTORCH**（LibTorch 势函数）的 LAMMPS，并完成端到端 MD 运行。

## 目录

1. [前置条件](#1-前置条件)
2. [获取 LAMMPS 源码](#2-获取-lammps-源码)
3. [安装 USER-MFFTORCH](#3-安装-user-mfftorch-到-lammps-源码树)
4. [修改 CMakeLists.txt](#4-修改-lammps-cmakeliststxt一次性)
5. [CMake 配置](#5-cmake-配置)
6. [编译](#6-编译)
7. [导出 core.pt](#7-导出-corepttorchscript-模型)
8. [运行 LAMMPS](#8-运行-lammps)
9. [一键测试脚本](#9-一键测试脚本)
10. [故障排查](#10-故障排查)

---

## 1. 前置条件

### 1.1 系统与工具

| 依赖 | 要求 |
|------|------|
| 操作系统 | Linux（推荐，用于 CUDA） |
| CMake | ≥ 3.20 |
| C++ 编译器 | GCC 7+ 或 Clang，支持 C++17 |
| CUDA | 11+（与 PyTorch/LibTorch 版本匹配） |
| Python | 3.8+（用于导出 `core.pt`） |

### 1.2 Python 环境

- 安装 PyTorch（含 CUDA 支持，与 GPU 驱动版本匹配）：
  ```bash
  pip install torch  # 或根据 CUDA 版本选择 torch+cu118 等
  ```
- 安装本仓库：
  ```bash
  pip install -e .  # 在 rebuild 仓库根目录
  ```

### 1.3 LibTorch 路径

USER-MFFTORCH 通过 `find_package(Torch)` 查找 LibTorch。推荐使用 **Python 内置的 LibTorch**：

```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

输出示例：`/path/to/python/site-packages/torch/share/cmake/Torch`。将该路径传给 CMake 的 `CMAKE_PREFIX_PATH`。

---

## 2. 获取 LAMMPS 源码

```bash
# 方式 A：下载官方源码
wget https://download.lammps.org/tars/lammps-22Jul2025.tar.gz
tar xzf lammps-22Jul2025.tar.gz
cd lammps-22Jul2025

# 方式 B：Git 克隆
git clone https://github.com/lammps/lammps.git
cd lammps
git checkout 22Jul2025  # 或 develop 等
```

---

## 3. 安装 USER-MFFTORCH 到 LAMMPS 源码树

在 **rebuild 仓库根目录** 执行：

```bash
bash scripts/install_user_mfftorch_into_lammps.sh /path/to/lammps
```

例如：

```bash
bash scripts/install_user_mfftorch_into_lammps.sh /root/lammps-22Jul2025
```

脚本会：

- 拷贝 `src/USER-MFFTORCH/` → `LAMMPS/src/USER-MFFTORCH/`
- 拷贝 `USER-MFFTORCH.cmake` → `LAMMPS/cmake/Modules/Packages/`（或 `cmake/Packages/`，视版本而定）

---

## 4. 修改 LAMMPS CMakeLists.txt（一次性）

LAMMPS 默认不识别 `USER-MFFTORCH`，需手动修改 `LAMMPS/cmake/CMakeLists.txt`。

### 4.1 添加 USER-MFFTORCH 到 STANDARD_PACKAGES

找到 `set(STANDARD_PACKAGES ...)`，在列表末尾添加 `USER-MFFTORCH`：

```cmake
set(STANDARD_PACKAGES
   ...
   VTK
   YAFF
   USER-MFFTORCH)  # 添加此行
```

### 4.2 添加 USER-MFFTORCH 到 PKG_WITH_INCL

找到 `foreach(PKG_WITH_INCL ...)` 的 include 列表（用于链接外部库的包，如 GRAPHICS、PYTHON、ML-IAP 等），在其中加入 `USER-MFFTORCH`。

**方式 A：加入现有 foreach 列表**

```cmake
foreach(PKG_WITH_INCL GRAPHICS KSPACE PYTHON ML-IAP ... USER-MFFTORCH)
  if(PKG_${PKG_WITH_INCL})
    include(Packages/${PKG_WITH_INCL})
  endif()
endforeach()
```

**方式 B：单独添加（若找不到合适列表）**

在 `foreach(PKG_WITH_INCL ...)` 附近添加：

```cmake
if(PKG_USER-MFFTORCH)
  include(Packages/USER-MFFTORCH)
  # 若报错找不到文件，改为：include(Modules/Packages/USER-MFFTORCH)
endif()
```

> `Packages/` 对应 `cmake/Packages/USER-MFFTORCH.cmake`；`Modules/Packages/` 对应 `cmake/Modules/Packages/USER-MFFTORCH.cmake`。安装脚本会拷贝到两个位置，按你 LAMMPS 版本的实际 include 路径选择其一。

---

## 5. CMake 配置

### 5.1 获取 LibTorch 路径

```bash
LIBTORCH_PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
```

### 5.2 选择 GPU 架构

根据你的 GPU 选择 `Kokkos_ARCH_XXX`：

| GPU | Kokkos_ARCH |
|-----|-------------|
| NVIDIA A100 | AMPERE80 |
| NVIDIA A30 / RTX 30 系列 | AMPERE86 |
| NVIDIA V100 | VOLTA70 |
| NVIDIA T4 | TURING75 |
| AMD GPU | 参见 Kokkos 文档 |

### 5.3 编译选项：Virial/应力计算

| 选项 | 默认 | 说明 |
|------|------|------|
| `MFF_ENABLE_VIRIAL` | OFF | 设为 ON 时，在 GPU 上计算 virial，可输出正确的压力/应力张量 |

- **默认（OFF）**：不计算 virial，`thermo_style` 中的 `press` 等只有动能贡献，数值不完整；吞吐量最高。
- **ON**：在 GPU 上用 Kokkos 计算 fdotr virial，只拷贝 6 个 double 回 CPU，**几乎无性能损失**；`press`、`pxx`、`pyy`、`pzz`、`pxy`、`pxz`、`pyz` 输出正确。

### 5.4 配置命令

**不启用 virial（默认，最高吞吐）：**

```bash
cd /path/to/lammps

cmake -S cmake -B build-mfftorch \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_AMPERE86=ON \
  -D PKG_USER-MFFTORCH=ON \
  -D CMAKE_PREFIX_PATH="$LIBTORCH_PREFIX" \
  -D CMAKE_BUILD_TYPE=Release
```

**启用 virial（压力/应力正确）：**

```bash
cmake -S cmake -B build-mfftorch \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_AMPERE86=ON \
  -D PKG_USER-MFFTORCH=ON \
  -D MFF_ENABLE_VIRIAL=ON \
  -D CMAKE_PREFIX_PATH="$LIBTORCH_PREFIX" \
  -D CMAKE_BUILD_TYPE=Release
```

**若 LibTorch 未被自动找到**，可显式指定：

```bash
cmake -S cmake -B build-mfftorch \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_AMPERE86=ON \
  -D PKG_USER-MFFTORCH=ON \
  -D MFF_ENABLE_VIRIAL=ON \
  -D CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -D CMAKE_BUILD_TYPE=Release
```

---

## 6. 编译

```bash
cmake --build build-mfftorch -j
```

编译成功后，可执行文件位于：

```
build-mfftorch/lmp
```

---

## 7. 导出 core.pt（TorchScript 模型）

LAMMPS 的 `pair_style mff/torch` 需要 `core.pt` 文件。可用两种方式：

### 7.1 方式 A：从真实 checkpoint 导出

```bash
mff-export-core \
  --checkpoint /path/to/model.pth \
  --elements H O \
  --device cuda \
  --dtype float32 \
  --e0-csv /path/to/fitted_E0.csv \
  --out core.pt
```

**--mode 参数**：支持 `pure-cartesian-ictd`、`pure-cartesian-ictd-save`、`spherical-save-cue`。不指定时自动从 checkpoint 读取。若 checkpoint 中保存了 `tensor_product_mode`（mff-train 会自动保存），则无需手动指定。

**spherical-save-cue 导出（方案 A，便携版）**：
- 默认导出为纯 PyTorch 实现，`core.pt` **无需 cuEquivariance 运行时**，可在任意 LibTorch 环境运行。
- 导出时使用 `force_naive`，将 cuEquivariance 自定义 ops 替换为纯 PyTorch 等价实现。

**max-radius 如何确定**：
- **新训练的 checkpoint**（mff-train 会保存 max_radius）：脚本会自动从 checkpoint 读取，无需手动指定。
- **旧 checkpoint 或未保存**：必须与训练时 `mff-train --max-radius` 一致（默认 5.0），且与 LAMMPS `pair_style mff/torch CUTOFF` 的 cutoff 一致。

**E0 默认行为**：
- `mff-export-core` 现在默认会把 E0 一起嵌入导出的 `core.pt`
- 若显式传 `--e0-csv`，则优先使用该文件
- 若不传 `--e0-csv`，新 checkpoint 会优先使用 checkpoint 中保存的 `atomic_energy_keys/atomic_energy_values`
- 若是老 checkpoint 且未保存 E0，则回退到本地 `fitted_E0.csv`
- 只有显式传 `--no-embed-e0` 时，才导出不带 E0 的纯网络能量

### 7.2 方式 B：生成 dummy 模型（用于测试）

```bash
python - <<'PY'
import torch
from molecular_force_field.interfaces.self_test_lammps_potential import _make_dummy_checkpoint_pure_cartesian_ictd
_make_dummy_checkpoint_pure_cartesian_ictd("dummy.pth", device=torch.device("cpu"))
print("dummy.pth 已生成")
PY

python molecular_force_field/scripts/export_libtorch_core.py \
  --checkpoint dummy.pth \
  --elements H O \
  --device cuda \
  --max-radius 5.0 \
  --out core.pt
```

---

## 8. 运行 LAMMPS

### 8.1 设置环境变量（LibTorch 动态库）

```bash
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"
```

### 8.2 最小 LAMMPS 输入示例

创建 `in.mfftorch`：

```lammps
units metal
atom_style atomic
boundary p p p

region box block 0 40 0 40 0 40
create_box 2 box
create_atoms 1 random 200 12345 box
create_atoms 2 random 100 12346 box
mass 1 1.008
mass 2 15.999

neighbor 1.0 bin

pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/core.pt H O

velocity all create 300 42
fix 1 all nve
thermo 20
run 200
```

### 8.3 输出压力/应力（virial）

若编译时启用了 `MFF_ENABLE_VIRIAL=ON`，可在 `thermo_style` 中加入压力与应力分量：

```lammps
thermo_style custom step temp pe ke etotal press pxx pyy pzz pxy pxz pyz
thermo 20
```

| 字段 | 含义 |
|------|------|
| `press` | 总压力（标量） |
| `pxx pyy pzz` | 压力张量对角分量 |
| `pxy pxz pyz` | 压力张量非对角分量（剪切应力） |

未启用 virial 时，`press` 等只有动能贡献，数值不完整。

### 8.4 使用 Kokkos GPU 运行

```bash
/path/to/lammps/build-mfftorch/lmp \
  -k on g 1 \
  -sf kk \
  -pk kokkos newton off neigh full \
  -in in.mfftorch
```

说明：

- `-k on g 1`：启用 Kokkos，使用 1 块 GPU
- `-sf kk`：将 `pair_style mff/torch` 映射到 Kokkos 变体 `mff/torch/kk`
- `-pk kokkos newton off neigh full`：Kokkos 使用 `neigh full` 时必须 `newton off`

### 8.5 仅 CPU 运行（无 Kokkos）

```bash
/path/to/lammps/build-mfftorch/lmp -in in.mfftorch
```

---

## 9. 一键测试脚本

### 9.1 快速 smoke 测试（dummy 模型）

```bash
# 在 rebuild 仓库根目录
bash lammps_user_mfftorch/examples/run_smoke_mfftorch.sh /path/to/lmp cuda
```

### 9.2 完整 GPU 测试（含 dummy 或真实模型）

```bash
# 使用 dummy 模型（pure-cartesian-ictd）
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lammps/build-mfftorch/lmp \
  --dummy-ictd \
  --elements H O \
  --dtype float32 \
  --cutoff 5.0 \
  --steps 200

# 使用 dummy 模型（spherical-save-cue，需 cuEquivariance）
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lammps/build-mfftorch/lmp \
  --dummy-cue \
  --elements H O \
  --cutoff 5.0 \
  --steps 200

# 使用真实模型
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lammps/build-mfftorch/lmp \
  --pth /path/to/model.pth \
  --elements H O \
  --e0-csv /path/to/fitted_E0.csv \
  --dtype float32 \
  --cutoff 5.0 \
  --steps 200
```

---

## 10. 故障排查

### 10.1 `torch/torch.h: No such file or directory`

- 原因：LibTorch 未被找到。
- 解决：确保 `CMAKE_PREFIX_PATH` 包含 `torch.utils.cmake_prefix_path` 的路径，且 `USER-MFFTORCH` 已加入 `PKG_WITH_INCL` 的 include 列表。

### 10.2 `Unrecognized pair style 'mff/torch'`

- 原因：USER-MFFTORCH 未正确编译或未启用。
- 解决：确认 `-D PKG_USER-MFFTORCH=ON`，且 `USER-MFFTORCH` 已加入 `STANDARD_PACKAGES`，`src/USER-MFFTORCH/` 存在且包含 `pair_mff_torch.cpp` 等文件。

### 10.3 `error while loading shared libraries: libtorch.so`

- 原因：运行时找不到 LibTorch 动态库。
- 解决：设置 `LD_LIBRARY_PATH`（见 8.1 节）。

### 10.4 `Must use 'newton off' with KOKKOS package option 'neigh full'`

- 原因：Kokkos 配置要求。
- 解决：在 `lmp` 命令行加上 `-pk kokkos newton off neigh full`。

### 10.5 `Kokkos_ARCH` 与 GPU 不匹配

- 现象：运行时报错或 JIT 编译耗时很长。
- 解决：根据 GPU 型号选择正确的 `Kokkos_ARCH_XXX`（见 5.2 节）。

### 10.6 能量/温度恒不变

- 若使用 dummy 模型，输出常数为预期行为（dummy 模型梯度近似为 0）。
- 使用真实训练模型后，能量和温度会随时间变化。

### 10.7 需要压力/应力但未启用 virial

- 现象：`thermo_style` 含 `press` 时，数值只有动能贡献，不完整。
- 解决：用 `-D MFF_ENABLE_VIRIAL=ON` 重新配置并编译，virial 在 GPU 上计算，几乎无性能损失。

---

## 11. 附录：目录结构速览

```
rebuild/
├── lammps_user_mfftorch/
│   ├── src/USER-MFFTORCH/          # 源码
│   ├── cmake/Modules/Packages/USER-MFFTORCH.cmake
│   ├── cmake/Packages/USER-MFFTORCH.cmake  # 部分版本
│   ├── examples/
│   └── docs/BUILD_AND_RUN.md       # 本文档
├── molecular_force_field/
│   └── scripts/
│       ├── export_libtorch_core.py  # 导出 core.pt
│       └── run_gpu_lammps_with_corept.sh
└── scripts/
    └── install_user_mfftorch_into_lammps.sh
```
