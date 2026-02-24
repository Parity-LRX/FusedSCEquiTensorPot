## 这是什么

这是一套可拷贝进 **LAMMPS 源码树**的自定义包：`USER-MFFTORCH`。

**完整编译与运行指南**：见 [docs/BUILD_AND_RUN.md](docs/BUILD_AND_RUN.md)。

它提供两个 pair style：

- `pair_style mff/torch`：纯 C++ + LibTorch（可先用 CPU neighbor list 跑通）
- `pair_style mff/torch/kk`：Kokkos+CUDA 数据准备 + LibTorch(CUDA) 推理（目标：全链路无 Python、无 Host 往返）

模型文件使用本仓库导出的 **TorchScript core**：`core.pt`（`torch.jit.save`），见 `molecular_force_field/scripts/export_libtorch_core.py`。

## 目录结构（拷贝到 LAMMPS 源码）

把本目录中的：

- `src/USER-MFFTORCH/` → 拷贝到 `LAMMPS/src/USER-MFFTORCH/`
- （你的版本为 22Jul2025）`cmake/Modules/Packages/USER-MFFTORCH.cmake` → 拷贝到 `LAMMPS/cmake/Modules/Packages/USER-MFFTORCH.cmake`

## LAMMPS CMake 编译示例

```bash
cmake -S /path/to/lammps/cmake -B build-lmp \
  -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_AMPERE86=ON \
  -D PKG_USER-MFFTORCH=ON \
  -D CMAKE_PREFIX_PATH="$(python - <<'PY'\nimport torch\nprint(torch.utils.cmake_prefix_path)\nPY\n)"
cmake --build build-lmp -j
```

此外你需要在 `LAMMPS/cmake/CMakeLists.txt` 里做 2 处一次性改动（否则 CMake 不会识别这个新包）：
- 把 `USER-MFFTORCH` 加进 `STANDARD_PACKAGES` 列表
- 把 `USER-MFFTORCH` 加进 `foreach(PKG_WITH_INCL ...)` 的 include 列表（这样会执行 `cmake/Modules/Packages/USER-MFFTORCH.cmake` 来链接 LibTorch）

如果运行时找不到 `libtorch.so` / `libc10.so`，给可执行文件设置 `LD_LIBRARY_PATH`（指向你的 Python venv 里 `torch/lib`）。

## LAMMPS 输入示例

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data

neighbor 1.0 bin

pair_style mff/torch/kk 5.0 cuda
pair_coeff * * /path/to/core.pt H O

velocity all create 300 42
fix 1 all nve
run 100
```

说明：
- `pair_style ... 5.0` 是 cutoff（Angstrom）。
- `pair_coeff * * core.pt H O` 的元素顺序必须与导出时一致（type→元素→Z 映射）。可用 `NULL` 跳过某个 type。

