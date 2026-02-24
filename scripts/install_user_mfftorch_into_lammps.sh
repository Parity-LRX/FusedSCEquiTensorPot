#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "用法: bash scripts/install_user_mfftorch_into_lammps.sh /path/to/lammps"
  exit 2
fi

LMP_ROOT="$1"
SRC_DIR="$LMP_ROOT/src"
CMAKE_PKG_DIR="$LMP_ROOT/cmake/Modules/Packages"
CMK_PACKAGES_DIR="$LMP_ROOT/cmake/Packages"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "找不到 LAMMPS src 目录: $SRC_DIR"
  exit 2
fi
if [[ ! -d "$CMAKE_PKG_DIR" ]]; then
  echo "找不到 LAMMPS CMake Packages 目录: $CMAKE_PKG_DIR"
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "拷贝 USER-MFFTORCH 源码到: $SRC_DIR/USER-MFFTORCH"
mkdir -p "$SRC_DIR/USER-MFFTORCH"
cp -v "$REPO_ROOT/lammps_user_mfftorch/src/USER-MFFTORCH/"* "$SRC_DIR/USER-MFFTORCH/"

echo "拷贝 CMake 包模块到: $CMAKE_PKG_DIR/USER-MFFTORCH.cmake"
cp -v "$REPO_ROOT/lammps_user_mfftorch/cmake/Modules/Packages/USER-MFFTORCH.cmake" \
  "$CMAKE_PKG_DIR/USER-MFFTORCH.cmake"

if [[ -d "$CMK_PACKAGES_DIR" ]]; then
  echo "拷贝 CMake 包模块到: $CMK_PACKAGES_DIR/USER-MFFTORCH.cmake"
  cp -v "$REPO_ROOT/lammps_user_mfftorch/cmake/Packages/USER-MFFTORCH.cmake" \
    "$CMK_PACKAGES_DIR/USER-MFFTORCH.cmake"
else
  echo "WARN: 未发现 $CMK_PACKAGES_DIR（你的 LAMMPS 版本正常）；将仅使用 $CMAKE_PKG_DIR 下的模块"
fi

echo "DONE. 现在可以在 LAMMPS CMake 中使用 -D PKG_USER-MFFTORCH=ON"

