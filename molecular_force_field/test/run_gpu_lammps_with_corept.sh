#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
GPU 机一键测试：
  pth(.pth) -> 导出 core.pt(TorchScript，可选 embed E0 / fp32) -> 用 LAMMPS(USER-MFFTORCH) 跑一段 MD

要求：
- 已编译好的 LAMMPS 可执行文件（包含 KOKKOS + USER-MFFTORCH）
- Python 环境里有 torch (CUDA) 且可导出 core.pt

用法：
  bash molecular_force_field/scripts/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --pth /path/to/model.pth \
    --elements H O \
    --e0-csv /path/to/fitted_E0.csv \
    --dtype float32 \
    --cutoff 5.0 \
    --steps 200

也支持自动生成 dummy pth（方式1）：
  bash molecular_force_field/scripts/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --elements H O \
    --dtype float32 \
    --cutoff 5.0 \
    --steps 200

参数：
  --lmp <path>        LAMMPS 可执行文件
  --pth <path>        真实 checkpoint (.pth)
  --dummy-ictd        不提供 --pth 时，自动生成 pure-cartesian-ictd dummy checkpoint
  --dummy-e0          在 --dummy-ictd 且未提供 --e0-csv 时，自动生成 fitted_E0.csv 用于测试 embed E0（默认开启）
  --no-dummy-e0       关闭自动生成 dummy E0
  --elements ...      元素顺序（对应 LAMMPS type 顺序）
  --e0-csv <path>     fitted_E0.csv（含 Atom,E0 列），提供则 embed 到 core.pt
  --dtype float32|float64   导出 core.pt 的精度（默认跟随 pth）
  --cutoff <A>        pair_style cutoff (Angstrom)
  --steps <N>         MD 步数（默认 200）
  --out-dir <dir>     输出目录（默认 mktemp）
  --gpu <g>           使用 GPU 数（默认 1）
  --n1 <int>          随机生成 type1 原子数（默认 2000）
  --n2 <int>          随机生成 type2 原子数（默认 1000）
  --box <float>       盒子边长（默认自动估计）

说明：
- 脚本会生成随机体系（create_atoms），直接用 `pair_style mff/torch`，并通过 `-sf kk` 走 Kokkos CUDA 路径。
- 若你有自己的体系文件，建议手动改生成 input 部分，把 read_data 换成你的 data。
EOF
}

LMP=""
PTH=""
DUMMY_ICTD=0
DUMMY_E0=1
ELEMENTS=()
E0CSV=""
DTYPE=""
CUTOFF="5.0"
STEPS="200"
OUT_DIR=""
GPU_N="1"
N1="2000"
N2="1000"
BOX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lmp) LMP="${2:-}"; shift 2;;
    --pth) PTH="${2:-}"; shift 2;;
    --dummy-ictd) DUMMY_ICTD=1; shift;;
    --dummy-e0) DUMMY_E0=1; shift;;
    --no-dummy-e0) DUMMY_E0=0; shift;;
    --elements)
      shift
      ELEMENTS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do ELEMENTS+=("$1"); shift; done
      ;;
    --e0-csv) E0CSV="${2:-}"; shift 2;;
    --dtype) DTYPE="${2:-}"; shift 2;;
    --cutoff) CUTOFF="${2:-}"; shift 2;;
    --steps) STEPS="${2:-}"; shift 2;;
    --out-dir) OUT_DIR="${2:-}"; shift 2;;
    --gpu) GPU_N="${2:-}"; shift 2;;
    --n1) N1="${2:-}"; shift 2;;
    --n2) N2="${2:-}"; shift 2;;
    --box) BOX="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "未知参数：$1"; usage; exit 2;;
  esac
done

if [[ -z "$LMP" || ${#ELEMENTS[@]} -eq 0 ]]; then
  echo "必须提供 --lmp --elements"
  usage
  exit 2
fi
if [[ -z "$PTH" && $DUMMY_ICTD -ne 1 ]]; then
  echo "必须提供 --pth，或使用 --dummy-ictd 自动生成"
  usage
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(python - <<'PY'
import tempfile
print(tempfile.mkdtemp(prefix="mff-corept-lmp-"))
PY
)"
fi
mkdir -p "$OUT_DIR"

CORE_PT="$OUT_DIR/core.pt"

if [[ -z "$PTH" && $DUMMY_ICTD -eq 1 ]]; then
  echo "[0/3] 生成 pure-cartesian-ictd dummy checkpoint"
  PTH="$OUT_DIR/dummy_ictd.pth"
  python - <<PY
import torch
from molecular_force_field.interfaces.self_test_lammps_potential import _make_dummy_checkpoint_pure_cartesian_ictd
out = r"$PTH"
_make_dummy_checkpoint_pure_cartesian_ictd(out, device=torch.device("cpu"))
print("wrote", out)
PY
fi

echo "[1/3] 导出 core.pt（TorchScript，embed E0 可选）"
if [[ $DUMMY_ICTD -eq 1 && -z "$E0CSV" && $DUMMY_E0 -eq 1 ]]; then
  # 为了测试 embed E0 功能，生成一个简单的 fitted_E0.csv（Atom,E0）
  E0CSV="$OUT_DIR/fitted_E0.csv"
  echo "Atom,E0" > "$E0CSV"
  # 目前主要用于 H/O 等常见元素的联机测试；未知元素会跳过（E0=0）。
  python - "${ELEMENTS[@]}" <<'PY' >> "$E0CSV"
import sys

sym2Z = {
  "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
  "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
}

seen = set()
for s in sys.argv[1:]:
  s = (s or "").strip()
  if not s or s in seen:
    continue
  seen.add(s)
  z = sym2Z.get(s)
  if z is None:
    continue
  # 随便给一个可辨识的测试值：E0 = -0.1 * Z (eV)
  print(f"{z},{-0.1*z:.8f}")
PY
  echo "[0/3] 已生成 dummy E0 CSV: $E0CSV"
fi

EXPORT_ARGS=(--checkpoint "$PTH" --elements "${ELEMENTS[@]}" --device cuda --max-radius "$CUTOFF" --out "$CORE_PT" --embed-e0)
if [[ -n "$DTYPE" ]]; then EXPORT_ARGS+=(--dtype "$DTYPE"); fi
if [[ -n "$E0CSV" ]]; then EXPORT_ARGS+=(--e0-csv "$E0CSV"); fi
python "$REPO_ROOT/molecular_force_field/scripts/export_libtorch_core.py" "${EXPORT_ARGS[@]}"

echo "[2/3] 写入 LAMMPS 输入文件"
if [[ -z "$BOX" ]]; then
  BOX="$(python - <<PY
import math
n = int("$N1")+int("$N2")
box = max(60.0, 3.0*(n**(1.0/3.0))*2.5)
print(f"{box:.3f}")
PY
)"
fi

cat > "$OUT_DIR/in.corept" <<EOF
units metal
atom_style atomic
boundary p p p

region box block 0 $BOX 0 $BOX 0 $BOX
create_box 2 box
create_atoms 1 random $N1 12345 box
create_atoms 2 random $N2 12346 box
mass 1 1.008
mass 2 15.999

neighbor 1.0 bin

pair_style mff/torch $CUTOFF cuda
pair_coeff * * $CORE_PT ${ELEMENTS[*]}

velocity all create 300 42
fix 1 all nve
thermo 20
run $STEPS
EOF

echo "[3/3] 运行 LAMMPS（Kokkos+CUDA）"
export LD_LIBRARY_PATH="$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
):${LD_LIBRARY_PATH:-}"

"$LMP" -k on g "$GPU_N" -sf kk -pk kokkos newton off neigh full -in "$OUT_DIR/in.corept"

echo "DONE. out_dir=$OUT_DIR"

