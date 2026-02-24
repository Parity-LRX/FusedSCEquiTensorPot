#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: bash run_smoke_mfftorch.sh /path/to/lmp [cuda|cpu]"
  exit 2
fi

LMP_EXE="$1"
DEVICE="${2:-cuda}"

OUT_DIR="${OUT_DIR:-$(pwd)/mfftorch_smoke_out}"
mkdir -p "$OUT_DIR"

echo "[1/3] 生成 dummy checkpoint + core.pt（若不存在）"
if [[ ! -f "$OUT_DIR/core.pt" ]]; then
  python - <<PY
import os
import torch
from molecular_force_field.interfaces.self_test_lammps_potential import _make_dummy_checkpoint_pure_cartesian_ictd

out_dir = r"$OUT_DIR"
ckpt = os.path.join(out_dir, "dummy.pth")
_make_dummy_checkpoint_pure_cartesian_ictd(ckpt, device=torch.device("cpu"))
print("dummy checkpoint:", ckpt)
PY
  python molecular_force_field/scripts/export_libtorch_core.py \
    --checkpoint "$OUT_DIR/dummy.pth" --device "$DEVICE" --elements H O --out "$OUT_DIR/core.pt"
fi

echo "[2/3] 写入 LAMMPS 输入文件（随机创建原子）"
cat > "$OUT_DIR/in.smoke" <<EOF
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

pair_style mff/torch 5.0 $DEVICE
pair_coeff * * $OUT_DIR/core.pt H O

velocity all create 300 42
fix 1 all nve
thermo 10
run 50
EOF

echo "[3/3] 运行 LAMMPS"
echo "OUT_DIR=$OUT_DIR"
echo "INPUT=$OUT_DIR/in.smoke"
if [[ "$DEVICE" == "cuda" ]]; then
  "$LMP_EXE" -k on g 1 -sf kk -pk kokkos newton off neigh full -in "$OUT_DIR/in.smoke"
else
  "$LMP_EXE" -in "$OUT_DIR/in.smoke"
fi

