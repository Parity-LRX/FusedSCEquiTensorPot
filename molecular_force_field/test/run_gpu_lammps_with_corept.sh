#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
One-click GPU test:
  pth(.pth) -> export core.pt (TorchScript, optional embed E0 / fp32) -> run MD with LAMMPS (USER-MFFTORCH)

Requirements:
- Built LAMMPS executable (with KOKKOS + USER-MFFTORCH)
- Python env with torch (CUDA) for exporting core.pt

Usage:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --pth /path/to/model.pth \
    --elements H O \
    --e0-csv /path/to/fitted_E0.csv \
    --dtype float32 \
    --cutoff 5.0 \
    --steps 200

Also supports auto dummy pth (option 1 pure-cartesian-ictd):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --elements H O \
    --dtype float32 \
    --cutoff 5.0 \
    --steps 200

Option 2 spherical-save-cue (requires cuEquivariance):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-cue \
    --elements H O \
    --cutoff 5.0 \
    --steps 200

Multi-GPU (4 GPUs, LAMMPS must be built with MPI):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /path/to/lmp \
    --dummy-ictd \
    --elements H O \
    --gpu 4 \
    --np 4 \
    --steps 200

  # SLURM: use srun:
  bash ... --gpu 4 --np 4 --mpi-cmd srun

Options:
  --lmp <path>        LAMMPS executable
  --pth <path>        Real checkpoint (.pth)
  --dummy-ictd        Auto-generate pure-cartesian-ictd dummy checkpoint when --pth not provided
  --dummy-cue         Auto-generate spherical-save-cue dummy checkpoint when --pth not provided (requires cuEquivariance)
  --dummy-e0          Auto-generate fitted_E0.csv for embed E0 test in dummy mode (default on)
  --no-dummy-e0       Disable auto dummy E0
  --elements ...      Element order (LAMMPS type order)
  --e0-csv <path>     fitted_E0.csv (Atom,E0 columns); embed into core.pt if provided
  --dtype float32|float64   core.pt export precision (default: follow pth)
  --cutoff <A>        pair_style cutoff (Angstrom)
  --mode <mode>       Model mode for export (e.g. spherical-save-cue); else from checkpoint
  --steps <N>         MD steps (default 200)
  --out-dir <dir>     Output dir (default mktemp)
  --gpu <g>           Number of GPUs (default 1)
  --np <N>            MPI processes for multi-GPU (default = --gpu)
  --mpi-cmd <cmd>     MPI launcher for multi-GPU (default mpirun; use srun for SLURM)
  --n1 <int>          Random type1 atoms (default 2000)
  --n2 <int>          Random type2 atoms (default 1000)
  --box <float>       Box side length (default auto)
  --native-ops        spherical-save-cue: keep native cuEquivariance ops (requires MFF_CUSTOM_OPS_LIB)

Notes:
- Script creates random system (create_atoms), uses pair_style mff/torch with -sf kk for Kokkos CUDA.
- Multi-GPU: --gpu 4 --np 4 uses mpirun -np 4; LAMMPS must be built with MPI.
- If multi-GPU hangs: set CUDA_VISIBLE_DEVICES=0,1 or reduce system with --n1 500 --n2 250.
- For custom system: edit input section, replace with read_data for your data file.
EOF
}

LMP=""
PTH=""
DUMMY_ICTD=0
DUMMY_CUE=0
DUMMY_E0=1
MODE=""
NATIVE_OPS=0
ELEMENTS=()
E0CSV=""
DTYPE=""
CUTOFF="5.0"
STEPS="200"
OUT_DIR=""
GPU_N="1"
NP=""          # MPI processes for multi-GPU; default = GPU_N
MPI_CMD="mpirun"
N1="2000"
N2="1000"
BOX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lmp) LMP="${2:-}"; shift 2;;
    --pth) PTH="${2:-}"; shift 2;;
    --dummy-ictd) DUMMY_ICTD=1; shift;;
    --dummy-cue) DUMMY_CUE=1; shift;;
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
    --mode) MODE="${2:-}"; shift 2;;
    --native-ops) NATIVE_OPS=1; shift;;
    --steps) STEPS="${2:-}"; shift 2;;
    --out-dir) OUT_DIR="${2:-}"; shift 2;;
    --gpu) GPU_N="${2:-}"; shift 2;;
    --np) NP="${2:-}"; shift 2;;
    --mpi-cmd) MPI_CMD="${2:-}"; shift 2;;
    --n1) N1="${2:-}"; shift 2;;
    --n2) N2="${2:-}"; shift 2;;
    --box) BOX="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 2;;
  esac
done

if [[ -z "$LMP" || ${#ELEMENTS[@]} -eq 0 ]]; then
  echo "Must provide --lmp --elements"
  usage
  exit 2
fi
if [[ -z "$PTH" && $DUMMY_ICTD -ne 1 && $DUMMY_CUE -ne 1 ]]; then
  echo "Must provide --pth, or use --dummy-ictd / --dummy-cue to auto-generate"
  usage
  exit 2
fi
if [[ $DUMMY_ICTD -eq 1 && $DUMMY_CUE -eq 1 ]]; then
  echo "Cannot use both --dummy-ictd and --dummy-cue"
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
  echo "[0/3] Generating pure-cartesian-ictd dummy checkpoint"
  PTH="$OUT_DIR/dummy_ictd.pth"
  python - <<PY
import torch
from molecular_force_field.interfaces.self_test_lammps_potential import _make_dummy_checkpoint_pure_cartesian_ictd
out = r"$PTH"
_make_dummy_checkpoint_pure_cartesian_ictd(out, device=torch.device("cpu"))
print("wrote", out)
PY
fi
if [[ -z "$PTH" && $DUMMY_CUE -eq 1 ]]; then
  echo "[0/3] Generating spherical-save-cue dummy checkpoint"
  PTH="$OUT_DIR/dummy_cue.pth"
  python - <<PY
import torch
from molecular_force_field.interfaces.self_test_lammps_potential import _make_dummy_checkpoint_spherical_save_cue
out = r"$PTH"
_make_dummy_checkpoint_spherical_save_cue(out, device=torch.device("cpu"))
print("wrote", out)
PY
fi

echo "[1/3] Exporting core.pt (TorchScript, embed E0 optional)"
if [[ ($DUMMY_ICTD -eq 1 || $DUMMY_CUE -eq 1) && -z "$E0CSV" && $DUMMY_E0 -eq 1 ]]; then
  # Generate simple fitted_E0.csv (Atom,E0) for embed E0 test
  E0CSV="$OUT_DIR/fitted_E0.csv"
  echo "Atom,E0" > "$E0CSV"
  # For H/O etc. integration test; unknown elements skipped (E0=0).
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
  # Simple test value: E0 = -0.1 * Z (eV)
  print(f"{z},{-0.1*z:.8f}")
PY
  echo "[0/3] Generated dummy E0 CSV: $E0CSV"
fi

EXPORT_ARGS=(--checkpoint "$PTH" --elements "${ELEMENTS[@]}" --device cuda --max-radius "$CUTOFF" --out "$CORE_PT" --embed-e0)
if [[ -n "$MODE" ]]; then EXPORT_ARGS+=(--mode "$MODE"); fi
if [[ $NATIVE_OPS -eq 1 ]]; then EXPORT_ARGS+=(--native-ops); fi
if [[ -n "$DTYPE" ]]; then EXPORT_ARGS+=(--dtype "$DTYPE"); fi
if [[ -n "$E0CSV" ]]; then EXPORT_ARGS+=(--e0-csv "$E0CSV"); fi
# Export always on cuda:0 to avoid MPI env interference
CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/molecular_force_field/cli/export_libtorch_core.py" "${EXPORT_ARGS[@]}"

echo "[2/3] Writing LAMMPS input file"
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

echo "[3/3] Running LAMMPS (Kokkos+CUDA)"
export LD_LIBRARY_PATH="$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
):${LD_LIBRARY_PATH:-}"

if [[ $NATIVE_OPS -eq 1 && -z "${MFF_CUSTOM_OPS_LIB:-}" ]]; then
  echo "[3/3] --native-ops: auto-detect cuEquivariance ops and libpython path"
  eval "$(python - <<'PY'
import pathlib, sys, sysconfig

extra_ld = []

# 1) Find the torch extension .so (registers TorchScript custom ops)
try:
    import cuequivariance_ops_torch
    pkg = pathlib.Path(cuequivariance_ops_torch.__file__).parent
    so_files = sorted(f for f in pkg.rglob("*.so") if "__pycache__" not in str(f))
    if so_files:
        print(f'export MFF_CUSTOM_OPS_LIB="{":".join(str(f) for f in so_files)}"')
except ImportError:
    print("echo 'WARNING: cuequivariance_ops_torch not installed'", flush=True)

# 2) Find libcue_ops.so and add its dir to LD_LIBRARY_PATH
try:
    import cuequivariance_ops
    pkg2 = pathlib.Path(cuequivariance_ops.__file__).parent
    for f in pkg2.rglob("libcue_ops*"):
        extra_ld.append(str(f.parent))
except ImportError:
    pass

# 3) Find libpython (needed by CPython extension in pure-C++ process)
libdir = sysconfig.get_config_var("LIBDIR") or ""
if libdir:
    extra_ld.append(libdir)

if extra_ld:
    dirs = ":".join(sorted(set(extra_ld)))
    print(f'export LD_LIBRARY_PATH="{dirs}:${{LD_LIBRARY_PATH:-}}"')
PY
)"
  echo "  MFF_CUSTOM_OPS_LIB=${MFF_CUSTOM_OPS_LIB:-<not set>}"
fi

# Multi-GPU: use MPI when np > 1
if [[ -z "$NP" ]]; then
  NP="$GPU_N"
fi
LMP_ARGS=(-k on g "$GPU_N" -sf kk -pk kokkos newton off neigh full -in "$OUT_DIR/in.corept")

if [[ "$NP" -gt 1 ]]; then
  echo "[3/3] Multi-GPU run: $MPI_CMD -np $NP $LMP ${LMP_ARGS[*]}"
  # Pass env vars for multi-GPU so each rank uses correct GPU; --bind-to none avoids binding conflicts
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  MPI_EXTRA=(-x LD_LIBRARY_PATH -x CUDA_DEVICE_ORDER)
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && MPI_EXTRA+=(-x CUDA_VISIBLE_DEVICES)
  "$MPI_CMD" -np "$NP" "${MPI_EXTRA[@]}" --bind-to none "$LMP" "${LMP_ARGS[@]}"
else
  echo "[3/3] Single-GPU run"
  "$LMP" "${LMP_ARGS[@]}"
fi

echo "DONE. out_dir=$OUT_DIR"

