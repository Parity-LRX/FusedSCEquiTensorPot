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

Runtime external field smoke test (rank-1):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --elements H O \
    --field-values 0.0 0.0 0.01 \
    --cutoff 5.0 \
    --steps 50

Runtime external field smoke test (rank-2 full 3x3, row-major):
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --elements H O \
    --field9-values 1 0 0 0 1 0 0 0 1 \
    --cutoff 5.0 \
    --steps 50

Runtime external field + physical tensor compute smoke test:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --dummy-phys-heads \
    --test-phys-compute \
    --elements H O \
    --field-values 0.0 0.0 0.01 \
    --cutoff 5.0 \
    --steps 50

No external field + physical tensor compute smoke test:
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /root/lammps-22Jul2025/build-mfftorch/lmp \
    --dummy-ictd \
    --dummy-phys-heads \
    --test-phys-compute \
    --elements H O \
    --cutoff 5.0 \
    --steps 50

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
  --dummy-phys-heads  In --dummy-ictd mode, also add dipole/polarizability physical tensor heads
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
  --field-values Ex Ey Ez   Enable runtime external field via pair_style field v_Ex v_Ey v_Ez
  --field6-values xx yy zz xy xz yz
                      Enable symmetric rank-2 runtime external field via pair_style field6
  --field9-values xx xy xz yx yy yz zx zy zz
                      Enable full rank-2 runtime external field via pair_style field9 (row-major)
  --test-phys-compute Inject compute mff/torch/phys checks into the generated LAMMPS input

Notes:
- Script creates random system (create_atoms), uses pair_style mff/torch with -sf kk for Kokkos CUDA.
- Multi-GPU: --gpu 4 --np 4 uses mpirun -np 4; LAMMPS must be built with MPI.
- If multi-GPU hangs: set CUDA_VISIBLE_DEVICES=0,1 or reduce system with --n1 500 --n2 250.
- For custom system: edit input section, replace with read_data for your data file.
- Four common test combinations:
  1) No field, no physical tensor:
     --dummy-ictd
  2) Field, no physical tensor:
     --dummy-ictd + one of --field-values / --field6-values / --field9-values
  3) No field, with physical tensor:
     --dummy-ictd --dummy-phys-heads --test-phys-compute
  4) Field, with physical tensor:
     --dummy-ictd --dummy-phys-heads --test-phys-compute + one field option
- --dummy-phys-heads currently adds fixed-schema heads only:
    dipole, dipole_per_atom, polarizability, polarizability_per_atom
  and is intended to validate the current mfftorch/LAMMPS export path.
- Current LAMMPS mfftorch physical tensor interface exposes only fixed-schema quantities:
    charge, dipole, polarizability, quadrupole
    charge_per_atom, dipole_per_atom, polarizability_per_atom, quadrupole_per_atom
  Missing heads are allowed: masks become 0 and corresponding outputs are filled with 0.
- Rank/l compatibility of the fixed schema:
    charge / charge_per_atom correspond to l=0 scalar outputs
    dipole / dipole_per_atom correspond to l=1 vector outputs
    polarizability is expected as a rank-2 Cartesian tensor (typically l=0+2)
    quadrupole is expected as a rank-2 traceless tensor (typically l=2)
- External field compatibility:
    --field-values tests rank-1 external tensor (l=1-like vector case)
    --field6-values / --field9-values test rank-2 external tensor input
- Arbitrary custom physical head names are not auto-exposed to LAMMPS yet.
  If your model trains a custom scalar/tensor head outside the fixed schema above,
  the model can still run, but compute mff/torch/phys will not automatically expose it.
- Current LibTorch/LAMMPS export path assumes channels_out == 1 for exposed physical heads.
EOF
}

LMP=""
PTH=""
DUMMY_ICTD=0
DUMMY_CUE=0
DUMMY_PHYS_HEADS=0
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
FIELD_VALUES=()
FIELD6_VALUES=()
FIELD9_VALUES=()
TEST_PHYS_COMPUTE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lmp) LMP="${2:-}"; shift 2;;
    --pth) PTH="${2:-}"; shift 2;;
    --dummy-ictd) DUMMY_ICTD=1; shift;;
    --dummy-cue) DUMMY_CUE=1; shift;;
    --dummy-phys-heads) DUMMY_PHYS_HEADS=1; shift;;
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
    --field-values)
      shift
      FIELD_VALUES=("${1:-}" "${2:-}" "${3:-}")
      shift 3
      ;;
    --field6-values)
      shift
      FIELD6_VALUES=("${1:-}" "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}")
      shift 6
      ;;
    --field9-values)
      shift
      FIELD9_VALUES=("${1:-}" "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}" "${7:-}" "${8:-}" "${9:-}")
      shift 9
      ;;
    --test-phys-compute) TEST_PHYS_COMPUTE=1; shift;;
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
if [[ $DUMMY_PHYS_HEADS -eq 1 && $DUMMY_ICTD -ne 1 ]]; then
  echo "--dummy-phys-heads currently requires --dummy-ictd"
  exit 2
fi
FIELD_MODE_COUNT=0
[[ ${#FIELD_VALUES[@]} -ne 0 ]] && FIELD_MODE_COUNT=$((FIELD_MODE_COUNT+1))
[[ ${#FIELD6_VALUES[@]} -ne 0 ]] && FIELD_MODE_COUNT=$((FIELD_MODE_COUNT+1))
[[ ${#FIELD9_VALUES[@]} -ne 0 ]] && FIELD_MODE_COUNT=$((FIELD_MODE_COUNT+1))
if [[ $FIELD_MODE_COUNT -gt 1 ]]; then
  echo "Use only one of --field-values, --field6-values, --field9-values"
  exit 2
fi
if [[ ${#FIELD_VALUES[@]} -ne 0 && ${#FIELD_VALUES[@]} -ne 3 ]]; then
  echo "--field-values expects exactly 3 values: Ex Ey Ez"
  exit 2
fi
if [[ ${#FIELD6_VALUES[@]} -ne 0 && ${#FIELD6_VALUES[@]} -ne 6 ]]; then
  echo "--field6-values expects exactly 6 values: xx yy zz xy xz yz"
  exit 2
fi
if [[ ${#FIELD9_VALUES[@]} -ne 0 && ${#FIELD9_VALUES[@]} -ne 9 ]]; then
  echo "--field9-values expects exactly 9 values in row-major order"
  exit 2
fi
if [[ ${#FIELD_VALUES[@]} -ne 0 && $DUMMY_CUE -eq 1 ]]; then
  echo "Runtime external field is currently supported only for pure-cartesian-ictd"
  exit 2
fi
if [[ ${#FIELD6_VALUES[@]} -ne 0 && $DUMMY_CUE -eq 1 ]]; then
  echo "Runtime external field is currently supported only for pure-cartesian-ictd"
  exit 2
fi
if [[ ${#FIELD9_VALUES[@]} -ne 0 && $DUMMY_CUE -eq 1 ]]; then
  echo "Runtime external field is currently supported only for pure-cartesian-ictd"
  exit 2
fi
if [[ $DUMMY_PHYS_HEADS -eq 1 ]]; then
  TEST_PHYS_COMPUTE=1
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
external_rank = None
if ${#FIELD_VALUES[@]} == 3:
    external_rank = 1
elif ${#FIELD6_VALUES[@]} == 6 or ${#FIELD9_VALUES[@]} == 9:
    external_rank = 2
physical_tensor_outputs = None
if $DUMMY_PHYS_HEADS == 1:
    physical_tensor_outputs = {
        "dipole": {"ls": [1], "channels_out": 1, "reduce": "sum"},
        "dipole_per_atom": {"ls": [1], "channels_out": 1, "reduce": "none"},
        "polarizability": {"ls": [0, 2], "channels_out": 1, "reduce": "sum"},
        "polarizability_per_atom": {"ls": [0, 2], "channels_out": 1, "reduce": "none"},
    }
_make_dummy_checkpoint_pure_cartesian_ictd(
    out,
    device=torch.device("cpu"),
    external_tensor_rank=external_rank,
    physical_tensor_outputs=physical_tensor_outputs,
)
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

FIELD_LMP="pair_style mff/torch $CUTOFF cuda"
if [[ ${#FIELD_VALUES[@]} -eq 3 ]]; then
  FIELD_LMP=$'variable Ex equal '"${FIELD_VALUES[0]}"$'\n'
  FIELD_LMP+=$'variable Ey equal '"${FIELD_VALUES[1]}"$'\n'
  FIELD_LMP+=$'variable Ez equal '"${FIELD_VALUES[2]}"$'\n'
  FIELD_LMP+=$'pair_style mff/torch '"$CUTOFF"$' cuda field v_Ex v_Ey v_Ez'
elif [[ ${#FIELD6_VALUES[@]} -eq 6 ]]; then
  FIELD_LMP=$'variable Txx equal '"${FIELD6_VALUES[0]}"$'\n'
  FIELD_LMP+=$'variable Tyy equal '"${FIELD6_VALUES[1]}"$'\n'
  FIELD_LMP+=$'variable Tzz equal '"${FIELD6_VALUES[2]}"$'\n'
  FIELD_LMP+=$'variable Txy equal '"${FIELD6_VALUES[3]}"$'\n'
  FIELD_LMP+=$'variable Txz equal '"${FIELD6_VALUES[4]}"$'\n'
  FIELD_LMP+=$'variable Tyz equal '"${FIELD6_VALUES[5]}"$'\n'
  FIELD_LMP+=$'pair_style mff/torch '"$CUTOFF"$' cuda field6 v_Txx v_Tyy v_Tzz v_Txy v_Txz v_Tyz'
elif [[ ${#FIELD9_VALUES[@]} -eq 9 ]]; then
  FIELD_LMP=$'variable Txx equal '"${FIELD9_VALUES[0]}"$'\n'
  FIELD_LMP+=$'variable Txy equal '"${FIELD9_VALUES[1]}"$'\n'
  FIELD_LMP+=$'variable Txz equal '"${FIELD9_VALUES[2]}"$'\n'
  FIELD_LMP+=$'variable Tyx equal '"${FIELD9_VALUES[3]}"$'\n'
  FIELD_LMP+=$'variable Tyy equal '"${FIELD9_VALUES[4]}"$'\n'
  FIELD_LMP+=$'variable Tyz equal '"${FIELD9_VALUES[5]}"$'\n'
  FIELD_LMP+=$'variable Tzx equal '"${FIELD9_VALUES[6]}"$'\n'
  FIELD_LMP+=$'variable Tzy equal '"${FIELD9_VALUES[7]}"$'\n'
  FIELD_LMP+=$'variable Tzz equal '"${FIELD9_VALUES[8]}"$'\n'
  FIELD_LMP+=$'pair_style mff/torch '"$CUTOFF"$' cuda field9 v_Txx v_Txy v_Txz v_Tyx v_Tyy v_Tyz v_Tzx v_Tzy v_Tzz'
fi

PHYS_LMP=""
DUMP_FREQ=10
THERMO_FREQ=20
if [[ "$STEPS" =~ ^[0-9]+$ ]]; then
  if (( STEPS > 0 && STEPS < DUMP_FREQ )); then DUMP_FREQ=$STEPS; fi
  if (( STEPS > 0 && STEPS < THERMO_FREQ )); then THERMO_FREQ=$STEPS; fi
fi
if [[ $DUMP_FREQ -le 0 ]]; then DUMP_FREQ=1; fi
if [[ $THERMO_FREQ -le 0 ]]; then THERMO_FREQ=1; fi
if [[ $TEST_PHYS_COMPUTE -eq 1 ]]; then
  PHYS_LMP+=$'compute mffg all mff/torch/phys global\n'
  PHYS_LMP+=$'compute mffgm all mff/torch/phys global/mask\n'
  PHYS_LMP+=$'compute mffd all mff/torch/phys global dipole\n'
  PHYS_LMP+=$'compute mffdx all mff/torch/phys global dipole x\n'
  PHYS_LMP+=$'compute mffp all mff/torch/phys global polarizability\n'
  PHYS_LMP+=$'compute mffpxx all mff/torch/phys global polarizability xx\n'
  PHYS_LMP+=$'compute mffa all mff/torch/phys atom\n'
  PHYS_LMP+=$'compute mffad all mff/torch/phys atom dipole\n'
  PHYS_LMP+=$'compute mffadx all mff/torch/phys atom dipole x\n'
  PHYS_LMP+=$'thermo_style custom step pe c_mffgm[2] c_mffgm[3] c_mffdx c_mffpxx\n'
  PHYS_LMP+=$'dump 1 all custom '"$DUMP_FREQ"$' dump.phys id type x y z c_mffadx c_mffad[1] c_mffad[2] c_mffad[3] c_mffa[1] c_mffa[2] c_mffa[3] c_mffa[4]\n'
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

$FIELD_LMP
pair_coeff * * $CORE_PT ${ELEMENTS[*]}

velocity all create 300 42
fix 1 all nve
$PHYS_LMP
thermo $THERMO_FREQ
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
        joined = ":".join(str(f) for f in so_files)
        print('export MFF_CUSTOM_OPS_LIB="{}"'.format(joined))
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
    print('export LD_LIBRARY_PATH="{}:${{LD_LIBRARY_PATH:-}}"'.format(dirs))
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

if [[ $TEST_PHYS_COMPUTE -eq 1 ]]; then
  if [[ ! -s "$OUT_DIR/dump.phys" ]]; then
    echo "ERROR: expected physical tensor dump file was not created: $OUT_DIR/dump.phys"
    exit 1
  fi
  echo "[3/3] Physical tensor dump written: $OUT_DIR/dump.phys"
fi

echo "DONE. out_dir=$OUT_DIR"

