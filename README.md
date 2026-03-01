# FusedSCEquiTensorPot

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**FusedSCEquiTensorPot** is an E(3)-equivariant neural potential for predicting molecular energies and forces. Built with PyTorch, it supports **eight equivariant tensor product modes**, including e3nn-based spherical harmonics, channelwise spherical backends, and multiple self-implemented Cartesian tensor product methods.

## ✨ Features

- **Eight Equivariant Tensor Product Modes**: 
  - `spherical`: e3nn-based spherical harmonics (strictly equivariant, default, standard implementation)
  - `spherical-save`: channelwise edge convolution (e3nn backend; fewer params)
  - `spherical-save-cue`: channelwise edge convolution (cuEquivariance backend; optional dependency)
  - `partial-cartesian`: Cartesian tensor products with CG coefficients (strictly equivariant, -17.4% params)
  - `partial-cartesian-loose`: Optimized Cartesian tensor products (approximate equivariance, faster)
  - `pure-cartesian`: Pure Cartesian \(3^L\) representation (strictly equivariant, very slow, not recommended)
  - `pure-cartesian-sparse`: Sparse pure Cartesian with δ/ε contractions (strictly equivariant, -29.6% params)
  - `pure-cartesian-ictd`: ICTD irreps internal representation (strictly equivariant, -72.1% params, fastest, best for memory)
  
- **E(3)-Equivariant**: All modes maintain rotational equivariance and parity conservation
  
- **Complete Workflow**:
  - Data preprocessing from Extended XYZ format with PBC support
  - Training with dynamic loss weight adjustment, SWA, EMA, stress training
  - Evaluation: static metrics, MD, NEB, phonon spectrum
  - LAMMPS integration: LibTorch (USER-MFFTORCH), ML-IAP, fix external
  
- **CLI Commands**:
  - `mff-preprocess` - Data preprocessing
  - `mff-train` - Training
  - `mff-evaluate` - Evaluation (static/MD/NEB/phonon)
  - `mff-export-core` - Export LibTorch core.pt (USER-MFFTORCH)
  - `mff-lammps` - Generate LAMMPS fix external script
  - `python -m molecular_force_field.cli.export_mliap` - Export ML-IAP format
  - `torchrun -m molecular_force_field.cli.inference_ddp` - Large-scale multi-GPU inference (pure-cartesian-ictd only)
  
- **Easy to Use**:
  - Simple command-line interface
  - Python API for custom workflows
  - Automatic data preprocessing
  - Checkpoint management with mode detection
  
- **GPU Support**: Full CUDA acceleration for training and inference

## Installation

```bash
pip install -e .
```

Or install dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

### Recommended (Linux CUDA): pinned cu128 toolchain

Some dependencies (PyTorch CUDA wheels, PyG `torch_scatter` wheels) require pip flags like
`--index-url` / `-f`, which **cannot be enforced from `setup.py`**. For a reproducible setup
with cuEquivariance + PyG wheels, use the provided script:

```bash
bash scripts/install_pt271_cu128.sh
pip install -e .
```

### Optional: cuEquivariance backend (for `tensor_product_mode=spherical-save-cue`)

This project supports an additional channelwise spherical backend powered by NVIDIA **cuEquivariance**.

Install via extras (recommended):

```bash
pip install -e ".[cue]"
```

Or via requirements files:

```bash
pip install -r requirements.txt
pip install -r requirements-cue.txt
```

Notes:
- `cuequivariance-ops-torch-cu12` (CUDA kernels) is **Linux CUDA only**. On macOS you can still install `cuequivariance-torch` for CPU fallback.
- If you select `--tensor-product-mode spherical-save-cue` without the dependency installed, the CLI will raise a clear ImportError with install instructions.

### Optional: PyG (torch-scatter, torch-cluster)

Faster neighbor list and scatter operations. Install via extras:

```bash
pip install -e ".[pyg]"
```

## Quick Start

### 1. Data Preprocessing

Preprocess your Extended XYZ file:

```bash
mff-preprocess --input-file data.xyz --output-dir data --max-atom 40
```

This will:
- Extract data blocks from XYZ file
- Split into training and validation sets
- Fit baseline atomic energies
- Save preprocessed data to HDF5 and CSV formats
- Precompute neighbor lists and write `processed_{train,val}.h5` by default

To skip neighbor list preprocessing (for quick sanity-check):

```bash
mff-preprocess --input-file data.xyz --output-dir data --max-atom 40 --skip-h5
```

### 2. Training

Train a model (default: spherical mode):

```bash
mff-train --data-dir data --epochs 1000 --batch-size 8 --device cuda
```

Train with Cartesian mode (strictly equivariant):

```bash
mff-train --data-dir data --epochs 1000 --batch-size 8 --device cuda --tensor-product-mode partial-cartesian
```

Train with different tensor product modes:

```bash
# Partial-Cartesian (strictly equivariant, -17.4% params)
mff-train --data-dir data --epochs 1000 --batch-size 8 --device cuda --tensor-product-mode partial-cartesian

# Partial-Cartesian-Loose (fastest, approximate equivariance)
mff-train --data-dir data --epochs 1000 --batch-size 8 --device cuda --tensor-product-mode partial-cartesian-loose

# Pure-Cartesian-Sparse (strictly equivariant, -29.6% params)
mff-train --data-dir data --epochs 1000 --batch-size 8 --device cuda --tensor-product-mode pure-cartesian-sparse

# Pure-Cartesian-ICTD (strictly equivariant, -72.1% params, best for memory)
mff-train --data-dir data --epochs 1000 --batch-size 8 --device cuda --tensor-product-mode pure-cartesian-ictd
```

By default, dynamic loss weights `a/b` are clamped to `[1, 1000]` (they change during training). You can override the range:

```bash
mff-train --data-dir data --a 10.0 --b 100.0 --update-param 750 --weight-a-growth 1.05 --weight-b-decay 0.98 --a-max 1000 --b-min 1 --b-max 1000 
```

Optional: override baseline atomic energies (E0):

```bash
# from CSV (Atom,E0)
mff-train --data-dir data --atomic-energy-file data/fitted_E0.csv

# or directly from CLI
mff-train --data-dir data --atomic-energy-keys 1 6 7 8 --atomic-energy-values -430.53 -821.03 -1488.19 -2044.35
```

### 3. Evaluation

Evaluate a trained model (use the same tensor-product-mode as training):

```bash
# For spherical mode (default)
mff-evaluate --checkpoint combined_model.pth --test-prefix test --output-prefix test --tensor-product-mode spherical --use-h5

# For partial-cartesian mode
mff-evaluate --checkpoint combined_model.pth --test-prefix test --output-prefix test --tensor-product-mode partial-cartesian --use-h5

# For partial-cartesian-loose mode
mff-evaluate --checkpoint combined_model.pth --test-prefix test --output-prefix test --tensor-product-mode partial-cartesian-loose --use-h5

# For pure-cartesian-sparse mode
mff-evaluate --checkpoint combined_model.pth --test-prefix test --output-prefix test --tensor-product-mode pure-cartesian-sparse --use-h5

# For pure-cartesian-ictd mode
mff-evaluate --checkpoint combined_model.pth --test-prefix test --output-prefix test --tensor-product-mode pure-cartesian-ictd --use-h5
```

Outputs include:
- `test_loss.csv`
- `test_energy.csv`
- `test_force.csv`

Optional: use `--compile e3trans` to accelerate evaluation with `torch.compile`.

For molecular dynamics simulation:

```bash
mff-evaluate --checkpoint combined_model.pth --md-sim
```

For NEB (Nudged Elastic Band) calculations:

```bash
mff-evaluate --checkpoint combined_model.pth --neb
```

For phonon spectrum (Hessian, vibrational frequencies):

```bash
mff-evaluate --checkpoint combined_model.pth --phonon --phonon-input structure.xyz
```

Optional: stress training (PBC with stress/virial in XYZ):

```bash
mff-train --data-dir data -c 0.1 --input-file pbc_with_stress.xyz
```

## LAMMPS Integration

FusedSCEquiTensorPot supports three LAMMPS integration methods:

| Method | Speed | Requirements | Use Case |
|--------|-------|---------------|----------|
| **USER-MFFTORCH (LibTorch pure C++)** | Fastest, no Python/GIL | LAMMPS built with KOKKOS + USER-MFFTORCH | HPC, clusters, production |
| **ML-IAP unified** | Faster (~1.7x vs fix external) | LAMMPS built with ML-IAP | Recommended, GPU support |
| **fix external / pair_style python** | Slower | Standard LAMMPS + Python | Quick validation, no ML-IAP |

### LibTorch Interface (USER-MFFTORCH, HPC Recommended)

**USER-MFFTORCH** loads TorchScript models via LibTorch C++ API directly. **No Python at runtime**, suitable for HPC and production deployment.

1. **Export core.pt** (one-time, requires Python):
   ```bash
   mff-export-core --checkpoint model.pth --elements H O --device cuda \
     --max-radius 5.0 --embed-e0 --e0-csv fitted_E0.csv --out core.pt
   ```

2. **Build LAMMPS**: Enable `PKG_KOKKOS` and `PKG_USER-MFFTORCH`. See [lammps_user_mfftorch/docs/BUILD_AND_RUN.md](lammps_user_mfftorch/docs/BUILD_AND_RUN.md).

3. **Run** (pure LAMMPS, no Python):
   ```bash
   lmp -k on g 1 -sf kk -pk kokkos newton off neigh full -in in.mfftorch
   ```

**LAMMPS input example**:
```lammps
pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/core.pt H O
```

**Model support**: `pure-cartesian-ictd` series and `spherical-save-cue` only.

### ML-IAP Interface

Export ML-IAP format (requires LAMMPS built with ML-IAP):

```bash
python -m molecular_force_field.cli.export_mliap checkpoint.pth --elements H O \
  --atomic-energy-keys 1 8 --atomic-energy-values -13.6 -75.0 --max-radius 5.0 --output model-mliap.pt
```

Supported models: `spherical`, `spherical-save`, `spherical-save-cue`, `pure-cartesian-ictd`, `pure-cartesian-ictd-save`.

See [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md) for full documentation.

## Python API

The library supports **eight tensor product modes**. Here's how to use them in Python:

### Basic Usage (Spherical Mode - Default)

```python
from molecular_force_field.models import E3_TransformerLayer_multi, MainNet
from molecular_force_field.data import H5Dataset
from molecular_force_field.data.collate import collate_fn_h5
from molecular_force_field.training.trainer import Trainer
from molecular_force_field.utils.config import ModelConfig
from torch.utils.data import DataLoader
import torch

# Load dataset
train_dataset = H5Dataset('train')
val_dataset = H5Dataset('val')

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn_h5
)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = ModelConfig()

model = MainNet(
    input_size=config.input_dim_weight,
    hidden_sizes=config.main_hidden_sizes4,
    output_size=1
).to(device)

# Spherical mode (default, e3nn-based)
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

# Train
trainer = Trainer(
    model=model,
    e3trans=e3trans,
    train_loader=train_loader,
    val_loader=val_loader,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    device=device,
    config=config,
)

trainer.run_training()
```

### Using Different Tensor Product Modes

```python
from molecular_force_field.models import (
    E3_TransformerLayer_multi,           # spherical mode
    CartesianTransformerLayer,           # partial-cartesian mode
    CartesianTransformerLayerLoose,      # partial-cartesian-loose mode
    PureCartesianTransformerLayer,       # pure-cartesian mode
    PureCartesianSparseTransformerLayer, # pure-cartesian-sparse mode
    PureCartesianICTDTransformerLayer,  # pure-cartesian-ictd mode
    MainNet
)
from molecular_force_field.utils.config import ModelConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = ModelConfig()

# Choose tensor product mode
tensor_product_mode = "pure-cartesian-ictd"  # Options: spherical, spherical-save, spherical-save-cue,
                                              # partial-cartesian, partial-cartesian-loose,
                                              # pure-cartesian, pure-cartesian-sparse,
                                              # pure-cartesian-ictd, pure-cartesian-ictd-save

if tensor_product_mode == 'spherical':
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
elif tensor_product_mode == 'partial-cartesian':
    e3trans = CartesianTransformerLayer(
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
        device=device
    ).to(device)
elif tensor_product_mode == 'partial-cartesian-loose':
    e3trans = CartesianTransformerLayerLoose(
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
        device=device
    ).to(device)
elif tensor_product_mode == 'pure-cartesian-sparse':
    e3trans = PureCartesianSparseTransformerLayer(
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
        max_rank_other=1,  # Restrict to rank≤1 interactions
        k_policy='k0',     # Delta contraction policy
        device=device
    ).to(device)
elif tensor_product_mode == 'pure-cartesian-ictd':
    e3trans = PureCartesianICTDTransformerLayer(
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
        ictd_tp_path_policy='full',  # Path pruning: 'full' or 'max_rank_other'
        ictd_tp_max_rank_other=None, # Max rank for sparse paths (if path_policy='max_rank_other')
        device=device
    ).to(device)
elif tensor_product_mode == 'pure-cartesian':
    e3trans = PureCartesianTransformerLayer(
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
        device=device
    ).to(device)

# Initialize main network
model = MainNet(
    input_size=config.input_dim_weight,
    hidden_sizes=config.main_hidden_sizes4,
    output_size=1
).to(device)

# Continue with training...
```

## Project Structure

```
rebuild/
├── molecular_force_field/         # Main package
│   ├── models/                    # Model definitions (eight tensor product modes)
│   │   ├── e3nn_layers.py        # Spherical mode (e3nn-based)
│   │   ├── cartesian_e3_layers.py    # Partial-cartesian modes
│   │   ├── pure_cartesian*.py    # Pure-cartesian modes
│   │   ├── mlp.py, losses.py
│   │   └── ...
│   ├── data/                      # Dataset and preprocessing
│   ├── utils/                     # Configuration, graph utilities
│   ├── training/                  # Trainer
│   ├── evaluation/                # Evaluator, ASE Calculator
│   ├── interfaces/                # LAMMPS potential, ML-IAP
│   │   ├── lammps_potential.py   # fix external / pair_style python
│   │   └── lammps_mliap.py      # ML-IAP unified
│   └── cli/                       # Command-line interfaces
│       ├── train.py, evaluate.py, preprocess.py
│       ├── lammps_interface.py   # fix external interface
│       ├── export_libtorch_core.py  # Export LibTorch core.pt
│       ├── export_mliap.py        # Export ML-IAP format
│       └── inference_ddp.py      # Large-scale multi-GPU inference
└── lammps_user_mfftorch/          # LAMMPS LibTorch package (USER-MFFTORCH)
    ├── src/USER-MFFTORCH/         # pair_style mff/torch source
    └── docs/BUILD_AND_RUN.md      # Build and run guide
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- e3nn >= 0.5.0
- ASE >= 3.22.0
- See `requirements.txt` for full list

## 🎯 Choosing Tensor Product Mode

The library supports **eight equivariant tensor product modes**, each optimized for different use cases:

1. **`spherical`**: e3nn-based spherical harmonics (default, standard implementation)
2. **`spherical-save`**: channelwise edge conv (e3nn backend; fewer params)
3. **`spherical-save-cue`**: channelwise edge conv (cuEquivariance backend; optional, GPU accelerated)
4. **`partial-cartesian`**: Cartesian coordinates + CG coefficients (strictly equivariant)
5. **`partial-cartesian-loose`**: Approximate equivariant (norm product approximation)
6. **`pure-cartesian`**: Pure Cartesian \(3^L\) representation (strictly equivariant, very slow)
7. **`pure-cartesian-sparse`**: Sparse pure Cartesian (strictly equivariant, parameter-optimized)
8. **`pure-cartesian-ictd`**: ICTD irreps internal representation (strictly equivariant, fastest, fewest parameters)

All modes maintain O(3) equivariance (including rotation and reflection). Performance comparison:

| Mode | Equivariance | Speed (CPU)* | Speed (GPU)** | Parameters* | Equivariance Error* | Use Case |
|------|--------------|-------------|---------------|-------------|---------------------|----------|
| `spherical` | ✅ Strict | 1.00x (baseline) | 1.00x (baseline) | 100% (baseline) | ~1e-15 | Default, maximum compatibility, research/publication |
| `partial-cartesian` | ✅ Strict | 0.16x-1.06x | 0.75x (lmax=2) | 82.6% (-17.4%) | ~1e-14 | Strict equivariance with fewer parameters |
| `partial-cartesian-loose` | ⚠️ Approximate | 0.17x-1.37x | 1.15x (lmax=2) | 82.7% (-17.3%) | ~1e-15 | Fast iteration (CPU, lmax≤3), approximate equivariance acceptable |
| `pure-cartesian-sparse` | ✅ Strict | 0.53x-1.39x | **1.17x (lmax=2)** | 70.4% (-29.6%) | ~1e-15 | Best balance: fewer params, stable performance |
| `pure-cartesian-ictd` | ✅ Strict | **1.58x-4.12x (fastest)** | **2.10x (lmax=2, fastest)** | **27.9% (-72.1%)** | ~1e-7 | **Best overall**: fewest params, fastest on CPU/GPU, strictly equivariant |
| `pure-cartesian` | ✅ Strict | 0.02x-0.36x (slowest) | 0.06x (lmax=2, fails at lmax≥4) | 514.0% (+414%) | ~1e-14 | ❌ Not recommended (too slow, too many params) |

*CPU benchmark: channels=64, lmax=0-6, 32 atoms, 256 edges, float64. Speed shown is total training time (forward+backward) acceleration ratio relative to spherical.  
**GPU benchmark: channels=64, lmax=0-6, 32 atoms, 256 edges, RTX 3090, float64. Speed shown is total training time (forward+backward) acceleration ratio relative to spherical.  
All modes pass O(3) equivariance tests (including parity/reflection, error < 1e-6).

### Quick Recommendations

#### CPU Environment (Recommended)
- **Speed + Memory**: Use `pure-cartesian-ictd` (**1.58x-4.12x faster**, 72.1% fewer parameters, all lmax)
- **High Precision**: Use `spherical` or `pure-cartesian-sparse` (equivariance error ~1e-15)
- **Best Balance**: Use `pure-cartesian-sparse` (0.53x-1.39x, 29.6% fewer params, strict equivariance)
- **Standard Baseline**: Use `spherical` (highest precision, standard implementation)

#### GPU Environment (Recommended for Training)
- **Speed + Memory**: Use `pure-cartesian-ictd` (**2.10x faster**, 72.1% fewer parameters, lmax≤3)
- **High Precision**: Use `spherical` or `pure-cartesian-sparse` (equivariance error ~1e-15)
- **Best Balance**: Use `pure-cartesian-sparse` (**1.17x faster**, 29.6% fewer params, strict equivariance)
- **Avoid**: `pure-cartesian` (too slow, fails at lmax≥4)

For detailed performance comparison and recommendations, see [USAGE.md](USAGE.md#tensor-product-mode-comparison).

### Real-World Task Performance

**Dataset**: Five nitrogen oxide and carbon structure reaction pathways from NEB (Nudged Elastic Band) calculations, filtered to fmax=0.2, totaling 2,788 structures. Test set: 1-2 complete or incomplete structures per reaction.

**Test Configuration**: 64 channels, lmax=2, float64

<table>
<thead>
<tr>
<th style="text-align:center">Method</th>
<th style="text-align:center">Configuration</th>
<th style="text-align:center">Mode</th>
<th style="text-align:center">Energy RMSE<br/>(mev/atom)</th>
<th style="text-align:center">Force RMSE<br/>(mev/Å)</th>
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

**Key Findings**:
- **Energy Accuracy**: FSCETP achieves **66.2% lower** energy RMSE than MACE (64ch) (0.044 vs 0.13 mev/atom)
- **Force Accuracy**: FSCETP achieves **36.2% lower** force RMSE than MACE (64ch) (7.4 vs 11.6 mev/Å)
- **Best Performance**: `spherical` and `partial-cartesian` modes show the best accuracy (Energy: 0.044-0.045, Force: 7.4)
- **Efficiency**: `pure-cartesian-ictd` achieves competitive accuracy (Energy: 0.046, Force: 9.0) with **72.1% fewer parameters** and **2.10x faster** training speed

## 📚 Documentation

- [USAGE.md](USAGE.md) - Full CLI and hyperparameter reference (Chinese)
- [USAGE_EN.md](USAGE_EN.md) - Full CLI and hyperparameter reference (English)
- [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md) - LAMMPS integration guide (LibTorch, ML-IAP, fix external)
- [lammps_user_mfftorch/docs/BUILD_AND_RUN.md](lammps_user_mfftorch/docs/BUILD_AND_RUN.md) - LibTorch interface build and run


## 📄 License

MIT License

## 🙏 Acknowledgments

This framework implements **eight equivariant tensor product modes**:
- **`spherical` and `spherical-save` modes**: Built on [e3nn](https://github.com/e3nn/e3nn) for spherical harmonics-based tensor products
- **`spherical-save-cue` mode**: Uses [cuEquivariance](https://github.com/NVIDIA/cuEquivariance) for GPU-accelerated channelwise spherical convolution
- **`partial-cartesian` and `partial-cartesian-loose` modes**: Partially use e3nn's Clebsch-Gordan coefficients and Irreps framework
- **Self-implemented Cartesian modes**: `pure-cartesian`, `pure-cartesian-sparse`, `pure-cartesian-ictd` are independently implemented without e3nn dependencies

Other dependencies and inspirations:
- Uses [ASE](https://wiki.fysik.dtu.dk/ase/) for molecular simulations
- Inspired by NequIP, MACE, and other equivariant neural potentials

## Citation

If you use this library in your research, please cite:

```bibtex
@software{fused_sc_equitensorpot,
  title = {FusedSCEquiTensorPot},
  version = {0.1.0},
  url = {https://github.com/Parity-LRX/FusedSCEquiTensorPot}
}
```

