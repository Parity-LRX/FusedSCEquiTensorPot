# FusedSCEquiTensorPot

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**FusedEquiSCTensorPot** is an E(3)-equivariant neural potential for predicting molecular energies and forces. Built with PyTorch and e3nn, it supports multiple tensor product modes including spherical harmonics and Cartesian implementations.

## ✨ Features

- **Multiple Tensor Product Modes**: 
  - `spherical`: e3nn-based spherical harmonics (strictly equivariant, default)
  - `partial-cartesian`: Cartesian tensor products with CG coefficients (strictly equivariant, -17.4% params)
  - `partial-cartesian-loose`: Optimized Cartesian tensor products (approximate equivariance, fastest)
  - `pure-cartesian-sparse`: Sparse pure Cartesian with δ/ε contractions (strictly equivariant, -29.6% params)
  - `pure-cartesian-ictd`: ICTD trace-chain invariants (strictly equivariant, -72.1% params, best for memory)
  
- **E(3)-Equivariant**: All modes maintain rotational equivariance and parity conservation
  
- **Complete Workflow**:
  - Data preprocessing from Extended XYZ format with PBC support
  - Training with dynamic loss weight adjustment, SWA, EMA support
  - Evaluation with detailed metrics
  - Molecular dynamics (MD) simulation via ASE
  - Nudged Elastic Band (NEB) calculations
  
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

Optional: clamp dynamic loss weights `a/b` (they change during training):

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

For molecular dynamics simulation:

```bash
mff-evaluate --checkpoint combined_model.pth --md-sim
```

For NEB (Nudged Elastic Band) calculations:

```bash
mff-evaluate --checkpoint combined_model.pth --neb
```

## Python API

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

## Project Structure

```
molecular_force_field/
├── models/              # Model definitions
│   ├── e3nn_layers.py  # E3NN-based layers
│   ├── mlp.py          # MLP networks
│   └── losses.py       # Loss functions
├── data/               # Dataset and preprocessing
│   ├── datasets.py     # Dataset classes
│   ├── collate.py      # Collate functions
│   └── preprocessing.py # Data preprocessing
├── utils/              # Utility functions
│   ├── graph_utils.py  # Graph operations
│   ├── tensor_utils.py # Tensor utilities
│   └── config.py       # Configuration management
├── training/           # Training utilities
│   ├── trainer.py      # Trainer class
│   └── schedulers.py   # Learning rate schedulers
├── evaluation/         # Evaluation utilities
│   ├── evaluator.py    # Static evaluation
│   └── calculator.py   # ASE Calculator wrapper
└── cli/                # Command-line interfaces
    ├── train.py        # Training CLI
    ├── evaluate.py     # Evaluation CLI
    └── preprocess.py   # Preprocessing CLI
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- e3nn >= 0.5.0
- ASE >= 3.22.0
- See `requirements.txt` for full list

## 🎯 Choosing Tensor Product Mode

The library supports **six tensor product modes**, each optimized for different use cases:

| Mode | Equivariance | Speed* | Parameters* | Equivariance Error* | Use Case |
|------|--------------|--------|-------------|---------------------|----------|
| `spherical` | ✅ Strict | 1.00x (baseline) | 100% (baseline) | 4.41e-07 | Default, maximum compatibility, research/publication |
| `partial-cartesian` | ✅ Strict | 1.08x | 82.6% (-17.4%) | 1.83e-07 | Strict equivariance with fewer parameters |
| `partial-cartesian-loose` | ⚠️ Approximate | **0.64x (fastest)** | 82.7% (-17.3%) | 6.52e-08 | Fast iteration, approximate equivariance acceptable |
| `pure-cartesian-sparse` | ✅ Strict | 1.02x | 70.4% (-29.6%) | 3.03e-07 | Best balance: fewer params, near-baseline speed |
| `pure-cartesian-ictd` | ✅ Strict | **0.67x (fastest)** | **27.9% (-72.1%)** | 1.08e-07 | **Best for memory**: fewest params, fast, strictly equivariant |
| `pure-cartesian` | ✅ Strict | 9.56x (slowest) | 514.0% (+414%) | 2.20e-07 | ❌ Not recommended (too slow, too many params) |

*Benchmark results on CPU, channels=64, lmax=2, 64 atoms, 512 edges. All modes pass O(3) equivariance tests (including parity/reflection, error < 1e-6).

### Quick Recommendations

- **First time / Research**: Use `spherical` (default)
- **Memory constrained**: Use `pure-cartesian-ictd` (72.1% fewer parameters, 0.67x speed, fastest)
- **Speed priority**: Use `partial-cartesian-loose` (0.64x, fastest, but approximate equivariance)
- **Best balance**: Use `pure-cartesian-sparse` (29.6% fewer params, 1.02x speed, strictly equivariant)
- **Strict equivariance + fewer params**: Use `partial-cartesian` or `pure-cartesian-sparse`

For detailed performance comparison and recommendations, see [USAGE.md](USAGE.md#张量积模式对比).

## 📚 Documentation

For full CLI and hyperparameter documentation, see [USAGE.md](USAGE.md).


## 📄 License

MIT License

## 🙏 Acknowledgments

- Built on [e3nn](https://github.com/e3nn/e3nn) for equivariant neural networks
- Uses [ASE](https://wiki.fysik.dtu.dk/ase/) for molecular simulations
- Inspired by NequIP, MACE, and other equivariant neural potentials

## Citation

If you use this library in your research, please cite:

```bibtex
@software{fused_sc_equitensorpot,
  title = {FusedEquiSCTensorPot},
  version = {0.1.0},
  url = {https://github.com/Parity-LRX/FusedSCEquiTensorPot}
}
```

