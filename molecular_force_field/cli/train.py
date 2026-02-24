"""Command-line interface for training."""

import argparse
import os
import random
import torch
import torch.distributed as dist
import numpy as np
import logging
import time
from datetime import timedelta
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from logging.handlers import RotatingFileHandler

from molecular_force_field.models import (
    E3_TransformerLayer_multi,
    CartesianTransformerLayer,
    CartesianTransformerLayerLoose,
    PureCartesianTransformerLayer,
    PureCartesianSparseTransformerLayer,
    PureCartesianICTDTransformerLayer,
    MainNet,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDTransformerLayerFull,
)
from molecular_force_field.models.e3nn_layers_channelwise import (
    E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
)
from molecular_force_field.data import H5Dataset
from molecular_force_field.data.collate import collate_fn_h5
from molecular_force_field.training.trainer import Trainer
from molecular_force_field.data.preprocessing import save_to_h5_parallel
from molecular_force_field.utils.config import ModelConfig


def check_and_preprocess_data(data_dir, train_prefix, val_prefix, max_radius, num_workers,
                              input_file=None, train_input_file=None, valid_input_file=None, seed=42):
    """
    Ensure train/val data exist: use existing H5, run H5-only preprocessing, or full pipeline from XYZ.

    Data resolution order:
        1. If processed_<prefix>.h5 exist in data_dir -> use as-is.
        2. If read_<prefix>.h5 exist -> run H5 preprocessing only.
        3. If train_input_file and valid_input_file are both given -> use as train/val, no split; run full preprocessing.
        4. If input_file is given -> split 90/10 train/val, then run full preprocessing.
        5. Otherwise -> return False and log error.

    Args:
        data_dir: Directory for data files (and fitted_E0.csv).
        train_prefix: Filename prefix for training set (e.g. 'train' -> processed_train.h5).
        val_prefix: Filename prefix for validation set (e.g. 'val' -> processed_val.h5).
        max_radius: Max radius for neighbor search in H5 preprocessing.
        num_workers: Number of workers for H5 preprocessing.
        input_file: Optional single XYZ path; triggers 90/10 train/val split then preprocessing.
        train_input_file: Optional training XYZ path; must be used together with valid_input_file (no split).
        valid_input_file: Optional validation XYZ path; must be used together with train_input_file.
        seed: Random seed for train/val split when input_file is used (default: 42).

    Returns:
        True if train/val data are ready, False otherwise.
    """
    train_processed = os.path.join(data_dir, f'processed_{train_prefix}.h5')
    val_processed = os.path.join(data_dir, f'processed_{val_prefix}.h5')
    train_raw = os.path.join(data_dir, f'read_{train_prefix}.h5')
    val_raw = os.path.join(data_dir, f'read_{val_prefix}.h5')

    # Case 1: Preprocessed train/val H5 already exist in data_dir
    if os.path.exists(train_processed) and os.path.exists(val_processed):
        logging.info(f"Found preprocessed data in {data_dir}/")
        return True

    # Case 2: Raw read_* H5 exist; run H5 preprocessing only (no XYZ extraction)
    if os.path.exists(train_raw) and os.path.exists(val_raw):
        logging.info(f"Found raw data in {data_dir}/, running H5 preprocessing...")
        save_to_h5_parallel(train_prefix, max_radius, num_workers, data_dir=data_dir)
        save_to_h5_parallel(val_prefix, max_radius, num_workers, data_dir=data_dir)
        return True

    # Case 3a: Two XYZ files given (train + valid); use as-is, no split; run full preprocessing
    if train_input_file and valid_input_file and os.path.exists(train_input_file) and os.path.exists(valid_input_file):
        logging.info(f"Using specified train/valid files (no auto split): train={train_input_file}, valid={valid_input_file}")
        from molecular_force_field.data.preprocessing import (
            extract_data_blocks,
            fit_baseline_energies,
            compute_correction,
            save_set
        )
        import pandas as pd
        import numpy as np
        os.makedirs(data_dir, exist_ok=True)
        train_blocks, train_energy, train_raw_energy, train_cells, train_pbcs, train_stresses = extract_data_blocks(train_input_file)
        val_blocks, val_energy, val_raw_energy, val_cells, val_pbcs, val_stresses = extract_data_blocks(valid_input_file)
        logging.info(f"Train frames: {len(train_blocks)}, Valid frames: {len(val_blocks)}")
        train_atoms = []
        for block in train_blocks:
            train_atoms.extend([int(row[3]) for row in block])
        uniq = sorted({a for a in train_atoms if a > 0})
        if not uniq:
            raise ValueError("No valid atomic numbers found in training blocks; cannot fit baseline energies.")
        keys = np.asarray(uniq, dtype=np.int64)
        initial_values = np.array([-0.01] * len(keys), dtype=np.float64)
        fitted_values = fit_baseline_energies(train_blocks, train_raw_energy, keys, initial_values)
        fitted_e0_path = os.path.join(data_dir, 'fitted_E0.csv')
        pd.DataFrame({'Atom': keys, 'E0': fitted_values}).to_csv(fitted_e0_path, index=False)
        logging.info(f"Saved {fitted_e0_path}")
        train_correction = compute_correction(train_blocks, train_raw_energy, keys, fitted_values)
        val_correction = compute_correction(val_blocks, val_raw_energy, keys, fitted_values)
        save_set(train_prefix, np.arange(len(train_blocks)), train_blocks, train_raw_energy,
                 train_correction, train_cells, pbc_list=train_pbcs, stress_list=train_stresses, output_dir=data_dir)
        save_set(val_prefix, np.arange(len(val_blocks)), val_blocks, val_raw_energy,
                 val_correction, val_cells, pbc_list=val_pbcs, stress_list=val_stresses, output_dir=data_dir)
        save_to_h5_parallel(train_prefix, max_radius, num_workers, data_dir=data_dir)
        save_to_h5_parallel(val_prefix, max_radius, num_workers, data_dir=data_dir)
        logging.info(f"Preprocessing completed! Data saved to {data_dir}/")
        return True

    # Case 3b: Single XYZ file given; 90/10 train/val split then full preprocessing
    if input_file and os.path.exists(input_file):
        logging.info(f"No preprocessed data found. Running preprocessing on {input_file} (auto 90/10 split)...")
        from molecular_force_field.data.preprocessing import (
            extract_data_blocks,
            fit_baseline_energies,
            compute_correction,
            save_set
        )
        import numpy as np
        import pandas as pd

        os.makedirs(data_dir, exist_ok=True)
        all_blocks, all_energy, all_raw_energy, all_cells, all_pbcs, all_stresses = extract_data_blocks(input_file)
        logging.info(f"Total frames: {len(all_blocks)}")

        # 90/10 train/val split (fixed ratio)
        data_size = len(all_blocks)
        indices = np.arange(data_size)
        np.random.seed(seed)
        train_size = int(0.90 * data_size)
        val_size = data_size - train_size
        val_indices = np.random.choice(indices, size=val_size, replace=False)
        train_mask = ~np.isin(indices, val_indices)
        train_indices = indices[train_mask]
        logging.info(f"Split: {len(train_indices)} Train, {len(val_indices)} Val")

        train_blocks = [all_blocks[i] for i in train_indices]
        train_raw_E = [all_raw_energy[i] for i in train_indices]
        val_blocks = [all_blocks[i] for i in val_indices]
        val_raw_E = [all_raw_energy[i] for i in val_indices]

        # Fit E0 on train set only; keys derived from train atoms (no hardcoding)
        train_atoms = []
        for block in train_blocks:
            train_atoms.extend([int(row[3]) for row in block])  # row[3] = atomic number
        uniq = sorted({a for a in train_atoms if a > 0})
        if not uniq:
            raise ValueError("No valid atomic numbers found in training blocks; cannot fit baseline energies.")
        keys = np.asarray(uniq, dtype=np.int64)
        initial_values = np.array([-0.01] * len(keys), dtype=np.float64)
        fitted_values = fit_baseline_energies(train_blocks, train_raw_E, keys, initial_values)

        fitted_e0_path = os.path.join(data_dir, 'fitted_E0.csv')
        pd.DataFrame({'Atom': keys, 'E0': fitted_values}).to_csv(fitted_e0_path, index=False)
        logging.info(f"Saved {fitted_e0_path}")

        train_correction = compute_correction(train_blocks, train_raw_E, keys, fitted_values)
        val_correction = compute_correction(val_blocks, val_raw_E, keys, fitted_values)
        save_set('train', train_indices, train_blocks, train_raw_E, train_correction, all_cells, pbc_list=all_pbcs, stress_list=all_stresses, output_dir=data_dir)
        save_set('val', val_indices, val_blocks, val_raw_E, val_correction, all_cells, pbc_list=all_pbcs, stress_list=all_stresses, output_dir=data_dir)
        save_to_h5_parallel('train', max_radius, num_workers, data_dir=data_dir)
        save_to_h5_parallel('val', max_radius, num_workers, data_dir=data_dir)
        logging.info(f"Preprocessing completed! Data saved to {data_dir}/")
        return True

    # Case 4: No data and no input file; cannot proceed
    logging.error(
        f"No data files found in {data_dir}/ and no input file specified.\n"
        f"Please either:\n"
        f"  1. Run 'mff-preprocess --input-file <xyz_file> --output-dir {data_dir}' first, or\n"
        f"  2. Use '--input-file <xyz_file>' to automatically preprocess data."
    )
    return False


def setup_logging():
    """Configure logging: console (filtered) and rotating file (full)."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    class ConsoleFilter(logging.Filter):
        """Suppress per-epoch training logs on console; keep validation and lifecycle messages."""
        def filter(self, record):
            msg = record.getMessage()
            if 'Epoch' in msg and 'Validation' not in msg and 'Training started' not in msg and 'Training completed' not in msg and 'Early stopping' not in msg:
                return False
            return True

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(ConsoleFilter())
    
    log_filename = f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = RotatingFileHandler(
        filename=log_filename,
        maxBytes=1000 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train molecular force field model')
    parser.add_argument('--train-prefix', type=str, default='train',
                        help='Prefix for training data files')
    parser.add_argument('--val-prefix', type=str, default='val',
                        help='Prefix for validation data files')
    parser.add_argument('--max-radius', type=float, default=5.0,
                        help='Maximum radius for neighbor search')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--min-learning-rate', type=float, default=2e-5,
                        help='Minimum learning rate')
    parser.add_argument('--warmup-batches', type=int, default=1000,
                        help='Number of warmup batches for learning rate (default: 1000)')
    parser.add_argument('--lr-decay-patience', type=int, default=1000,
                        help='Patience (in batches) before learning rate decay (default: 1000)')
    parser.add_argument('--lr-decay-factor', type=float, default=0.98,
                        help='Learning rate decay factor (default: 0.98)')
    parser.add_argument('--warmup-start-ratio', type=float, default=0.1,
                        help='Starting learning rate ratio during warmup (0.1 means start at 10%% of target LR, default: 0.1)')
    parser.add_argument('--checkpoint', type=str, default='combined_model.pth',
                        help='Checkpoint path')
    parser.add_argument('--reset-loss-weights', action='store_true', default=False,
                        help='When loading checkpoint, ignore saved loss weights (a, b) and use '
                             'values from command line arguments (--a, --b) instead. '
                             'Default: False (use checkpoint weights)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for data preprocessing')
    parser.add_argument('--mp-context', type=str, default='auto',
                        choices=['auto', 'fork', 'spawn'],
                        help='Multiprocessing start method for DataLoader workers. '
                             '"auto" forces "spawn" when validation compile is enabled (safer with CUDA/compile), '
                             'otherwise uses the default OS method (often "fork" on Linux, faster).')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float32', 'float64', 'float', 'double'],
                        help='Default dtype for tensors (float32 or float64, default: float64)')
    parser.add_argument('--matmul-precision', type=str, default='high',
                        choices=['highest', 'high', 'medium'],
                        help='Float32 matmul precision. "high" (default) enables TF32 on Ampere+ GPUs for ~2x matmul speedup. Use "highest" for strict FP32.')
    parser.add_argument('--dump-frequency', type=int, default=250,
                        help='Frequency (in batches) for validation and model saving (default: 250)')
    parser.add_argument('--train-eval-sample-ratio', type=float, default=0.2,
                        help='Ratio of training set to evaluate during validation (0.0-1.0, default: 0.2). '
                             'Set to 1.0 for full evaluation, or lower (e.g., 0.2 for 20%%) for faster validation. '
                             'Useful for large datasets.')
    parser.add_argument('--energy-log-frequency', type=int, default=100,
                        help='Frequency (in batches) to log energy predictions (default: 100)')
    parser.add_argument('--energy-weight', '-a', type=float, default=1.0,
                        help='Initial weight for energy loss (default: 1.0)')
    parser.add_argument('--force-weight', '-b', type=float, default=10.0,
                        help='Initial weight for force loss (default: 10.0)')
    parser.add_argument('--update-param', type=int, default=1000,
                        help='Interval (in batches) to update loss weights a and b (default: 1000)')
    parser.add_argument('--weight-a-growth', type=float, default=1.05,
                        help='Growth factor for energy weight a at each update (default: 1.05, meaning 5%% growth). '
                             'Recommended: 1.005 (slow), 1.01 (medium), 1.02 (fast), 1.05 (very fast)')
    parser.add_argument('--weight-b-decay', type=float, default=0.98,
                        help='Decay factor for force weight b at each update (default: 0.98, meaning 2%% decay). '
                             'Recommended: 0.995 (slow), 0.99 (medium), 0.98 (fast)')
    parser.add_argument('--a-min', type=float, default=1.0,
                        help='Minimum clamp for dynamic energy weight a (default: 1.0).')
    parser.add_argument('--a-max', type=float, default=1000.0,
                        help='Maximum clamp for dynamic energy weight a (default: 1000.0).')
    parser.add_argument('--b-min', type=float, default=1.0,
                        help='Minimum clamp for dynamic force weight b (default: 1.0).')
    parser.add_argument('--b-max', type=float, default=1000.0,
                        help='Maximum clamp for dynamic force weight b (default: 1000.0).')
    parser.add_argument('--swa-start-epoch', type=int, default=None,
                        help='Epoch to start SWA (Stochastic Weight Averaging) for loss weights. '
                             'After this epoch, a and b will be set to --swa-a and --swa-b values directly. '
                             'If not set, continuous linear growth/decay will be used (default: None).')
    parser.add_argument('--swa-a', type=float, default=None,
                        help='Energy weight a after SWA starts (default: None, must be set if --swa-start-epoch is set)')
    parser.add_argument('--swa-b', type=float, default=None,
                        help='Force weight b after SWA starts (default: None, must be set if --swa-start-epoch is set)')
    parser.add_argument('--ema-start-epoch', type=int, default=None,
                        help='Epoch to start EMA (Exponential Moving Average) for e3trans weights. '
                             'If not set, EMA is disabled (default: None). Recommended: start at ~60%%-80%% of total epochs.')
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay factor in (0, 1). Larger -> smoother but slower (default: 0.999).')
    parser.add_argument('--use-ema-for-validation', action='store_true',
                        help='Use EMA weights for validation forward pass (default: False).')
    parser.add_argument('--save-ema-model', action='store_true',
                        help='Save EMA model weights into checkpoints (default: False).')
    parser.add_argument('--save-val-csv', action='store_true', default=False,
                        help='Save validation energy and force predictions to CSV files (default: True). '
                             'Files: val_energy_epoch{N}_batch{M}.csv and val_force_epoch{N}_batch{M}.csv')
    parser.add_argument('--no-save-val-csv', dest='save_val_csv', action='store_false',
                        help='Disable saving validation CSV files to reduce I/O overhead.')
    parser.add_argument('--log-val-batch-energy', action='store_true', default=False,
                        help='Log validation batch energy predictions to console (default: False). '
                             'If False, energy predictions are only logged to file, not console.')
    parser.add_argument('--force-shift-value', type=float, default=1.0,
                        help='Scaling factor for force labels (default: 1.0)')
    parser.add_argument('--stress-weight', '-c', type=float, default=0.0,
                        help='Weight for stress loss (default: 0.0, disabled). '
                             'Set > 0 to enable stress training via cell strain derivative. '
                             'Requires stress/virial data in the training XYZ files.')
    parser.add_argument('--c-min', type=float, default=0.0,
                        help='Minimum clamp for stress weight c (default: 0.0).')
    parser.add_argument('--c-max', type=float, default=1000.0,
                        help='Maximum clamp for stress weight c (default: 1000.0).')

    # Atomic reference energies (E0)
    parser.add_argument('--atomic-energy-file', type=str, default=None,
                        help='CSV file with columns Atom,E0 to load atomic reference energies. '
                             'If not set, defaults to <data-dir>/fitted_E0.csv (generated by least squares on train set).')
    parser.add_argument('--atomic-energy-keys', type=int, nargs='+', default=None,
                        help='Atomic numbers for custom atomic reference energies (must match --atomic-energy-values length). '
                             'Example: --atomic-energy-keys 1 6 7 8')
    parser.add_argument('--atomic-energy-values', type=float, nargs='+', default=None,
                        help='Atomic reference energies (E0) in eV corresponding to --atomic-energy-keys. '
                             'Example: --atomic-energy-values -430.53 -821.03 -1488.19 -2044.35')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience in epochs (default: 20)')
    parser.add_argument('--vhat-clamp-interval', type=int, default=2000,
                        help='Interval (in batches) to clamp v_hat (default: 2000)')
    parser.add_argument('--max-vhat-growth', type=float, default=5.0,
                        help='Maximum growth factor for v_hat (default: 5.0)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='Maximum gradient norm (default: 0.5)')
    parser.add_argument('--grad-log-interval', type=int, default=500,
                        help='Interval (in batches) to log gradient statistics (default: 100)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing preprocessed data files (default: data)')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Single XYZ file for automatic preprocessing; program will split 90/10 train/val (optional)')
    parser.add_argument('--train-input-file', type=str, default=None,
                        help='Training set XYZ file. If both --train-input-file and --valid-input-file are set, '
                             'use them as train/valid directly (no auto split).')
    parser.add_argument('--valid-input-file', type=str, default=None,
                        help='Validation set XYZ file. Use together with --train-input-file to specify train/valid datasets.')
    parser.add_argument('--train-data', type=str, default=None,
                        help='Path to preprocessed training H5 file (e.g. processed_train.h5). '
                             'If both --train-data and --valid-data are set, use these files and skip preprocessing.')
    parser.add_argument('--valid-data', type=str, default=None,
                        help='Path to preprocessed validation H5 file. Use together with --train-data.')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training (DDP)')
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Local rank for distributed training (set automatically by torchrun)')
    parser.add_argument('--backend', type=str, default='nccl',
                        choices=['nccl', 'gloo'],
                        help='Distributed backend (default: nccl for GPU, gloo for CPU)')
    parser.add_argument('--init-method', type=str, default='env://',
                        help='Distributed initialization method (default: env://)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Model architecture hyperparameters
    parser.add_argument('--max-atomvalue', type=int, default=10,
                        help='Maximum atomic number in atom embedding (default: 10)')
    parser.add_argument('--embedding-dim', type=int, default=16,
                        help='Atom embedding dimension (default: 16)')
    parser.add_argument('--embed-size', type=int, nargs='+', default=[128, 128, 128],
                        help='Hidden layer sizes for readout MLP (default: 128 128 128)')
    parser.add_argument('--output-size', type=int, default=8,
                        help='Output size for atom readout MLP (default: 8)')
    parser.add_argument('--lmax', type=int, default=2,
                        help='Maximum L value for spherical harmonics in irreps (default: 2). Controls the highest order of irreducible representations.')
    parser.add_argument('--irreps-output-conv-channels', type=int, default=None,
                        help='Number of channels for irreps_output_conv (e.g., 64 for lmax=2 gives "64x0e + 64x1o + 64x2e"). If not set, uses channel_in from config (default: 64)')
    parser.add_argument('--function-type', type=str, default='gaussian',
                        choices=['gaussian', 'bessel', 'fourier', 'cosine', 'smooth_finite'],
                        help='Basis function type for radial basis (default: gaussian). Options: gaussian, bessel, fourier, cosine, smooth_finite')
    parser.add_argument('--tensor-product-mode', type=str, default='spherical',
                        choices=['spherical', 'spherical-save', 'spherical-save-cue', 'partial-cartesian', 'partial-cartesian-loose', 'pure-cartesian', 'pure-cartesian-sparse', 'pure-cartesian-ictd', 'pure-cartesian-ictd-save'],
                        help='Tensor product mode: "spherical" uses e3nn spherical harmonics (default), '
                             '"spherical-save" uses channelwise edge convolution (e3nn backend; fewer params, same irreps), '
                             '"spherical-save-cue" uses channelwise edge convolution (cuEquivariance backend; requires cuequivariance-torch), '
                             '"partial-cartesian" uses Cartesian tensor products with EquivariantTensorProduct (strictly equivariant), '
                             '"partial-cartesian-loose" uses non-strictly-equivariant Cartesian tensor products (norm product approximation, not strictly equivariant), '
                             '"pure-cartesian" uses full rank Cartesian tensors (3^L) with delta/epsilon contractions (most pure), '
                             '"pure-cartesian-sparse" uses a sparse pure-cartesian delta/epsilon tensor product (O(3) strict) by restricting rank-rank paths, '
                             '"pure-cartesian-ictd" uses pure_cartesian_ictd_layers_full (ICTD, DDP supported). '
                             '"pure-cartesian-ictd-save" uses pure_cartesian_ictd_layers (original ICTD, same readout, DDP supported). '
                             'Note: ICTD inference is typically ~3x faster than spherical-save.')
    parser.add_argument('--max-rank-other', type=int, default=1,
                        help='Max rank for sparse tensor product in pure-cartesian-sparse mode (default: 1). '
                             'Only interactions where min(L1, L2) <= max_rank_other are allowed. '
                             'Larger values allow more interactions but increase parameters and computation.')
    parser.add_argument('--k-policy', type=str, default='k0',
                        choices=['k0', 'k1', 'both'],
                        help='K policy for sparse tensor product in pure-cartesian-sparse mode (default: k0). '
                             'k0: only k=0 (promotes higher rank), k1: only k=1 (contracts to lower rank), both: keep both')
    parser.add_argument('--num-interaction', type=int, default=2,
                        help='Number of message-passing steps (conv layers) per block (default: 2). '
                             'Used by: pure-cartesian, pure-cartesian-ictd, pure-cartesian-ictd-save, pure-cartesian-sparse, '
                             'partial-cartesian, partial-cartesian-loose, spherical, spherical-save, spherical-save-cue. Must be >= 2.')

    # ICTD path pruning controls (pure-cartesian-ictd and pure-cartesian-ictd-save)
    parser.add_argument('--ictd-tp-path-policy', type=str, default='full',
                        choices=['full', 'max_rank_other'],
                        help='Path policy for ICTD tensor products in pure-cartesian-ictd mode (default: full). '
                             'full: keep all CG-allowed (l1,l2->l3) paths; '
                             'max_rank_other: keep only paths with min(l1,l2) <= --ictd-tp-max-rank-other.')
    parser.add_argument('--ictd-tp-max-rank-other', type=int, default=None,
                        help='Used when --ictd-tp-path-policy=max_rank_other. '
                             'Keeps only paths with min(l1,l2) <= this value (e.g. 1 keeps scalar/vector couplings).')

    # Validation acceleration (evaluation-only; training uses double backward and is NOT compiled)
    parser.add_argument('--compile-val', type=str, default='none',
                        choices=['none', 'e3trans'],
                        help='Enable torch.compile during validation only. '
                             '"e3trans" compiles the eval forward used in validate(). '
                             'Training forward/backward is NOT compiled (double backward unsupported).')
    parser.add_argument('--compile-val-mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode for validation (default: reduce-overhead)')
    parser.add_argument('--compile-val-fullgraph', action='store_true',
                        help='Pass fullgraph=True to torch.compile for validation (may fail more often).')
    parser.add_argument('--compile-val-dynamic', action='store_true',
                        help='Pass dynamic=True to torch.compile for validation.')
    parser.add_argument('--compile-val-precache', action='store_true',
                        help='Run one eager forward on the first validation batch before compiling (recommended for ICTD).')
    
    args = parser.parse_args()

    # --- Dataset option validation (pairs must be both set or both unset) ---
    if (args.train_data is None) != (args.valid_data is None):
        raise ValueError("Must specify both --train-data and --valid-data together, or neither.")
    if (args.train_input_file is None) != (args.valid_input_file is None):
        raise ValueError("Must specify both --train-input-file and --valid-input-file together, or neither.")

    # --- Rank / world size (updated later if --distributed) ---
    rank = 0
    world_size = 1
    local_rank = 0

    # --- Loss weight bounds ---
    if args.a_min is not None and args.a_max is not None and args.a_min > args.a_max:
        raise ValueError("--a-min must be <= --a-max")
    if args.b_min is not None and args.b_max is not None and args.b_min > args.b_max:
        raise ValueError("--b-min must be <= --b-max")
    if args.num_interaction < 2:
        raise ValueError(f"--num-interaction must be >= 2, got {args.num_interaction}")
    
    # --- SWA (Stochastic Weight Averaging) ---
    if args.swa_start_epoch is not None:
        if args.swa_a is None or args.swa_b is None:
            raise ValueError("--swa-a and --swa-b must be set when --swa-start-epoch is set")
        if args.swa_start_epoch < 1:
            raise ValueError("--swa-start-epoch must be >= 1")
        if rank == 0:
            logging.info(f"SWA enabled: Will switch to a={args.swa_a}, b={args.swa_b} at epoch {args.swa_start_epoch}")

    # --- EMA (Exponential Moving Average) ---
    if args.ema_start_epoch is not None:
        if args.ema_start_epoch < 1:
            raise ValueError("--ema-start-epoch must be >= 1")
        if not (0.0 < args.ema_decay < 1.0):
            raise ValueError("--ema-decay must be in (0, 1)")
        if rank == 0:
            logging.info(f"EMA enabled: Will start at epoch {args.ema_start_epoch} with decay={args.ema_decay}")
            if args.use_ema_for_validation:
                logging.info("  Using EMA model for validation")
            if args.save_ema_model:
                logging.info("  Will save EMA model in checkpoint")
    
    # --- Random seed ---
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # TF32: faster float32 matmul on Ampere+ (removes inductor warning, ~2x matmul speed)
        torch.set_float32_matmul_precision(args.matmul_precision)
    
    # --- Distributed training (DDP) ---
    if args.distributed:
        if args.local_rank == -1:
            args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        
        if args.local_rank == -1:
            raise ValueError("--local-rank must be set when --distributed is enabled. "
                           "Use 'torchrun' or set LOCAL_RANK environment variable.")
        
        dist.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            timeout=timedelta(hours=2)
        )
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = args.local_rank
        
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
        
        if rank == 0:
            setup_logging()
            logging.info(f"Distributed training enabled: {world_size} GPUs")
            logging.info(f"Using device: {device} (rank {rank}, local_rank {local_rank})")
            logging.info(f"Data directory: {args.data_dir}")
        else:
            logging.basicConfig(level=logging.WARNING)
    else:
        setup_logging()

    # Log scatter backend (helps diagnose performance regressions when torch_scatter is broken).
    try:
        from molecular_force_field.utils.scatter import scatter_backend, require_torch_scatter

        if not args.distributed or rank == 0:
            logging.info("Scatter backend: %s", scatter_backend())

        # For cuEquivariance backend, torch_scatter is strongly recommended for speed.
        if args.tensor_product_mode == "spherical-save-cue":
            require_torch_scatter(reason="tensor_product_mode='spherical-save-cue' aims for maximum speed.")
    except Exception:
        pass

    # --- Default dtype ---
    if args.dtype == 'float64' or args.dtype == 'double':
        torch.set_default_dtype(torch.float64)
        if rank == 0:
            logging.info("Using dtype: float64")
    elif args.dtype == 'float32' or args.dtype == 'float':
        torch.set_default_dtype(torch.float32)
        if rank == 0:
            logging.info("Using dtype: float32")
    
    # --- Device (non-distributed) ---
    if not args.distributed:
        if args.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        
        logging.info(f"Using device: {device}")
        logging.info(f"Data directory: {args.data_dir}")
    
    # --- Data: resolve source (custom H5 paths vs data_dir + prefix) and preprocess if needed ---
    use_custom_data_paths = (args.train_data is not None and args.valid_data is not None)
    if use_custom_data_paths:
        if not os.path.exists(args.train_data):
            logging.error(f"Training data file not found: {args.train_data}")
            return
        if not os.path.exists(args.valid_data):
            logging.error(f"Validation data file not found: {args.valid_data}")
            return
        if rank == 0:
            logging.info(f"Using custom datasets: train={args.train_data}, valid={args.valid_data}")
        if args.distributed:
            dist.barrier()
    elif args.distributed:
        train_processed = os.path.join(args.data_dir, f'processed_{args.train_prefix}.h5')
        val_processed = os.path.join(args.data_dir, f'processed_{args.val_prefix}.h5')
        
        if not (os.path.exists(train_processed) and os.path.exists(val_processed)):
            if rank == 0:
                logging.error(
                    f"Preprocessed data not found in {args.data_dir}/\n"
                    f"In distributed mode, you must preprocess data first:\n"
                    f"  mff-preprocess --input-file {args.input_file or '<your_xyz_file>'} --output-dir {args.data_dir}\n"
                    f"Or run single-GPU training first (without --distributed) to auto-preprocess."
                )
            dist.destroy_process_group()
            return
        
        if rank == 0:
            logging.info(f"Found preprocessed data in {args.data_dir}/")
        dist.barrier()
    else:
        data_ready = check_and_preprocess_data(
            args.data_dir,
            args.train_prefix,
            args.val_prefix,
            args.max_radius,
            args.num_workers,
            input_file=args.input_file,
            train_input_file=args.train_input_file,
            valid_input_file=args.valid_input_file,
            seed=args.seed
        )
        if not data_ready:
            logging.error("Data preparation failed. Exiting.")
            return

    # --- Build datasets (from custom paths or data_dir + prefix) ---
    if use_custom_data_paths:
        train_dataset = H5Dataset('train', data_dir=args.data_dir, file_path=args.train_data)
        val_dataset = H5Dataset('val', data_dir=args.data_dir, file_path=args.valid_data)
    else:
        train_dataset = H5Dataset(args.train_prefix, data_dir=args.data_dir)
        val_dataset = H5Dataset(args.val_prefix, data_dir=args.data_dir)

    # --- DataLoaders and distributed samplers ---
    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_num_workers = max(1, args.num_workers // 2)
    val_num_workers = max(1, args.num_workers // 4)
    if args.mp_context == "spawn":
        mp_ctx = "spawn"
    elif args.mp_context == "fork":
        mp_ctx = "fork"
    else:
        # auto: only force spawn when validation compile is enabled on CUDA to avoid fork deadlocks
        mp_ctx = "spawn" if (args.compile_val != "none" and torch.cuda.is_available() and train_num_workers > 0) else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn_h5,
        num_workers=train_num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        multiprocessing_context=mp_ctx,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn_h5,
        num_workers=val_num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        multiprocessing_context=mp_ctx,
    )
    
    # --- Model config ---
    config_dtype = torch.float64 if args.dtype in ['float64', 'double'] else torch.float32
    config = ModelConfig(
        dtype=config_dtype,
        max_atomvalue=args.max_atomvalue,
        embedding_dim=args.embedding_dim,
        embed_size=args.embed_size,
        output_size=args.output_size,
        lmax=args.lmax,
        irreps_output_conv_channels=args.irreps_output_conv_channels,
        function_type=args.function_type,
        max_radius=args.max_radius
    )
    
    # --- Atomic reference energies (E0) ---
    if args.atomic_energy_keys is not None or args.atomic_energy_values is not None:
        if args.atomic_energy_keys is None or args.atomic_energy_values is None:
            raise ValueError("Both --atomic-energy-keys and --atomic-energy-values must be provided together.")
        if len(args.atomic_energy_keys) != len(args.atomic_energy_values):
            raise ValueError("--atomic-energy-keys and --atomic-energy-values must have the same length.")
        config.atomic_energy_keys = torch.tensor(args.atomic_energy_keys, dtype=torch.long)
        config.atomic_energy_values = torch.tensor(args.atomic_energy_values, dtype=config.dtype)
        if rank == 0:
            logging.info("Using custom atomic reference energies from CLI:")
            for k, v in zip(args.atomic_energy_keys, args.atomic_energy_values):
                logging.info(f"  Atom {k}: {v:.8f} eV")
    else:
        # Default behavior: load least-squares fitted E0 from fitted_E0.csv
        e0_path = args.atomic_energy_file or os.path.join(args.data_dir, 'fitted_E0.csv')
        config.load_atomic_energies_from_file(e0_path)
    
    # --- Log hyperparameters ---
    if rank == 0:
        logging.info("=" * 80)
        logging.info("Model Hyperparameters:")
        logging.info(f"  max_atomvalue: {config.max_atomvalue}")
        logging.info(f"  embedding_dim: {config.embedding_dim}")
        logging.info(f"  embed_size: {config.embed_size}")
        logging.info(f"  output_size: {config.output_size}")
        logging.info(f"  lmax: {config.lmax}")
        logging.info(f"  irreps_output_conv: {config.get_irreps_output_conv()}")
        logging.info(f"  function_type: {config.function_type}")
        logging.info(f"  max_radius: {config.max_radius}")
        logging.info(f"  dtype: {config.dtype}")
        logging.info("=" * 80)
    
    # Initialize models
    model = MainNet(
        input_size=config.input_dim_weight,
        hidden_sizes=config.main_hidden_sizes4,
        output_size=1
    ).to(device)
    
    # Initialize model based on tensor product mode
    if args.tensor_product_mode == 'pure-cartesian':
        logging.info("Using PURE Cartesian mode (rank tensors 3^L with delta/epsilon contractions), num_interaction=%d", args.num_interaction)
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
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            lmax=config.lmax,
            device=device
        ).to(device)
    elif args.tensor_product_mode == 'pure-cartesian-ictd':
        logging.info("Using PURE Cartesian ICTD mode (pure_cartesian_ictd_layers_full, DDP sync), num_interaction=%d", args.num_interaction)
        logging.info(f"  ictd_tp_path_policy={args.ictd_tp_path_policy}, ictd_tp_max_rank_other={args.ictd_tp_max_rank_other}")
        e3trans = PureCartesianICTDTransformerLayerFull(
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
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            lmax=config.lmax,
            ictd_tp_path_policy=args.ictd_tp_path_policy,
            ictd_tp_max_rank_other=args.ictd_tp_max_rank_other,
            internal_compute_dtype=config.dtype,
            device=device,
        ).to(device)
    elif args.tensor_product_mode == 'pure-cartesian-ictd-save':
        logging.info("Using PURE Cartesian ICTD mode (pure_cartesian_ictd_layers, save/original), num_interaction=%d", args.num_interaction)
        logging.info(f"  ictd_tp_path_policy={args.ictd_tp_path_policy}, ictd_tp_max_rank_other={args.ictd_tp_max_rank_other}")
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
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            lmax=config.lmax,
            ictd_tp_path_policy=args.ictd_tp_path_policy,
            ictd_tp_max_rank_other=args.ictd_tp_max_rank_other,
            internal_compute_dtype=config.dtype,
            device=device,
        ).to(device)
    elif args.tensor_product_mode == 'pure-cartesian-sparse':
        logging.info("Using PURE Cartesian SPARSE mode (δ/ε path-sparse within 3^L, O(3) strict)")
        logging.info(f"  max_rank_other={args.max_rank_other}, k_policy={args.k_policy}, num_interaction={args.num_interaction}")
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
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            lmax=config.lmax,
            max_rank_other=args.max_rank_other,
            k_policy=args.k_policy,
            device=device
        ).to(device)
    elif args.tensor_product_mode == 'partial-cartesian':
        logging.info("Using Partial-Cartesian tensor product mode (strict), num_interaction=%d", args.num_interaction)
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
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            lmax=config.lmax,
            device=device
        ).to(device)
    elif args.tensor_product_mode == 'partial-cartesian-loose':
        logging.info("Using Partial-Cartesian LOOSE mode (non-strictly-equivariant, norm product approximation), num_interaction=%d", args.num_interaction)
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
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            lmax=config.lmax,
            device=device
        ).to(device)
    elif args.tensor_product_mode == 'spherical-save':
        logging.info("Using Spherical (channelwise conv) tensor product mode (e3nn_layers_channelwise), num_interaction=%d", args.num_interaction)
        e3trans = E3_TransformerLayer_multi_channelwise(
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
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            device=device
        ).to(device)
    elif args.tensor_product_mode == 'spherical-save-cue':
        logging.info("Using Spherical (channelwise conv) tensor product mode (cuEquivariance backend), num_interaction=%d", args.num_interaction)
        # Detect optional dependency early with a clear message.
        try:
            import cuequivariance_torch  # noqa: F401
        except Exception as e:
            raise ImportError(
                "tensor_product_mode='spherical-save-cue' requires cuEquivariance.\n"
                "Install one of:\n"
                "  pip install -e \".[cue]\"\n"
                "  pip install -r requirements-cue.txt\n"
                "Notes: CUDA kernels package (cuequivariance-ops-torch-cu12) is Linux CUDA only.\n"
                f"Original import error: {e}"
            ) from e
        from molecular_force_field.models.cue_layers_channelwise import (
            E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise_cue,
        )
        e3trans = E3_TransformerLayer_multi_channelwise_cue(
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
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            device=device,
        ).to(device)
    else:  # spherical (default)
        logging.info("Using Spherical harmonics tensor product mode (e3nn), num_interaction=%d", args.num_interaction)
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
            num_interaction=args.num_interaction,
            function_type_main=config.function_type,
            device=device
        ).to(device)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        e3trans=e3trans,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=config,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        initial_learning_rate_for_weight=args.warmup_start_ratio,
        warmup_batches=args.warmup_batches,
        patience_opim=args.lr_decay_patience,
        gamma_value=args.lr_decay_factor,
        epoch_numbers=args.epochs,
        checkpoint_path=args.checkpoint,
        use_checkpoint_loss_weights=not args.reset_loss_weights,
        dump_frequency=args.dump_frequency,
        energy_log_frequency=args.energy_log_frequency,
        vhat_clamp_interval=args.vhat_clamp_interval,
        max_vhat_growth_factor=args.max_vhat_growth,
        max_norm_value=args.max_grad_norm,
        gradient_log_interval=args.grad_log_interval,
        a=args.energy_weight,
        b=args.force_weight,
        update_param=args.update_param,
        weight_a_growth=args.weight_a_growth,
        weight_b_decay=args.weight_b_decay,
        a_min=args.a_min,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
        swa_start_epoch=args.swa_start_epoch,
        swa_a=args.swa_a,
        swa_b=args.swa_b,
        ema_start_epoch=args.ema_start_epoch,
        ema_decay=args.ema_decay,
        use_ema_for_validation=args.use_ema_for_validation,
        save_ema_model=args.save_ema_model,
        force_shift_value=args.force_shift_value,
        c=args.stress_weight,
        c_min=args.c_min,
        c_max=args.c_max,
        patience=args.patience,
        atomic_energy_keys=config.atomic_energy_keys,
        atomic_energy_values=config.atomic_energy_values,
        distributed=args.distributed,
        rank=rank,
        world_size=world_size,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        save_val_csv=args.save_val_csv,
        train_eval_sample_ratio=args.train_eval_sample_ratio,
        log_val_batch_energy_to_console=args.log_val_batch_energy,
        tensor_product_mode=args.tensor_product_mode,
        compile_val=args.compile_val,
        compile_val_mode=args.compile_val_mode,
        compile_val_fullgraph=args.compile_val_fullgraph,
        compile_val_dynamic=args.compile_val_dynamic,
        compile_val_precache=args.compile_val_precache,
    )
    
    # Start training
    if rank == 0:
        logging.info("Starting training...")
    trainer.run_training()
    if rank == 0:
        logging.info("Training completed!")
    
    # Cleanup distributed training
    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()