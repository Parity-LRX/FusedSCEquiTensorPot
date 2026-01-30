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
from molecular_force_field.data import H5Dataset
from molecular_force_field.data.collate import collate_fn_h5
from molecular_force_field.training.trainer import Trainer
from molecular_force_field.data.preprocessing import save_to_h5_parallel
from molecular_force_field.utils.config import ModelConfig


def check_and_preprocess_data(data_dir, train_prefix, val_prefix, max_radius, num_workers, input_file=None, seed=42):
    """
    Check if preprocessed data exists, and run preprocessing if needed.
    
    Args:
        data_dir: Directory containing data files
        train_prefix: Prefix for training data
        val_prefix: Prefix for validation data
        max_radius: Maximum radius for neighbor search
        num_workers: Number of workers for preprocessing
        input_file: Input XYZ file for preprocessing (optional)
        seed: Random seed for train/val split (default: 42)
    
    Returns:
        True if data is ready, False otherwise
    """
    train_processed = os.path.join(data_dir, f'processed_{train_prefix}.h5')
    val_processed = os.path.join(data_dir, f'processed_{val_prefix}.h5')
    train_raw = os.path.join(data_dir, f'read_{train_prefix}.h5')
    val_raw = os.path.join(data_dir, f'read_{val_prefix}.h5')
    
    # Case 1: Preprocessed files exist
    if os.path.exists(train_processed) and os.path.exists(val_processed):
        logging.info(f"Found preprocessed data in {data_dir}/")
        return True
    
    # Case 2: Raw files exist but not preprocessed
    if os.path.exists(train_raw) and os.path.exists(val_raw):
        logging.info(f"Found raw data in {data_dir}/, running H5 preprocessing...")
        save_to_h5_parallel(train_prefix, max_radius, num_workers, data_dir=data_dir)
        save_to_h5_parallel(val_prefix, max_radius, num_workers, data_dir=data_dir)
        return True
    
    # Case 3: No data files, need to run full preprocessing
    if input_file and os.path.exists(input_file):
        logging.info(f"No preprocessed data found. Running preprocessing on {input_file}...")
        # Import here to avoid circular import
        from molecular_force_field.data.preprocessing import (
            extract_data_blocks,
            fit_baseline_energies,
            compute_correction,
            save_set
        )
        import numpy as np
        import pandas as pd
        
        # Create output directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Extract data
        all_blocks, all_energy, all_raw_energy, all_cells, all_pbcs = extract_data_blocks(input_file)
        logging.info(f"Total frames: {len(all_blocks)}")
        
        # Split train/val (95/5)
        data_size = len(all_blocks)
        indices = np.arange(data_size)
        np.random.seed(seed)
        train_size = int(0.95 * data_size)
        val_size = data_size - train_size
        val_indices = np.random.choice(indices, size=val_size, replace=False)
        train_mask = ~np.isin(indices, val_indices)
        train_indices = indices[train_mask]
        
        logging.info(f"Split: {len(train_indices)} Train, {len(val_indices)} Val")
        
        train_blocks = [all_blocks[i] for i in train_indices]
        train_raw_E = [all_raw_energy[i] for i in train_indices]
        val_blocks = [all_blocks[i] for i in val_indices]
        val_raw_E = [all_raw_energy[i] for i in val_indices]
        
        # Fit baseline energies (E0) on TRAIN only.
        # IMPORTANT: do NOT hardcode element keys here; derive them from the training set
        # to support the full periodic table / arbitrary compositions.
        train_atoms = []
        for block in train_blocks:
            # block rows: [x, y, z, A, fx, fy, fz]
            train_atoms.extend([int(row[3]) for row in block])
        uniq = sorted({a for a in train_atoms if a > 0})
        if not uniq:
            raise ValueError("No valid atomic numbers found in training blocks; cannot fit baseline energies.")

        keys = np.asarray(uniq, dtype=np.int64)
        initial_values = np.array([-0.01] * len(keys), dtype=np.float64)
        fitted_values = fit_baseline_energies(train_blocks, train_raw_E, keys, initial_values)
        
        # Save fitted energies
        fitted_e0_path = os.path.join(data_dir, 'fitted_E0.csv')
        pd.DataFrame({'Atom': keys, 'E0': fitted_values}).to_csv(fitted_e0_path, index=False)
        logging.info(f"Saved {fitted_e0_path}")
        
        # Compute corrections
        train_correction = compute_correction(train_blocks, train_raw_E, keys, fitted_values)
        val_correction = compute_correction(val_blocks, val_raw_E, keys, fitted_values)
        
        # Save sets
        save_set('train', train_indices, train_blocks, train_raw_E, train_correction, all_cells, pbc_list=all_pbcs, output_dir=data_dir)
        save_set('val', val_indices, val_blocks, val_raw_E, val_correction, all_cells, pbc_list=all_pbcs, output_dir=data_dir)
        
        # Preprocess H5
        save_to_h5_parallel('train', max_radius, num_workers, data_dir=data_dir)
        save_to_h5_parallel('val', max_radius, num_workers, data_dir=data_dir)
        
        logging.info(f"Preprocessing completed! Data saved to {data_dir}/")
        return True
    
    # Case 4: No data at all
    logging.error(
        f"No data files found in {data_dir}/ and no input file specified.\n"
        f"Please either:\n"
        f"  1. Run 'mff-preprocess --input-file <xyz_file> --output-dir {data_dir}' first, or\n"
        f"  2. Use '--input-file <xyz_file>' to automatically preprocess data."
    )
    return False


def setup_logging():
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # Custom filter to suppress epoch training logs on console but keep validation logs
    class ConsoleFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            # Suppress epoch training logs, but allow validation and important messages
            if 'Epoch' in msg and 'Validation' not in msg and 'Training started' not in msg and 'Training completed' not in msg and 'Early stopping' not in msg:
                return False
            # Note: Validation batch energy logs are controlled by log level:
            # - logging.debug: only goes to file (console shows INFO+)
            # - logging.info: goes to both file and console (if --log-val-batch-energy is set)
            return True
    
    # Console handler - only show validation and important messages
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(ConsoleFilter())
    
    log_filename = f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = RotatingFileHandler(
        filename=log_filename,
        maxBytes=1000 * 1024 * 1024,  # 1GB per file
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Root logger
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
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float32', 'float64', 'float', 'double'],
                        help='Default dtype for tensors (float32 or float64, default: float64)')
    parser.add_argument('--dump-frequency', type=int, default=250,
                        help='Frequency (in batches) for validation and model saving (default: 250)')
    parser.add_argument('--train-eval-sample-ratio', type=float, default=0.2,
                        help='Ratio of training set to evaluate during validation (0.0-1.0, default: 0.2). '
                             'Set to 1.0 for full evaluation, or lower (e.g., 0.2 for 20%%) for faster validation. '
                             'Useful for large datasets.')
    parser.add_argument('--energy-log-frequency', type=int, default=10,
                        help='Frequency (in batches) to log energy predictions (default: 10)')
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
                        help='Input XYZ file for automatic preprocessing (optional)')
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
                        choices=['spherical', 'partial-cartesian', 'partial-cartesian-loose', 'pure-cartesian', 'pure-cartesian-sparse', 'pure-cartesian-ictd'],
                        help='Tensor product mode: "spherical" uses e3nn spherical harmonics (default), '
                             '"partial-cartesian" uses Cartesian tensor products with EquivariantTensorProduct (strictly equivariant), '
                             '"partial-cartesian-loose" uses non-strictly-equivariant Cartesian tensor products (norm product approximation, not strictly equivariant), '
                             '"pure-cartesian" uses full rank Cartesian tensors (3^L) with delta/epsilon contractions (most pure), '
                             '"pure-cartesian-sparse" uses a sparse pure-cartesian delta/epsilon tensor product (O(3) strict) by restricting rank-rank paths, '
                             '"pure-cartesian-ictd" uses pure-cartesian message passing but ICTD trace-chain invariants for readout')
    parser.add_argument('--max-rank-other', type=int, default=1,
                        help='Max rank for sparse tensor product in pure-cartesian-sparse mode (default: 1). '
                             'Only interactions where min(L1, L2) <= max_rank_other are allowed. '
                             'Larger values allow more interactions but increase parameters and computation.')
    parser.add_argument('--k-policy', type=str, default='k0',
                        choices=['k0', 'k1', 'both'],
                        help='K policy for sparse tensor product in pure-cartesian-sparse mode (default: k0). '
                             'k0: only k=0 (promotes higher rank), k1: only k=1 (contracts to lower rank), both: keep both')

    # ICTD path pruning controls (pure-cartesian-ictd only)
    parser.add_argument('--ictd-tp-path-policy', type=str, default='full',
                        choices=['full', 'max_rank_other'],
                        help='Path policy for ICTD tensor products in pure-cartesian-ictd mode (default: full). '
                             'full: keep all CG-allowed (l1,l2->l3) paths; '
                             'max_rank_other: keep only paths with min(l1,l2) <= --ictd-tp-max-rank-other.')
    parser.add_argument('--ictd-tp-max-rank-other', type=int, default=None,
                        help='Used when --ictd-tp-path-policy=max_rank_other. '
                             'Keeps only paths with min(l1,l2) <= this value (e.g. 1 keeps scalar/vector couplings).')
    
    args = parser.parse_args()

    # Initialize rank and world_size early (before validation that uses rank)
    # These will be updated later if distributed training is enabled
    rank = 0
    world_size = 1
    local_rank = 0

    # Validate a/b clamp ranges
    if args.a_min is not None and args.a_max is not None and args.a_min > args.a_max:
        raise ValueError("--a-min must be <= --a-max")
    if args.b_min is not None and args.b_max is not None and args.b_min > args.b_max:
        raise ValueError("--b-min must be <= --b-max")
    
    # Validate SWA parameters
    if args.swa_start_epoch is not None:
        if args.swa_a is None or args.swa_b is None:
            raise ValueError("--swa-a and --swa-b must be set when --swa-start-epoch is set")
        if args.swa_start_epoch < 1:
            raise ValueError("--swa-start-epoch must be >= 1")
        if rank == 0:
            logging.info(f"SWA enabled: Will switch to a={args.swa_a}, b={args.swa_b} at epoch {args.swa_start_epoch}")

    # Validate EMA parameters
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
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # For reproducibility with CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Initialize distributed training if enabled
    if args.distributed:
        # Get local_rank from environment if not provided
        if args.local_rank == -1:
            args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        
        if args.local_rank == -1:
            raise ValueError("--local-rank must be set when --distributed is enabled. "
                           "Use 'torchrun' or set LOCAL_RANK environment variable.")
        
        # Initialize process group with extended timeout for preprocessing
        # Default NCCL timeout is 30 minutes, extend to 2 hours for large datasets
        dist.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            timeout=timedelta(hours=2)
        )
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = args.local_rank
        
        # Set device for this process
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
        
        # Only setup logging on rank 0
        if rank == 0:
            setup_logging()
            logging.info(f"Distributed training enabled: {world_size} GPUs")
            logging.info(f"Using device: {device} (rank {rank}, local_rank {local_rank})")
            logging.info(f"Data directory: {args.data_dir}")
        else:
            # Minimal logging setup for non-rank-0 processes
            logging.basicConfig(level=logging.WARNING)
    else:
        setup_logging()
        # rank, world_size, local_rank already initialized above
    
    # Set default dtype before creating any tensors
    if args.dtype == 'float64' or args.dtype == 'double':
        torch.set_default_dtype(torch.float64)
        if rank == 0:
            logging.info("Using dtype: float64")
    elif args.dtype == 'float32' or args.dtype == 'float':
        torch.set_default_dtype(torch.float32)
        if rank == 0:
            logging.info("Using dtype: float32")
    
    # Device setup (for non-distributed case)
    if not args.distributed:
        if args.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        
        logging.info(f"Using device: {device}")
        logging.info(f"Data directory: {args.data_dir}")
    
    # Check and preprocess data if needed
    if args.distributed:
        # In distributed mode, only check if data exists (no preprocessing)
        # All ranks check independently to avoid synchronization issues
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
        
        # Barrier to ensure all processes are ready
        dist.barrier()
    else:
        # Single process mode: can auto-preprocess
        data_ready = check_and_preprocess_data(
            args.data_dir, 
            args.train_prefix, 
            args.val_prefix, 
            args.max_radius, 
            args.num_workers,
            args.input_file,
            args.seed
        )
        if not data_ready:
            logging.error("Data preparation failed. Exiting.")
            return
    
    # Load datasets
    train_dataset = H5Dataset(args.train_prefix, data_dir=args.data_dir)
    val_dataset = H5Dataset(args.val_prefix, data_dir=args.data_dir)
    
    # Create distributed samplers if using distributed training
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
        shuffle = False  # Shuffle is handled by DistributedSampler
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn_h5,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn_h5,
        num_workers=max(1, args.num_workers // 4),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Model configuration
    # Convert dtype string to torch.dtype
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
    
    # Atomic reference energies (E0)
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
    
    # Log model hyperparameters
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
        logging.info("Using PURE Cartesian mode (rank tensors 3^L with delta/epsilon contractions)")
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
    elif args.tensor_product_mode == 'pure-cartesian-ictd':
        logging.info("Using PURE Cartesian ICTD mode (trace-chain invariants for readout)")
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
            function_type_main=config.function_type,
            lmax=config.lmax,
            ictd_tp_path_policy=args.ictd_tp_path_policy,
            ictd_tp_max_rank_other=args.ictd_tp_max_rank_other,
            internal_compute_dtype=config.dtype,
            device=device,
        ).to(device)
    elif args.tensor_product_mode == 'pure-cartesian-sparse':
        logging.info("Using PURE Cartesian SPARSE mode (δ/ε path-sparse within 3^L, O(3) strict)")
        logging.info(f"  max_rank_other={args.max_rank_other}, k_policy={args.k_policy}")
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
            max_rank_other=args.max_rank_other,
            k_policy=args.k_policy,
            device=device
        ).to(device)
    elif args.tensor_product_mode == 'partial-cartesian':
        logging.info("Using Partial-Cartesian tensor product mode (strict)")
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
    elif args.tensor_product_mode == 'partial-cartesian-loose':
        logging.info("Using Partial-Cartesian LOOSE mode (non-strictly-equivariant, norm product approximation)")
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
    else:  # spherical (default)
        logging.info("Using Spherical harmonics tensor product mode (e3nn)")
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