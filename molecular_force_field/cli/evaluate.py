"""Command-line interface for evaluation."""

import argparse
import torch
import logging
import os
import numpy as np
from torch.utils.data import DataLoader

from molecular_force_field.models import (
    E3_TransformerLayer_multi,
    CartesianTransformerLayer,
    CartesianTransformerLayerLoose,
    PureCartesianTransformerLayer,
    PureCartesianSparseTransformerLayer,
    PureCartesianICTDTransformerLayer,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDTransformerLayerFull,
)
from molecular_force_field.models.e3nn_layers_channelwise import (
    E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
)
from molecular_force_field.data import OnTheFlyDataset, H5Dataset
from molecular_force_field.data.collate import on_the_fly_collate, collate_fn_h5
from molecular_force_field.data.preprocessing import extract_data_blocks, compute_correction, save_set
from molecular_force_field.evaluation.evaluator import Evaluator
from molecular_force_field.evaluation.calculator import MyE3NNCalculator, DDPCalculator
from molecular_force_field.utils.config import ModelConfig


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate molecular force field model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Path to input XYZ file. If provided, will automatically convert to read/energy/cell files for test set evaluation. '
                             'Note: When using this option, you MUST provide atomic reference energies via --atomic-energy-file or --atomic-energy-keys/--atomic-energy-values '
                             '(typically use the fitted_E0.csv generated during training/preprocessing).')
    parser.add_argument('--test-prefix', type=str, default='test',
                        help='Prefix for test data files')
    parser.add_argument('--data-dir', type=str, default='.',
                        help='Directory containing data files (default: current directory)')
    parser.add_argument('--read-file', type=str, default=None,
                        help='Path to read HDF5 file (for on-the-fly dataset)')
    parser.add_argument('--energy-file', type=str, default=None,
                        help='Path to energy HDF5 file (for on-the-fly dataset)')
    parser.add_argument('--cell-file', type=str, default=None,
                        help='Path to cell HDF5 file (for on-the-fly dataset)')
    parser.add_argument('--max-atom', type=int, default=1,
                        help='Maximum number of atoms (for padding, used when --input-file is provided)')
    parser.add_argument('--elements', type=str, nargs='+', default=None,
                        help='Element symbols to recognize (default: None, recognizes all elements from periodic table). '
                             'Used when --input-file is provided.')
    parser.add_argument('--max-radius', type=float, default=5.0,
                        help='Maximum radius for neighbor search')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--output-prefix', type=str, default='test',
                        help='Prefix for output files')
    parser.add_argument('--use-h5', action='store_true',
                        help='Use H5Dataset instead of OnTheFlyDataset')
    parser.add_argument('--md-sim', action='store_true',
                        help='Run MD simulation')
    parser.add_argument('--neb', action='store_true',
                        help='Run NEB calculation')
    parser.add_argument('--phonon', action='store_true',
                        help='Calculate phonon spectrum (Hessian matrix)')
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float32', 'float64', 'float', 'double'],
                        help='Default dtype for tensors (float32 or float64, default: float64)')

    # Inference acceleration (evaluation only)
    parser.add_argument('--compile', type=str, default='none',
                        choices=['none', 'e3trans'],
                        help='Enable torch.compile for evaluation inference. '
                             '"e3trans" compiles the transformer module only (default: none). '
                             'NOTE: torch.compile does NOT support double backward; training with force loss is unaffected here.')
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode (default: reduce-overhead)')
    parser.add_argument('--compile-fullgraph', action='store_true',
                        help='Pass fullgraph=True to torch.compile (may fail more often).')
    parser.add_argument('--compile-dynamic', action='store_true',
                        help='Pass dynamic=True to torch.compile.')
    parser.add_argument('--compile-precache', action='store_true',
                        help='Run one eager forward before compiling to warm ICTD caches on CUDA (recommended).')
    
    # MD simulation parameters
    parser.add_argument('--md-input', type=str, default='start_structure.xyz',
                        help='Input structure file for MD simulation (default: start_structure.xyz)')
    parser.add_argument('--md-temperature', type=float, default=300.0,
                        help='MD simulation temperature in Kelvin (default: 300.0)')
    parser.add_argument('--md-timestep', type=float, default=1.0,
                        help='MD timestep in femtoseconds (default: 1.0)')
    parser.add_argument('--md-friction', type=float, default=0.01,
                        help='Langevin friction coefficient (default: 0.01)')
    parser.add_argument('--md-steps', type=int, default=10000,
                        help='Number of MD steps (default: 10000)')
    parser.add_argument('--md-relax-fmax', type=float, default=0.05,
                        help='Force convergence for initial relaxation in eV/Å (default: 0.05)')
    parser.add_argument('--md-log-interval', type=int, default=10,
                        help='Interval for logging MD status (default: 10)')
    parser.add_argument('--md-output', type=str, default='md_traj.xyz',
                        help='Output trajectory file for MD (default: md_traj.xyz)')
    parser.add_argument('--md-no-relax', action='store_true',
                        help='Skip initial relaxation before MD')
    
    # NEB parameters
    parser.add_argument('--neb-initial', type=str, default='initial.xyz',
                        help='Initial structure for NEB (default: initial.xyz)')
    parser.add_argument('--neb-final', type=str, default='final.xyz',
                        help='Final structure for NEB (default: final.xyz)')
    parser.add_argument('--neb-images', type=int, default=10,
                        help='Number of intermediate NEB images (default: 10)')
    parser.add_argument('--neb-fmax', type=float, default=0.05,
                        help='Force convergence for NEB in eV/Å (default: 0.05)')
    parser.add_argument('--neb-output', type=str, default='neb.traj',
                        help='Output trajectory file for NEB (default: neb.traj)')
    
    # Phonon calculation parameters
    parser.add_argument('--phonon-input', type=str, default='structure.xyz',
                        help='Input structure file for phonon calculation (default: structure.xyz)')
    parser.add_argument('--phonon-relax-fmax', type=float, default=0.01,
                        help='Force convergence for structure relaxation before phonon calculation in eV/Å (default: 0.01)')
    parser.add_argument('--phonon-output', type=str, default='phonon',
                        help='Output prefix for phonon files (default: phonon). Will generate phonon_frequencies.txt and phonon_hessian.npy')
    parser.add_argument('--phonon-no-relax', action='store_true',
                        help='Skip structure relaxation before phonon calculation')
    
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
                             '"spherical-save" uses channelwise edge convolution (e3nn backend; fewer params). '
                             '"spherical-save-cue" uses channelwise edge convolution (cuEquivariance backend; requires cuequivariance-torch). '
                             '"partial-cartesian" uses Cartesian tensor products (strictly equivariant), '
                             '"partial-cartesian-loose" uses non-strictly-equivariant Cartesian tensor products (norm product approximation, not strictly equivariant), '
                             '"pure-cartesian" uses full rank Cartesian tensors (3^L) with delta/epsilon contractions (most pure), '
                             '"pure-cartesian-sparse" uses a sparse pure-cartesian delta/epsilon tensor product (O(3) strict) by restricting rank-rank paths, '
                             '"pure-cartesian-ictd" uses pure_cartesian_ictd_layers_full (ICTD, DDP supported). '
                             '"pure-cartesian-ictd-save" uses pure_cartesian_ictd_layers (original ICTD, DDP supported). '
                             'Note: ICTD inference is typically ~3x faster than spherical-save.')
    parser.add_argument('--num-interaction', type=int, default=2,
                        help='Number of message-passing steps per block (default: 2). '
                             'Used by all tensor-product modes. Must match the value used at training. Must be >= 2.')

    parser.add_argument('--mp-context', type=str, default='auto',
                        choices=['auto', 'fork', 'spawn'],
                        help='Multiprocessing start method for DataLoader workers. '
                             '"auto" forces "spawn" when --compile is enabled (safer with CUDA/compile), '
                             'otherwise uses the default OS method (often "fork" on Linux, faster).')

    # Atomic reference energies (E0)
    parser.add_argument('--atomic-energy-file', type=str, default=None,
                        help='CSV file with columns Atom,E0 to load atomic reference energies. '
                             'If not set, defaults to fitted_E0.csv in the current directory. '
                             'Note: When using --input-file, this file is REQUIRED (cannot fit from test data). '
                             'Typically generated during preprocessing/training via least-squares fitting.')
    parser.add_argument('--atomic-energy-keys', type=int, nargs='+', default=None,
                        help='Atomic numbers for custom atomic reference energies (must match --atomic-energy-values length).')
    parser.add_argument('--atomic-energy-values', type=float, nargs='+', default=None,
                        help='Atomic reference energies (E0) in eV corresponding to --atomic-energy-keys.')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Log scatter backend (helps diagnose performance regressions when torch_scatter is broken).
    try:
        from molecular_force_field.utils.scatter import scatter_backend, require_torch_scatter

        logging.info("Scatter backend: %s", scatter_backend())
        if args.tensor_product_mode == "spherical-save-cue":
            require_torch_scatter(reason="tensor_product_mode='spherical-save-cue' aims for maximum speed.")
    except Exception:
        pass
    
    # Set default dtype before creating any tensors
    if args.dtype == 'float64' or args.dtype == 'double':
        torch.set_default_dtype(torch.float64)
        logging.info("Using dtype: float64")
    elif args.dtype == 'float32' or args.dtype == 'float':
        torch.set_default_dtype(torch.float32)
        logging.info("Using dtype: float32")
    
    # Device setup（MD + torchrun 时提前 init dist，保证各 rank 用对应 GPU）
    use_ddp_md = args.md_sim and os.environ.get("RANK") is not None
    if use_ddp_md:
        import torch.distributed as dist
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', rank)}" if torch.cuda.is_available() else "cpu")
        if rank != 0:
            logging.basicConfig(level=logging.WARNING)
        logging.info(f"Using device: {device} (rank {rank}/{world_size})")
    else:
        rank = 0
        world_size = 1
        if args.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        logging.info(f"Using device: {device}")
    if args.num_interaction < 2:
        raise ValueError(f"--num-interaction must be >= 2, got {args.num_interaction}")

    def _maybe_compile_e3trans(e3trans_module, *, precache_batch=None):
        if args.compile == 'none':
            return e3trans_module
        if args.compile != 'e3trans':
            raise ValueError(f"Unknown --compile={args.compile!r}")
        if not hasattr(torch, "compile"):
            logging.warning("torch.compile not available in this PyTorch; continuing without compile.")
            return e3trans_module

        # Optional CUDA cache warmup (avoid CPU work / graph breaks in ICTD direction_harmonics_fast)
        if args.compile_precache and precache_batch is not None:
            try:
                with torch.no_grad():
                    (pos, A, batch_idx, _force, _target, edge_src, edge_dst, edge_shifts, cell, _stress) = precache_batch
                    _ = e3trans_module(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
            except Exception as e:
                logging.warning(f"Compile precache forward failed; continuing. Error: {e}")

        try:
            logging.info(
                "Compiling e3trans with torch.compile(mode=%s, fullgraph=%s, dynamic=%s)",
                args.compile_mode, args.compile_fullgraph, args.compile_dynamic
            )
            return torch.compile(
                e3trans_module,
                mode=args.compile_mode,
                fullgraph=args.compile_fullgraph,
                dynamic=args.compile_dynamic,
            )
        except Exception as e:
            logging.warning(f"torch.compile failed; continuing without compile. Error: {e}")
            return e3trans_module
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
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
    # If --input-file is provided, atomic energies are required (cannot fit from data)
    if args.input_file is not None:
        if args.atomic_energy_keys is None or args.atomic_energy_values is None:
            # Try to load from file
            e0_path = args.atomic_energy_file or 'fitted_E0.csv'
            if not os.path.exists(e0_path):
                raise ValueError(
                    f"When using --input-file, atomic reference energies are REQUIRED. Use one of the following:\n"
                    f"  1. Use --atomic-energy-file to specify a CSV file path (typically use fitted_E0.csv generated during training/preprocessing)\n"
                    f"  2. Use --atomic-energy-keys and --atomic-energy-values to provide directly\n"
                    f"  File not found: {e0_path}\n"
                    f"  Note: fitted_E0.csv is typically auto-generated during preprocessing (mff-preprocess) or training (mff-train)."
                )
            # Load from file and verify it was successful (not using defaults)
            loaded_successfully = config.load_atomic_energies_from_file(e0_path)
            if not loaded_successfully:
                raise ValueError(
                    f"When using --input-file, atomic reference energy file must be loaded successfully.\n"
                    f"File {e0_path} exists but has incorrect format or failed to load.\n"
                    f"Please ensure the file contains 'Atom' and 'E0' columns, or use --atomic-energy-keys and --atomic-energy-values to provide directly."
                )
            logging.info(f"Loaded atomic reference energies from file: {e0_path}")
        else:
            if len(args.atomic_energy_keys) != len(args.atomic_energy_values):
                raise ValueError("--atomic-energy-keys and --atomic-energy-values must have the same length.")
            config.atomic_energy_keys = torch.tensor(args.atomic_energy_keys, dtype=torch.long)
            config.atomic_energy_values = torch.tensor(args.atomic_energy_values, dtype=config.dtype)
            logging.info("Using atomic reference energies provided via CLI.")
    else:
        # Normal case: can use defaults or load from file
        if args.atomic_energy_keys is not None or args.atomic_energy_values is not None:
            if args.atomic_energy_keys is None or args.atomic_energy_values is None:
                raise ValueError("Both --atomic-energy-keys and --atomic-energy-values must be provided together.")
            if len(args.atomic_energy_keys) != len(args.atomic_energy_values):
                raise ValueError("--atomic-energy-keys and --atomic-energy-values must have the same length.")
            config.atomic_energy_keys = torch.tensor(args.atomic_energy_keys, dtype=torch.long)
            config.atomic_energy_values = torch.tensor(args.atomic_energy_values, dtype=config.dtype)
            logging.info("Using custom atomic reference energies from CLI.")
        else:
            e0_path = args.atomic_energy_file or 'fitted_E0.csv'
            config.load_atomic_energies_from_file(e0_path)
    
    # Log model hyperparameters
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
            internal_compute_dtype=config.dtype,
            device=device,
        ).to(device)
    elif args.tensor_product_mode == 'pure-cartesian-ictd-save':
        logging.info("Using PURE Cartesian ICTD mode (pure_cartesian_ictd_layers, save/original), num_interaction=%d", args.num_interaction)
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
            internal_compute_dtype=config.dtype,
            device=device,
        ).to(device)
    elif args.tensor_product_mode == 'pure-cartesian-sparse':
        logging.info("Using PURE Cartesian SPARSE mode (δ/ε path-sparse within 3^L, O(3) strict), num_interaction=%d", args.num_interaction)
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
            irreps_input=config.get_irreps_output_conv(),  # Fixed: should match train.py
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
    
    e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
    e3trans.eval()
    
    logging.info("Loaded model from checkpoint.")
    
    # Skip static evaluation if only running NEB or MD
    skip_static_eval = args.neb or args.md_sim
    
    if not skip_static_eval:
        # Convert XYZ file to required data files if --input-file is provided
        if args.input_file is not None:
            if not os.path.exists(args.input_file):
                raise FileNotFoundError(f"Input file not found: {args.input_file}")
            
            # Warn if user also provided read/energy/cell files (will be overridden)
            if args.read_file is not None or args.energy_file is not None or args.cell_file is not None:
                logging.warning(
                    "Detected both --input-file and --read-file/--energy-file/--cell-file arguments. "
                    "Will use files generated from XYZ conversion, ignoring user-provided file paths."
                )
            
            # Warn if user specified --use-h5 (will use OnTheFlyDataset instead)
            if args.use_h5:
                logging.warning(
                    "Detected both --input-file and --use-h5 arguments. "
                    "XYZ conversion generates raw data files (read/energy/cell), will use OnTheFlyDataset for evaluation. "
                    "To use H5Dataset, run preprocessing step first to generate processed_*.h5 files."
                )
                args.use_h5 = False  # Force use of OnTheFlyDataset
            
            logging.info(f"Converting data from XYZ file: {args.input_file}")
            
            # Extract data blocks from XYZ file
            all_blocks, all_energy, all_raw_energy, all_cells, all_pbcs, all_stresses = extract_data_blocks(
                args.input_file, elements=args.elements
            )
            logging.info(f"Extracted {len(all_blocks)} structures")
            
            # Use all data (no train/val split)
            all_indices = np.arange(len(all_blocks))
            
            # Compute correction energies using user-provided atomic energies
            # Verify atomic energies are set
            if config.atomic_energy_keys is None or config.atomic_energy_values is None:
                raise ValueError(
                    "Atomic reference energies not set. When using --input-file, atomic reference energies are REQUIRED."
                )
            
            # Convert atomic energy keys and values to numpy arrays
            atomic_keys = config.atomic_energy_keys.cpu().numpy() if isinstance(config.atomic_energy_keys, torch.Tensor) else config.atomic_energy_keys
            atomic_values = config.atomic_energy_values.cpu().numpy() if isinstance(config.atomic_energy_values, torch.Tensor) else config.atomic_energy_values
            
            logging.info("Computing correction energies using user-provided atomic reference energies...")
            logging.info(f"Atomic reference energy keys: {atomic_keys}")
            logging.info(f"Atomic reference energy values: {atomic_values}")
            all_correction = compute_correction(all_blocks, all_raw_energy, atomic_keys, atomic_values)
            
            # Save test set files
            logging.info(f"Saving test data files to {args.data_dir}/...")
            save_set(
                args.test_prefix,
                all_indices,
                all_blocks,
                all_raw_energy,
                all_correction,
                all_cells,
                pbc_list=all_pbcs,
                stress_list=all_stresses,
                max_atom=args.max_atom,
                output_dir=args.data_dir
            )
            
            # Set file paths to generated files
            args.read_file = os.path.join(args.data_dir, f'read_{args.test_prefix}.h5')
            args.energy_file = os.path.join(args.data_dir, f'raw_energy_{args.test_prefix}.h5')
            args.cell_file = os.path.join(args.data_dir, f'cell_{args.test_prefix}.h5')
            
            logging.info(f"Data conversion completed. Generated files:")
            logging.info(f"  - {args.read_file}")
            logging.info(f"  - {args.energy_file}")
            logging.info(f"  - {args.cell_file}")
        
        # Static evaluation
        if args.use_h5:
            dataset = H5Dataset(args.test_prefix, data_dir=args.data_dir)
            collate_fn = collate_fn_h5
        else:
            if args.read_file is None:
                args.read_file = f'read_{args.test_prefix}.h5'
            if args.energy_file is None:
                args.energy_file = f'raw_energy_{args.test_prefix}.h5'
            if args.cell_file is None:
                args.cell_file = f'cell_{args.test_prefix}.h5'
            
            dataset = OnTheFlyDataset(
                args.read_file,
                args.energy_file,
                args.cell_file,
                max_radius=args.max_radius
            )
            collate_fn = on_the_fly_collate
        
        if args.mp_context == "spawn":
            mp_ctx = "spawn"
        elif args.mp_context == "fork":
            mp_ctx = "fork"
        else:
            mp_ctx = "spawn" if (args.compile != "none" and device.type == "cuda") else None

        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=10,
            pin_memory=True,
            multiprocessing_context=mp_ctx,
        )

        # Optionally precache+compile using the first batch (evaluation inference only)
        if args.compile != 'none':
            try:
                first_batch = next(iter(data_loader))
                if first_batch is not None:
                    # Move batch to device
                    first_batch = tuple(x.to(device) if torch.is_tensor(x) else x for x in first_batch)
                    e3trans = _maybe_compile_e3trans(e3trans, precache_batch=first_batch)
                else:
                    e3trans = _maybe_compile_e3trans(e3trans, precache_batch=None)
            except StopIteration:
                e3trans = _maybe_compile_e3trans(e3trans, precache_batch=None)
        
        evaluator = Evaluator(
            model=e3trans,
            dataset=dataset,
            device=device,
            atomic_energy_keys=config.atomic_energy_keys,
            atomic_energy_values=config.atomic_energy_values,
        )
        
        logging.info("Starting static evaluation...")
        metrics = evaluator.evaluate(data_loader, output_prefix=args.output_prefix)
        logging.info(f"Evaluation completed! Energy RMSE: {metrics['energy_rmse']:.6f}")
    
    # Phonon calculation
    if args.phonon:
        from ase import Atoms
        from ase.io import read, write
        from ase.optimize import BFGS
        from ase.data import atomic_masses, atomic_numbers
        from ase.neighborlist import neighbor_list
        
        ref_energies_dict = {
            k.item(): v.item()
            for k, v in zip(config.atomic_energy_keys, config.atomic_energy_values)
        }
        calc = MyE3NNCalculator(e3trans, ref_energies_dict, device, args.max_radius)
        
        logging.info("=" * 60)
        logging.info("Phonon Calculation")
        logging.info("=" * 60)
        logging.info(f"  Input structure: {args.phonon_input}")
        logging.info(f"  Relaxation fmax: {args.phonon_relax_fmax} eV/Å")
        logging.info(f"  Output prefix: {args.phonon_output}")
        logging.info("=" * 60)
        
        # Read structure
        if not os.path.exists(args.phonon_input):
            raise FileNotFoundError(f"Input structure file not found: {args.phonon_input}")
        
        atoms = read(args.phonon_input)
        atoms.calc = calc
        
        # Relax structure if needed
        if not args.phonon_no_relax:
            logging.info("Relaxing structure before phonon calculation...")
            optimizer = BFGS(atoms, logfile=None)
            optimizer.run(fmax=args.phonon_relax_fmax)
            logging.info(f"Relaxation completed. Final forces: max={atoms.get_forces().max():.6f} eV/Å")
        else:
            logging.info("Skipping structure relaxation (--phonon-no-relax specified)")
        
        # Get atomic positions and numbers
        pos = torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype(), device=device)
        A = torch.tensor([atomic_numbers[symbol] for symbol in atoms.get_chemical_symbols()], 
                         dtype=torch.long, device=device)
        batch_idx = torch.zeros(len(atoms), dtype=torch.long, device=device)
        
        # Build graph (neighbors) using on-the-fly ASE neighbor list
        cell_array = atoms.cell.array
        pbc_flags = atoms.pbc
        if cell_array is None or np.abs(cell_array).sum() <= 1e-9:
            cell_array = np.eye(3) * 100.0
            pbc_flags = [False, False, False]
        atoms_nl = atoms.copy()
        atoms_nl.cell = cell_array
        atoms_nl.pbc = pbc_flags

        edge_src_np, edge_dst_np, edge_shifts_np = neighbor_list('ijS', atoms_nl, args.max_radius)
        edge_src = torch.tensor(edge_src_np, dtype=torch.long, device=device)
        edge_dst = torch.tensor(edge_dst_np, dtype=torch.long, device=device)
        edge_shifts = torch.tensor(edge_shifts_np, dtype=torch.float64, device=device)
        cell = torch.tensor(cell_array, dtype=torch.get_default_dtype(), device=device).unsqueeze(0)
        
        # Compute Hessian
        logging.info("Computing Hessian matrix...")
        hessian = evaluator.compute_hessian(
            pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell
        )
        
        # Get atomic masses
        masses = torch.tensor([atomic_masses[atomic_numbers[symbol]] for symbol in atoms.get_chemical_symbols()],
                             dtype=torch.get_default_dtype())
        
        # Compute phonon spectrum
        frequencies = evaluator.compute_phonon_spectrum(
            hessian, masses, output_prefix=args.phonon_output
        )
        
        logging.info("Phonon calculation completed!")
    
    # MD simulation or NEB calculation
    if args.md_sim or args.neb:
        from ase import Atoms
        from ase.io import read, write
        from ase.optimize import BFGS
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase import units
        from ase.md import MDLogger
        
        ref_energies_dict = {
            k.item(): v.item()
            for k, v in zip(config.atomic_energy_keys, config.atomic_energy_values)
        }
        # DDP MD：非 rank0 进入 worker 循环并退出；rank0 用 DDPCalculator 跑 MD
        if args.md_sim and use_ddp_md and world_size > 1:
            if rank != 0:
                from molecular_force_field.cli.inference_ddp import run_one_ddp_inference_from_ase_atoms
                dtype_calc = next(e3trans.parameters()).dtype
                while True:
                    e, f = run_one_ddp_inference_from_ase_atoms(
                        None, e3trans, args.max_radius, device, dtype_calc,
                        return_forces=True, atomic_energies_dict=ref_energies_dict,
                    )
                    if e is None:
                        break
                dist.destroy_process_group()
                return
            calc = DDPCalculator(e3trans, ref_energies_dict, device, args.max_radius)
        else:
            calc = MyE3NNCalculator(e3trans, ref_energies_dict, device, args.max_radius)
        
        if args.neb:
            # Validate fmax parameter
            if args.neb_fmax <= 0:
                raise ValueError(f"Invalid --neb-fmax value: {args.neb_fmax}. Must be positive.")
            
            # Create logfile for NEB optimization (records fmax and optimization details)
            neb_logfile = args.neb_output.replace('.traj', '.log').replace('.xyz', '.log')
            if neb_logfile == args.neb_output:
                neb_logfile = 'neb.log'
            
            logging.info("=" * 60)
            logging.info("NEB Calculation")
            logging.info("=" * 60)
            logging.info(f"  Initial structure: {args.neb_initial}")
            logging.info(f"  Final structure:   {args.neb_final}")
            logging.info(f"  Number of images:  {args.neb_images}")
            logging.info(f"  Force threshold:   {args.neb_fmax} eV/Å")
            logging.info(f"  Output trajectory: {args.neb_output}")
            logging.info(f"  Optimization log: {neb_logfile} (contains fmax for each step)")
            logging.info("=" * 60)
            
            # Validate input files exist
            if not os.path.exists(args.neb_initial):
                raise FileNotFoundError(f"NEB initial structure file not found: {args.neb_initial}")
            if not os.path.exists(args.neb_final):
                raise FileNotFoundError(f"NEB final structure file not found: {args.neb_final}")
            
            from ase.mep import NEB
            from ase.optimize import FIRE
            
            try:
                initial_structure = read(args.neb_initial)
                final_structure = read(args.neb_final)
            except Exception as e:
                logging.error(f"Error reading structure files: {e}")
                return
            
            images = [initial_structure]
            images += [initial_structure.copy() for _ in range(args.neb_images)]
            images += [final_structure]
            
            neb = NEB(images, climb=True, method='improvedtangent', allow_shared_calculator=True)
            neb.interpolate()
            
            for image in images:
                # Use cell from initial structure if available, otherwise use large box
                if initial_structure.cell.any():
                    image.set_cell(initial_structure.cell)
                    image.set_pbc(initial_structure.pbc)
                else:
                    image.set_cell([100, 100, 100])
                    image.set_pbc(False)
                image.set_calculator(calc)
            
            optimizer = FIRE(neb, trajectory=args.neb_output, logfile=neb_logfile)
            
            def log_neb_status():
                energies = [img.get_potential_energy() for img in images]
                # Use NEB's projected forces (same as optimizer uses)
                neb_forces = neb.get_forces()
                n_images = len(images)
                n_atoms = len(images[0])
                forces_reshaped = neb_forces.reshape(n_images, n_atoms, 3)
                intermediate_forces = forces_reshaped[1:-1]
                max_force = (intermediate_forces**2).sum(axis=2).max()**0.5
                logging.info(
                    f"NEB step {optimizer.nsteps:4d} | E_min = {min(energies):.6f} eV | "
                    f"E_max = {max(energies):.6f} eV | "
                    f"Barrier = {max(energies) - energies[0]:.4f} eV | "
                    f"F_max = {max_force:.4f} eV/Å"
                )
            
            optimizer.attach(log_neb_status, interval=1)
            
            # Log fmax value before optimization to confirm it's being used
            logging.info(f"Starting NEB optimization with fmax={args.neb_fmax} eV/Å...")
            
            try:
                optimizer.run(fmax=args.neb_fmax)
                logging.info("NEB optimization completed successfully!")
                
                # Print final results
                energies = [img.get_potential_energy() for img in images]
                barrier = max(energies) - energies[0]
                reverse_barrier = max(energies) - energies[-1]
                logging.info("=" * 60)
                logging.info("NEB Results:")
                logging.info(f"  Forward barrier:  {barrier:.4f} eV")
                logging.info(f"  Reverse barrier:  {reverse_barrier:.4f} eV")
                logging.info(f"  Reaction energy:  {energies[-1] - energies[0]:.4f} eV")
                logging.info("=" * 60)
            except Exception as e:
                logging.critical(f"Error during NEB optimization: {e}")
        
        if args.md_sim:
            logging.info("=" * 60)
            logging.info("MD Simulation")
            logging.info("=" * 60)
            logging.info(f"  Input structure:  {args.md_input}")
            logging.info(f"  Temperature:      {args.md_temperature} K")
            logging.info(f"  Timestep:         {args.md_timestep} fs")
            logging.info(f"  Friction:         {args.md_friction}")
            logging.info(f"  Total steps:      {args.md_steps}")
            logging.info(f"  Log interval:     {args.md_log_interval}")
            logging.info(f"  Output file:      {args.md_output}")
            logging.info(f"  Skip relaxation:  {args.md_no_relax}")
            logging.info("=" * 60)
            
            # Validate input file exists
            if not os.path.exists(args.md_input):
                raise FileNotFoundError(f"MD input structure file not found: {args.md_input}")
            
            try:
                atoms = read(args.md_input)
                logging.info(f"Loaded {len(atoms)} atoms from {args.md_input}")
            except Exception as e:
                logging.error(f"Error reading {args.md_input}: {e}")
                return
            
            atoms.set_calculator(calc)
            
            # Initial relaxation (optional)
            if not args.md_no_relax:
                logging.info(f"Relaxing structure (fmax={args.md_relax_fmax} eV/Å)...")
                opt = BFGS(atoms, logfile='md_relax.log')
                opt.run(fmax=args.md_relax_fmax)
                logging.info("Relaxation completed.")
                write('relaxed_structure.xyz', atoms)
                logging.info("Saved relaxed structure to relaxed_structure.xyz")
            
            # Initialize velocities
            logging.info(f"Initializing velocities at {args.md_temperature} K...")
            MaxwellBoltzmannDistribution(atoms, temperature_K=args.md_temperature)
            
            # Setup Langevin dynamics
            dyn = Langevin(
                atoms, 
                args.md_timestep * units.fs, 
                temperature_K=args.md_temperature, 
                friction=args.md_friction
            )
            
            # Calculate total simulation time
            total_time_fs = args.md_steps * args.md_timestep
            total_time_ps = total_time_fs / 1000
            logging.info(f"Starting MD: {args.md_steps} steps = {total_time_ps:.2f} ps")
            
            # Log file
            md_log_file = args.md_output.replace('.xyz', '_log.txt').replace('.traj', '_log.txt')
            if md_log_file == args.md_output:
                md_log_file = 'md_log.txt'
            
            def print_status():
                time_fs = dyn.nsteps * args.md_timestep
                time_ps = time_fs / 1000
                logging.info(
                    f"Step {dyn.nsteps:6d} | t = {time_ps:8.3f} ps | "
                    f"T = {atoms.get_temperature():6.1f} K | "
                    f"E_pot = {atoms.get_potential_energy():10.4f} eV | "
                    f"E_tot = {atoms.get_total_energy():10.4f} eV"
                )
            
            dyn.attach(MDLogger(dyn, atoms, md_log_file, header=True, mode="w"), interval=args.md_log_interval)
            dyn.attach(print_status, interval=args.md_log_interval)
            dyn.attach(lambda: write(args.md_output, atoms, append=True), interval=args.md_log_interval)
            
            try:
                dyn.run(args.md_steps)
                logging.info("=" * 60)
                logging.info("MD simulation completed successfully!")
                logging.info(f"  Final temperature: {atoms.get_temperature():.1f} K")
                logging.info(f"  Final energy:      {atoms.get_potential_energy():.4f} eV")
                logging.info(f"  Trajectory saved:  {args.md_output}")
                logging.info(f"  Log saved:         {md_log_file}")
                logging.info("=" * 60)
            except Exception as e:
                logging.critical(f"Error during MD simulation: {e}")
            finally:
                if use_ddp_md and world_size > 1:
                    from molecular_force_field.cli.inference_ddp import run_one_ddp_inference_from_ase_atoms
                    dtype_calc = next(e3trans.parameters()).dtype
                    run_one_ddp_inference_from_ase_atoms(
                        None, e3trans, args.max_radius, device, dtype_calc,
                        return_forces=True, atomic_energies_dict=ref_energies_dict,
                    )
                    dist.destroy_process_group()


if __name__ == '__main__':
    main()