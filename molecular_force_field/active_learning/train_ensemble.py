"""Train multiple models with different random seeds for active learning ensemble."""

import glob
import logging
import os
import subprocess
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)


def train_ensemble(
    data_dir: str,
    work_dir: str,
    n_models: int = 4,
    base_seed: int = 42,
    train_args: Optional[List[str]] = None,
) -> List[str]:
    """
    Train n_models with different random seeds. mff-train saves to work_dir/checkpoint/model_{j}_*.pth.

    Args:
        data_dir: Data directory (processed_train.h5, processed_val.h5)
        work_dir: Output directory for checkpoints
        n_models: Number of models to train
        base_seed: Base seed; model j uses seed = base_seed + j
        train_args: Additional args for mff-train (e.g. ["--epochs", "100"])

    Returns:
        List of checkpoint paths (actual saved paths)
    """
    os.makedirs(work_dir, exist_ok=True)
    train_args = train_args or []
    checkpoint_paths = []
    ckpt_subdir = os.path.join(work_dir, "checkpoint")
    data_dir_abs = os.path.abspath(data_dir)
    for j in range(n_models):
        seed = base_seed + j
        ckpt_path = os.path.join(work_dir, f"model_{j}.pth")
        cmd = [
            sys.executable,
            "-m",
            "molecular_force_field.cli.train",
            "--data-dir",
            data_dir_abs,
            "--checkpoint",
            ckpt_path,
            "--seed",
            str(seed),
        ] + train_args
        logger.info(f"Training model {j}/{n_models} (seed={seed})...")
        ret = subprocess.run(cmd, cwd=work_dir)
        if ret.returncode != 0:
            raise RuntimeError(f"mff-train failed for model {j} (exit {ret.returncode})")
        # mff-train saves to checkpoint/model_{j}_spherical.pth (or _spherical_save etc.)
        pattern = os.path.join(ckpt_subdir, f"model_{j}_*.pth")
        found = glob.glob(pattern)
        if found:
            checkpoint_paths.append(found[0])
        else:
            # fallback: try exact path
            fallback = os.path.join(ckpt_subdir, f"model_{j}_spherical.pth")
            if os.path.exists(fallback):
                checkpoint_paths.append(fallback)
            else:
                raise FileNotFoundError(f"Checkpoint not found: {pattern}")
    return checkpoint_paths
