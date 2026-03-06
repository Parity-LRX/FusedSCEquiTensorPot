"""Active learning main loop (DPGen2-style), with multi-stage scheduling
and multi-layer candidate filtering.

Filtering layers
----------------
0. **Fail recovery** – optionally promote a bounded number of high-deviation
   (fail) frames so the model can learn from boundary regions.
1. **Uncertainty gate** – keep frames whose ``max_devi_f`` falls within
   ``[level_f_lo, level_f_hi)`` (DPGen2-style trust window).
2. **Diversity selection** – use structural fingerprints (SOAP, deviation-
   histogram, …) + Farthest Point Sampling to pick a maximally diverse
   subset and cap the DFT budget.
"""

import logging
import os
from typing import Callable, List, Optional, Union

import numpy as np
from ase.io import read as ase_read, write as ase_write

from molecular_force_field.active_learning.conf_selector import ConfSelector
from molecular_force_field.active_learning.data_merge import merge_training_data
from molecular_force_field.active_learning.diversity_selector import (
    DiversitySelector,
    load_per_atom_devi,
    parse_max_devi_f,
)
from molecular_force_field.active_learning.model_devi import ModelDeviCalculator
from molecular_force_field.active_learning.pes_coverage import evaluate_pes_coverage
from molecular_force_field.active_learning.stage_scheduler import (
    ExplorationStage,
    StageScheduler,
    make_single_stage_scheduler,
)
from molecular_force_field.active_learning.train_ensemble import train_ensemble

logger = logging.getLogger(__name__)


def _run_one_stage(
    stage_idx: int,
    stage: ExplorationStage,
    scheduler: StageScheduler,
    work_dir: str,
    data_dir: str,
    explore_fn: Callable,
    label_fn: Callable,
    n_models: int,
    device: str,
    e0_path: str,
    max_radius: float,
    train_args: Optional[list],
    global_iter_offset: int,
    diversity_selector: Optional[DiversitySelector] = None,
    fail_strategy: str = "discard",
    fail_max_select: int = 0,
    explore_structures: Optional[List[str]] = None,
    n_gpu: int = 1,
    max_parallel: int = 0,
    nnodes: int = 1,
    master_addr: str = "auto",
    master_port: int = 29500,
    launcher: str = "auto",
) -> int:
    """Run iterations for one stage until converged or max_iters reached.

    Returns the number of iterations actually run.
    """
    stage_name = stage.name or f"stage_{stage_idx}"
    iter_dir = os.path.join(work_dir, "iterations")
    os.makedirs(iter_dir, exist_ok=True)

    for local_iter in range(stage.max_iters):
        global_iter = global_iter_offset + local_iter
        logger.info(
            f"=== Stage [{stage_idx}] {stage_name}  "
            f"local_iter {local_iter + 1}/{stage.max_iters}  "
            f"(global iter {global_iter + 1}) ==="
        )
        iter_path = os.path.join(iter_dir, f"iter_{global_iter}")
        os.makedirs(iter_path, exist_ok=True)

        # ---- 1. Train ensemble ----
        checkpoints = train_ensemble(
            data_dir=data_dir,
            work_dir=iter_path,
            n_models=n_models,
            base_seed=42 + global_iter * 100,
            train_args=train_args,
            n_gpu=n_gpu,
            max_parallel=max_parallel,
            nnodes=nnodes,
            master_addr=master_addr,
            master_port=master_port,
            launcher=launcher,
        )

        # ---- 2. Explore (single or multi-structure) ----
        if explore_structures and len(explore_structures) > 1:
            combined_atoms = []
            for s_idx, struct_path in enumerate(explore_structures):
                sub_traj = os.path.join(iter_path, f"explore_traj_{s_idx}.xyz")
                explore_fn(
                    global_iter, checkpoints[0], stage,
                    input_structure=struct_path, output_traj=sub_traj,
                )
                if os.path.exists(sub_traj):
                    sub_atoms = ase_read(sub_traj, index=":")
                    combined_atoms.extend(sub_atoms)
                    logger.info(
                        f"  Structure {s_idx} ({os.path.basename(struct_path)}): "
                        f"{len(sub_atoms)} frames"
                    )
            traj_path = os.path.join(iter_path, "explore_traj.xyz")
            if not combined_atoms:
                raise FileNotFoundError(
                    "Multi-structure exploration produced no frames."
                )
            ase_write(traj_path, combined_atoms, format="extxyz")
            logger.info(
                f"Combined {len(combined_atoms)} frames from "
                f"{len(explore_structures)} structures"
            )
        else:
            traj_path = explore_fn(global_iter, checkpoints[0], stage)
            if not os.path.exists(traj_path):
                raise FileNotFoundError(f"Exploration failed: {traj_path}")

        # ---- 3. Model deviation (+ per-atom data for diversity) ----
        import torch
        calc = ModelDeviCalculator(
            checkpoint_paths=checkpoints,
            device=torch.device(device),
            atomic_energy_file=e0_path,
        )
        model_devi_path = os.path.join(iter_path, "model_devi.out")
        calc.compute_from_trajectory(traj_path, output_path=model_devi_path)

        # ---- 4. Layer 0 + 1: uncertainty gate (+ fail recovery) ----
        selector = ConfSelector(
            level_f_lo=stage.level_f_lo,
            level_f_hi=stage.level_f_hi,
            conv_accuracy=stage.conv_accuracy,
            fail_strategy=fail_strategy,
            fail_max_select=fail_max_select,
        )
        candidate_path = os.path.join(iter_path, "candidate.xyz")
        candidate_ids, _, _, converged = selector.select(
            traj_path, model_devi_path, output_candidate_path=candidate_path
        )

        scheduler.increment_iter(stage_idx)

        if converged:
            logger.info(
                f"Stage [{stage_idx}] {stage_name} converged at local_iter {local_iter + 1}."
            )
            scheduler.mark_converged(stage_idx)
            return local_iter + 1

        if not os.path.exists(candidate_path) or not candidate_ids:
            logger.warning("No candidates selected; continuing to next iteration.")
            continue

        # ---- 5. Layer 2: diversity selection ----
        if diversity_selector is not None and diversity_selector.metric != "none":
            candidate_atoms = ase_read(candidate_path, index=":")
            if len(candidate_atoms) > diversity_selector.max_select:
                all_max_devi = parse_max_devi_f(model_devi_path)
                candidate_max_devi = (
                    all_max_devi[np.array(candidate_ids)]
                    if len(all_max_devi) > 0
                    else None
                )

                per_atom_devi_cand = None
                if diversity_selector.metric == "devi_hist":
                    per_atom_path = model_devi_path.replace(".out", "_per_atom.txt")
                    if os.path.exists(per_atom_path):
                        all_per_atom = load_per_atom_devi(per_atom_path)
                        per_atom_devi_cand = [all_per_atom[i] for i in candidate_ids]

                diverse_local_ids = diversity_selector.select(
                    candidate_atoms,
                    max_devi_f=candidate_max_devi,
                    per_atom_devi=per_atom_devi_cand,
                )
                diverse_atoms = [candidate_atoms[i] for i in diverse_local_ids]
                ase_write(candidate_path, diverse_atoms, format="extxyz")
                logger.info(
                    f"Layer 2 (diversity): {len(candidate_atoms)} -> "
                    f"{len(diverse_atoms)} candidates"
                )

        # ---- 6. Label ----
        labeled_path = os.path.join(iter_path, "labeled.xyz")
        label_fn(candidate_path, labeled_path, iter_path, checkpoint_path=checkpoints[0])

        # ---- 7. Merge ----
        merge_training_data(
            data_dir=data_dir,
            new_xyz_path=labeled_path,
            e0_csv_path=e0_path,
            max_radius=max_radius,
        )

    logger.warning(
        f"Stage [{stage_idx}] {stage_name} reached max_iters={stage.max_iters} without convergence."
    )
    return stage.max_iters


def run_active_learning_loop(
    work_dir: str,
    data_dir: str,
    explore_fn: Callable,
    label_fn: Callable,
    n_models: int = 4,
    n_iterations: int = 20,
    level_f_lo: float = 0.05,
    level_f_hi: float = 0.50,
    conv_accuracy: float = 0.9,
    pre_eval: bool = True,
    explore_structure: Optional[str] = None,
    device: str = "cuda",
    atomic_energy_file: Optional[str] = None,
    max_radius: float = 5.0,
    train_args: Optional[list] = None,
    scheduler: Optional[StageScheduler] = None,
    diversity_selector: Optional[DiversitySelector] = None,
    fail_strategy: str = "discard",
    fail_max_select: int = 0,
    explore_structures: Optional[List[str]] = None,
    n_gpu: int = 1,
    max_parallel: int = 0,
    nnodes: int = 1,
    master_addr: str = "auto",
    master_port: int = 29500,
    launcher: str = "auto",
) -> None:
    """Run DPGen2-style active learning loop with multi-layer filtering.

    Parameters
    ----------
    diversity_selector :
        A :class:`DiversitySelector` instance.  ``None`` disables Layer 2
        (diversity) filtering — backward compatible.
    fail_strategy :
        ``"discard"`` (default) or ``"sample_topk"`` (Layer 0).
    fail_max_select :
        Number of fail frames to promote when ``fail_strategy="sample_topk"``.
    n_gpu :
        Number of GPUs for training each ensemble model (1 = single process).
    max_parallel :
        Max ensemble models trained simultaneously.  0 = auto
        (``available_gpus // n_gpu``).
    nnodes :
        Number of nodes for multi-node DDP (default 1 = single node).
    master_addr :
        Rendezvous address (``'auto'`` resolves via SLURM or hostname).
    master_port :
        Base rendezvous port (default 29500).
    launcher :
        ``'auto'`` / ``'local'`` / ``'slurm'``.

    Other parameters are unchanged from the original API.
    """
    os.makedirs(work_dir, exist_ok=True)
    e0_path = atomic_energy_file or os.path.join(data_dir, "fitted_E0.csv")

    if pre_eval:
        logger.info("Running PES coverage evaluation...")
        evaluate_pes_coverage(
            dataset_path=data_dir,
            output_path=os.path.join(work_dir, "pes_coverage_report.json"),
        )

    # Build scheduler if not provided (backward-compatible single-stage path)
    if scheduler is None:
        scheduler = make_single_stage_scheduler(
            level_f_lo=level_f_lo,
            level_f_hi=level_f_hi,
            conv_accuracy=conv_accuracy,
            max_iters=n_iterations,
            name="stage_0",
        )
        # Wrap old-style explore_fn (2 args) to new 3-arg signature
        _raw_explore_fn = explore_fn
        import inspect
        sig = inspect.signature(_raw_explore_fn)
        if len(sig.parameters) < 3:
            def explore_fn(iter_idx, ckpt, stage):  # noqa: F811
                return _raw_explore_fn(iter_idx, ckpt)

    logger.info(
        f"Starting active learning: {len(scheduler)} stage(s), "
        f"{sum(s.max_iters for s in scheduler.stages)} max total iterations."
    )
    if diversity_selector is not None and diversity_selector.metric != "none":
        logger.info(
            f"Diversity filter: metric={diversity_selector.metric}, "
            f"max_select={diversity_selector.max_select}"
        )
    if fail_strategy != "discard":
        logger.info(f"Fail recovery: strategy={fail_strategy}, max_select={fail_max_select}")
    if n_gpu > 1 or max_parallel > 1 or nnodes > 1:
        parts = [f"{n_gpu} GPU(s)/model"]
        if nnodes > 1:
            parts.append(f"{nnodes} node(s)")
        parts.append(f"max_parallel={'auto' if max_parallel == 0 else max_parallel}")
        if launcher != "auto":
            parts.append(f"launcher={launcher}")
        logger.info(f"Training config: {', '.join(parts)}")
    if explore_structures and len(explore_structures) > 1:
        logger.info(
            f"Multi-structure exploration: {len(explore_structures)} structures"
        )
        for i, sp in enumerate(explore_structures):
            logger.info(f"  [{i}] {sp}")

    global_iter_offset = 0
    for stage_idx, stage in scheduler:
        stage_name = stage.name or f"stage_{stage_idx}"
        logger.info(
            f">>> Entering stage [{stage_idx}] {stage_name}  "
            f"T={stage.temperature}K  steps={stage.nsteps}  "
            f"level_f=[{stage.level_f_lo}, {stage.level_f_hi}]  "
            f"conv_accuracy={stage.conv_accuracy}  max_iters={stage.max_iters}"
        )
        iters_run = _run_one_stage(
            stage_idx=stage_idx,
            stage=stage,
            scheduler=scheduler,
            work_dir=work_dir,
            data_dir=data_dir,
            explore_fn=explore_fn,
            label_fn=label_fn,
            n_models=n_models,
            device=device,
            e0_path=e0_path,
            max_radius=max_radius,
            train_args=train_args,
            global_iter_offset=global_iter_offset,
            diversity_selector=diversity_selector,
            fail_strategy=fail_strategy,
            fail_max_select=fail_max_select,
            explore_structures=explore_structures,
            n_gpu=n_gpu,
            max_parallel=max_parallel,
            nnodes=nnodes,
            master_addr=master_addr,
            master_port=master_port,
            launcher=launcher,
        )
        global_iter_offset += iters_run

    logger.info(scheduler.summary())
    logger.info("Active learning loop finished.")
