"""Active learning main loop (DPGen2-style), with multi-stage scheduling."""

import logging
import os
from typing import Callable, List, Optional, Union

from molecular_force_field.active_learning.conf_selector import ConfSelector
from molecular_force_field.active_learning.data_merge import merge_training_data
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
) -> int:
    """
    Run iterations for one stage until converged or max_iters reached.

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

        checkpoints = train_ensemble(
            data_dir=data_dir,
            work_dir=iter_path,
            n_models=n_models,
            base_seed=42 + global_iter * 100,
            train_args=train_args,
        )

        traj_path = explore_fn(global_iter, checkpoints[0], stage)
        if not os.path.exists(traj_path):
            raise FileNotFoundError(f"Exploration failed: {traj_path}")

        import torch
        calc = ModelDeviCalculator(
            checkpoint_paths=checkpoints,
            device=torch.device(device),
            atomic_energy_file=e0_path,
        )
        model_devi_path = os.path.join(iter_path, "model_devi.out")
        calc.compute_from_trajectory(traj_path, output_path=model_devi_path)

        selector = ConfSelector(
            level_f_lo=stage.level_f_lo,
            level_f_hi=stage.level_f_hi,
            conv_accuracy=stage.conv_accuracy,
        )
        candidate_path = os.path.join(iter_path, "candidate.xyz")
        _, _, _, converged = selector.select(
            traj_path, model_devi_path, output_candidate_path=candidate_path
        )

        scheduler.increment_iter(stage_idx)

        if converged:
            logger.info(
                f"Stage [{stage_idx}] {stage_name} converged at local_iter {local_iter + 1}."
            )
            scheduler.mark_converged(stage_idx)
            return local_iter + 1

        if not os.path.exists(candidate_path):
            logger.warning("No candidates selected; continuing to next iteration.")
            continue

        labeled_path = os.path.join(iter_path, "labeled.xyz")
        label_fn(candidate_path, labeled_path, iter_path, checkpoint_path=checkpoints[0])
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
) -> None:
    """
    Run DPGen2-style active learning loop, with optional multi-stage scheduling.

    Args
    ----
    scheduler:
        A StageScheduler with one or more ExplorationStage objects.
        If None, a single stage is created from the flat parameters
        (n_iterations, level_f_lo, level_f_hi, conv_accuracy).
        The explore_fn and label_fn receive ``stage`` as a third argument
        when a scheduler is provided (see note below).

    explore_fn signature (multi-stage):
        explore_fn(global_iter_idx, checkpoint_path, stage: ExplorationStage) -> traj_path

    explore_fn signature (single-stage backward compat):
        explore_fn(iter_idx, checkpoint_path) -> traj_path
        Will be wrapped automatically.

    label_fn signature:
        label_fn(candidate_path, output_path, work_dir, checkpoint_path=None) -> output_path
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
        )
        global_iter_offset += iters_run

    logger.info(scheduler.summary())
    logger.info("Active learning loop finished.")
