"""Active learning module for DPGen2-style iterative training."""

__all__ = [
    "evaluate_pes_coverage",
    "ModelDeviCalculator",
    "ConfSelector",
    "merge_training_data",
    "ExplorationStage",
    "StageScheduler",
    "make_single_stage_scheduler",
]

# Lazy imports to avoid loading heavy deps (dscribe, etc.) at package load
def __getattr__(name):
    if name == "evaluate_pes_coverage":
        from molecular_force_field.active_learning.pes_coverage import evaluate_pes_coverage
        return evaluate_pes_coverage
    if name == "ModelDeviCalculator":
        from molecular_force_field.active_learning.model_devi import ModelDeviCalculator
        return ModelDeviCalculator
    if name == "ConfSelector":
        from molecular_force_field.active_learning.conf_selector import ConfSelector
        return ConfSelector
    if name == "merge_training_data":
        from molecular_force_field.active_learning.data_merge import merge_training_data
        return merge_training_data
    if name == "ExplorationStage":
        from molecular_force_field.active_learning.stage_scheduler import ExplorationStage
        return ExplorationStage
    if name == "StageScheduler":
        from molecular_force_field.active_learning.stage_scheduler import StageScheduler
        return StageScheduler
    if name == "make_single_stage_scheduler":
        from molecular_force_field.active_learning.stage_scheduler import make_single_stage_scheduler
        return make_single_stage_scheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
