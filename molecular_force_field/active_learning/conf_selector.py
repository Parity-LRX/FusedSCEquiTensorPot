"""Configuration selector for active learning (DPGen2-style trust levels)."""

import logging
from typing import List, Optional, Tuple

import numpy as np
from ase.io import read, write

logger = logging.getLogger(__name__)


class ConfSelector:
    """
    Select candidate configurations based on model deviation.
    Uses level_f_lo / level_f_hi (force deviation) and optional level_v_lo / level_v_hi.
    """

    def __init__(
        self,
        level_f_lo: float = 0.05,
        level_f_hi: float = 0.50,
        level_v_lo: Optional[float] = None,
        level_v_hi: Optional[float] = None,
        conv_accuracy: float = 0.9,
    ):
        self.level_f_lo = level_f_lo
        self.level_f_hi = level_f_hi
        self.level_v_lo = level_v_lo
        self.level_v_hi = level_v_hi
        self.conv_accuracy = conv_accuracy

    def select(
        self,
        traj_path: str,
        model_devi_path: str,
        output_candidate_path: str = "candidate.xyz",
    ) -> Tuple[List[int], List[int], List[int], bool]:
        """
        Select configurations from trajectory using model_devi.out.

        Returns:
            (candidate_indices, accurate_indices, fail_indices, converged)
        """
        atoms_list = read(traj_path, index=":")
        max_devi_f = []
        with open(model_devi_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    max_devi_f.append(float(parts[1]))
        max_devi_f = np.array(max_devi_f)
        if len(max_devi_f) != len(atoms_list):
            raise ValueError(
                f"model_devi.out has {len(max_devi_f)} lines but trajectory has {len(atoms_list)} frames"
            )

        id_candidate = np.where(
            (max_devi_f >= self.level_f_lo) & (max_devi_f < self.level_f_hi)
        )[0].tolist()
        id_accurate = np.where(max_devi_f < self.level_f_lo)[0].tolist()
        id_fail = np.where(max_devi_f >= self.level_f_hi)[0].tolist()

        n_total = len(max_devi_f)
        n_accurate = len(id_accurate)
        converged = (n_accurate / n_total) >= self.conv_accuracy if n_total > 0 else True

        if id_candidate:
            candidate_atoms = [atoms_list[i] for i in id_candidate]
            write(output_candidate_path, candidate_atoms, format="extxyz")
            logger.info(
                f"Selected {len(id_candidate)} candidates -> {output_candidate_path}"
            )
        logger.info(
            f"accurate={len(id_accurate)}, candidate={len(id_candidate)}, fail={len(id_fail)}, "
            f"converged={converged}"
        )
        return id_candidate, id_accurate, id_fail, converged
