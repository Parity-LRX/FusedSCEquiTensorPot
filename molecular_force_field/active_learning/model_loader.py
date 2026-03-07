"""Load E3NN model from checkpoint for active learning (model deviation, etc.)."""

import logging
import os
from typing import Dict, Optional, Tuple

import torch

from molecular_force_field.utils.config import ModelConfig

logger = logging.getLogger(__name__)


def build_e3trans_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    atomic_energy_file: Optional[str] = None,
    tensor_product_mode: Optional[str] = None,
    num_interaction: int = 2,
) -> Tuple[torch.nn.Module, ModelConfig]:
    """
    Build e3trans model from checkpoint and load state_dict.

    Returns:
        (e3trans, config)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mode = tensor_product_mode or ckpt.get("tensor_product_mode")
    if mode is None:
        raise ValueError(
            "tensor_product_mode not in checkpoint. Specify via --tensor-product-mode."
        )
    max_radius = float(ckpt.get("max_radius", 5.0))

    external_tensor_rank = ckpt.get("external_tensor_rank")
    if external_tensor_rank is None:
        state = ckpt.get("e3trans_state_dict", {})
        if "e3_conv_emb.external_tensor_scale_by_l" in state:
            external_tensor_rank = 1

    config = ModelConfig(dtype=torch.float64, max_radius=max_radius)
    if atomic_energy_file and os.path.exists(atomic_energy_file):
        config.load_atomic_energies_from_file(atomic_energy_file)
    elif config.atomic_energy_keys is None:
        config.atomic_energy_keys = torch.tensor([1, 6, 7, 8], dtype=torch.long)
        config.atomic_energy_values = torch.tensor(
            [-430.53, -821.03, -1488.19, -2044.35], dtype=torch.float64
        )
        logger.warning("Using default atomic energies; pass --atomic-energy-file for fitted E0.")

    cfg = config
    dtype = cfg.dtype
    k_spherical = dict(
        max_embed_radius=cfg.max_radius,
        main_max_radius=cfg.max_radius_main,
        main_number_of_basis=cfg.number_of_basis_main,
        irreps_input=cfg.get_irreps_output_conv(),
        irreps_query=cfg.get_irreps_query_main(),
        irreps_key=cfg.get_irreps_key_main(),
        irreps_value=cfg.get_irreps_value_main(),
        irreps_output=cfg.get_irreps_output_conv_2(),
        irreps_sh=cfg.get_irreps_sh_transformer(),
        hidden_dim_sh=cfg.get_hidden_dim_sh(),
        hidden_dim=cfg.emb_number_main_2,
        channel_in2=cfg.channel_in2,
        embedding_dim=cfg.embedding_dim,
        max_atomvalue=cfg.max_atomvalue,
        output_size=cfg.output_size,
        embed_size=cfg.embed_size,
        main_hidden_sizes3=cfg.main_hidden_sizes3,
        num_layers=cfg.num_layers,
        num_interaction=num_interaction,
        function_type_main=cfg.function_type,
        device=device,
    )
    k_cartesian = dict(
        max_embed_radius=cfg.max_radius,
        main_max_radius=cfg.max_radius_main,
        main_number_of_basis=cfg.number_of_basis_main,
        hidden_dim_conv=cfg.channel_in,
        hidden_dim_sh=cfg.get_hidden_dim_sh(),
        hidden_dim=cfg.emb_number_main_2,
        channel_in2=cfg.channel_in2,
        embedding_dim=cfg.embedding_dim,
        max_atomvalue=cfg.max_atomvalue,
        output_size=cfg.output_size,
        embed_size=cfg.embed_size,
        main_hidden_sizes3=cfg.main_hidden_sizes3,
        num_layers=cfg.num_layers,
        num_interaction=num_interaction,
        function_type_main=cfg.function_type,
        lmax=cfg.lmax,
        device=device,
    )

    from molecular_force_field.models import (
        E3_TransformerLayer_multi,
        CartesianTransformerLayer,
        CartesianTransformerLayerLoose,
        PureCartesianTransformerLayer,
        PureCartesianSparseTransformerLayer,
        PureCartesianICTDTransformerLayer,
    )
    from molecular_force_field.models.e3nn_layers_channelwise import (
        E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
    )
    from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
        PureCartesianICTDTransformerLayer as PureCartesianICTDTransformerLayerFull,
    )

    if mode == "spherical":
        e3trans = E3_TransformerLayer_multi(**k_spherical)
    elif mode == "spherical-save":
        e3trans = E3_TransformerLayer_multi_channelwise(**k_spherical)
    elif mode == "spherical-save-cue":
        from molecular_force_field.models.cue_layers_channelwise import (
            E3_TransformerLayer_multi as E3_TransformerLayer_multi_cue,
        )
        e3trans = E3_TransformerLayer_multi_cue(**k_spherical)
    elif mode == "partial-cartesian":
        e3trans = CartesianTransformerLayer(**k_cartesian)
    elif mode == "partial-cartesian-loose":
        e3trans = CartesianTransformerLayerLoose(**k_cartesian)
    elif mode == "pure-cartesian":
        e3trans = PureCartesianTransformerLayer(**k_cartesian)
    elif mode == "pure-cartesian-sparse":
        e3trans = PureCartesianSparseTransformerLayer(
            **k_cartesian, max_rank_other=1, k_policy="k0"
        )
    elif mode == "pure-cartesian-ictd":
        ictd_kwargs = dict(
            ictd_tp_path_policy="full",
            ictd_tp_max_rank_other=None,
            internal_compute_dtype=dtype,
        )
        if external_tensor_rank is not None:
            ictd_kwargs["external_tensor_rank"] = external_tensor_rank
        e3trans = PureCartesianICTDTransformerLayerFull(
            **k_cartesian,
            **ictd_kwargs,
        )
    elif mode == "pure-cartesian-ictd-save":
        e3trans = PureCartesianICTDTransformerLayer(
            **k_cartesian,
            ictd_tp_path_policy="full",
            ictd_tp_max_rank_other=None,
            internal_compute_dtype=dtype,
        )
    else:
        raise ValueError(
            f"Unsupported tensor_product_mode: {mode}. "
            "Supported: spherical, spherical-save, spherical-save-cue, "
            "partial-cartesian, partial-cartesian-loose, pure-cartesian, "
            "pure-cartesian-sparse, pure-cartesian-ictd, pure-cartesian-ictd-save"
        )

    e3trans = e3trans.to(device=device, dtype=dtype)
    state = ckpt.get("e3trans_ema_state_dict") or ckpt["e3trans_state_dict"]
    e3trans.load_state_dict(state, strict=True)
    e3trans.eval()
    return e3trans, config
