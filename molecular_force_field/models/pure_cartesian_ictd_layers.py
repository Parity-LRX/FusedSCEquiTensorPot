"""
Pure-Cartesian ICTD-IRREPS layers (internal representation is irreps, NOT 3^L).

Goal (speed):
  - Replace internal 3^L rank-tensor features with an irreps representation of size
    sum_{l=0..lmax}(2l+1) = (lmax+1)^2 per channel.
  - Compute edge direction features WITHOUT spherical harmonics:
      use harmonic polynomials / ICTD basis (see `ictd_irreps.direction_harmonics`).
  - Perform tensor products in the SAME harmonic basis using coupling tensors computed
    by polynomial multiplication + trace-chain projection (see `ictd_irreps`).

This provides an SO(3)-equivariant irreps message passing stack without ever materializing
3^L tensors in the forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from molecular_force_field.utils.scatter import scatter
from e3nn.math import soft_one_hot_linspace
import math

from molecular_force_field.models.ictd_irreps import (
    HarmonicElementwiseProduct,
    HarmonicFullyConnectedTensorProduct,
    direction_harmonics,
    direction_harmonics_all,
)
from molecular_force_field.models.mlp import RobustScalarWeightedSum
from molecular_force_field.models.mlp import MainNet


def _irreps_total_dim(channels: int, lmax: int) -> int:
    return channels * (lmax + 1) ** 2


def _split_irreps(x: torch.Tensor, channels: int, lmax: int) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    idx = 0
    for l in range(lmax + 1):
        d = channels * (2 * l + 1)
        blk = x[..., idx: idx + d]
        idx += d
        out[l] = blk.view(*x.shape[:-1], channels, 2 * l + 1)
    return out


def _merge_irreps(blocks: dict[int, torch.Tensor], channels: int, lmax: int) -> torch.Tensor:
    parts = []
    for l in range(lmax + 1):
        parts.append(blocks[l].reshape(*blocks[l].shape[:-2], channels * (2 * l + 1)))
    return torch.cat(parts, dim=-1)

def _irreps_elementwise_tensor_product_0e(x1: torch.Tensor, x2: torch.Tensor, channels: int, lmax: int) -> torch.Tensor:
    """
    Irreps analogue of e3nn ElementwiseTensorProduct filtered to 0e:
    for each l-block, contract over m (angular index) to produce one scalar per channel.

    x: (..., channels*(lmax+1)^2) arranged as concat over l=0..lmax of (channels, 2l+1).
    Returns: (..., channels*(lmax+1)) arranged as concat over l=0..lmax of (channels).
    """
    b1 = _split_irreps(x1, channels, lmax)
    b2 = _split_irreps(x2, channels, lmax)
    outs = []
    for l in range(lmax + 1):
        # (..., channels, 2l+1) -> (..., channels)
        # e3nn-style component normalization: divide by sqrt(2l+1)
        outs.append((b1[l] * b2[l]).sum(dim=-1) / ((2 * l + 1) ** 0.5))
    return torch.cat(outs, dim=-1)


def _apply_channel_adapter_per_l(x_l: torch.Tensor, adapter: nn.Module) -> torch.Tensor:
    """
    Apply a channel adapter Linear/Identity to an irreps block.

    x_l: (..., Cin, 2l+1)
    adapter: maps Cin -> Cout (or Identity)
    Returns: (..., Cout, 2l+1)
    """
    if isinstance(adapter, nn.Identity):
        return x_l
    # move channel to last dim for nn.Linear
    # (..., Cin, m) -> (..., m, Cin) -> Linear -> (..., m, Cout) -> (..., Cout, m)
    y = adapter(x_l.movedim(-2, -1))
    return y.movedim(-1, -2)


def _elementwise_tensor_product_0e_blocks(
    b1: dict[int, torch.Tensor],
    b2: dict[int, torch.Tensor],
    muls_by_l: dict[int, int],
    lmax: int,
) -> torch.Tensor:
    """
    Generalized elementwise 0e invariant builder aligned with e3nn's
    ElementwiseTensorProduct(..., ["0e"], normalization="component") semantics:

      out_l[c] = sum_m x_l[c,m] * y_l[c,m] / sqrt(2l+1)

    b1[l], b2[l]: (..., mul_l, 2l+1)
    Returns concatenated (..., sum_l mul_l) in l=0..lmax order.
    """
    outs = []
    for l in range(lmax + 1):
        x = b1[l]
        y = b2[l]
        # component normalization like e3nn
        outs.append((x * y).sum(dim=-1) / math.sqrt(2 * l + 1))
    return torch.cat(outs, dim=-1)


class ICTDIrrepsE3Conv(nn.Module):
    """
    First convolution in ICTD-irreps space (channelwise form: no neighbor in TP).

      scalar(Ai) ⊗ Y_l(n) -> f_in (output_size copies per l)
      then TP(f_in, edge_Y; weights(r)) -> channels_out irreps  [second operand is edge geometry only, mul=1]
      scatter_sum to receivers, normalize by avg_num_neighbors (global).
    """

    def __init__(
        self,
        max_radius: float,
        number_of_basis: int,
        channels_out: int,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        output_size: int = 8,
        lmax: int = 2,
        function_type: str = "gaussian",
        # ICTD tensor-product path control (e3nn-instructions-like)
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        # Normalize messages by this (default None = use num_edges/num_nodes at runtime)
        avg_num_neighbors: float | None = None,
        # Internal computation dtype for ICTD operations (default: float64 for stability)
        internal_compute_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_out = channels_out
        self.output_size = output_size
        self.lmax = lmax
        self.function_type = function_type
        self.avg_num_neighbors = avg_num_neighbors

        self.atom_embedding = nn.Embedding(max_atomvalue, embedding_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_size),
        )

        # Channelwise: (node ⊗ edge_Y) only; second input has mul=1 (edge geometry)
        self.tp2 = HarmonicFullyConnectedTensorProduct(
            mul_in1=output_size,
            mul_in2=1,
            mul_out=channels_out,
            lmax=lmax,
            internal_weights=True,
            path_policy=ictd_tp_path_policy,
            max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
        )
        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp2.num_paths),
        )

        self.output_dim = _irreps_total_dim(channels_out, lmax)

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, *, precomputed_n=None, precomputed_edge_length=None, precomputed_Y_list=None):
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        if precomputed_n is None or precomputed_edge_length is None:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
            edge_length = edge_vec.norm(dim=1)
            n = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1)
        else:
            n = precomputed_n
            edge_length = precomputed_edge_length

        Ai = self.atom_mlp(self.atom_embedding(A.long()))  # (N, output_size)
        n = n.to(dtype=Ai.dtype)
        edge_length = edge_length.to(dtype=Ai.dtype)

        if precomputed_Y_list is None:
            Y_list = direction_harmonics_all(n, self.lmax)
        else:
            Y_list = precomputed_Y_list
        Y = {l: Y_list[l] for l in range(self.lmax + 1)}  # (E, 2l+1)

        f_in = {l: Ai[edge_src].unsqueeze(-1) * Y[l].unsqueeze(-2) for l in range(self.lmax + 1)}  # (E, output_size, 2l+1)
        # Second operand: edge geometry only (mul=1), no neighbor
        x2 = {l: Y_list[l].unsqueeze(-2) for l in range(self.lmax + 1)}  # (E, 1, 2l+1)

        emb = soft_one_hot_linspace(edge_length, 0.0, self.max_radius, self.number_of_basis, basis=self.function_type, cutoff=True)
        emb = emb.mul(self.number_of_basis ** 0.5).to(dtype=Ai.dtype)
        gates = self.fc(emb)
        out_blocks = self.tp2(f_in, x2, gates)
        edge_features = _merge_irreps(out_blocks, self.channels_out, self.lmax)

        num_nodes = pos.size(0)
        out = scatter(edge_features, edge_dst, dim=0, dim_size=num_nodes, reduce="sum")
        avg = self.avg_num_neighbors if self.avg_num_neighbors is not None else (float(edge_src.numel()) / float(max(num_nodes, 1)))
        out = out / max(avg, 1e-8)
        return out


class PureCartesianICTDTransformerLayer(nn.Module):
    """
    Pure-Cartesian Transformer layer with ICTD (trace-chain) invariants for readout.

    ICTD-irreps internal model:
      - node features are stored as irreps blocks l=0..lmax (2l+1 dims each)
      - edge direction irreps Y_l(n) are computed from Cartesian n without spherical harmonics
      - tensor products use harmonic-basis CG tensors computed by polynomial multiplication + trace-chain projection
    """

    def __init__(
        self,
        max_embed_radius: float,
        main_max_radius: float,
        main_number_of_basis: int,
        hidden_dim_conv: int,
        hidden_dim_sh: int,
        hidden_dim: int,
        channel_in2: int = 32,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        output_size: int = 8,
        embed_size=None,
        main_hidden_sizes3=None,
        num_layers: int = 1,
        num_interaction: int = 2,
        device=None,
        function_type_main: str = "gaussian",
        lmax: int = 2,
        ictd_Lmax: int = 6,
        # ICTD tensor-product path control (e3nn-instructions-like)
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        # Keep these for backward compatibility; currently unused in ICTD mode.
        max_rank_other: int = 1,
        k_policy: str = "k0",
        # Internal computation dtype for ICTD operations (default: float64 for stability)
        internal_compute_dtype: torch.dtype = torch.float64,
        # Optional: allow per-l multiplicities for the "product_5-like" scalar invariant vector.
        # If None: keep current behavior (mul_l = channels for all l).
        # If provided: dict l->mul_l for l=0..lmax; used only for the readout invariants.
        product5_muls_by_l: dict[int, int] | None = None,
    ):
        super().__init__()
        if embed_size is None:
            embed_size = [128, 128, 128]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.lmax = int(lmax)
        self.channels = int(hidden_dim_conv)
        self.irreps_dim = _irreps_total_dim(self.channels, self.lmax)

        self.num_interaction = int(num_interaction)
        if self.num_interaction < 2:
            raise ValueError(f"num_interaction must be >= 2, got {self.num_interaction}")

        self.max_radius = float(max_embed_radius)
        self.number_of_basis = int(main_number_of_basis)
        self.function_type = str(function_type_main)

        # conv1
        self.e3_conv_emb = ICTDIrrepsE3Conv(
            max_radius=max_embed_radius,
            number_of_basis=main_number_of_basis,
            channels_out=self.channels,
            embedding_dim=embedding_dim,
            max_atomvalue=max_atomvalue,
            output_size=output_size,
            lmax=self.lmax,
            function_type=function_type_main,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
        )

        # conv2..convN: node irreps (mul=C) x edge Y_l (mul=1) -> node irreps (mul=C), per-edge weights
        self.tp2_layers = nn.ModuleList()
        self.fc2_layers = nn.ModuleList()
        for _ in range(self.num_interaction - 1):
            tp2 = HarmonicFullyConnectedTensorProduct(
                mul_in1=self.channels,
                mul_in2=1,
                mul_out=self.channels,
                lmax=self.lmax,
                internal_weights=True,
                path_policy=ictd_tp_path_policy,
                max_rank_other=ictd_tp_max_rank_other,
                internal_compute_dtype=internal_compute_dtype,
            )
            fc2 = nn.Sequential(
                nn.Linear(main_number_of_basis, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, tp2.num_paths),
            )
            self.tp2_layers.append(tp2)
            self.fc2_layers.append(fc2)

        # Readout invariants:
        #  - scalars: per-l channel Gram -> 32
        #  - norms: per-l per-channel L2 over m
        combined_channels = self.channels * self.num_interaction
        scalar_channels = (self.num_interaction - 1) * 32
        self.W_read = nn.ParameterList([
            nn.Parameter(torch.randn(scalar_channels, combined_channels, combined_channels) * 0.02)
            for _ in range(self.lmax + 1)
        ])
        self.readout_linear = nn.Sequential(
            nn.Linear(scalar_channels + (self.lmax + 1) * combined_channels, embed_size[0]),
            nn.SiLU(),
            nn.Linear(embed_size[0], 17),
        )
        self.weighted_sum = RobustScalarWeightedSum(17, init_weights="zero")
        # Match e3nn-style product_5:
        # T = cat([f1..fn, scalars]); ElementwiseTensorProduct(T,T)->0e
        if product5_muls_by_l is None:
            self.product5_muls_by_l = {l: self.channels for l in range(self.lmax + 1)}
        else:
            # Validate keys and values
            self.product5_muls_by_l = {int(k): int(v) for k, v in product5_muls_by_l.items()}
            for l in range(self.lmax + 1):
                if l not in self.product5_muls_by_l:
                    raise ValueError(f"product5_muls_by_l missing l={l} (must cover 0..lmax)")
                if self.product5_muls_by_l[l] <= 0:
                    raise ValueError(f"product5_muls_by_l[{l}] must be positive, got {self.product5_muls_by_l[l]}")

        # Optional per-l channel adapters for f1..fn to match desired muls_by_l.
        self._p5_adapt = nn.ModuleList()
        for _ in range(self.num_interaction):
            layer_adapt = nn.ModuleDict()
            for l in range(self.lmax + 1):
                out_ch = self.product5_muls_by_l[l]
                if out_ch == self.channels:
                    layer_adapt[str(l)] = nn.Identity()
                else:
                    layer_adapt[str(l)] = nn.Linear(self.channels, out_ch, bias=False)
            self._p5_adapt.append(layer_adapt)

        # HarmonicElementwiseProduct replaces manual _irreps_elementwise_tensor_product_0e.
        self.product_5 = HarmonicElementwiseProduct(
            lmax=self.lmax,
            mul=combined_channels,
            irreps_out="0e",
            internal_compute_dtype=internal_compute_dtype,
        )

        sum_mul = sum(self.product5_muls_by_l[l] for l in range(self.lmax + 1))
        self.proj_total = MainNet(self.num_interaction * sum_mul + scalar_channels, embed_size, 17)

    def forward(
        self,
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        *,
        precomputed_edge_vec=None,
        return_combined_features: bool = False,
        sync_after_scatter: callable | None = None,
    ):
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]

        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec[sort_idx]
        else:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)
        n = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1)

        # conv1
        # compute Y_l once and reuse
        Y_list = direction_harmonics_all(n.to(dtype=next(self.parameters()).dtype), self.lmax)
        f1 = self.e3_conv_emb(
            pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
            precomputed_n=n,
            precomputed_edge_length=edge_length,
            precomputed_Y_list=Y_list,
        )  # (N, C*(lmax+1)^2)
        if sync_after_scatter is not None:
            f1 = sync_after_scatter(f1)
        features = [f1]

        # conv2..convN: node irreps x edge Y_l -> node irreps (channelwise); scatter_sum then / avg_num_neighbors
        num_nodes = pos.size(0)
        avg_num_neighbors = float(edge_src.numel()) / float(max(num_nodes, 1))
        emb_base = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis=self.function_type,
            cutoff=True,
        ).mul(self.number_of_basis ** 0.5)
        for tp2, fc2 in zip(self.tp2_layers, self.fc2_layers):
            f_prev = features[-1]
            n = n.to(dtype=f_prev.dtype)
            edge_length = edge_length.to(dtype=f_prev.dtype)
            Y = {l: Y_list[l].to(dtype=f_prev.dtype).unsqueeze(-2) for l in range(self.lmax + 1)}  # (E,1,2l+1)
            emb = emb_base.to(dtype=f_prev.dtype)
            gates = fc2(emb)

            x1 = _split_irreps(f_prev, self.channels, self.lmax)
            x1e = {l: x1[l][edge_src] for l in range(self.lmax + 1)}
            edge_blocks = tp2(x1e, Y, gates)  # dict l -> (E, C, 2l+1)
            edge_flat = _merge_irreps(edge_blocks, self.channels, self.lmax)
            f_next = scatter(edge_flat, edge_dst, dim=0, dim_size=num_nodes, reduce="sum") / max(avg_num_neighbors, 1e-8)
            if sync_after_scatter is not None:
                f_next = sync_after_scatter(f_next)
            features.append(f_next)

        f_combine = torch.cat(features, dim=-1)  # (N, nC*(lmax+1)^2)

        xb = _split_irreps(f_combine, self.channels * self.num_interaction, self.lmax)
        scalars = torch.zeros(f_combine.shape[0], (self.num_interaction - 1) * 32, device=f_combine.device, dtype=f_combine.dtype)
        for l in range(self.lmax + 1):
            t = xb[l]  # (N,nC,2l+1)
            # e3nn-style component normalization: divide by sqrt(2l+1)
            gram = torch.einsum("ncm,ndm->ncd", t, t) / ((2 * l + 1) ** 0.5)  # (N,2C,2C)
            scalars = scalars + torch.einsum("ocd,ncd->no", self.W_read[l], gram)

        # Build T = cat(features, scalars) per l, then product_5(T, T) → 0e.
        # Scalars are appended to the l=0 block so they also go through normalized EWP.
        all_identity = all(
            isinstance(self._p5_adapt[i][str(l)], nn.Identity)
            for i in range(self.num_interaction)
            for l in range(self.lmax + 1)
        )
        splits = [_split_irreps(f, self.channels, self.lmax) for f in features]
        T_blocks: dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            parts = []
            for i in range(len(features)):
                b_l = splits[i][l]
                if not all_identity:
                    b_l = _apply_channel_adapter_per_l(b_l, self._p5_adapt[i][str(l)])
                parts.append(b_l)
            T_blocks[l] = torch.cat(parts, dim=-2)
        T_blocks[0] = torch.cat([T_blocks[0], scalars.unsqueeze(-1)], dim=-2)
        f_prod5 = self.product_5(T_blocks, T_blocks)

        product_proj = self.proj_total(f_prod5)
        e_out = self.weighted_sum(product_proj)
        out = e_out.sum(dim=-1, keepdim=True)
        if return_combined_features:
            return out, f_combine
        return out

