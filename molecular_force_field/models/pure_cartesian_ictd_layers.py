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
from torch_scatter import scatter
from e3nn.math import soft_one_hot_linspace

from molecular_force_field.models.ictd_irreps import (
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


class ICTDIrrepsE3Conv(nn.Module):
    """
    First convolution in ICTD-irreps space:
      scalar(Ai) ⊗ Y_l(n) -> output_size copies of each l
      then (irreps) ⊗ scalar(Aj) with per-edge radial weights -> channels_out irreps
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
    ):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_out = channels_out
        self.output_size = output_size
        self.lmax = lmax
        self.function_type = function_type

        self.atom_embedding = nn.Embedding(max_atomvalue, embedding_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_size),
        )

        self.tp2 = HarmonicFullyConnectedTensorProduct(
            mul_in1=output_size,
            mul_in2=output_size,
            mul_out=channels_out,
            lmax=lmax,
            internal_weights=True,
            path_policy=ictd_tp_path_policy,
            max_rank_other=ictd_tp_max_rank_other,
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

        x2 = {l: torch.zeros(edge_src.shape[0], self.output_size, 2 * l + 1, device=Ai.device, dtype=Ai.dtype) for l in range(self.lmax + 1)}
        x2[0] = Ai[edge_dst].unsqueeze(-1)  # scalar only

        emb = soft_one_hot_linspace(edge_length, 0.0, self.max_radius, self.number_of_basis, basis=self.function_type, cutoff=True)
        emb = emb.mul(self.number_of_basis ** 0.5).to(dtype=Ai.dtype)
        gates = self.fc(emb)
        out_blocks = self.tp2(f_in, x2, gates)
        edge_features = _merge_irreps(out_blocks, self.channels_out, self.lmax)

        num_nodes = pos.size(0)
        neighbor_count = scatter(torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes).clamp(min=1).to(edge_features.dtype)
        out = scatter(edge_features, edge_dst, dim=0, dim_size=num_nodes).div(neighbor_count.unsqueeze(-1))
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
        )

        # conv2: node irreps (mul=C) x edge Y_l (mul=1) -> node irreps (mul=C), per-edge weights
        self.tp2 = HarmonicFullyConnectedTensorProduct(
            mul_in1=self.channels,
            mul_in2=1,
            mul_out=self.channels,
            lmax=self.lmax,
            internal_weights=True,
            path_policy=ictd_tp_path_policy,
            max_rank_other=ictd_tp_max_rank_other,
        )
        self.fc2 = nn.Sequential(
            nn.Linear(main_number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp2.num_paths),
        )

        # Readout invariants:
        #  - scalars: per-l channel Gram -> 32
        #  - norms: per-l per-channel L2 over m
        self.W_read = nn.ParameterList([
            nn.Parameter(torch.randn(32, self.channels * 2, self.channels * 2) * 0.02)
            for _ in range(self.lmax + 1)
        ])
        self.readout_linear = nn.Sequential(
            nn.Linear(32 + (self.lmax + 1) * (self.channels * 2), embed_size[0]),
            nn.SiLU(),
            nn.Linear(embed_size[0], 17),
        )
        self.weighted_sum = RobustScalarWeightedSum(17, init_weights="zero")
        # Match e3nn-style product_5:
        # T = cat([f1, f2, scalars]); ElementwiseTensorProduct(T,T)->0e
        self.proj_total = MainNet(2 * (self.channels * (self.lmax + 1)) + 32, embed_size, 17)

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]

        # Precompute edge geometry once and reuse for conv1 + conv2
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

        # conv2: node irreps x edge Y_l -> node irreps
        n = n.to(dtype=f1.dtype)
        edge_length = edge_length.to(dtype=f1.dtype)
        # reuse Y_list (computed above) and only reshape
        Y = {l: Y_list[l].to(dtype=f1.dtype).unsqueeze(-2) for l in range(self.lmax + 1)}  # (E,1,2l+1)
        emb = soft_one_hot_linspace(edge_length, 0.0, self.max_radius, self.number_of_basis, basis=self.function_type, cutoff=True)
        emb = emb.mul(self.number_of_basis ** 0.5).to(dtype=f1.dtype)
        gates = self.fc2(emb)

        x1 = _split_irreps(f1, self.channels, self.lmax)
        x1e = {l: x1[l][edge_src] for l in range(self.lmax + 1)}
        edge_blocks = self.tp2(x1e, Y, gates)  # dict l -> (E, C, 2l+1)
        edge_flat = _merge_irreps(edge_blocks, self.channels, self.lmax)
        num_nodes = pos.size(0)
        neighbor_count = scatter(torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes).clamp(min=1).to(edge_flat.dtype)
        f2 = scatter(edge_flat, edge_dst, dim=0, dim_size=num_nodes).div(neighbor_count.unsqueeze(-1))

        f_combine = torch.cat([f1, f2], dim=-1)  # (N, 2C*(lmax+1)^2)

        xb = _split_irreps(f_combine, self.channels * 2, self.lmax)
        scalars = torch.zeros(f_combine.shape[0], 32, device=f_combine.device, dtype=f_combine.dtype)
        for l in range(self.lmax + 1):
            t = xb[l]  # (N,2C,2l+1)
            # e3nn-style component normalization: divide by sqrt(2l+1)
            gram = torch.einsum("ncm,ndm->ncd", t, t) / ((2 * l + 1) ** 0.5)  # (N,2C,2C)
            scalars = scalars + torch.einsum("ocd,ncd->no", self.W_read[l], gram)

        inv1 = _irreps_elementwise_tensor_product_0e(f1, f1, self.channels, self.lmax)
        inv2 = _irreps_elementwise_tensor_product_0e(f2, f2, self.channels, self.lmax)
        inv3 = scalars * scalars
        f_prod5 = torch.cat([inv1, inv2, inv3], dim=-1)

        product_proj = self.proj_total(f_prod5)
        e_out = self.weighted_sum(product_proj)
        return e_out.sum(dim=-1, keepdim=True)

