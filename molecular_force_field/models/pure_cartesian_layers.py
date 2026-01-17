"""
Pure Cartesian E(3) layers (no irreps, no spherical harmonics, no CG).

Representation choice (A-route, "most pure"):
  - Keep rank tensors directly with 3^L Cartesian components.
  - Store node features as a flat vector concatenating ranks L=0..Lmax:
      rank 0: channels * 1
      rank 1: channels * 3
      rank 2: channels * 9
      ...
      total per-channel dim = sum_{L=0..Lmax} 3^L

Edge features:
  - Use powers of the normalized edge direction n = r/||r||:
      rank L edge tensor = n^{⊗L}  (full Cartesian outer power)

Equivariant bilinear mixing:
  - Use PureCartesianTensorProduct (δ/ε contractions) for rank mixing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn.math import soft_one_hot_linspace
from torch_scatter import scatter

from molecular_force_field.models.mlp import MainNet, RobustScalarWeightedSum
from molecular_force_field.models.pure_cartesian import (
    PureCartesianTensorProductO3,
    edge_rank_powers,
    merge_by_rank,
    split_by_rank,
    total_dim,
    total_dim_o3,
    split_by_rank_o3,
    merge_by_rank_o3,
)


class PureCartesianE3Conv(nn.Module):
    """
    First pure-Cartesian convolution:
      - node scalar features (from atom embedding) x edge direction powers -> rank features
      - then bilinear with neighbor scalar and radial weights to produce channels_out

    This mirrors the structure of CartesianE3Conv, but with rank tensors (3^L) instead of irreps (2l+1).
    """

    def __init__(
        self,
        max_radius: float,
        number_of_basis: int,
        channels_out: int,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        output_size: int = 8,
        Lmax: int = 2,
        function_type: str = "gaussian",
    ):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_out = channels_out
        self.output_size = output_size
        self.Lmax = Lmax
        self.function_type = function_type

        # Node scalar features
        self.atom_embedding = nn.Embedding(max_atomvalue, embedding_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_size),
        )

        # tp2: (rank features, scalar) -> rank features, with per-edge weights
        # Inputs:
        #   x1: output_size channels over ranks
        #   x2: output_size channels rank0 only
        # Output:
        #   channels_out over ranks
        self.tp2 = PureCartesianTensorProductO3(C1=output_size, C2=output_size, Cout=channels_out, Lmax=Lmax, internal_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp2.weight_numel),
        )

        self.output_dim = total_dim_o3(channels_out, Lmax)

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        # Datasets may provide float64 positions/cell; keep module dtype consistent
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)

        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs

        num_nodes = pos.size(0)
        Ai = self.atom_mlp(self.atom_embedding(A.long()))  # (N, output_size) scalar channels
        # Match geometry dtype to feature dtype to avoid float/double promotion
        edge_vec = edge_vec.to(dtype=Ai.dtype)
        edge_length = edge_vec.norm(dim=1)

        # Edge rank powers (pure Cartesian)
        e = edge_rank_powers(edge_vec, self.Lmax, normalize=True)

        # tp1 (pure): scalar(Ai) * n^{⊗L}
        # Build O(3)-graded blocks: start with s=0 only, s=1 zeros.
        f1_blocks = {}
        f1_blocks[(0, 0)] = Ai[edge_src]
        f1_blocks[(1, 0)] = torch.zeros_like(Ai[edge_src])
        for L in range(1, self.Lmax + 1):
            base = Ai[edge_src].view(-1, self.output_size, *([1] * L)) * e[L].view(-1, 1, *([3] * L))
            f1_blocks[(0, L)] = base
            f1_blocks[(1, L)] = torch.zeros_like(base)

        f_in = merge_by_rank_o3(f1_blocks, self.output_size, self.Lmax)

        # Radial basis + weights
        emb = soft_one_hot_linspace(
            edge_length, 0.0, self.max_radius, self.number_of_basis, basis=self.function_type, cutoff=True
        ).mul(self.number_of_basis ** 0.5)
        # e3nn may produce float64 depending on internal defaults; keep dtype consistent
        emb = emb.to(dtype=Ai.dtype)
        weights = self.fc(emb)

        # Neighbor scalar (rank0 only) as second input (E, output_size * 1) but tp expects full ranks.
        # We'll pack rank0 and zeros for higher ranks.
        x2_blocks = {(0, 0): Ai[edge_dst], (1, 0): torch.zeros_like(Ai[edge_dst])}
        for L in range(1, self.Lmax + 1):
            z = torch.zeros(Ai.size(0), self.output_size, *([3] * L), device=Ai.device, dtype=Ai.dtype)[edge_dst]
            x2_blocks[(0, L)] = z
            x2_blocks[(1, L)] = torch.zeros_like(z)
        x2 = merge_by_rank_o3(x2_blocks, self.output_size, self.Lmax)

        edge_features = self.tp2(f_in, x2, weights)

        # Aggregate to nodes
        neighbor_count = scatter(torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes).clamp(min=1).to(edge_features.dtype)
        out = scatter(edge_features, edge_dst, dim=0, dim_size=num_nodes).div(neighbor_count.unsqueeze(-1))
        return out


class PureCartesianE3Conv2(nn.Module):
    """
    Second pure-Cartesian convolution:
      node rank features x edge rank powers -> node rank features
    """

    def __init__(
        self,
        max_radius: float,
        number_of_basis: int,
        channels_in: int,
        channels_out: int,
        Lmax: int = 2,
        function_type: str = "gaussian",
    ):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.Lmax = Lmax
        self.function_type = function_type

        # tp: (node ranks, edge ranks) -> node ranks, with per-edge weights
        self.tp = PureCartesianTensorProductO3(C1=channels_in, C2=1, Cout=channels_out, Lmax=Lmax, internal_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel),
        )

        self.output_dim = total_dim_o3(channels_out, Lmax)

    def forward(self, f_in, pos, batch, edge_src, edge_dst, edge_shifts, cell):
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)

        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_vec = edge_vec.to(dtype=f_in.dtype)
        edge_length = edge_vec.norm(dim=1)

        e = edge_rank_powers(edge_vec, self.Lmax, normalize=True)

        # Pack edge ranks with C2=1 channels, graded (s=0 only)
        e_blocks = {(0, 0): e[0].view(-1, 1), (1, 0): torch.zeros_like(e[0].view(-1, 1))}
        for L in range(1, self.Lmax + 1):
            base = e[L].view(-1, 1, *([3] * L))
            e_blocks[(0, L)] = base
            e_blocks[(1, L)] = torch.zeros_like(base)
        e_flat = merge_by_rank_o3(e_blocks, 1, self.Lmax)

        # Radial basis -> per-edge weights
        emb = soft_one_hot_linspace(
            edge_length, 0.0, self.max_radius, self.number_of_basis, basis=self.function_type, cutoff=True
        ).mul(self.number_of_basis ** 0.5)
        emb = emb.to(dtype=f_in.dtype)
        weights = self.fc(emb)

        num_nodes = pos.size(0)
        edge_features = self.tp(f_in[edge_src], e_flat, weights)

        neighbor_count = scatter(torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes).clamp(min=1).to(edge_features.dtype)
        out = scatter(edge_features, edge_dst, dim=0, dim_size=num_nodes).div(neighbor_count.unsqueeze(-1))
        return out


class PureCartesianInvariantBilinear(nn.Module):
    """
    Build invariant scalars from rank features using pure δ-contractions:
      For each rank L: inner product over tensor indices -> (channels, channels) Gram matrix
      Then learnable bilinear weights produce out_channels scalars.
    """

    def __init__(self, channels: int, out_channels: int, Lmax: int):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.Lmax = Lmax

        # One bilinear weight per rank: (out, ch, ch)
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(out_channels, channels, channels) * 0.02)
            for _ in range(Lmax + 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 2 * channels * sum 3^L)  (O(3)-graded: true + pseudo)
        blocks = split_by_rank_o3(x, self.channels, self.Lmax)
        outs = []
        for L in range(self.Lmax + 1):
            t0 = blocks[(0, L)]  # (N, ch, 3..)
            t1 = blocks[(1, L)]  # (N, ch, 3..)
            t0_flat = t0.reshape(t0.shape[0], self.channels, -1)
            t1_flat = t1.reshape(t1.shape[0], self.channels, -1)
            # δ-contraction over tensor indices -> channel Gram matrices (even scalars)
            gram0 = torch.einsum("nci,ndi->ncd", t0_flat, t0_flat)  # (N, ch, ch)
            gram1 = torch.einsum("nci,ndi->ncd", t1_flat, t1_flat)  # (N, ch, ch)
            gram = gram0 + gram1
            outs.append(torch.einsum("ocd,ncd->no", self.W[L], gram))
        return sum(outs)  # (N, out_channels)


class PureCartesianTransformerLayer(nn.Module):
    """
    Pure Cartesian replacement model (A-route).
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
    ):
        super().__init__()
        if embed_size is None:
            embed_size = [128, 128, 128]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.Lmax = lmax
        self.channels = hidden_dim_conv
        self.feature_dim = total_dim_o3(self.channels, self.Lmax)

        self.e3_conv_emb = PureCartesianE3Conv(
            max_radius=max_embed_radius,
            number_of_basis=main_number_of_basis,
            channels_out=self.channels,
            embedding_dim=embedding_dim,
            max_atomvalue=max_atomvalue,
            output_size=output_size,
            Lmax=self.Lmax,
            function_type=function_type_main,
        )
        self.e3_conv_emb2 = PureCartesianE3Conv2(
            max_radius=max_embed_radius,
            number_of_basis=main_number_of_basis,
            channels_in=self.channels,
            channels_out=self.channels,
            Lmax=self.Lmax,
            function_type=function_type_main,
        )

        combined_channels = self.channels * 2
        self.combined_dim = total_dim_o3(combined_channels, self.Lmax)

        # product_3: invariant bilinear to 32 scalars
        self.product_3 = PureCartesianInvariantBilinear(channels=combined_channels, out_channels=32, Lmax=self.Lmax)

        # product_5: invariants from T=T_high + scalars
        # We'll use simple per-rank self-inner invariants to produce readout features:
        self.readout_linear = nn.Sequential(
            nn.Linear(32 + (self.Lmax + 1) * combined_channels, embed_size[0]),
            nn.SiLU(),
            nn.Linear(embed_size[0], 17),
        )
        self.weighted_sum = RobustScalarWeightedSum(17, init_weights="zero")

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]

        f1 = self.e3_conv_emb(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        f2 = self.e3_conv_emb2(f1, pos, batch, edge_src, edge_dst, edge_shifts, cell)

        f_combine = torch.cat([f1, f2], dim=-1)  # (N, combined_dim)

        f_prod3 = self.product_3(f_combine)  # (N, 32)

        # Additional simple invariants per rank: channelwise L2 norms
        blocks = split_by_rank_o3(f_combine, self.channels * 2, self.Lmax)
        invs = []
        # Use only true-tensor (s=0) norms for invariants; pseudoscalars would flip under reflections.
        for L in range(self.Lmax + 1):
            t = blocks[(0, L)]
            t = t.reshape(t.shape[0], self.channels * 2, -1)
            invs.append((t * t).sum(dim=-1))
        invs_cat = torch.cat(invs, dim=-1)  # (N, (Lmax+1)*combined_channels)

        readout_in = torch.cat([f_prod3, invs_cat], dim=-1)
        proj = self.readout_linear(readout_in)
        e_out = self.weighted_sum(proj)
        return e_out.sum(dim=-1, keepdim=True)

