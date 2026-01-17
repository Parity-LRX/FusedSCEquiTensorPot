"""
ICTD/trace-chain based irreps (2l+1) utilities and tensor products WITHOUT spherical harmonics.

Goal:
  - Provide a small-lmax (<=6) SO(3)-irreps representation built from harmonic polynomials
    (a.k.a. STF tensors / Laplacian kernel), derived purely from Cartesian algebra.
  - Provide Clebsch-Gordan-like coupling tensors computed in THIS basis using only:
      - polynomial multiplication (in monomial coefficient space)
      - trace-chain / harmonic projection (via Laplacian kernel)
    No e3nn spherical_harmonics and no e3nn wigner_3j are used here.

Important:
  - The basis for each l is fixed by our construction (harmonic nullspace + weighted orthonormalization).
    The CG tensors are computed consistently in the same basis, so equivariance is exact by construction.
  - We target lmax<=6 for speed and simplicity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from molecular_force_field.models.ictd_fast import (
    _counts_list,
    _build_laplacian_matrix,
    _build_r2k_lift,
)


def sym_dim(L: int) -> int:
    """dim Sym^L(R^3) = (L+2 choose 2)."""
    return (L + 2) * (L + 1) // 2


def _double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    out = 1
    for k in range(n, 0, -2):
        out *= k
    return out


def _gaussian_moment(n: int) -> float:
    # E[x^n] for x~N(0,1)
    if n % 2 == 1:
        return 0.0
    return float(_double_factorial(n - 1))


@lru_cache(maxsize=None)
def _gram_gaussian(L: int) -> torch.Tensor:
    """
    O(3)-invariant Gram matrix on Sym^L (monomial coefficient basis).

    For monomials x^a y^b z^c (with a+b+c=L),
      <m_{abc}, m_{a'b'c'}> = E[x^{a+a'}] E[y^{b+b'}] E[z^{c+c'}]
    under isotropic Gaussian measure, which is rotation-invariant.
    """
    counts = _counts_list(L)
    D = len(counts)
    G = torch.zeros(D, D, dtype=torch.float64)
    for i, (a, b, c) in enumerate(counts):
        for j, (a2, b2, c2) in enumerate(counts):
            G[i, j] = _gaussian_moment(a + a2) * _gaussian_moment(b + b2) * _gaussian_moment(c + c2)
    return G


def _harmonic_basis_t(L: int, device=None, dtype=None) -> torch.Tensor:
    """
    Harmonic basis in monomial coefficient (t_{abc}) coordinates.

    Returns B_t with shape (Dsym(L), 2L+1) such that columns are orthonormal under
    weighted inner product <t,u> = t^T diag(w) u, where w = multinomial counts.
    """
    if L == 0:
        return torch.ones(1, 1, device=device, dtype=dtype)
    if L == 1:
        return torch.eye(3, device=device, dtype=dtype)

    # Build harmonic subspace as nullspace of Laplacian on Sym^L -> Sym^{L-2}
    Delta = _build_laplacian_matrix(L, dtype=torch.float64)  # (D_{L-2}, D_L)
    # nullspace via SVD: Delta = U S V^T, nullspace basis are last columns of V
    _, s, vh = torch.linalg.svd(Delta, full_matrices=True)
    rank = int((s > 1e-12).sum().item())
    B = vh[rank:].T.contiguous()  # (D_L, dim_null) == (D_L, 2L+1)

    # Orthonormalize under O(3)-invariant Gram matrix G_L
    G = _gram_gaussian(L)  # (D_L,D_L) float64
    M = B.T @ G @ B
    # symmetric PSD; use eigh and whitening
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.clamp(evals, min=1e-14)
    W = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
    B_ortho = (B @ W).contiguous()

    B_ortho = B_ortho.to(device=device, dtype=dtype)
    return B_ortho


@dataclass(frozen=True)
class HarmonicProjectors:
    """
    Projection matrices for the symmetric trace chain on Sym^L:
      Sym^L ~= ⊕_{k=0..floor(L/2)} r^{2k} Harm^{l},  l=L-2k

    We return projectors that map monomial coefficient vectors t_L (dim Sym^L)
    to harmonic coordinates c_l in the canonical basis B_t(l) (dim 2l+1).

      c_l = P_{L->l} t_L
    """

    Lmax: int
    P: Dict[Tuple[int, int], torch.Tensor]  # (L,l) -> (2l+1, Dsym(L))


@lru_cache(maxsize=None)
def build_harmonic_projectors(Lmax: int) -> HarmonicProjectors:
    """
    Build all P_{L->l} on CPU/float64 for stability; move to device/dtype at runtime.
    """
    P: Dict[Tuple[int, int], torch.Tensor] = {}
    for L in range(Lmax + 1):
        D_L = sym_dim(L)
        GL = _gram_gaussian(L)               # (D_L,D_L)

        for k in range(L // 2 + 1):
            l = L - 2 * k
            # Harmonic basis at degree l in t-coords
            B_l = _harmonic_basis_t(l, dtype=torch.float64)  # (D_l, 2l+1)
            # Lift to degree L via r^{2k}
            M = _build_r2k_lift(l, k, dtype=torch.float64)   # (D_L, D_l)
            V = (M @ B_l).contiguous()                       # (D_L, 2l+1)

            # Weighted least squares projection onto span(V) under <.,.>_L with diag(wL):
            # c = (V^T W V)^{-1} V^T W t
            G = V.T @ GL @ V  # (2l+1,2l+1)
            # Stabilize: symmetric positive definite for small L; use solve.
            Pinv = torch.linalg.solve(G, V.T @ GL)  # (2l+1, D_L)
            P[(L, l)] = Pinv.contiguous()

    return HarmonicProjectors(Lmax=Lmax, P=P)


def direction_harmonics(n: torch.Tensor, l: int) -> torch.Tensor:
    """
    Compute harmonic (irrep) coordinates of the symmetric tensor n^{⊗l} in our basis.

    n: (..., 3) unit vector (or any vector; scaling changes non-homogeneously for trace-chain, so use unit).
    Returns: (..., 2l+1)

    Derivation:
      Polynomial p(x,y,z) = (n_x x + n_y y + n_z z)^l
      has monomial coefficients t_{abc} = multinomial(l;a,b,c) n_x^a n_y^b n_z^c.
      Project to harmonic coordinates via:
        c = B_l^T W_l t    (since B_l orthonormal under W_l)
    """
    if l == 0:
        return torch.ones(*n.shape[:-1], 1, device=n.device, dtype=n.dtype)
    counts = _counts_list(l)
    # t_{abc}
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    t_list = []
    for (a, b, c) in counts:
        coef = math.factorial(l) / (math.factorial(a) * math.factorial(b) * math.factorial(c))
        t_list.append((nx**a) * (ny**b) * (nz**c) * float(coef))
    t = torch.stack(t_list, dim=-1)  # (..., Dsym(l))
    B = _harmonic_basis_t(l, device=n.device, dtype=n.dtype)  # (Dsym, 2l+1)
    # coords under Gram: c = B^T G t
    G = _gram_gaussian(l).to(device=n.device, dtype=n.dtype)  # (Dsym, Dsym)
    c = torch.einsum("...d,md,mc->...c", t, G, B)
    return c


@dataclass(frozen=True)
class CGKey:
    l1: int
    l2: int
    l3: int


@lru_cache(maxsize=None)
def build_cg_tensor(l1: int, l2: int, l3: int) -> torch.Tensor:
    """
    Build the coupling tensor C_{m1,m2,m3} in OUR harmonic basis.

    Semantics:
      Given harmonic coefficient vectors a in R^{2l1+1}, b in R^{2l2+1},
      define polynomial product at degree L=l1+l2, then project to the trace-chain block l3.
      The result is a harmonic coefficient vector c in R^{2l3+1}:
        c[m3] = sum_{m1,m2} a[m1] b[m2] C[m1,m2,m3]

    This is an SO(3)-equivariant intertwiner by construction.
    """
    L = l1 + l2
    if not (abs(l1 - l2) <= l3 <= l1 + l2) or ((l1 + l2 + l3) % 2 == 1):
        # parity selection for harmonic products: l1+l2-l3 must be even (k integer).
        return torch.zeros(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1, dtype=torch.float64)

    proj = build_harmonic_projectors(Lmax=L)
    P_L_l3 = proj.P[(L, l3)]  # (2l3+1, Dsym(L))

    # Bases in t-coords
    B1 = _harmonic_basis_t(l1, dtype=torch.float64)  # (D1, 2l1+1)
    B2 = _harmonic_basis_t(l2, dtype=torch.float64)  # (D2, 2l2+1)

    counts1 = _counts_list(l1)
    counts2 = _counts_list(l2)
    countsL = _counts_list(L)
    idxL = {t: i for i, t in enumerate(countsL)}
    D1, D2, DL = len(counts1), len(counts2), len(countsL)

    C = torch.zeros(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1, dtype=torch.float64)

    # For each basis vector e_{m1}, e_{m2} produce t_L via convolution in (a,b,c) counts
    for m1 in range(2 * l1 + 1):
        t1 = B1[:, m1]  # (D1,)
        for m2 in range(2 * l2 + 1):
            t2 = B2[:, m2]  # (D2,)
            tL = torch.zeros(DL, dtype=torch.float64)
            for i, (a, b, c) in enumerate(counts1):
                if t1[i] == 0:
                    continue
                for j, (a2, b2, c2) in enumerate(counts2):
                    v = t1[i] * t2[j]
                    if v == 0:
                        continue
                    k = idxL[(a + a2, b + b2, c + c2)]
                    tL[k] += v
            # project to l3 coeffs
            c3 = (P_L_l3 @ tL)  # (2l3+1,)
            C[m1, m2, :] = c3

    return C.contiguous()


class HarmonicFullyConnectedTensorProduct(nn.Module):
    """
    Fully-connected tensor product in harmonic/ICTD basis (SO(3) irreps, no spherical harmonics).

    Representation:
      input features are a dict l -> (..., mul_l, 2l+1) (mul_l is multiplicity/channels for that l).
      output is similarly l -> (..., mul_out_l, 2l+1).

    We follow the same "W[mul_out, mul1, mul2]" weight structure per (l1,l2->l3) path.
    """

    def __init__(self, mul_in1: int, mul_in2: int, mul_out: int, lmax: int, internal_weights: bool = True):
        super().__init__()
        self.mul_in1 = mul_in1
        self.mul_in2 = mul_in2
        self.mul_out = mul_out
        self.lmax = lmax
        self.internal_weights = internal_weights

        # Enumerate all valid (l1,l2,l3) with parity selection (even step)
        self.paths: List[Tuple[int, int, int]] = []
        for l1 in range(lmax + 1):
            for l2 in range(lmax + 1):
                for l3 in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                    if (l1 + l2 + l3) % 2 == 1:
                        continue
                    self.paths.append((l1, l2, l3))

        self.num_paths = len(self.paths)
        self.weight_numel = self.num_paths * mul_out * mul_in1 * mul_in2

        if internal_weights:
            # (P, mul_out, mul1, mul2)
            self.weight = nn.Parameter(torch.randn(self.num_paths, mul_out, mul_in1, mul_in2) * 0.02)
        else:
            self.register_parameter("weight", None)

    def forward(
        self,
        x1: Dict[int, torch.Tensor],
        x2: Dict[int, torch.Tensor],
        weights: torch.Tensor | None = None,
    ) -> Dict[int, torch.Tensor]:
        # Determine batch shape from any present block
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype

        if self.internal_weights:
            w_param = self.weight.to(device=device, dtype=dtype)  # (P, o, i, j)
        else:
            assert weights is not None
            w = weights
            if w.shape[-1] not in (self.weight_numel, self.num_paths):
                raise ValueError(f"weights last-dim must be weight_numel={self.weight_numel} or num_paths={self.num_paths}, got {w.shape[-1]}")

        # init output
        out: Dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            out[l] = torch.zeros(*batch_shape, self.mul_out, 2 * l + 1, device=device, dtype=dtype)

        idx = 0
        for p_idx, (l1, l2, l3) in enumerate(self.paths):
            if self.internal_weights and (weights is None or weights.shape[-1] == self.num_paths):
                # gated shared weights: gate is (...,) scalar per path
                gate = 1.0
                if weights is not None:
                    gate = weights[..., p_idx].to(device=device, dtype=dtype)
                Wp = w_param[p_idx]  # (o,i,j)
            else:
                # full per-example weights: (..., weight_numel)
                assert weights is not None
                block = self.mul_out * self.mul_in1 * self.mul_in2
                Wp = weights[..., idx: idx + block].view(*batch_shape, self.mul_out, self.mul_in1, self.mul_in2)
                idx += block
                gate = 1.0

            a = x1.get(l1)
            b = x2.get(l2)
            if a is None or b is None:
                continue

            C = build_cg_tensor(l1, l2, l3).to(device=device, dtype=dtype)  # (m1,m2,m3)
            # a: (..., mul1, m1), b: (..., mul2, m2), Wp: (..., mul_out, mul1, mul2)
            if self.internal_weights and (weights is None or weights.shape[-1] == self.num_paths):
                out_l3 = torch.einsum("...im,...jn,mnk,oij->...ok", a, b, C, Wp)
                if not isinstance(gate, float):
                    out_l3 = out_l3 * gate[..., None, None]
            else:
                out_l3 = torch.einsum("...im,...jn,mnk,...oij->...ok", a, b, C, Wp)
            out[l3] = out[l3] + out_l3

        return out

