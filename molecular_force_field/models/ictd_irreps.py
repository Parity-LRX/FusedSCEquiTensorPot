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


@lru_cache(maxsize=None)
def _dir_monomial_exps_coefs(l: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute monomial exponents and multinomial coefficients for degree l.

    Returns:
      exps: (Dsym(l), 3) int64, rows are (a,b,c)
      coefs: (Dsym(l),) float64, multinomial(l; a,b,c)
    """
    counts = _counts_list(l)
    exps = torch.tensor(counts, dtype=torch.int64)  # (D,3)
    if l == 0:
        coefs = torch.ones(1, dtype=torch.float64)
    else:
        coefs_list = []
        for (a, b, c) in counts:
            coefs_list.append(math.factorial(l) / (math.factorial(a) * math.factorial(b) * math.factorial(c)))
        coefs = torch.tensor(coefs_list, dtype=torch.float64)
    return exps, coefs


@lru_cache(maxsize=None)
def _dir_proj_cpu_f64(l: int) -> torch.Tensor:
    """
    Precompute P_l = G_l @ B_l on CPU float64:
      t (..., Dsym) -> c (..., 2l+1) via c = t @ P_l
    """
    if l == 0:
        return torch.ones(1, 1, dtype=torch.float64)
    B = _harmonic_basis_t(l, dtype=torch.float64)  # (Dsym, 2l+1)
    G = _gram_gaussian(l)  # (Dsym, Dsym) float64
    return (G @ B).contiguous()  # (Dsym, 2l+1)


_dir_proj_cache_by_dev_dtype: Dict[Tuple[str, str, int], torch.Tensor] = {}


def direction_harmonics_fast(n: torch.Tensor, l: int) -> torch.Tensor:
    """
    Faster version of direction_harmonics with:
      - vectorized monomial evaluation
      - cached projection matrix (G@B) per (device,dtype,l)
    """
    if l == 0:
        return torch.ones(*n.shape[:-1], 1, device=n.device, dtype=n.dtype)

    key = (str(n.device), str(n.dtype), int(l))
    P = _dir_proj_cache_by_dev_dtype.get(key)
    if P is None:
        P = _dir_proj_cpu_f64(l).to(device=n.device, dtype=n.dtype)
        _dir_proj_cache_by_dev_dtype[key] = P

    exps, coefs = _dir_monomial_exps_coefs(l)
    exps = exps.to(device=n.device)
    coefs = coefs.to(device=n.device, dtype=n.dtype)

    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    a = exps[:, 0]
    b = exps[:, 1]
    c = exps[:, 2]
    # (..., Dsym)
    t = (nx.unsqueeze(-1) ** a) * (ny.unsqueeze(-1) ** b) * (nz.unsqueeze(-1) ** c)
    t = t * coefs
    # (..., 2l+1)
    return t @ P


def direction_harmonics_all(n: torch.Tensor, lmax: int) -> List[torch.Tensor]:
    """
    Compute direction harmonics for all l=0..lmax.
    Returns a list Y where Y[l] has shape (..., 2l+1).
    """
    return [direction_harmonics_fast(n, l) for l in range(int(lmax) + 1)]


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

    def __init__(
        self,
        mul_in1: int,
        mul_in2: int,
        mul_out: int,
        lmax: int,
        internal_weights: bool = True,
        *,
        # e3nn-instructions-like control: explicitly choose which (l1,l2,l3) paths exist.
        # If provided, this is the most precise "pruning" mechanism.
        allowed_paths: List[Tuple[int, int, int]] | None = None,
        # Convenience policy to generate allowed_paths.
        # - "full": keep all CG-allowed paths
        # - "max_rank_other": keep paths with min(l1,l2) <= max_rank_other (like sparse heuristic)
        path_policy: str = "full",
        max_rank_other: int | None = None,
        # Internal computation dtype for CG tensors and projections (default: float64 for stability)
        internal_compute_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.mul_in1 = mul_in1
        self.mul_in2 = mul_in2
        self.mul_out = mul_out
        self.lmax = lmax
        self.internal_weights = internal_weights
        self.internal_compute_dtype = internal_compute_dtype

        # Enumerate all valid (l1,l2,l3) with parity selection (even step)
        all_paths: List[Tuple[int, int, int]] = []
        for l1 in range(lmax + 1):
            for l2 in range(lmax + 1):
                for l3 in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                    if (l1 + l2 + l3) % 2 == 1:
                        continue
                    all_paths.append((l1, l2, l3))

        if allowed_paths is not None:
            allowed_set = set(allowed_paths)
            self.paths = [p for p in all_paths if p in allowed_set]
        else:
            if path_policy == "full":
                self.paths = all_paths
            elif path_policy == "max_rank_other":
                if max_rank_other is None:
                    raise ValueError("path_policy='max_rank_other' requires max_rank_other")
                self.paths = [p for p in all_paths if min(p[0], p[1]) <= int(max_rank_other)]
            else:
                raise ValueError(f"Unknown path_policy={path_policy!r}")

        self.num_paths = len(self.paths)
        self.weight_numel = self.num_paths * mul_out * mul_in1 * mul_in2

        if internal_weights:
            # (P, mul_out, mul1, mul2)
            self.weight = nn.Parameter(torch.randn(self.num_paths, mul_out, mul_in1, mul_in2) * 0.02)
        else:
            self.register_parameter("weight", None)

        # Cache CG tensors to avoid per-forward .to(device,dtype) allocations.
        #
        # build_cg_tensor(l1,l2,l3) returns a CPU float64 tensor (and is itself lru_cached),
        # but calling .to(device,dtype) for every path on every forward is costly (especially
        # for higher lmax with many paths). We keep:
        # - a CPU float64 list (built lazily) for the paths
        # - per-(device,dtype) converted lists for fast reuse
        self._cg_cpu_f64: List[torch.Tensor] | None = None
        self._cg_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

        # Group paths by (l1,l2). This enables an e3nn-like factorization:
        #   1) build the (l1,l2) tensor-product basis once (NO mul_out)
        #   2) apply per-path (mul_out,mul1,mul2) weights as a separate contraction
        # This avoids repeating the expensive m-contractions for every output channel and path.
        #
        # Each group stores:
        #   - l1, l2
        #   - p_indices: indices into self.paths (and self.weight / gates vector)
        #   - l3_list: l3 per path in group (aligned with p_indices)
        #   - segments: list of (p_idx, l3, start, end) into concatenated K_total
        self._groups: List[Dict[str, object]] = []
        groups_tmp: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for p_idx, (l1, l2, l3) in enumerate(self.paths):
            groups_tmp.setdefault((l1, l2), []).append((p_idx, l3))
        for (l1, l2), items in sorted(groups_tmp.items()):
            p_indices = [p for (p, _) in items]
            l3_list = [l3 for (_, l3) in items]
            segments = []
            start = 0
            for p_idx, l3 in items:
                kdim = 2 * l3 + 1
                segments.append((p_idx, l3, start, start + kdim))
                start += kdim
            self._groups.append(
                {
                    "l1": l1,
                    "l2": l2,
                    "p_indices": p_indices,
                    "l3_list": l3_list,
                    "segments": segments,
                    "k_total": start,
                }
            )

        # Cache per-group projection matrices per (device,dtype) matching _groups:
        #   U_g: (m1*m2, K_total), where K_total = sum_{paths in group} (2*l3+1)
        self._proj_group_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    def _get_cg_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        # Use internal_compute_dtype for CG tensors (for numerical stability)
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._cg_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached

        if self._cg_cpu_f64 is None:
            # Build CPU float64 tensors once, in path order.
            self._cg_cpu_f64 = [build_cg_tensor(l1, l2, l3) for (l1, l2, l3) in self.paths]

        cg_list = [C.to(device=device, dtype=compute_dtype) for C in self._cg_cpu_f64]
        self._cg_cache_by_dev_dtype[key] = cg_list
        return cg_list

    def _get_proj_group_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        """
        Returns a list of projection matrices U_g, one per (l1,l2) group:
          U_g: (m1*m2, K_total)
        such that for tensor-product coefficients t_{m1,m2} (flattened to m1*m2),
        the concatenated outputs for all paths in the group are:
          y_concat = t_flat @ U_g
        """
        # Use internal_compute_dtype for projection matrices (for numerical stability)
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._proj_group_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached

        cg_list = self._get_cg_list(device=device, dtype=dtype)
        proj_list: List[torch.Tensor] = []
        for g in self._groups:
            l1 = int(g["l1"])  # type: ignore[arg-type]
            l2 = int(g["l2"])  # type: ignore[arg-type]
            segments = g["segments"]  # type: ignore[assignment]
            k_total = int(g["k_total"])  # type: ignore[arg-type]

            m1 = 2 * l1 + 1
            m2 = 2 * l2 + 1
            U = torch.zeros(m1 * m2, k_total, device=device, dtype=compute_dtype)
            for p_idx, _l3, s, e in segments:  # type: ignore[misc]
                C = cg_list[int(p_idx)]  # (m1,m2,kdim)
                U[:, int(s): int(e)] = C.reshape(m1 * m2, int(e) - int(s))
            proj_list.append(U)

        self._proj_group_cache_by_dev_dtype[key] = proj_list
        return proj_list

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
        compute_dtype = self.internal_compute_dtype

        if self.internal_weights:
            # Assume module has already been moved to the right device/dtype by caller.
            w_param = self.weight  # (P, o, i, j)
        else:
            assert weights is not None
            w = weights
            if w.shape[-1] not in (self.weight_numel, self.num_paths):
                raise ValueError(f"weights last-dim must be weight_numel={self.weight_numel} or num_paths={self.num_paths}, got {w.shape[-1]}")

        # Make weights/gates device+dtype consistent once (avoid per-path .to()).
        # Only convert when needed; calling .to() unconditionally can add overhead.
        if weights is not None and (weights.device != device or weights.dtype != dtype):
            weights = weights.to(device=device, dtype=dtype)

        # init output
        out: Dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            out[l] = torch.zeros(*batch_shape, self.mul_out, 2 * l + 1, device=device, dtype=dtype)

        # Fast path: internal_weights + per-path scalar gates (this is what our models use).
        if self.internal_weights and (weights is None or weights.shape[-1] == self.num_paths):
            proj_list = self._get_proj_group_list(device=device, dtype=dtype)
            for g_idx, g in enumerate(self._groups):
                l1 = int(g["l1"])  # type: ignore[arg-type]
                l2 = int(g["l2"])  # type: ignore[arg-type]
                segments = g["segments"]  # type: ignore[assignment]
                k_total = int(g["k_total"])  # type: ignore[arg-type]

                a = x1.get(l1)
                b = x2.get(l2)
                if a is None or b is None:
                    continue

                # e3nn-like factorization:
                # 1) project to concatenated k space once: (..., i, j, K_total)
                # 2) batch channel mixing for all paths in group, then segment and accumulate
                m1 = 2 * l1 + 1
                m2 = 2 * l2 + 1
                U = proj_list[g_idx]  # (m1*m2, K_total) in compute_dtype
                # Convert inputs to compute_dtype for numerical stability
                a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
                b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
                # Optimized outer product: (..., i, m1) x (..., j, m2) -> (..., i, j, m1, m2)
                # Using broadcast instead of einsum for better performance
                t_mn = (a_comp.unsqueeze(-2).unsqueeze(-1) * b_comp.unsqueeze(-3).unsqueeze(-2))  # (..., i, j, m1, m2)
                t_flat = t_mn.reshape(*batch_shape, self.mul_in1, self.mul_in2, m1 * m2)
                y = torch.matmul(t_flat, U)  # (..., i, j, K_total) in compute_dtype
                if y.shape[-1] != k_total:
                    raise RuntimeError("ICTD TP projection produced wrong K_total")

                # Process each path in the group (projection is already done once for the group)
                # The key optimization is that projection (y) is computed once per group
                i, j = self.mul_in1, self.mul_in2
                for p_idx, l3, s, e in segments:  # type: ignore[misc]
                    Wp = w_param[int(p_idx)]  # (o,i,j)
                    Wp_comp = Wp.to(dtype=compute_dtype) if Wp.dtype != compute_dtype else Wp
                    y_seg = y[..., :, :, int(s): int(e)]  # (..., i, j, kdim) in compute_dtype
                    # GEMM-style channel mixing (optimized)
                    kdim = int(e) - int(s)
                    y2 = y_seg.movedim(-1, -3).contiguous().view(*y_seg.shape[:-3], kdim, i * j)  # (..., k, ij)
                    W2 = Wp_comp.contiguous().view(Wp_comp.shape[0], i * j)  # (o, ij)
                    out_seg = torch.matmul(y2, W2.transpose(0, 1)).movedim(-1, -2)  # (..., o, k) in compute_dtype
                    # Convert back to output dtype
                    out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                    if weights is not None:
                        gate = weights[..., int(p_idx)]
                        out_seg = out_seg * gate[..., None, None]
                    out[int(l3)] = out[int(l3)] + out_seg
        # Fast path: external full per-example weights (..., weight_numel).
        # Still uses the e3nn-like factorization (projection first, then channel mixing).
        elif weights is not None and weights.shape[-1] == self.weight_numel:
            proj_list = self._get_proj_group_list(device=device, dtype=dtype)
            # Reshape once:
            #   weights_full: (..., P, o, i, j)
            weights_full = weights.view(*batch_shape, self.num_paths, self.mul_out, self.mul_in1, self.mul_in2)
            for g_idx, g in enumerate(self._groups):
                l1 = int(g["l1"])  # type: ignore[arg-type]
                l2 = int(g["l2"])  # type: ignore[arg-type]
                segments = g["segments"]  # type: ignore[assignment]
                k_total = int(g["k_total"])  # type: ignore[arg-type]

                a = x1.get(l1)
                b = x2.get(l2)
                if a is None or b is None:
                    continue

                m1 = 2 * l1 + 1
                m2 = 2 * l2 + 1
                U = proj_list[g_idx]  # (m1*m2, K_total) in compute_dtype
                # Convert inputs to compute_dtype for numerical stability
                a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
                b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
                # Optimized outer product: (..., i, m1) x (..., j, m2) -> (..., i, j, m1, m2)
                # Using broadcast instead of einsum for better performance
                t_mn = (a_comp.unsqueeze(-2).unsqueeze(-1) * b_comp.unsqueeze(-3).unsqueeze(-2))  # (..., i, j, m1, m2)
                t_flat = t_mn.reshape(*batch_shape, self.mul_in1, self.mul_in2, m1 * m2)
                y = torch.matmul(t_flat, U)  # (..., i, j, K_total) in compute_dtype
                if y.shape[-1] != k_total:
                    raise RuntimeError("ICTD TP projection produced wrong K_total")

                # Batch channel mixing for all paths in this group
                num_paths_in_group = len(segments)
                # Extract weights for this group: (..., P_g, o, i, j)
                p_indices = [int(p_idx) for p_idx, _, _, _ in segments]
                W_stack = weights_full[..., p_indices, :, :, :]  # (..., P_g, o, i, j)
                W_stack_comp = W_stack.to(dtype=compute_dtype) if W_stack.dtype != compute_dtype else W_stack
                
                # Reshape y for batched matmul: (..., i*j, K_total)
                i, j = self.mul_in1, self.mul_in2
                y_reshaped = y.permute(*range(len(batch_shape)), -3, -2, -1).reshape(*batch_shape, i * j, k_total)  # (..., ij, K)
                
                # Reshape W_stack: (..., P_g, o, i*j)
                W_reshaped = W_stack_comp.reshape(*batch_shape, num_paths_in_group, self.mul_out, i * j)  # (..., P_g, o, ij)
                
                # Batched matmul: (..., ij, K) @ (..., P_g, ij, o) -> (..., P_g, o, K)
                # We need: (..., 1, ij, K) @ (..., P_g, ij, o) -> (..., P_g, 1, o, K) -> (..., P_g, o, K)
                y_expanded = y_reshaped.unsqueeze(-3)  # (..., 1, ij, K)
                W_transposed = W_reshaped.transpose(-2, -1)  # (..., P_g, ij, o)
                out_group = torch.matmul(y_expanded, W_transposed)  # (..., P_g, 1, o, K) in compute_dtype
                out_group = out_group.squeeze(-2).permute(*range(len(batch_shape)), 0, 1, 2)  # (..., P_g, o, K_total)
                # Convert back to output dtype
                out_group = out_group.to(dtype=dtype) if out_group.dtype != dtype else out_group
                
                # Segment and accumulate to output
                for seg_idx, (p_idx, l3, s, e) in enumerate(segments):  # type: ignore[misc]
                    kdim = int(e) - int(s)
                    out_seg = out_group[..., seg_idx, :, int(s): int(e)]  # (..., o, kdim)
                    out[int(l3)] = out[int(l3)] + out_seg
        else:
            # Fallback: original per-path loop (supports external per-example weights).
            cg_list = self._get_cg_list(device=device, dtype=dtype)
            idx = 0
            for p_idx, (l1, l2, l3) in enumerate(self.paths):
                if self.internal_weights:
                    gate = 1.0
                    if weights is not None and weights.shape[-1] == self.num_paths:
                        gate = weights[..., p_idx]
                    Wp = w_param[p_idx]  # (o,i,j)
                else:
                    assert weights is not None
                    block = self.mul_out * self.mul_in1 * self.mul_in2
                    Wp = weights[..., idx: idx + block].view(*batch_shape, self.mul_out, self.mul_in1, self.mul_in2)
                    idx += block
                    gate = 1.0

                a = x1.get(l1)
                b = x2.get(l2)
                if a is None or b is None:
                    continue

                # Convert to compute_dtype for numerical stability
                a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
                b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
                Wp_comp = Wp.to(dtype=compute_dtype) if Wp.dtype != compute_dtype else Wp
                C = cg_list[p_idx]  # (m1,m2,m3) in compute_dtype
                out_l3 = torch.einsum("...im,...jn,mnk,oij->...ok", a_comp, b_comp, C, Wp_comp)
                # Convert back to output dtype
                out_l3 = out_l3.to(dtype=dtype) if out_l3.dtype != dtype else out_l3
                if not isinstance(gate, float):
                    out_l3 = out_l3 * gate[..., None, None]
                out[l3] = out[l3] + out_l3

        return out

