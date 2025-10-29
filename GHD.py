# GHD.py
# -----------------------------------------------------------
# GHD (Generalized Harmonic Deformation) minimal viable implementation (differentiable)
# - Fixed mixed Laplacian weights: cot=1.0, dis=0.3, std=0.3
# - Does not rely on a BiV template (only requires a base_shape OBJ)
# - Supports: GH<->verts projection/reconstruction; mesh reconstruction from GH+Affine (differentiable)
# - Supports: save/load npz; recover GH, R, s, T from GT mesh (Umeyama similarity transform)
# -----------------------------------------------------------

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import cot_laplacian
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

try:
    import trimesh
    _HAS_TRIMESH = True
except Exception:
    _HAS_TRIMESH = False


# ------------------------------
# Configuration
# ------------------------------
@dataclass
class GHDConfig:
    base_shape_path: str
    device: str = "cuda:0"
    num_basis: int = 49
    basis_cache: Optional[str] = None
    normalize_lap: bool = True
    mix_cot: float = 1.0
    mix_dis: float = 0.3
    mix_std: float = 0.3


# ------------------------------
# Mixed Laplacian
# ------------------------------
def _to_scipy_coo(sp_tensor: torch.Tensor) -> coo_matrix:
    sp = sp_tensor.coalesce()
    idx = sp.indices().cpu().numpy()
    val = sp.values().cpu().numpy()
    return coo_matrix((val, (idx[0], idx[1])), shape=sp.size())


def _sym_laplacian_from_edges(weights, edges, n_verts, normalize=True, device="cpu"):
    A = torch.sparse_coo_tensor(edges, weights, (n_verts, n_verts),
                                dtype=torch.float32, device=device)
    A = A + A.transpose(0, 1)
    deg = torch.sparse.sum(A, dim=1).to_dense()
    if normalize:
        invsqrt = (deg + 1e-8).pow(-0.5)
        D_inv_sqrt = torch.sparse_coo_tensor(
            torch.stack([torch.arange(n_verts, device=device)]*2),
            invsqrt, (n_verts, n_verts), device=device
        )
        A = torch.sparse.mm(D_inv_sqrt, torch.sparse.mm(A, D_inv_sqrt))
        I = torch.sparse_coo_tensor(
            torch.stack([torch.arange(n_verts, device=device)]*2),
            torch.ones(n_verts, device=device),
            (n_verts, n_verts), device=device
        )
        L = I - A
    else:
        D = torch.sparse_coo_tensor(
            torch.stack([torch.arange(n_verts, device=device)]*2),
            deg, (n_verts, n_verts), device=device
        )
        L = D - A
    return L.coalesce()


def build_mixed_laplacian(mesh: Meshes, cfg: GHDConfig) -> coo_matrix:
    V = mesh.verts_packed()
    F = mesh.faces_packed()
    n_verts = V.shape[0]

    cot_w, _ = cot_laplacian(V, F)
    edges = cot_w.coalesce().indices()
    cot_vals = cot_w.coalesce().values()

    vi = V[edges[0]]
    vj = V[edges[1]]
    dist = (vi - vj).pow(2).sum(-1).sqrt()
    dis_vals = torch.exp(-dist.pow(2) / 2.0)

    uni_vals = torch.ones_like(cot_vals)

    L_cot = _sym_laplacian_from_edges(cot_vals, edges, n_verts, normalize=cfg.normalize_lap, device=V.device)
    L_dis = _sym_laplacian_from_edges(dis_vals, edges, n_verts, normalize=cfg.normalize_lap, device=V.device)
    L_uni = _sym_laplacian_from_edges(uni_vals, edges, n_verts, normalize=cfg.normalize_lap, device=V.device)

    L_mix = cfg.mix_cot * L_cot + cfg.mix_dis * L_dis + cfg.mix_std * L_uni
    return _to_scipy_coo(L_mix.cpu())


# ------------------------------
# Umeyama similarity transform
# ------------------------------
def umeyama_similarity(src, dst, eps=1e-8):
    B, N, _ = src.shape
    src_mean = src.mean(1, keepdim=True)
    dst_mean = dst.mean(1, keepdim=True)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = torch.bmm(dst_c.transpose(1, 2), src_c) / N
    U, S, Vt = torch.linalg.svd(cov)
    V = Vt.transpose(1, 2)
    R = torch.bmm(U, V.transpose(1, 2))
    detR = torch.det(R)
    mask = (detR < 0).view(B, 1, 1)
    U[:, :, -1] *= (1 - 2 * mask.squeeze(-1).squeeze(-1))
    R = torch.bmm(U, V.transpose(1, 2))
    var_src = (src_c ** 2).sum((1, 2)) / N
    s = (S.sum(1) / (var_src + eps)).view(B, 1)
    T = dst_mean.squeeze(1) - s * torch.bmm(src_mean, R.transpose(1, 2)).squeeze(1)
    return R, s, T


# ------------------------------
# GHD main class
# ------------------------------
class GHD(nn.Module):
    def __init__(self, cfg: GHDConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.base = load_objs_as_meshes([cfg.base_shape_path], device=self.device)
        self.base_V = self.base.verts_packed().shape[0]

        # ---- Load/normalize basis ----
        if cfg.basis_cache and os.path.isfile(cfg.basis_cache):
            eigval, eigvec = torch.load(cfg.basis_cache, map_location=self.device)
        else:
            eigval, eigvec = self._compute_eigs_and_cache(cfg.basis_cache)

        eigval = torch.as_tensor(eigval, dtype=torch.float32, device=self.device).squeeze()
        eigvec = torch.as_tensor(eigvec, dtype=torch.float32, device=self.device).squeeze()

        # Compatible with [K,N] / [1,N,K]
        if eigvec.ndim == 3 and eigvec.shape[0] == 1:
            eigvec = eigvec.squeeze(0)
        if eigvec.shape[0] == self.base_V:
            pass
        elif eigvec.shape[1] == self.base_V:
            eigvec = eigvec.transpose(0, 1)
        else:
            raise ValueError(f"basis shape {eigvec.shape} not compatible with base_V={self.base_V}")

        eigvec = eigvec[:, :cfg.num_basis]
        eigval = eigval[:cfg.num_basis]

        self.register_buffer("eigval", eigval)
        self.register_buffer("eigvec", eigvec)

        self.R = nn.Parameter(torch.zeros(1, 3, device=self.device))
        self.s = nn.Parameter(torch.ones(1, 1, device=self.device))
        self.T = nn.Parameter(torch.zeros(1, 3, device=self.device))
        self.GH = nn.Parameter(torch.zeros(1, cfg.num_basis, 3, device=self.device))

    def _compute_eigs_and_cache(self, cache_path):
        L_mix = build_mixed_laplacian(self.base, self.cfg)
        k = max(self.cfg.num_basis, 64)
        w, v = eigsh(L_mix, k=k, which="SM")
        eigval = torch.from_numpy(w).float()
        eigvec = torch.from_numpy(v).float()
        if cache_path:
            torch.save((eigval, eigvec), cache_path)
        return eigval, eigvec

    def gh_to_verts(self, GH):
        return torch.einsum('nk,bkc->bnc', self.eigvec, GH)

    def verts_to_gh(self, disp):
        return torch.einsum('nk,bnc->bkc', self.eigvec, disp)

    def render(self, GH=None, R=None, s=None, T=None):
        if GH is None: GH = self.GH
        B = GH.shape[0]
        disp = self.gh_to_verts(GH)
        baseV = self.base.verts_padded().expand(B, -1, -1)
        V = baseV + disp
        if R is None:
            Rmat = axis_angle_to_matrix(self.R).expand(B, -1, -1)
        else:
            if R.ndim == 2: Rmat = axis_angle_to_matrix(R)
            else: Rmat = R
        s_use = self.s.expand(B, 1) if s is None else s.view(B, 1)
        T_use = self.T.expand(B, -1) if T is None else T.view(B, 3)
        V_aff = torch.bmm(V, Rmat.transpose(1, 2)) * s_use.view(B, 1, 1) + T_use.view(B, 1, 3)
        F = self.base.faces_padded().expand(B, -1, -1)
        return Meshes(verts=V_aff, faces=F)

    @staticmethod
    def save_npz(path, GH, R, s, T):
        data = dict(
            GH=GH.detach().cpu().numpy(),
            R=R.detach().cpu().numpy(),
            s=s.detach().cpu().numpy(),
            T=T.detach().cpu().numpy(),
        )
        np.savez_compressed(path, **data)

    @staticmethod
    def load_npz(path, device="cpu"):
        z = np.load(path)
        GH = torch.from_numpy(z["GH"]).float().to(device)
        R = torch.from_numpy(z["R"]).float().to(device)
        s = torch.from_numpy(z["s"]).float().to(device)
        T = torch.from_numpy(z["T"]).float().to(device)
        return GH, R, s, T

    @torch.no_grad()
    def invert_from_mesh(self, mesh):
        Vt = mesh.verts_padded()
        B, N, _ = Vt.shape
        assert N == self.base_V
        Vb = self.base.verts_padded().expand(B, -1, -1)
        Rmat, s, T = umeyama_similarity(Vb, Vt)
        Vt_back = torch.bmm((Vt - T.view(B, 1, 3)) / (s.view(B, 1, 1) + 1e-8), Rmat)
        disp = Vt_back - Vb
        GH = self.verts_to_gh(disp)
        R_axis = matrix_to_axis_angle(Rmat)
        return dict(GH=GH, R=R_axis, s=s, T=T)

    @staticmethod
    def export_mesh(mesh, out_path):
        if not _HAS_TRIMESH:
            raise RuntimeError("需要安装 trimesh")
        V = mesh.verts_packed().cpu().numpy()
        F = mesh.faces_packed().cpu().numpy()
        m = trimesh.Trimesh(vertices=V, faces=F, process=False)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        m.export(out_path)