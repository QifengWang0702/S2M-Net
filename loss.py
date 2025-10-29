# loss.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes
import pytorch3d.ops as p3d_ops
from pytorch3d.loss import (
    chamfer_distance,
    mesh_laplacian_smoothing,
    mesh_edge_loss,
    mesh_normal_consistency,
)

# =========================
# A. Parameter regression loss (differentiable)
# =========================
def npz_param_loss(
    GH_pred, R_pred, s_pred, T_pred,
    GH_gt,  R_gt,  s_gt,  T_gt,
    w_GH=1.0, w_R=1.0, w_s=1.0, w_T=1.0,
    reg_GH=1e-6
):
    """
    Supervise GHD parameters (used when training with GH labels).
    """
    L_GH = F.mse_loss(GH_pred, GH_gt)
    L_R  = F.mse_loss(R_pred,  R_gt)
    L_s  = F.mse_loss(s_pred,  s_gt)
    L_T  = F.mse_loss(T_pred,  T_gt)
    L = w_GH*L_GH + w_R*L_R + w_s*L_s + w_T*L_T + reg_GH*(GH_pred**2).mean()
    logs = {'L_param': L.detach(), 'L_GH': L_GH.detach(), 'L_R': L_R.detach(), 'L_s': L_s.detach(), 'L_T': L_T.detach()}
    return L, logs


# =========================
# B. 3D differentiable geometric losses (for training)
# =========================
def chamfer_loss3d(mesh_pred: Meshes, mesh_gt: Meshes, n_samples: int = 20000):
    """
    Returns:
      L_ch  — Chamfer (point-to-point); lower is better
      L_n   — Normal term (point-normal discrepancy); can be used as a light regularizer
    Units: identical to mesh vertex units (typically pixels/normalized units during training)
    """
    x, nx = p3d_ops.sample_points_from_meshes(mesh_pred, n_samples, return_normals=True)
    y, ny = p3d_ops.sample_points_from_meshes(mesh_gt,   n_samples, return_normals=True)
    L_ch, L_n = chamfer_distance(x, y, x_normals=nx, y_normals=ny)
    return L_ch, L_n

def mesh_regularizers(mesh_pred: Meshes) -> Dict[str, torch.Tensor]:
    regs = {}
    regs['lap']  = mesh_laplacian_smoothing(mesh_pred, method='cot')
    regs['edge'] = mesh_edge_loss(mesh_pred)
    regs['nrm']  = mesh_normal_consistency(mesh_pred)
    return regs

# ---- Voxel soft Dice (usable for both training and evaluation) ----
@dataclass
class VoxelDiceConfig:
    grid_size: int = 64         # For training, 48–72 is reasonable; for validation, 96/128
    bbox_scale: float = 1.05
    alpha: float = 200.0
    reduction: str = "mean"
    # Memory friendly options
    surf_samples_max: int = 30000
    knn_chunk_points: int = 120_000

def _mesh_bbox(mesh: Meshes):
    V = mesh.verts_padded()
    return V.min(dim=1).values, V.max(dim=1).values

def _grid_points(vmin: torch.Tensor, vmax: torch.Tensor, G: int) -> torch.Tensor:
    B = vmin.shape[0]; device = vmin.device
    xs = torch.linspace(0.0, 1.0, G, device=device)
    grid = torch.stack(torch.meshgrid(xs, xs, xs, indexing='ij'), dim=-1)  # [G,G,G,3]
    grid = grid.reshape(1, G**3, 3).repeat(B, 1, 1)
    return vmin[:, None, :] + grid * (vmax[:, None, :] - vmin[:, None, :])

def _soft_occupancy_from_surface(
    mesh: Meshes,
    pts: torch.Tensor,             # [B, M, 3]
    alpha: float,
    surf_samples_max: int = 30000,
    knn_chunk_points: int = 120_000,
) -> torch.Tensor:
    B, M, _ = pts.shape
    # Reference surface points
    K = min(surf_samples_max, max(10000, M // 2))
    surf_pts = p3d_ops.sample_points_from_meshes(mesh, K, return_normals=False)  # [B,K,3]

    occ_list = []
    for start in range(0, M, knn_chunk_points):
        Q = pts[:, start:start + knn_chunk_points, :]   # [B, m, 3]
        d2, _, _ = p3d_ops.knn_points(Q, surf_pts, K=1, return_nn=False)
        d = torch.sqrt(torch.clamp_min(d2.squeeze(-1), 1e-12))  # [B, m]
        occ = torch.exp(-alpha * d)
        occ_list.append(occ)
    return torch.cat(occ_list, dim=1)  # [B, M]

def dice3d_voxel_soft(mesh_pred: Meshes, mesh_gt: Meshes, cfg: VoxelDiceConfig = VoxelDiceConfig()) -> torch.Tensor:
    vmin_p, vmax_p = _mesh_bbox(mesh_pred)
    vmin_g, vmax_g = _mesh_bbox(mesh_gt)
    vmin = torch.minimum(vmin_p, vmin_g)
    vmax = torch.maximum(vmax_p, vmax_g)

    center = (vmin + vmax) / 2
    half   = (vmax - vmin) * 0.5 * cfg.bbox_scale
    vmin = center - half
    vmax = center + half

    pts = _grid_points(vmin, vmax, cfg.grid_size)  # [B, G^3, 3]
    occp = _soft_occupancy_from_surface(mesh_pred, pts, cfg.alpha, cfg.surf_samples_max, cfg.knn_chunk_points)
    occg = _soft_occupancy_from_surface(mesh_gt,   pts, cfg.alpha, cfg.surf_samples_max, cfg.knn_chunk_points)

    inter = (occp * occg).sum(dim=1)
    sum_  = (occp.pow(2) + occg.pow(2)).sum(dim=1) + 1e-6
    dice  = (2.0 * inter) / sum_
    loss  = 1.0 - dice

    if cfg.reduction == "mean":
        return loss.mean()
    elif cfg.reduction == "sum":
        return loss.sum()
    else:
        return loss


# ---- Loss weights (set a component to 0 to perform an ablation) ----
@dataclass
class LossWeights:
    ch: float = 1.0        # Chamfer
    nrm_surf: float = 0.1  # Normal term attached to Chamfer
    dice3d: float = 1.0    # Voxel soft Dice
    lap: float = 0.1       # Laplacian
    edge: float = 0.1      # Edge length
    nrm: float = 0.05      # Face normal consistency

def combine_3d_losses(
    mesh_pred: Meshes,
    mesh_gt:   Meshes,
    weights: Dict[str, float] | LossWeights | None = None,
    voxel_cfg: VoxelDiceConfig = VoxelDiceConfig(),
    n_samples_ch: int = 20000
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined 3D loss during training. For ablation, set the corresponding weight to 0.
    """
    if weights is None:
        weights = LossWeights()
    if isinstance(weights, dict):
        weights = LossWeights(**weights)

    L_ch, L_nsurf = chamfer_loss3d(mesh_pred, mesh_gt, n_samples=n_samples_ch)  # Lower is better
    L_dice = dice3d_voxel_soft(mesh_pred, mesh_gt, voxel_cfg)                   # 1 - Dice
    regs = mesh_regularizers(mesh_pred)

    total = (weights.ch * L_ch +
             weights.nrm_surf * L_nsurf +
             weights.dice3d * L_dice +
             weights.lap * regs['lap'] +
             weights.edge * regs['edge'] +
             weights.nrm * regs['nrm'])

    logs = dict(
        L_total=total.detach(),
        L_ch=L_ch.detach(),
        L_nsurf=L_nsurf.detach(),
        L_dice3d=L_dice.detach(),
        L_lap=regs['lap'].detach(),
        L_edge=regs['edge'].detach(),
        L_nrm=regs['nrm'].detach(),
    )
    return total, logs


# =========================
# C. Evaluation: CD / HD / EMD / Dice3D (supports spacing → mm)
# =========================
def _scale_mesh(mesh: Meshes, spacing_xyz: Optional[Iterable[float]]) -> Meshes:
    """
    Scale mesh vertices per-axis by spacing (sx, sy, sz) to convert units from pixel/normalized to mm.
    Used only in evaluation; training typically remains in original units.
    """
    if spacing_xyz is None:
        return mesh
    sx, sy, sz = [float(v) for v in spacing_xyz]
    V = mesh.verts_padded()
    scale = torch.tensor([[sx, sy, sz]], device=V.device, dtype=V.dtype)
    V_mm = V * scale[:, None, :]
    return Meshes(verts=list(V_mm), faces=list(mesh.faces_padded()))

@torch.no_grad()
def _sample_points(mesh: Meshes, n: int = 50000) -> torch.Tensor:
    pts = p3d_ops.sample_points_from_meshes(mesh, n, return_normals=False)  # [B,n,3]
    return pts

@torch.no_grad()
def _hausdorff_from_points(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x,y: [B,N,3]
    HD = max( max_i min_j ||x_i - y_j||, max_j min_i ||y_j - x_i|| )
    """
    # knn_points computes the distance from each point to the nearest neighbor in the other set
    d_xy, _, _ = p3d_ops.knn_points(x, y, K=1, return_nn=False)  # [B,N,1]
    d_yx, _, _ = p3d_ops.knn_points(y, x, K=1, return_nn=False)
    # sqrt to convert squared distance to distance
    d1 = torch.sqrt(torch.clamp_min(d_xy.squeeze(-1), 1e-12)).max(dim=1).values
    d2 = torch.sqrt(torch.clamp_min(d_yx.squeeze(-1), 1e-12)).max(dim=1).values
    return torch.maximum(d1, d2)  # [B]

@torch.no_grad()
def _sinkhorn_emd_cost(x: torch.Tensor, y: torch.Tensor, reg: float = 0.02, iters: int = 50) -> torch.Tensor:
    """
    Approximate EMD (Sinkhorn): returns the transport cost per batch (lower is better).
    x,y: [B,N,3] and [B,M,3]; N and M can differ (downsampling both to the same n tends to be more stable).
    """
    B, N, _ = x.shape
    M = y.shape[1]
    # Cost matrix: ||xi - yj||_2
    # Note memory usage: can be chunked; N and M are recommended to be <= ~50k for comfort.
    C = torch.cdist(x, y, p=2)  # [B,N,M]
    K = torch.exp(-C / reg)     # [B,N,M]

    # Uniform distributions
    a = torch.full((B, N), 1.0 / N, device=x.device, dtype=x.dtype)
    b = torch.full((B, M), 1.0 / M, device=x.device, dtype=x.dtype)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    # Sinkhorn iterations
    for _ in range(iters):
        u = a / (K @ v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
        v = b / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)

    # Transport matrix
    P = (u.unsqueeze(-1) * K) * v.unsqueeze(1)   # [B,N,M]
    emd = (P * C).sum(dim=(1,2))                 # [B]
    return emd

# =========================
# C. Evaluation: CD / HD / Dice3D / IoU3D / Laplace (no spacing / no EMD)
# =========================
@dataclass
class Eval3DConfig:
    n_points_cd: int = 50000
    n_points_hd: int = 50000
    voxel_dice: VoxelDiceConfig = VoxelDiceConfig(grid_size=128, bbox_scale=1.05, alpha=200.0)

@torch.no_grad()
def _sample_points(mesh: Meshes, n: int = 50000) -> torch.Tensor:
    return p3d_ops.sample_points_from_meshes(mesh, n, return_normals=False)  # [B,n,3]

@torch.no_grad()
def _hausdorff_from_points(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d_xy, _, _ = p3d_ops.knn_points(x, y, K=1, return_nn=False)  # [B,N,1]
    d_yx, _, _ = p3d_ops.knn_points(y, x, K=1, return_nn=False)
    d1 = torch.sqrt(torch.clamp_min(d_xy.squeeze(-1), 1e-12)).max(dim=1).values
    d2 = torch.sqrt(torch.clamp_min(d_yx.squeeze(-1), 1e-12)).max(dim=1).values
    return torch.maximum(d1, d2)  # [B]

@torch.no_grad()
def _soft_occ_stats_for_iou(
    mesh_pred: Meshes, mesh_gt: Meshes, cfg: VoxelDiceConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reuse voxel soft occupancy to return per-batch Dice and IoU (soft versions).
    Dice = 2*inter / (sumA + sumB)
    IoU  = inter / (sumA + sumB - inter)
    """
    # Compute the same voxel grid & occupancy as in dice3d_voxel_soft
    vmin_p, vmax_p = _mesh_bbox(mesh_pred)
    vmin_g, vmax_g = _mesh_bbox(mesh_gt)
    vmin = torch.minimum(vmin_p, vmin_g)
    vmax = torch.maximum(vmax_p, vmax_g)

    center = (vmin + vmax) / 2
    half   = (vmax - vmin) * 0.5 * cfg.bbox_scale
    vmin = center - half
    vmax = center + half

    pts = _grid_points(vmin, vmax, cfg.grid_size)  # [B, G^3, 3]
    occp = _soft_occupancy_from_surface(mesh_pred, pts, cfg.alpha, cfg.surf_samples_max, cfg.knn_chunk_points)
    occg = _soft_occupancy_from_surface(mesh_gt,   pts, cfg.alpha, cfg.surf_samples_max, cfg.knn_chunk_points)

    inter = (occp * occg).sum(dim=1)                         # [B]
    sumA  = (occp.pow(2)).sum(dim=1)                         # [B]
    sumB  = (occg.pow(2)).sum(dim=1)                         # [B]
    dice  = (2.0 * inter) / (sumA + sumB + 1e-6)             # [B]
    iou   = inter / (sumA + sumB - inter + 1e-6)             # [B]
    return dice, iou

@torch.no_grad()
def eval_metrics_3d(
    mesh_pred: Meshes,
    mesh_gt:   Meshes,
    cfg: Eval3DConfig = Eval3DConfig(),
) -> Dict[str, float]:
    """
    No spacing conversion and no EMD:
    Returns: CD, HD, Dice3D, IoU3D, Laplace
    Units: the same as the mesh coordinate units (no mm conversion).
    """
    # CD (sampling)
    xp = _sample_points(mesh_pred, cfg.n_points_cd)   # [B,N,3]
    xg = _sample_points(mesh_gt,   cfg.n_points_cd)
    cd, _ = chamfer_distance(xp, xg)                  # scalar (batch mean)

    # HD (supremum of bidirectional nearest neighbor distances with sampling)
    xp_hd = _sample_points(mesh_pred, cfg.n_points_hd)
    xg_hd = _sample_points(mesh_gt,   cfg.n_points_hd)
    hd = _hausdorff_from_points(xp_hd, xg_hd).mean()  # batch mean

    # Dice3D & IoU3D (soft voxel)
    dice_b, iou_b = _soft_occ_stats_for_iou(mesh_pred, mesh_gt, cfg.voxel_dice)
    dice = float(dice_b.mean().detach().item())
    iou  = float(iou_b.mean().detach().item())

    # Laplace (cot)
    lap = float(mesh_laplacian_smoothing(mesh_pred, method='cot').detach().item())

    return dict(
        CD = float(cd.detach().item()),
        HD = float(hd.detach().item()),
        Dice3D = dice,
        IoU3D  = iou,
        Laplace = lap
    )


# =========================
# D. 2D radial (differentiable) — compatible with your existing training script
# =========================
@dataclass
class Radial2DSoftCfg:
    num_planes: int = 9
    angle_offset_deg: float = 0.0
    grid_hw: Tuple[int,int] = (160, 160)
    border_scale: float = 1.10
    alpha: float = 200.0
    ref_dir: Tuple[float,float,float] = (0.0, 0.0, 1.0)
    reduction: str = "mean"

def _normalize(v: torch.Tensor, eps: float=1e-12):
    n = v.norm(dim=-1, keepdim=True)
    return v / (n.clamp_min(eps))

def _plane_uv_axes_Y(n_xy: torch.Tensor):
    B = n_xy.shape[0]; device = n_xy.device
    a = torch.tensor([0.0,1.0,0.0], device=device).view(1,3).repeat(B,1)
    e_a = _normalize(a)
    e_b = _normalize(torch.cross(n_xy, e_a, dim=-1))
    return e_a, e_b

def _radial_planes_Y(B:int, ref_dir: torch.Tensor, num_planes:int, offset_rad: float, device):
    u0 = ref_dir / (ref_dir.norm()+1e-12)
    a = torch.tensor([0.0,1.0,0.0], device=device)
    v0 = torch.cross(a, u0, dim=0); v0 = v0 / (v0.norm()+1e-12)
    thetas = torch.linspace(0, 2*np.pi, num_planes, device=device, dtype=torch.float32) + offset_rad
    normals = [torch.cos(th)*u0 + torch.sin(th)*v0 for th in thetas]
    n = torch.stack(normals, dim=0).unsqueeze(0).repeat(B,1,1)   # [B,P,3]
    return n

def _project_bbox_to_plane(vmin, vmax, n_xy, border_scale):
    xmin = torch.minimum(vmin[:,0], vmax[:,0]); xmax = torch.maximum(vmin[:,0], vmax[:,0])
    zmin = torch.minimum(vmin[:,2], vmax[:,2]); zmax = torch.maximum(vmin[:,2], vmax[:,2])
    du = (xmax - xmin) * border_scale
    dv = (zmax - zmin) * border_scale
    umin = -du/2; umax = du/2; vmin2 = -dv/2; vmax2 = dv/2
    return umin, umax, vmin2, vmax2

def _grid_in_plane(o: torch.Tensor, e_a: torch.Tensor, e_b: torch.Tensor,
                   bounds: Tuple[torch.Tensor, ...], H:int, W:int):
    B = o.shape[0]; device = o.device
    umin, umax, vmin, vmax = bounds
    uu = torch.linspace(0, 1, W, device=device).view(1,1,W)
    vv = torch.linspace(0, 1, H, device=device).view(1,H,1)
    U = umin[:,None,None] + (umax-umin)[:,None,None]*uu
    V = vmin[:,None,None] + (vmax-vmin)[:,None,None]*vv
    U = U.expand(B,H,W); V = V.expand(B,H,W)
    p = o[:,None,None,:] + U[...,None]*e_b[:,None,None,:] + V[...,None]*e_a[:,None,None,:]
    return p.reshape(B, H*W, 3)

def _soft_occ_on_plane(mesh: Meshes, pts_plane: torch.Tensor, alpha: float) -> torch.Tensor:
    B, M, _ = pts_plane.shape
    K = min(60000, max(10000, M // 2))
    surf_pts = p3d_ops.sample_points_from_meshes(mesh, K, return_normals=False)  # [B,K,3]
    out = []
    chunk = 120_000
    for s in range(0, M, chunk):
        Q = pts_plane[:, s:s+chunk, :]
        d2, _, _ = p3d_ops.knn_points(Q, surf_pts, K=1, return_nn=False)
        d = torch.sqrt(torch.clamp_min(d2.squeeze(-1), 1e-12))
        out.append(torch.exp(-alpha * d))
    return torch.cat(out, dim=1)

def radial_soft_dice_loss2d(
    mesh_pred: Meshes, mesh_gt: Meshes,
    cfg: Radial2DSoftCfg = Radial2DSoftCfg()
) -> torch.Tensor:
    device = mesh_pred.device
    B = len(mesh_pred)

    Vg = mesh_gt.verts_padded()
    o  = Vg.mean(dim=1)  # [B,3]

    vmin_p, vmax_p = _mesh_bbox(mesh_pred)
    vmin_g, vmax_g = _mesh_bbox(mesh_gt)
    vmin = torch.minimum(vmin_p, vmin_g); vmax = torch.maximum(vmax_p, vmax_g)

    ref = torch.tensor(cfg.ref_dir, device=device, dtype=torch.float32)
    if ref.norm() < 1e-6: ref = torch.tensor([0.0,0.0,1.0], device=device)
    ref[1] = 0.0
    if ref.norm() < 1e-6: ref = torch.tensor([1.0,0.0,0.0], device=device)

    n_xy = _radial_planes_Y(B, ref, cfg.num_planes, np.deg2rad(cfg.angle_offset_deg), device)  # [B,P,3]

    H, W = cfg.grid_hw
    losses = []
    for p in range(cfg.num_planes):
        n = n_xy[:, p, :]
        e_a, e_b = _plane_uv_axes_Y(n)
        bounds = _project_bbox_to_plane(vmin, vmax, n, cfg.border_scale)
        pts = _grid_in_plane(o, e_a, e_b, bounds, H, W)  # [B,H*W,3]

        occp = _soft_occ_on_plane(mesh_pred, pts, cfg.alpha)
        occg = _soft_occ_on_plane(mesh_gt,   pts, cfg.alpha)

        inter = (occp*occg).sum(dim=1)
        denom = (occp.pow(2)+occg.pow(2)).sum(dim=1) + 1e-6
        dice  = (2.0*inter)/denom
        losses.append(1.0 - dice)

    loss = torch.stack(losses, dim=1).mean(dim=1)
    if cfg.reduction == "mean":
        return loss.mean()
    elif cfg.reduction == "sum":
        return loss.sum()
    else:
        return loss


# =========================
# E. 2D radial (non-differentiable evaluation + visualization), compatible with the original script
# =========================
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import trimesh

@dataclass
class RadialEvalConfig:
    num_planes: int = 37
    angle_offset_deg: float = 0.0
    image_hw: Tuple[int, int] = (256, 256)
    border_margin: float = 0.05
    visualize_indices: Tuple[int, ...] = (0, 7, 14, 21, 28, 35)
    dpi: int = 110
    axis_point: Optional[Iterable[float]] = None
    ref_dir: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    quiet: bool = True

def _normalize_np(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v); return v / max(n, eps)

def _plane_basis_np(axis_dir: np.ndarray, plane_normal_xy: np.ndarray):
    e_a = axis_dir
    e_b = np.cross(plane_normal_xy, e_a)
    e_b = _normalize_np(e_b)
    return e_a, e_b

def _section_polys_2d_np(tri: trimesh.Trimesh, plane_origin: np.ndarray, plane_normal: np.ndarray, axis_dir: np.ndarray) -> List[np.ndarray]:
    sec = tri.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if sec is None: return []
    e_a, e_b = _plane_basis_np(axis_dir, plane_normal)
    pls_3d = sec.discrete
    if len(pls_3d) == 0: return []
    out = []
    for P in pls_3d:
        rel = (P - plane_origin[None, :])
        u = rel @ e_b
        v = rel @ e_a
        out.append(np.stack([u, v], axis=1))
    return out

def _rasterize_np(polys: List[np.ndarray], H: int, W: int, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    if len(polys) == 0: return np.zeros((H, W), dtype=bool)
    umin, umax, vmin, vmax = bounds
    uu = np.linspace(umin, umax, W)
    vv = np.linspace(vmax, vmin, H)
    grid_u, grid_v = np.meshgrid(uu, vv)
    pts = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1)
    mask = np.zeros((H, W), dtype=bool)
    for poly in polys:
        if poly.shape[0] < 3: continue
        if np.linalg.norm(poly[0] - poly[-1]) > 1e-6:
            poly = np.vstack([poly, poly[0]])
        path = MplPath(poly)
        inside = path.contains_points(pts, radius=1e-9)
        mask |= inside.reshape(H, W)
    return mask

def _bounds_two_np(polys_a: List[np.ndarray], polys_b: List[np.ndarray], margin_ratio: float):
    allp = []
    for S in (polys_a, polys_b):
        for p in S:
            if len(p): allp.append(p)
    if len(allp) == 0:
        return (-1.0, 1.0, -1.0, 1.0)
    xy = np.vstack(allp)
    umin, vmin = xy.min(axis=0); umax, vmax = xy.max(axis=0)
    du, dv = umax - umin, vmax - vmin
    pad_u = du * margin_ratio; pad_v = dv * margin_ratio
    return (umin - pad_u, umax + pad_u, vmin - pad_v, vmax + pad_v)

def dice_iou_ce_2d_np(a: np.ndarray, b: np.ndarray):
    inter = np.logical_and(a, b).sum()
    sa, sb = a.sum(), b.sum()
    dice = (2 * inter) / (sa + sb + 1e-8)
    union = sa + sb - inter
    iou = inter / (union + 1e-8)
    pa = a.astype(np.float32); pb = b.astype(np.float32); eps = 1e-8
    ce = - (pa * np.log(pb + eps) + (1 - pa) * np.log(1 - pb + eps)).mean()
    return float(dice), float(iou), float(ce)

def radial_2d_metrics(
    gt_obj: str,
    pred_obj: str,
    cfg: RadialEvalConfig,
    save_png: Optional[str] = None
) -> Dict[str, float]:
    gt = trimesh.load_mesh(gt_obj, process=False)
    pr = trimesh.load_mesh(pred_obj, process=False)
    if not isinstance(gt, trimesh.Trimesh): gt = gt.dump(concatenate=True)
    if not isinstance(pr, trimesh.Trimesh): pr = pr.dump(concatenate=True)

    a = np.array([0.0, 1.0, 0.0], dtype=float); a = _normalize_np(a)
    o = gt.vertices.mean(axis=0) if cfg.axis_point is None else np.array(cfg.axis_point, dtype=float)

    r0_xy = np.array(cfg.ref_dir, dtype=float); r0_xy[1] = 0.0
    if np.linalg.norm(r0_xy) < 1e-8: r0_xy = np.array([0.0,0.0,1.0], float)
    r0_xy = _normalize_np(r0_xy)
    u0 = r0_xy; v0 = _normalize_np(np.cross(a, u0))

    H, W = cfg.image_hw
    dice_arr, iou_arr, ce_arr = [], [], []
    vis_angles = set(cfg.visualize_indices)
    fig, axs, vis_col = (None, None, 0)
    if save_png is not None and len(vis_angles)>0:
        fig, axs = plt.subplots(1, len(vis_angles), figsize=(3.2*len(vis_angles), 3.2), dpi=cfg.dpi)
        if len(vis_angles)==1: axs=[axs]
    ang0 = np.deg2rad(cfg.angle_offset_deg)

    for i in range(cfg.num_planes):
        th = 2.0*np.pi*i/cfg.num_planes + ang0
        n = np.cos(th)*u0 + np.sin(th)*v0; n = _normalize_np(n)
        gt_pl = _section_polys_2d_np(gt, o, n, a)
        pr_pl = _section_polys_2d_np(pr, o, n, a)
        bounds = _bounds_two_np(gt_pl, pr_pl, cfg.border_margin)
        mask_gt = _rasterize_np(gt_pl, H, W, bounds)
        mask_pr = _rasterize_np(pr_pl, H, W, bounds)
        d,j,ce = dice_iou_ce_2d_np(mask_gt, mask_pr)
        dice_arr.append(d); iou_arr.append(j); ce_arr.append(ce)

        if fig is not None and i in vis_angles:
            umin, umax, vmin, vmax = bounds
            ax = axs[vis_col]
            ax.set_xlim(umin, umax); ax.set_ylim(vmax, vmin)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            for p in gt_pl: ax.plot(p[:,0], p[:,1], color='lime', lw=2)
            for p in pr_pl: ax.plot(p[:,0], p[:,1], color='red',  lw=2)
            ax.set_title(f"{i} | D={d:.3f}")
            vis_col += 1

    if fig is not None and save_png is not None:
        fig.tight_layout(); fig.savefig(save_png, bbox_inches='tight'); plt.close(fig)

    out = dict(
        dice_mean=float(np.mean(dice_arr)), dice_std=float(np.std(dice_arr)),
        iou_mean=float(np.mean(iou_arr)),   iou_std=float(np.std(iou_arr)),
        ce_mean=float(np.mean(ce_arr)),     ce_std=float(np.std(ce_arr))
    )
    return out