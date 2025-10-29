# train.py
# -*- coding: utf-8 -*-
import os, time, json, math, random, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ==== Project modules ====
from data import RadialNPZDataset, collate_batch
from net import build_model, ModelCfg
from loss import (
    npz_param_loss,
    combine_3d_losses,
    VoxelDiceConfig,
    radial_2d_metrics,
    RadialEvalConfig,
    eval_metrics_3d, Eval3DConfig,
    LossWeights
)

# ==== 3D ====
import pytorch3d.ops as p3d_ops

# ==== GHD (minimal differentiable implementation) ====
from GHD import GHD, GHDConfig

# ==== Optional logging (SAFE-optional dependency) ====
try:
    import swanlab
    _HAS_SWANLAB = True
except Exception:
    _HAS_SWANLAB = False

# ==== Visualization ====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message=".*nested_tensor.*")
warnings.filterwarnings("ignore", message=".*Mtl file does not exist.*")
warnings.filterwarnings("ignore", message=".*No mtl file provided.*")
warnings.filterwarnings("ignore", message=r".*torch\.sparse\.SparseTensor\(.*\) is deprecated.*")


# ===================== Configuration =====================
class CFG:
    # Data & labels
    data_root     = './FeEcho4D-Dataset'
    gh_label_dir  = './FeEcho4D-Dataset/gh_labels'
    table_csv     = None

    # GHD
    base_shape    = './FeEcho4D-Dataset/gh_labels/LV_tmp.obj'
    basis_cache   = './FeEcho4D-Dataset/gh_labels/basis_cache.pt'
    num_basis     = 49

    # Training
    device        = 'cuda:0'
    img_size      = (256,256)
    S             = 37
    batch_size    = 2
    val_batch     = 2
    epochs        = 300
    num_workers   = 4

    # Optimization
    lr            = 1e-4
    wd            = 1e-4
    grad_clip     = 5.0

    # LR scheduling
    warmup_epochs = 3
    min_lr        = 1e-6
    plateau_factor= 0.5
    plateau_patience = 5
    plateau_cooldown = 1
    plateau_threshold = 1e-4

    # Global loss weights
    w_param       = 1.0
    w_3d          = 1.0
    w_2d          = 0.0

    # 3D composite inner weights (aligned with loss.py: LossWeights)
    loss_weights  = LossWeights(ch=1.0, nrm_surf=0.1, dice3d=1.0, lap=0.1, edge=0.1, nrm=0.05)
    voxel_cfg     = VoxelDiceConfig(grid_size=64, bbox_scale=1.05, alpha=200.0)

    # 2D radial soft silhouette (participates in backprop)
    planes_for_loss = 8
    grid_2d         = 32
    alpha_2d        = 120.0
    pts_per_mesh_2d = 1500
    ref_dir_xy      = (0.0, 0.0, 1.0)
    axis_y          = (0.0, 1.0, 0.0)

    # Evaluation / saving
    out_root       = './FeEcho4D-Results/s2m-net_res'
    save_max_per_epoch = 10
    save_vis_every = 10

    radial_eval = RadialEvalConfig(
        num_planes=37, angle_offset_deg=0.0, image_hw=(256,256),
        border_margin=0.05, visualize_indices=(0,7,14,21,28,35),
        dpi=110, axis_point=None, ref_dir=(0.0,0.0,1.0), quiet=True
    )

    # ===== Model (aligned with net.ModelCfg) =====
    encoder_name     = 'unet'      # 'cnn' | 'unet' | 'resunet' | 'transunet' | 'sam'
    in_per_slice     = 5
    num_basis        = 49
    d_model          = 512
    use_radial_trans = True
    nhead            = 8
    dim_ff           = 1024
    num_layers       = 4
    dropout          = 0.1

    model_type       = 'slice2mesh'
    attn_num_samples = 2
    attn_num_slices  = 4
    attn_alpha       = 0.6

    use_loss_switches = False
    loss_switches = dict(
        chamfer=True,
        normal_surface=True,
        voxel_dice=True,
        laplacian=True,
        edge=True,
        normal_consistency=True
    )

    # Test-set evaluation
    eval_with_emd   = False
    eval_points_cd  = 2000
    eval_points_hd  = 2000
    eval_points_emd = 2000
    eval_voxel_dice_grid = 128


# ===================== Utilities =====================
def _to_cpu_safe(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).cpu()
    if isinstance(x, list):
        return [_to_cpu_safe(e) for e in x]
    if isinstance(x, tuple):
        return tuple(_to_cpu_safe(e) for e in x)
    if isinstance(x, dict):
        return {k: _to_cpu_safe(v) for k, v in x.items()}
    return x

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])

def save_cfg(cfg: CFG, run_dir: Path):
    # Dump configuration to JSON, flattening dataclasses
    d = {k:v for k,v in cfg.__dict__.items() if not k.startswith("__") and not callable(v)}
    if isinstance(cfg.radial_eval, RadialEvalConfig):
        d['radial_eval'] = cfg.radial_eval.__dict__
    if isinstance(cfg.loss_weights, LossWeights):
        d['loss_weights'] = cfg.loss_weights.__dict__
    with open(run_dir/"cfg.json", "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
        
def subset_batch_for_save(batch: Dict[str, Any], idx_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Slice a batch by idx_tensor:
    - For tensors: move idx to the same device and use index_select
    - For list/tuple: slice by a CPU index list
    - Others: return as-is
    """
    idx_cpu = idx_tensor.detach().cpu().tolist()
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.index_select(0, idx_tensor.to(v.device))
        elif isinstance(v, (list, tuple)):
            out[k] = [v[i] for i in idx_cpu]
        else:
            out[k] = v
    return out

# ========= Safe render check without changing GHD (per-sample try/except) =========
@torch.no_grad()
def _try_render_one(ghd: GHD,
                    GH_1: torch.Tensor, R_1: torch.Tensor,
                    s_1: torch.Tensor, T_1: torch.Tensor):
    """
    Render a single sample; return (ok: bool, err_msg: str|None)
    Only catches 'Padding of faces must be at the end'; re-raises other errors.
    """
    try:
        m = ghd.render(GH_1, R_1, s_1, T_1)
        nv = m.num_verts_per_mesh()
        nf = m.num_faces_per_mesh()
        ok = bool((nv > 0).all().item() and (nf > 0).all().item())
        return ok, None
    except Exception as e:
        msg = str(e)
        if "Padding of faces must be at the end" in msg:
            return False, msg
        raise

@torch.no_grad()
def safe_valid_mask_by_render(ghd: GHD,
                              GH: torch.Tensor, R: torch.Tensor,
                              s: torch.Tensor, T: torch.Tensor,
                              batch_meta: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """
    Render each item of a batch, return a boolean valid mask (same device as GH).
    Samples that trigger faces padding error -> False and print a hint. GHD is unchanged.
    """
    B = GH.shape[0]
    device = GH.device
    valid = torch.zeros(B, dtype=torch.bool, device=device)
    for b in range(B):
        ok, err = _try_render_one(ghd, GH[b:b+1], R[b:b+1], s[b:b+1], T[b:b+1])
        if not ok and err is not None:
            pid = None
            tfrm = None
            if batch_meta is not None:
                try:
                    pid = batch_meta.get('patient', [None]*B)[b]
                    tfrm = batch_meta.get('time', [None]*B)[b]
                    tfrm = int(tfrm.item()) if torch.is_tensor(tfrm) else tfrm
                except Exception:
                    pass
            print(f"[faces-pad] skip sample b={b} pid={pid} time={tfrm}: {err}")
        valid[b] = ok
    return valid

# ========= Parameter sanitization & mesh validity mask =========
def _sanitize_params(GH_p, R_p, s_p, T_p):
    GH_p = torch.nan_to_num(GH_p).clamp(-3.0, 3.0)
    R_p  = torch.nan_to_num(R_p).clamp(-3.0, 3.0)
    s_p  = torch.nan_to_num(s_p)
    s_p  = torch.clamp(s_p, min=1e-3)
    T_p  = torch.nan_to_num(T_p).clamp(-5.0, 5.0)
    return GH_p, R_p, s_p, T_p

def _valid_mesh_mask(mesh_pred, mesh_gt):
    nv_p = mesh_pred.num_verts_per_mesh()
    nf_p = mesh_pred.num_faces_per_mesh()
    nv_g = mesh_gt.num_verts_per_mesh()
    nf_g = mesh_gt.num_faces_per_mesh()
    valid = (nv_p > 0) & (nf_p > 0) & (nv_g > 0) & (nf_g > 0)
    return valid.to(dtype=torch.bool, device=nv_p.device)  # ★


def build_loaders(cfg: CFG):
    """
    Construct train/val/test datasets and dataloaders with collate function.
    """
    ds_tr = RadialNPZDataset(cfg.data_root, cfg.gh_label_dir, split='train',
                             img_size=cfg.img_size, target_S=cfg.S, enable_aug=True,
                             table_csv=cfg.table_csv)
    ds_va = RadialNPZDataset(cfg.data_root, cfg.gh_label_dir, split='val',
                             img_size=cfg.img_size, target_S=cfg.S, enable_aug=False,
                             table_csv=cfg.table_csv)
    ds_te = RadialNPZDataset(cfg.data_root, cfg.gh_label_dir, split='test',
                             img_size=cfg.img_size, target_S=cfg.S, enable_aug=False,
                             table_csv=cfg.table_csv)

    print(f"[Train] count={len(ds_tr)}")
    print(f"[Val  ] count={len(ds_va)}")
    print(f"[Test ] count={len(ds_te)}")

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_batch, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.val_batch, shuffle=False,
                       num_workers=max(1,cfg.num_workers//2), pin_memory=True, collate_fn=collate_batch, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=1, shuffle=False,
                       num_workers=max(1,cfg.num_workers//2), pin_memory=True, collate_fn=collate_batch, drop_last=False)

    print(f"[Data] train={len(ds_tr)}  val={len(ds_va)}  test={len(ds_te)}")
    print(f"[Cfg ] batch(train/val)={cfg.batch_size}/{cfg.val_batch}  S={cfg.S}  img={cfg.img_size}  K={cfg.num_basis}")
    try:
        b0 = next(iter(dl_tr))
        print("[Batch Shapes] x:", tuple(b0['x'].shape),
              " GH:", tuple(b0['GH'].shape),
              " R:", tuple(b0['R'].shape),
              " s:", tuple(b0['s'].shape),
              " T:", tuple(b0['T'].shape))
    except StopIteration:
        pass
    return dl_tr, dl_va, dl_te


# ============== 2D differentiable radial silhouette (SAFE: handle empty meshes) ==============
def _plane_uv_basis(axis_dir: torch.Tensor, plane_normal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    e_a = axis_dir / (axis_dir.norm(dim=-1, keepdim=True) + 1e-8)
    e_b = torch.linalg.cross(plane_normal, e_a)
    e_b = e_b / (e_b.norm(dim=-1, keepdim=True) + 1e-8)
    return e_a, e_b

def _soft_mask_from_points(u: torch.Tensor, v: torch.Tensor,
                           umin: torch.Tensor, umax: torch.Tensor,
                           vmin: torch.Tensor, vmax: torch.Tensor,
                           H:int, W:int, alpha: float) -> torch.Tensor:
    device = u.device
    uu = torch.linspace(umin, umax, W, device=device)
    vv = torch.linspace(vmax, vmin, H, device=device)
    grid_u, grid_v = torch.meshgrid(uu, vv, indexing='xy')
    grid_u = grid_u.t().contiguous()
    grid_v = grid_v.t().contiguous()
    G = H * W
    gu = grid_u.reshape(G)
    gv = grid_v.reshape(G)
    P = u.shape[-1]
    chunks = 20000
    dmin2 = torch.full((G,), 1e6, device=device)
    for st in range(0, P, chunks):
        up = u[..., st:st+chunks]
        vp = v[..., st:st+chunks]
        du = gu[:, None] - up[None, :]
        dv = gv[:, None] - vp[None, :]
        d2 = du*du + dv*dv
        dmin2 = torch.minimum(dmin2, d2.min(dim=1).values)
    d = torch.sqrt(torch.clamp_min(dmin2, 1e-12))
    occ = torch.exp(-alpha * d).reshape(H, W)
    return occ

def radial_soft_silhouette_loss(
    mesh_pred, mesh_gt,
    num_planes:int, grid:int,
    alpha: float,
    pts_per_mesh:int,
    ref_dir_xy: Tuple[float,float,float],
    axis_y: Tuple[float,float,float]
) -> Tuple[torch.Tensor, Dict[str,float]]:
    device = mesh_pred.device

    # SAFE early-exit: if either mesh is empty, return zero loss and keep training flow
    if (mesh_pred.num_faces_per_mesh() == 0).any() or (mesh_gt.num_faces_per_mesh() == 0).any() \
       or (mesh_pred.num_verts_per_mesh() == 0).any() or (mesh_gt.num_verts_per_mesh() == 0).any():
        zero = torch.zeros((), device=device)
        return zero, dict(rad2d_dice=0.0, rad2d_ce=0.0)

    # SAFE sampling try/except
    try:
        pts_p = p3d_ops.sample_points_from_meshes(mesh_pred, pts_per_mesh, return_normals=False)
        pts_g = p3d_ops.sample_points_from_meshes(mesh_gt,   pts_per_mesh, return_normals=False)
    except ValueError as e:
        if "Meshes are empty" in str(e):
            zero = torch.zeros((), device=device)
            return zero, dict(rad2d_dice=0.0, rad2d_ce=0.0)
        raise

    B, P, _ = pts_p.shape
    a = torch.tensor(axis_y, dtype=torch.float32, device=device); a = a / (a.norm() + 1e-8)
    r0 = torch.tensor(ref_dir_xy, dtype=torch.float32, device=device); r0 = r0 / (r0.norm() + 1e-8)
    v0 = torch.linalg.cross(a, r0); v0 = v0 / (v0.norm() + 1e-8)

    thetas = torch.linspace(0, 2.0*math.pi, num_planes, device=device)
    dice_list, ce_list = [], []
    for theta in thetas:
        n = torch.cos(theta)*r0 + torch.sin(theta)*v0
        n = n / (n.norm() + 1e-8)
        o = pts_g.mean(dim=1, keepdim=True)
        e_a, e_b = _plane_uv_basis(a, n)
        rel_p = pts_p - o; rel_g = pts_g - o
        u_pred = (rel_p @ e_b); v_pred = (rel_p @ e_a)
        u_gt   = (rel_g @ e_b); v_gt   = (rel_g @ e_a)
        up = torch.cat([u_pred, u_gt], dim=1)
        vp = torch.cat([v_pred, v_gt], dim=1)
        umin = up.min(); umax = up.max()
        vmin = vp.min(); vmax = vp.max()
        du = umax - umin; dv = vmax - vmin
        umin = umin - du*0.05; umax = umax + du*0.05
        vmin = vmin - dv*0.05; vmax = vmax + dv*0.05

        dice_b, ce_b = [], []
        for b in range(B):
            occ_p = _soft_mask_from_points(u_pred[b], v_pred[b], umin, umax, vmin, vmax, grid, grid, alpha)
            occ_g = _soft_mask_from_points(u_gt[b],   v_gt[b],   umin, umax, vmin, vmax, grid, grid, alpha)
            inter = (occ_p*occ_g).sum()
            denom = (occ_p.pow(2)+occ_g.pow(2)).sum() + 1e-6
            dice  = (2*inter/denom)
            eps=1e-6
            ce = - (occ_g.clamp(min=eps,max=1-eps)*torch.log(occ_p.clamp(min=eps,max=1-eps)) +
                    (1-occ_g).clamp(min=eps,max=1-eps)*torch.log((1-occ_p).clamp(min=eps,max=1-eps))).mean()
            dice_b.append(dice); ce_b.append(ce)
        dice_list.append(torch.stack(dice_b).mean())
        ce_list.append(torch.stack(ce_b).mean())

    dice_mean = torch.stack(dice_list).mean()
    ce_mean   = torch.stack(ce_list).mean()
    loss_2d   = (1.0 - dice_mean) + 0.5*ce_mean
    logs = dict(rad2d_dice=float(dice_mean.detach().cpu().item()),
                rad2d_ce=float(ce_mean.detach().cpu().item()))
    return loss_2d, logs


# ============== Export: mesh + radial PNGs (SAFE: skip empty meshes) ==============
@torch.no_grad()
def save_mesh_and_radial_pngs(cfg: CFG, ghd: GHD, batch: Dict[str,Any],
                              GH_p: torch.Tensor, R_p: torch.Tensor, s_p: torch.Tensor, T_p: torch.Tensor,
                              out_dir_epoch: Path, max_save:int):
    out_dir_epoch.mkdir(parents=True, exist_ok=True)
    saved = 0
    for b in range(GH_p.shape[0]):
        if saved >= max_save: break
        pid = batch['patient'][b]
        t   = int(batch['time'][b].item())

        mesh_pred = ghd.render(GH_p[b:b+1], R_p[b:b+1], s_p[b:b+1], T_p[b:b+1])
        if (mesh_pred.num_faces_per_mesh() == 0).any() or (mesh_pred.num_verts_per_mesh() == 0).any():
            continue

        Vp = mesh_pred.verts_packed().detach().cpu().numpy()
        Fp = mesh_pred.faces_packed().detach().cpu().numpy()
        import trimesh
        pred_trimesh = trimesh.Trimesh(vertices=Vp, faces=Fp, process=False)
        obj_pred = out_dir_epoch / f"{pid}_time{t:03d}_pred.obj"
        pred_trimesh.export(obj_pred)

        gt_obj = batch.get('mesh_path',[None]*GH_p.shape[0])[b]
        if gt_obj is not None and os.path.exists(gt_obj):
            png_path = out_dir_epoch / f"{pid}_time{t:03d}_radial.png"
            try:
                _ = radial_2d_metrics(gt_obj, str(obj_pred), cfg.radial_eval, save_png=str(png_path))
            except Exception as e:
                print(f"[warn] radial_2d_metrics failed for {pid} t{t}: {e}")
        saved += 1


# ============== Saliency (input gradient) overlay visualization ==============
@torch.no_grad()
def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def _norm01(a: np.ndarray) -> np.ndarray:
    a = a - a.min()
    m = a.max() + 1e-8
    return a / m

def save_saliency_grid(x: torch.Tensor,
                       sal: torch.Tensor,
                       out_png: Path,
                       num_samples:int = 2,
                       num_slices:int = 4,
                       alpha: float = 0.6):
    B,S,C,H,W = x.shape
    idx_b = random.sample(range(B), k=min(num_samples, B))
    fig, axes = plt.subplots(len(idx_b), num_slices, figsize=(3.2*num_slices, 3.2*len(idx_b)), dpi=120)
    if len(idx_b)==1:
        import numpy as _np
        axes = _np.expand_dims(axes, 0)
    for row, b in enumerate(idx_b):
        pick = sorted(random.sample(range(S), k=min(num_slices, S)))
        for col, s in enumerate(pick):
            img = _to_numpy(x[b, s, 0])
            att = _to_numpy(sal[b, s])
            img = _norm01(img); att = _norm01(att)
            ax = axes[row, col]
            ax.imshow(img, cmap='gray'); ax.imshow(att, cmap='jet', alpha=alpha)
            ax.axis('off'); ax.set_title(f"b{b}-s{s}")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)

def compute_saliency(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.clone().detach().requires_grad_(True)
    GH_p, R_p, s_p, T_p = model(x)
    scalar = (GH_p**2).mean() + (R_p**2).mean() + (s_p**2).mean() + (T_p**2).mean()
    scalar.backward()
    g = x.grad  # [B,S,C,H,W]
    g = g.abs().sum(dim=2)
    B,S,H,W = g.shape
    g = g.reshape(B*S, -1)
    g = g / (g.max(dim=1, keepdim=True).values + 1e-12)
    g = g.reshape(B,S,H,W)
    return g.detach()


# ============== LR schedulers (Warmup + Cosine + Plateau) ==============
def build_schedulers(optimizer: torch.optim.Optimizer, cfg: CFG):
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return float(epoch + 1) / float(max(1, cfg.warmup_epochs))
        progress = (epoch - cfg.warmup_epochs) / float(max(1, cfg.epochs - cfg.warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        base = cfg.min_lr / cfg.lr
        return base + (1.0 - base) * cosine

    sch_epoch = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    sch_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.plateau_factor,
        patience=cfg.plateau_patience, threshold=cfg.plateau_threshold,
        cooldown=cfg.plateau_cooldown, min_lr=cfg.min_lr
    )
    return sch_epoch, sch_plateau


# ======== Parameter counting ========
def param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def try_encoder_param_count(model: nn.Module) -> Optional[int]:
    for attr in ['encoder','backbone','slice_encoder','stem','feature_extractor']:
        if hasattr(model, attr) and isinstance(getattr(model, attr), nn.Module):
            return param_count(getattr(model, attr))
    return None


# ========= Loss switch application (set a component to 0 to ablate) =========
def apply_loss_switches(base: LossWeights, switches: Dict[str,bool]) -> LossWeights:
    w = LossWeights(**base.__dict__)
    if not switches: return w
    if not switches.get('chamfer', True):            w.ch = 0.0
    if not switches.get('normal_surface', True):     w.nrm_surf = 0.0
    if not switches.get('voxel_dice', True):         w.dice3d = 0.0
    if not switches.get('laplacian', True):          w.lap = 0.0
    if not switches.get('edge', True):               w.edge = 0.0
    if not switches.get('normal_consistency', True): w.nrm = 0.0
    return w


# ===================== Test-set evaluation (run when best is refreshed) =====================
@torch.no_grad()
def evaluate_test_and_save_csv(
    cfg: CFG,
    model: nn.Module,
    ghd: GHD,
    dl_te: DataLoader,
    run_dir: Path,
    epoch_best: int
):
    model.eval()
    rows = []
    times = []

    eval_cfg = Eval3DConfig(
        n_points_cd = cfg.eval_points_cd,
        n_points_hd = cfg.eval_points_hd,
        voxel_dice = VoxelDiceConfig(grid_size=cfg.eval_voxel_dice_grid, bbox_scale=1.05, alpha=200.0)
    )

    for batch in tqdm(dl_te, desc="[Test Eval]", ncols=120):
        x  = batch['x'].to(cfg.device)
        GH = batch['GH'].to(cfg.device)
        R  = batch['R'].to(cfg.device)
        s  = batch['s'].to(cfg.device)
        T  = batch['T'].to(cfg.device)
        pid = str(batch.get('patient',['NA'])[0])
        tfrm= int(batch.get('time',[torch.tensor(0)])[0])

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        GH_p, R_p, s_p, T_p = model(x)
        mesh_pred_full = ghd.render(GH_p, R_p, s_p, T_p)
        mesh_gt_full   = ghd.render(GH,   R,   s,   T)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt_ms = (time.time() - t0) * 1000.0
        times.append(dt_ms)

        nv_p = mesh_pred_full.num_verts_per_mesh()
        nf_p = mesh_pred_full.num_faces_per_mesh()
        nv_g = mesh_gt_full.num_verts_per_mesh()
        nf_g = mesh_gt_full.num_faces_per_mesh()
        if (nv_p == 0).any() or (nf_p == 0).any() or (nv_g == 0).any() or (nf_g == 0).any():
            rows.append({
                'patient': pid, 'time_idx': tfrm,
                'dice3d': float('nan'), 'iou3d': float('nan'),
                'cd': float('nan'), 'hd': float('nan'),
                'laplace': float('nan'), 'time_ms': dt_ms
            })
            continue

        m = eval_metrics_3d(mesh_pred_full, mesh_gt_full, cfg=eval_cfg)

        rows.append({
            'patient': pid,
            'time_idx': tfrm,
            'dice3d': m['Dice3D'],
            'iou3d':  m['IoU3D'],
            'cd':     m['CD'],
            'hd':     m['HD'],
            'laplace': m['Laplace'],
            'time_ms': dt_ms
        })

    def _mean_std(key):
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and (np.isnan(r[key]) or np.isinf(r[key])))]
        arr = np.array(vals, dtype=float) if len(vals)>0 else np.array([np.nan])
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    mean_row = {'patient':'MEAN','time_idx':'-'}
    std_row  = {'patient':'STD','time_idx':'-'}
    for k in ['dice3d','iou3d','cd','hd','laplace','time_ms']:
        m, s = _mean_std(k)
        mean_row[k] = m; std_row[k] = s

    csv_path = run_dir / "test_metrics.csv"
    tmp_path = run_dir / "test_metrics.tmp.csv"
    fields = ['patient','time_idx','dice3d','iou3d','cd','hd','laplace','time_ms']
    with open(tmp_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(mean_row)
        w.writerow(std_row)
        f.write("\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp_path, csv_path)
    print(f"  ✓ Test metrics saved → {csv_path}")

    print(f"  [TEST] mean Dice3D={mean_row['dice3d']:.4f} | IoU3D={mean_row['iou3d']:.4f} | "
          f"CD={mean_row['cd']:.6f} | HD={mean_row['hd']:.6f} | Laplace={mean_row['laplace']:.6e} | "
          f"speed={mean_row['time_ms']:.1f}ms")


def main():
    # ===== Optional: enable anomaly detection for debugging =====
    # torch.autograd.set_detect_anomaly(True)

    cfg = CFG()

    run_dir = Path(cfg.out_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_cfg(cfg, run_dir)

    if _HAS_SWANLAB:
        try:
            swanlab.init(
                project="FeEchoFM-NPZ",
                run_name=f"{cfg.model_type}-{cfg.encoder_name}-rt{int(cfg.use_radial_trans)}-d{cfg.d_model}",
                mode="online"
            )
        except Exception as e:
            print(f"[warn] SwanLab init failed: {e}")
    else:
        print("[info] SwanLab not installed; fallback to console logs.")

    dl_tr, dl_va, dl_te = build_loaders(cfg)

    model = build_model(ModelCfg(
        encoder_name=cfg.encoder_name,
        in_per_slice=cfg.in_per_slice,
        num_basis=cfg.num_basis,
        d_model=cfg.d_model,
        use_radial_trans=cfg.use_radial_trans,
        nhead=cfg.nhead, dim_ff=cfg.dim_ff, num_layers=cfg.num_layers,
        dropout=cfg.dropout
    )).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sch_epoch, sch_plateau = build_schedulers(opt, cfg)

    total_params = param_count(model)
    enc_params = try_encoder_param_count(model)
    if enc_params is not None:
        print(f"[Params] encoder={enc_params/1e6:.2f} M | total={total_params/1e6:.2f} M")
        if _HAS_SWANLAB:
            swanlab.log({"model/params_encoder_M": enc_params/1e6, "model/params_total_M": total_params/1e6}, step=0)
    else:
        print(f"[Params] total={total_params/1e6:.2f} M (encoder param count not detected)")
        if _HAS_SWANLAB:
            swanlab.log({"model/params_total_M": total_params/1e6}, step=0)

    gcfg = GHDConfig(
        base_shape_path=cfg.base_shape,
        device=cfg.device,
        num_basis=cfg.num_basis,
        basis_cache=cfg.basis_cache,
        mix_cot=1.0, mix_dis=0.3, mix_std=0.3
    )
    ghd = GHD(gcfg)

    # ---- Train-side helpers: per-sample safe render without changing GHD ----
    @torch.no_grad()
    def _try_render_one(GH_1, R_1, s_1, T_1):
        try:
            m = ghd.render(GH_1, R_1, s_1, T_1)
            nv = m.num_verts_per_mesh()
            nf = m.num_faces_per_mesh()
            return bool((nv > 0).all().item() and (nf > 0).all().item()), None
        except Exception as e:
            msg = str(e)
            if "Padding of faces must be at the end" in msg:
                return False, msg
            raise

    @torch.no_grad()
    def safe_valid_mask_by_render(GH, R, s, T, batch_meta=None):
        B = GH.shape[0]
        device = GH.device
        valid = torch.zeros(B, dtype=torch.bool, device=device)
        for b in range(B):
            ok, err = _try_render_one(GH[b:b+1], R[b:b+1], s[b:b+1], T[b:b+1])
            if not ok and err is not None:
                pid, tfrm = None, None
                if batch_meta is not None:
                    try:
                        pid = batch_meta.get('patient', [None]*B)[b]
                        tfrm = batch_meta.get('time', [None]*B)[b]
                        tfrm = int(tfrm.item()) if torch.is_tensor(tfrm) else tfrm
                    except Exception:
                        pass
                print(f"[faces-pad] skip sample b={b} pid={pid} time={tfrm}: {err}")
            valid[b] = ok
        return valid

    best_val = 1e9

    for ep in range(1, cfg.epochs + 1):
        # ----------------- Train -----------------
        model.train()
        meters = dict(L=0.0, Lparam=0.0, L3d=0.0, L2d=0.0,
                      ch=0.0, dice3d=0.0, rad2d_dice=0.0, rad2d_ce=0.0, n=0)
        pbar = tqdm(dl_tr, desc=f"[Train {ep}/{cfg.epochs}]", ncols=140)

        for batch in pbar:
            x  = batch['x'].to(cfg.device)     # [B,S,5,H,W]
            GH = batch['GH'].to(cfg.device)    # [B,K,3]
            R  = batch['R'].to(cfg.device)     # [B,3]
            s  = batch['s'].to(cfg.device)     # [B,1]
            T  = batch['T'].to(cfg.device)     # [B,3]

            opt.zero_grad(set_to_none=True)

            # Forward pass: parameter regression loss
            GH_p, R_p, s_p, T_p = model(x)
            L_param, _ = npz_param_loss(
                GH_p, R_p, s_p, T_p, GH, R, s, T,
                w_GH=2.0, w_R=1.0, w_s=1.0, w_T=1.0, reg_GH=1e-6
            )
            GH_p, R_p, s_p, T_p = _sanitize_params(GH_p, R_p, s_p, T_p)

            # Entry-level validity check: per-sample render out-of-graph
            valid_pred = safe_valid_mask_by_render(GH_p, R_p, s_p, T_p, batch)
            valid_gt   = safe_valid_mask_by_render(GH,   R,   s,   T,   batch)
            valid = (valid_pred & valid_gt)

            # —— Per-sample geometry losses (in-graph; do not index_select whole batch) —— #
            L_3d_accum = torch.tensor(0.0, device=cfg.device)
            L_2d_accum = torch.tensor(0.0, device=cfg.device)
            eff = 0
            logs3d_acc = dict(L_total=0.0, L_ch=0.0, L_dice3d=0.0)
            logs2d_acc = dict(rad2d_dice=0.0, rad2d_ce=0.0)

            for b in range(x.size(0)):
                if not bool(valid[b].item()):
                    continue

                GH_p_b, R_p_b, s_p_b, T_p_b = GH_p[b:b+1], R_p[b:b+1], s_p[b:b+1], T_p[b:b+1]
                GH_b,   R_b,   s_b,   T_b   = GH[b:b+1],   R[b:b+1],   s[b:b+1],   T[b:b+1]

                # Per-sample mesh rendering (kept in graph)
                mesh_pred_b = ghd.render(GH_p_b, R_p_b, s_p_b, T_p_b)
                mesh_gt_b   = ghd.render(GH_b,   R_b,   s_b,   T_b)

                # Second-level validity check
                v2 = _valid_mesh_mask(mesh_pred_b, mesh_gt_b)
                if not bool(v2.all().item()):
                    continue

                # Geometry losses
                w3d_eff = apply_loss_switches(cfg.loss_weights, cfg.loss_switches) if cfg.use_loss_switches else cfg.loss_weights
                L_3d_b, logs3d_b = combine_3d_losses(
                    mesh_pred_b, mesh_gt_b, weights=w3d_eff.__dict__, voxel_cfg=cfg.voxel_cfg
                )
                L_2d_b, logs2d_b = radial_soft_silhouette_loss(
                    mesh_pred_b, mesh_gt_b,
                    num_planes=cfg.planes_for_loss, grid=cfg.grid_2d,
                    alpha=cfg.alpha_2d, pts_per_mesh=cfg.pts_per_mesh_2d,
                    ref_dir_xy=cfg.ref_dir_xy, axis_y=cfg.axis_y
                )

                L_3d_accum = L_3d_accum + L_3d_b
                L_2d_accum = L_2d_accum + L_2d_b
                eff += 1

                # Log accumulation (scalarized)
                logs3d_acc['L_total']   += float(L_3d_b.detach().item())
                logs3d_acc['L_ch']      += float(logs3d_b.get('L_ch', 0.0))
                logs3d_acc['L_dice3d']  += float(logs3d_b.get('L_dice3d', 0.0))
                logs2d_acc['rad2d_dice']+= float(logs2d_b.get('rad2d_dice', 0.0))
                logs2d_acc['rad2d_ce']  += float(logs2d_b.get('rad2d_ce', 0.0))

            if eff > 0:
                L_3d = L_3d_accum / eff
                L_2d = L_2d_accum / eff
            else:
                L_3d = torch.tensor(0.0, device=cfg.device)
                L_2d = torch.tensor(0.0, device=cfg.device)

            # Global loss and backward
            L = cfg.w_param * L_param + cfg.w_3d * L_3d + cfg.w_2d * L_2d
            L.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            # Running meters
            bs_full = x.size(0)
            meters['n']      += bs_full
            meters['L']      += float(L.item()) * bs_full
            meters['Lparam'] += float(L_param.item()) * bs_full
            meters['L3d']    += logs3d_acc['L_total'] * eff
            meters['L2d']    += float(L_2d.item()) * eff
            meters['ch']     += logs3d_acc['L_ch'] * eff
            meters['dice3d'] += logs3d_acc['L_dice3d'] * eff
            meters['rad2d_dice'] += logs2d_acc['rad2d_dice'] * eff
            meters['rad2d_ce']   += logs2d_acc['rad2d_ce'] * eff

            pbar.set_postfix(loss=meters['L']/max(1,meters['n']),
                             Chamfer=meters['ch']/max(1,meters['n']),
                             Dice2D=meters['rad2d_dice']/max(1,meters['n']),
                             lr=get_lr(opt))

        # Epoch-level warmup/cosine step
        sch_epoch.step()

        # Average meters
        for k in list(meters.keys()):
            if k != 'n':
                meters[k] /= max(1, meters['n'])

        # ----------------- Validation -----------------
        model.eval()
        val = dict(L=0.0, Lparam=0.0, L3d=0.0, L2d=0.0,
                   ch=0.0, dice3d=0.0, rad2d_dice=0.0, rad2d_ce=0.0, n=0)

        out_dir_ep = run_dir / f"epoch_{ep:03d}"
        out_dir_ep.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            saved_counter = 0
            vbar = tqdm(dl_va, desc=f"[Val   {ep}/{cfg.epochs}]", ncols=140)
            attn_batch_cached = None

            total_ms = 0.0
            cnt_ms   = 0

            for batch in vbar:
                x  = batch['x'].to(cfg.device)
                GH = batch['GH'].to(cfg.device)
                R  = batch['R'].to(cfg.device)
                s  = batch['s'].to(cfg.device)
                T  = batch['T'].to(cfg.device)

                # Inference time for the batch (single forward only)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t0 = time.time()
                GH_p, R_p, s_p, T_p = model(x)
                GH_p, R_p, s_p, T_p = _sanitize_params(GH_p, R_p, s_p, T_p)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                dt_ms = (time.time() - t0) * 1000.0
                total_ms += dt_ms; cnt_ms += x.size(0)

                # Parameter loss (no backward in val)
                L_param, _ = npz_param_loss(
                    GH_p, R_p, s_p, T_p, GH, R, s, T,
                    w_GH=2.0, w_R=1.0, w_s=1.0, w_T=1.0, reg_GH=1e-6
                )

                # Entry-level validity (val uses safe checks too)
                valid_pred = safe_valid_mask_by_render(GH_p, R_p, s_p, T_p, batch)
                valid_gt   = safe_valid_mask_by_render(GH,   R,   s,   T,   batch)
                valid = (valid_pred & valid_gt)

                # Per-sample geometry losses (no_grad)
                L_3d_accum = 0.0
                L_2d_accum = 0.0
                eff = 0
                logs3d_acc = dict(L_total=0.0, L_ch=0.0, L_dice3d=0.0)
                logs2d_acc = dict(rad2d_dice=0.0, rad2d_ce=0.0)
                valid_indices = []

                for b in range(x.size(0)):
                    if not bool(valid[b].item()):
                        continue

                    GH_p_b, R_p_b, s_p_b, T_p_b = GH_p[b:b+1], R_p[b:b+1], s_p[b:b+1], T_p[b:b+1]
                    GH_b,   R_b,   s_b,   T_b   = GH[b:b+1],   R[b:b+1],   s[b:b+1],   T[b:b+1]

                    mesh_pred_b = ghd.render(GH_p_b, R_p_b, s_p_b, T_p_b)
                    mesh_gt_b   = ghd.render(GH_b,   R_b,   s_b,   T_b)

                    v2 = _valid_mesh_mask(mesh_pred_b, mesh_gt_b)
                    if not bool(v2.all().item()):
                        continue

                    w3d_eff = apply_loss_switches(cfg.loss_weights, cfg.loss_switches) if cfg.use_loss_switches else cfg.loss_weights
                    L_3d_b, logs3d_b = combine_3d_losses(
                        mesh_pred_b, mesh_gt_b, weights=w3d_eff.__dict__, voxel_cfg=cfg.voxel_cfg
                    )
                    L_2d_b, logs2d_b = radial_soft_silhouette_loss(
                        mesh_pred_b, mesh_gt_b,
                        num_planes=cfg.planes_for_loss, grid=cfg.grid_2d,
                        alpha=cfg.alpha_2d, pts_per_mesh=cfg.pts_per_mesh_2d,
                        ref_dir_xy=cfg.ref_dir_xy, axis_y=cfg.axis_y
                    )

                    L_3d_accum += float(L_3d_b.item())
                    L_2d_accum += float(L_2d_b.item())
                    eff += 1
                    valid_indices.append(b)

                    logs3d_acc['L_total']   += float(L_3d_b.item())
                    logs3d_acc['L_ch']      += float(logs3d_b.get('L_ch', 0.0))
                    logs3d_acc['L_dice3d']  += float(logs3d_b.get('L_dice3d', 0.0))
                    logs2d_acc['rad2d_dice']+= float(logs2d_b.get('rad2d_dice', 0.0))
                    logs2d_acc['rad2d_ce']  += float(logs2d_b.get('rad2d_ce', 0.0))

                if eff > 0:
                    L_3d = L_3d_accum / eff
                    L_2d = L_2d_accum / eff
                else:
                    L_3d = 0.0
                    L_2d = 0.0

                L = cfg.w_param * float(L_param.item()) + cfg.w_3d * L_3d + cfg.w_2d * L_2d

                bs_full = x.size(0)
                val['n']      += bs_full
                val['L']      += L * bs_full
                val['Lparam'] += float(L_param.item()) * bs_full
                val['L3d']    += logs3d_acc['L_total'] * eff
                val['L2d']    += L_2d * eff
                val['ch']     += logs3d_acc['L_ch'] * eff
                val['dice3d'] += logs3d_acc['L_dice3d'] * eff
                val['rad2d_dice'] += logs2d_acc['rad2d_dice'] * eff
                val['rad2d_ce']   += logs2d_acc['rad2d_ce'] * eff

                # Export visualizations only for valid samples and within quota
                do_vis = (cfg.save_vis_every == 0) or (ep % cfg.save_vis_every == 0)
                if do_vis and saved_counter < cfg.save_max_per_epoch and len(valid_indices) > 0:
                    k = min(cfg.save_max_per_epoch - saved_counter, len(valid_indices))
                    idx_tensor = torch.tensor(valid_indices[:k], dtype=torch.long, device=cfg.device)
                    batch_v = subset_batch_for_save(batch, idx_tensor.cpu())
                    GH_p_v = GH_p.index_select(0, idx_tensor)
                    R_p_v  = R_p.index_select(0, idx_tensor)
                    s_p_v  = s_p.index_select(0, idx_tensor)
                    T_p_v  = T_p.index_select(0, idx_tensor)
                    save_mesh_and_radial_pngs(cfg, ghd, batch_v, GH_p_v, R_p_v, s_p_v, T_p_v, out_dir_ep, k)
                    saved_counter += k

                if attn_batch_cached is None and random.random() < 0.5:
                    attn_batch_cached = {k: _to_cpu_safe(v) for k, v in batch.items()}

                vbar.set_postfix(L=val['L']/max(1,val['n']),
                                 Chamfer=val['ch']/max(1,val['n']),
                                 Dice2D=val['rad2d_dice']/max(1,val['n']),
                                 lr=get_lr(opt))

        # Validation averages
        for k in list(val.keys()):
            if k != 'n':
                val[k] /= max(1, val['n'])

        avg_infer_ms = (total_ms / max(1, cnt_ms)) if cnt_ms > 0 else float('nan')
        if _HAS_SWANLAB:
            swanlab.log({"val/avg_infer_ms_per_sample": avg_infer_ms}, step=ep)
        print(f"[{ep:03d}] Val avg infer time: {avg_infer_ms:.1f} ms/sample")

        # —— Saliency —— #
        try:
            if attn_batch_cached is not None:
                with torch.enable_grad():
                    xb = attn_batch_cached['x'].to(cfg.device)
                    sal = compute_saliency(model, xb)
                    attn_png = out_dir_ep / f"saliency.png"
                    save_saliency_grid(
                        xb.detach().cpu(), sal.detach().cpu(),
                        attn_png, num_samples=cfg.attn_num_samples,
                        num_slices=cfg.attn_num_slices, alpha=cfg.attn_alpha
                    )
        except Exception as e:
            print(f"[warn] saliency save failed: {e}")

        # —— LR on plateau —— #
        sch_plateau.step(val['L'])

        # Console summary
        print(f"[{ep:03d}] "
              f"Train L={meters['L']:.4f} | Chamfer={meters['ch']:.6f} | 3D softDice={1-meters['dice3d']:.4f} "
              f"| 2D Dice={meters['rad2d_dice']:.4f} CE={meters['rad2d_ce']:.4f}")

        print(f"[{ep:03d}] "
              f" Val  L={val['L']:.4f} | Chamfer={val['ch']:.6f} | 3D softDice={1-val['dice3d']:.4f} "
              f"| 2D Dice={val['rad2d_dice']:.4f} CE={val['rad2d_ce']:.4f}")

        # SwanLab logs
        if _HAS_SWANLAB:
            swanlab.log({
                # train
                "train/L_total": meters['L'],
                "train/L_param": meters['Lparam'],
                "train/L_3d": meters['L3d'],
                "train/L_2d": meters['L2d'],
                "train/Chamfer": meters['ch'],
                "train/3D_softDice": 1.0 - meters['dice3d'],
                "train/2D_Dice": meters['rad2d_dice'],
                "train/2D_CE": meters['rad2d_ce'],
                "lr": get_lr(opt),
                # val
                "val/L_total": val['L'],
                "val/L_param": val['Lparam'],
                "val/L_3d": val['L3d'],
                "val/L_2d": val['L2d'],
                "val/Chamfer": val['ch'],
                "val/3D_softDice": 1.0 - val['dice3d'],
                "val/2D_Dice": val['rad2d_dice'],
                "val/2D_CE": val['rad2d_ce'],
            }, step=ep)

        # Save best by val/L_total
        if val['L'] < best_val:
            best_val = val['L']
            ckpt = run_dir / "best_model.pth"
            torch.save({'model': model.state_dict(),
                        'epoch': ep,
                        'val_L': best_val}, ckpt)
            print(f"  ✓ saved best → {ckpt}")
        
        # Periodic Test every 15 epochs
        if ep % 15 == 0:
            best_ckpt = run_dir / "best_model.pth"
            if best_ckpt.exists():
                state = torch.load(best_ckpt, map_location=cfg.device)
                model.load_state_dict(state['model'])
                print(f"[Periodic Test] Using best from epoch={state.get('epoch','?')}, val_L={state.get('val_L','?')}")
            else:
                print("[Periodic Test] best_model.pth not found, using current model.")

            evaluate_test_and_save_csv(cfg, model, ghd, dl_te, run_dir, epoch_best=ep)
        
        # Periodic checkpoint every 50 epochs
        if ep % 50 == 0:
            ckpt_ep = run_dir / f"epoch_{ep:03d}.pth"
            torch.save({
                'model': model.state_dict(),
                'epoch': ep,
                'val_L': float(val['L'])
            }, ckpt_ep)
            print(f"  ✓ periodic checkpoint saved → {ckpt_ep}")
    
    # ===== Final Test with best weights =====
    best_ckpt = run_dir / "best_model.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=cfg.device)
        model.load_state_dict(state['model'])
        print(f"[Final Test] Using best from epoch={state.get('epoch','?')}, val_L={state.get('val_L','?')}")
    else:
        print("[Final Test] best_model.pth not found, using last-epoch model.")

    evaluate_test_and_save_csv(cfg, model, ghd, dl_te, run_dir, epoch_best=-1)
    print("Done ✅")


if __name__ == "__main__":
    main()