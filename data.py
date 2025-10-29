# data.py
# -----------------------------------------------------------
# Unified data loading (radial PNG + GH npz + table.csv)
# Output batch:
#   x:[B,S,5,H,W], mask:[B,S,H,W] (placeholder), keep:[B,S], angle:[B,S],
#   GH:[B,K,3], R:[B,3], s:[B,1], T:[B,3],
#   image_raw_u8:[B,S,H,W], patient(list[str]), time:[B],
#   mesh_path(list[str or None]),
#   spacing:[B,3], times:[B], ED_time:[B], ES_time:[B], metrics(list[dict])
# Features:
#   - Read spacing(mm) from PatientXXX/info.cfg
#   - Dataset exposes self.patients (de-duplicated patient IDs)
#   - slice_degree_step controls sparse sampling by angle
# -----------------------------------------------------------

from __future__ import annotations
import os, re, math, csv
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import yaml  # If not available, fallback to a simple parser
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# ------------------------------
#       Basic Utilities
# ------------------------------
def _read_info_cfg(info_path: str):
    """
    Return {'spacing': (sx,sy,sz), 'patient': 'Patient001', ...}
    spacing unit: mm/px (pixel spacing)
    """
    p = Path(info_path)
    if not p.exists():
        return {"spacing": (1.0, 1.0, 1.0), "patient": p.parent.name}
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if _HAS_YAML:
        d = yaml.safe_load(txt)
        sx = float(d.get("spacing", {}).get("x_mm", 1.0))
        sy = float(d.get("spacing", {}).get("y_mm", 1.0))
        sz = float(d.get("spacing", {}).get("z_mm", 1.0))
        pid = d.get("patient", p.parent.name)
        return {"spacing": (sx, sy, sz), "patient": pid}
    # Simple fallback parsing (without yaml dependency)
    def _grab(key):
        m = re.search(rf"{key}:\s*([^\n]+)", txt)
        return m.group(1).strip() if m else None
    pid = _grab("patient") or p.parent.name
    sx = _grab("x_mm"); sy = _grab("y_mm"); sz = _grab("z_mm")
    try:
        sx, sy, sz = float(sx), float(sy), float(sz)
    except Exception:
        sx, sy, sz = 1.0, 1.0, 1.0
    return {"spacing": (sx, sy, sz), "patient": pid}


def _scan_radial_cases(data_root: Path,
                       require_mesh: bool = False,
                       patients: Optional[List[str]] = None) -> List[Dict[str,Any]]:
    """
    Traverse radial frames under Patient*/image and group by time;
    Return a list, each item:
      dict(patient, time, img_paths(list), mask_paths(list or None), mesh_path or None)
    """
    cases = []
    for pid_dir in sorted(data_root.glob("Patient*")):
        pid = pid_dir.name
        if patients and pid not in set(patients):
            continue

        img_dir, mask_dir = pid_dir/"image", pid_dir/"mask"
        if not img_dir.exists():
            continue

        # Collect time IDs
        time_ids = set()
        for p in img_dir.glob(f"{pid}_slice*time*.png"):
            m = re.search(r"time(\d+)", p.name)
            if m:
                time_ids.add(int(m.group(1)))

        for t in sorted(time_ids):
            # All slice PNGs at this time
            imgs = sorted(
                img_dir.glob(f"{pid}_slice*time{t:03d}.png"),
                key=lambda p: int(re.search(r"slice(\d+)", p.name).group(1))
            )
            if len(imgs) == 0:
                continue

            masks = None
            if mask_dir.exists():
                cand = [mask_dir/p.name for p in imgs]
                masks = [str(p) if p.exists() else None for p in cand]

            mp = pid_dir/"mesh"/f"{pid}_time{t:03d}.obj"
            mesh_path = str(mp) if mp.exists() else None
            if require_mesh and mesh_path is None:
                continue

            cases.append(dict(
                patient=pid, time=int(t),
                img_paths=[str(p) for p in imgs],
                mask_paths=masks,
                mesh_path=mesh_path
            ))
    return cases


def _select_evenly_by_count(xs: List[Any], target_len: int) -> List[Any]:
    """Uniformly sample xs to target_len (nearest neighbor to linspace), for angular sparsification/resampling."""
    if len(xs) == target_len:
        return list(xs)
    if len(xs) == 0 or target_len <= 0:
        return []
    idx = np.linspace(0, len(xs)-1, target_len)
    idx = np.round(idx).astype(int)
    return [xs[i] for i in idx]


def _read_gray01(png_path: str) -> np.ndarray:
    im = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(png_path)
    return (im.astype(np.float32) / 255.0)


def _radial_posenc(S: int, H: int, W: int, start_angle_deg: float = 0.0) -> np.ndarray:
    """Simple per-slice angular positional encoding (cosθ, sinθ, normalized index)."""
    enc = []
    for s in range(S):
        theta = (start_angle_deg + s * 360.0 / max(1, S)) * math.pi / 180.0
        cos_t = np.full((H, W), math.cos(theta), np.float32)
        sin_t = np.full((H, W), math.sin(theta), np.float32)
        idx_n = np.full((H, W), s / max(1, S-1), np.float32)
        enc.append(np.stack([cos_t, sin_t, idx_n], axis=0))
    return np.stack(enc, axis=0)  # (S,3,H,W)


def load_patient_table(table_csv: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    Optional patient-level CSV (spacing/times/ED/ES/volumes/strains etc.).
    If CSV does not provide spacing, will fallback to info.cfg.
    """
    table: Dict[str, Dict[str, Any]] = {}
    if table_csv is None:
        return table
    csv_path = Path(table_csv)
    if not csv_path.exists():
        print(f"[data] warn: table csv not found: {csv_path}")
        return table

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patient") or row.get("Patient") or row.get("PatientID")
            if not pid:
                continue

            def _get_float(key):
                v = row.get(key, None)
                if v in (None, ""): return None
                try: return float(v)
                except: return None

            def _get_int(key):
                v = row.get(key, None)
                if v in (None, ""): return None
                try: return int(float(v))
                except: return None

            entry = dict(
                spacing_x_mm=_get_float("spacing_x_mm"),
                spacing_y_mm=_get_float("spacing_y_mm"),
                spacing_z_mm=_get_float("spacing_z_mm"),
                times=_get_int("times"),
                ED_time=_get_int("ED_time"),
                ES_time=_get_int("ES_time"),
                EDV_ml=_get_float("EDV_ml"),
                ESV_ml=_get_float("ESV_ml"),
                SV_ml=_get_float("SV_ml"),
                EF_percent=_get_float("EF_%") or _get_float("EF") or _get_float("EF_percent"),
                GLS_percent=_get_float("GLS_%") or _get_float("GLS"),
                GCS_percent=_get_float("GCS_%") or _get_float("GCS"),
            )
            table[pid] = entry
    return table

# ------------------------------
#        Main Dataset 
# ------------------------------
class RadialNPZDataset(Dataset):
    """
    Read radial slice sequences, output x:[S,5,H,W], and (GH,R,s,T) labels (from gh_label_dir).
    Supports:
      - Read spacing(mm) from info.cfg / table.csv
      - slice_degree_step: sparse angular sampling (target S = round(360/step))
      - Expose self.patients (de-duplicated patient IDs)
    """
    def __init__(self,
                 data_root: str,
                 gh_label_dir: str,
                 split: str = "train",
                 img_size: Tuple[int,int] = (256,256),
                 target_S: int = 37,
                 slice_degree_step: Optional[float] = None,  # New: angle step (deg), e.g., 5/10/15/20/25
                 patients: Optional[List[str]] = None,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42,
                 enable_aug: bool = True,
                 table_csv: Optional[str] = None):
        super().__init__()
        self.root = Path(data_root)
        self.gh_dir = Path(gh_label_dir)
        self.H, self.W = int(img_size[0]), int(img_size[1])
        self.enable_aug = bool(enable_aug and split == "train")
        self.table = load_patient_table(table_csv)

        # Angle control: if slice_degree_step is set, determine target slice count by it
        if slice_degree_step is not None and slice_degree_step > 0:
            self.S = int(round(360.0 / float(slice_degree_step)))
        else:
            self.S = int(target_S)

        # Scan samples (all time frames of all Patients)
        items_all = _scan_radial_cases(self.root, require_mesh=False, patients=patients)

        # Record spacing / patients
        self.spacing_map: Dict[str, Tuple[float,float,float]] = {}
        self.patients: List[str] = sorted(list({it["patient"] for it in items_all}))  # de-duplicated patient list
        for pid in self.patients:
            # Prefer spacing from table.csv; otherwise from info.cfg
            tb = self.table.get(pid, {})
            sx = tb.get("spacing_x_mm", None)
            sy = tb.get("spacing_y_mm", None)
            sz = tb.get("spacing_z_mm", None)
            if sx is None or sy is None or sz is None:
                info = _read_info_cfg(str(self.root / pid / "info.cfg"))
                sx, sy, sz = info["spacing"]
            self.spacing_map[pid] = (float(sx or 1.0), float(sy or 1.0), float(sz or 1.0))

        # Split
        rng = np.random.RandomState(seed)
        idx = np.arange(len(items_all)); rng.shuffle(idx)
        n_val = int(round(len(idx)*val_ratio))
        n_te  = int(round(len(idx)*test_ratio))
        id_val = set(idx[:n_val]); id_test=set(idx[n_val:n_val+n_te])

        if split == "train":
            self.items = [items_all[i] for i in range(len(items_all)) if (i not in id_val and i not in id_test)]
        elif split == "val":
            self.items = [items_all[i] for i in sorted(list(id_val))]
        else:
            self.items = [items_all[i] for i in sorted(list(id_test))]

        if len(self.items) == 0:
            raise RuntimeError(f"[data] no items for split={split} under {data_root}")

    def __len__(self):
        return len(self.items)

    # ---- Augmentation ---- #
    def _color_aug(self, x01: np.ndarray) -> np.ndarray:
        if not self.enable_aug:
            return x01
        a = 1.0 + np.random.uniform(-0.15, 0.15)
        b = np.random.uniform(-0.10, 0.10)
        y = np.clip(a * x01 + b, 0.0, 1.0)
        return y

    # ---- Sampling ---- #
    def __getitem__(self, i):
        it = self.items[i]
        pid, time_id = it["patient"], int(it["time"])

        # Sparse angular sampling: uniformly select self.S slices at this time
        img_paths_full = list(it["img_paths"])
        img_paths = _select_evenly_by_count(img_paths_full, self.S)

        # Synchronize mask extraction (if available)
        if it.get("mask_paths", None) is not None:
            mask_paths_full = list(it["mask_paths"])
            mask_paths = _select_evenly_by_count(mask_paths_full, self.S)
        else:
            mask_paths = None

        # Read & resize to target resolution
        imgs = []
        for p in img_paths:
            im = _read_gray01(p)
            im = cv2.resize(im, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            im = self._color_aug(im)
            imgs.append(im)
        imgs = np.stack(imgs, 0)                               # (S,H,W)
        raw_u8 = (imgs*255.0 + 0.5).astype(np.uint8)

        # Positional encoding + present channel
        pos = _radial_posenc(self.S, self.H, self.W, 0.0)      # (S,3,H,W)
        present = np.ones((self.S,1,self.H,self.W), np.float32)
        x = np.concatenate([imgs[:,None], pos, present], axis=1)  # (S,5,H,W)

        angle = np.linspace(-np.pi, np.pi, self.S, dtype=np.float32)
        keep  = np.ones((self.S,), np.bool_)

        # GH labels
        gh_path = self.gh_dir / f"{pid}_time{time_id:03d}.npz"
        if not gh_path.exists():
            raise FileNotFoundError(f"Missing GH label npz: {gh_path}")
        lab = dict(np.load(str(gh_path), allow_pickle=True))
        GH = lab["GH"].astype(np.float32)
        if GH.ndim == 3 and GH.shape[0] == 1:
            GH = GH[0]
        R  = lab["R"].astype(np.float32).reshape(-1)
        s  = lab["s"].astype(np.float32).reshape(-1)
        T  = lab["T"].astype(np.float32).reshape(-1)

        # Mask
        if mask_paths is not None:
            masks = []
            for p in mask_paths:
                if p is None:
                    masks.append(np.zeros((self.H, self.W), np.uint8))
                else:
                    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        masks.append(np.zeros((self.H, self.W), np.uint8))
                    else:
                        m = cv2.resize(m, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                        masks.append(m)
            mask = np.stack(masks, 0)
        else:
            mask = np.zeros((self.S, self.H, self.W), np.uint8)

        # Patient-level info (prefer table.csv, otherwise info.cfg)
        tb = self.table.get(pid, {})
        sx, sy, sz = self.spacing_map.get(pid, (1.0,1.0,1.0))
        times   = int(tb.get("times", -1) or -1)
        ED_time = int(tb.get("ED_time", -1) or -1)
        ES_time = int(tb.get("ES_time", -1) or -1)
        metrics = dict(
            EDV_ml = tb.get("EDV_ml", None),
            ESV_ml = tb.get("ESV_ml", None),
            SV_ml  = tb.get("SV_ml", None),
            EF_percent  = tb.get("EF_percent", None),
            GLS_percent = tb.get("GLS_percent", None),
            GCS_percent = tb.get("GCS_percent", None),
        )

        sample = dict(
            x=torch.from_numpy(x).float(),                  # [S,5,H,W]
            mask=torch.from_numpy(mask.astype(np.int64)),   # [S,H,W]
            keep=torch.from_numpy(keep),                    # [S]
            angle=torch.from_numpy(angle),                  # [S]

            GH=torch.from_numpy(GH),                        # [K,3]
            R=torch.from_numpy(R),                          # [3]
            s=torch.from_numpy(s),                          # [1]
            T=torch.from_numpy(T),                          # [3]

            image_raw_u8=torch.from_numpy(raw_u8),          # [S,H,W]
            patient=pid,
            time=int(time_id),
            mesh_path=it.get("mesh_path", None),

            spacing=torch.tensor([sx,sy,sz], dtype=torch.float32),  # [3], mm/px
            times=int(times),
            ED_time=int(ED_time),
            ES_time=int(ES_time),
            metrics=metrics
        )
        return sample

# ------------------------------
#     collate & dataloaders 
# ------------------------------
def collate_batch(batch: List[Dict[str,Any]]) -> Dict[str,Any]:
    B = len(batch)
    S = batch[0]["x"].shape[0]
    C = batch[0]["x"].shape[1]
    H, W = batch[0]["x"].shape[-2:]

    x     = torch.stack([b["x"] for b in batch], 0)
    mask  = torch.stack([b["mask"] for b in batch], 0)
    keep  = torch.stack([b["keep"] for b in batch], 0)
    angle = torch.stack([b["angle"] for b in batch], 0)

    GH = torch.stack([b["GH"] for b in batch], 0)
    R  = torch.stack([b["R"]  for b in batch], 0)
    s  = torch.stack([b["s"]  for b in batch], 0)
    T  = torch.stack([b["T"]  for b in batch], 0)

    raw = torch.stack([b["image_raw_u8"] for b in batch], 0)
    patient = [b["patient"] for b in batch]
    time    = torch.tensor([b["time"] for b in batch], dtype=torch.int32)
    mesh_path = [b["mesh_path"] for b in batch]

    spacing = torch.stack([b["spacing"] for b in batch], 0)
    times_arr = torch.tensor([b["times"] for b in batch], dtype=torch.int32)
    ED_arr    = torch.tensor([b["ED_time"] for b in batch], dtype=torch.int32)
    ES_arr    = torch.tensor([b["ES_time"] for b in batch], dtype=torch.int32)
    metrics   = [b["metrics"] for b in batch]

    return dict(
        x=x, mask=mask, keep=keep, angle=angle,
        GH=GH, R=R, s=s, T=T,
        image_raw_u8=raw,
        patient=patient, time=time, mesh_path=mesh_path,
        spacing=spacing, times=times_arr, ED_time=ED_arr, ES_time=ES_arr, metrics=metrics,
        S=S, C=C, H=H, W=W
    )


def build_dataloaders(data_root: str,
                      gh_label_dir: str,
                      table_csv: Optional[str] = None,
                      img_size: Tuple[int,int]=(256,256),
                      target_S: int = 37,
                      slice_degree_step: Optional[float] = None,  # New
                      batch_size: int = 2,
                      num_workers: int = 4,
                      patients: Optional[List[str]] = None,
                      seed: int = 42,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      enable_aug: bool = True):
    """
    If slice_degree_step is set (e.g., 10), uniformly select round(360/10)=36 slices per full circle;
    otherwise use target_S (e.g., 37).
    """
    ds_train = RadialNPZDataset(
        data_root=data_root, gh_label_dir=gh_label_dir, split="train",
        img_size=img_size, target_S=target_S, slice_degree_step=slice_degree_step,
        patients=patients, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
        enable_aug=enable_aug, table_csv=table_csv
    )
    ds_val = RadialNPZDataset(
        data_root=data_root, gh_label_dir=gh_label_dir, split="val",
        img_size=img_size, target_S=target_S, slice_degree_step=slice_degree_step,
        patients=patients, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
        enable_aug=False, table_csv=table_csv
    )
    ds_test = RadialNPZDataset(
        data_root=data_root, gh_label_dir=gh_label_dir, split="test",
        img_size=img_size, target_S=target_S, slice_degree_step=slice_degree_step,
        patients=patients, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
        enable_aug=False, table_csv=table_csv
    )

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          collate_fn=collate_batch, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=max(1,batch_size//2), shuffle=False,
                        num_workers=max(1,num_workers//2), pin_memory=True,
                        collate_fn=collate_batch, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=max(1,batch_size//2), shuffle=False,
                         num_workers=max(1,num_workers//2), pin_memory=True,
                         collate_fn=collate_batch, drop_last=False)
    return ds_train, ds_val, ds_test, dl_train, dl_val, dl_test