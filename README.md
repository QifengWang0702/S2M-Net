🚧 **Under Development**  ⚠

# 🫀 Sparse Slice-to-Mesh (S²M-Net)

> End-to-End Sparse Slice-to-Mesh Reconstruction for Fetal Cardiac Ultrasound Anatomy
> Our dataset **FeEcho4D** can be viewed at 🔗 [FeEcho4D](https://feecho4d.github.io/Website/)  
> The non end-to-end reconstruction method is available at 🔗 [GHDHeart](https://github.com/Luo-Yihao/GHDHeart)


---

## 🌐 Project Structure

```bash
├── train.py                 # Main training / validation / testing pipeline
├── net.py                   # Model: encoder, transformer, regression head
├── loss.py                  # 3D + 2D geometric loss functions
├── data.py                  # Dataset loader (RadialNPZDataset)
├── GHD.py                   # Graph Harmonic Deformation implementation
├── environment_feechofm.yml # Conda environment configuration
└── FeEcho4D-Results/        # Output directory for checkpoints and results
```

## 🧬 Dataset Structure

```bash
FeEcho4D-Dataset/
├── Patient0XX/
│   ├── image
│   └──  mask
└──gh_labels/
    ├── LV_tmp.obj
    ├── basis_cache.pt
    └── Patient0XX_time0XX.npz
```
Each .npz file contains:
	•	x → [37, 5, 256, 256] radial slice sequence
	•	GH, R, s, T → GHD coefficients and transformation parameters
	•	patient, time (optional, used for naming and export)


## ⚙️ Environment Setup

Create the environment via Conda:
```bash
conda env create -f environment_feechofm.yml
conda activate feechofm
```
Optional dependencies (for TransUNet or SAM encoders):
```bash
pip install timm
```
If PyTorch3D is missing or incompatible:
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## 🚀 Training

Edit key paths and parameters in train.py (CFG class):
```bash
data_root   = '/path/to/FeEcho4D-Dataset'
encoder_name= 'unet'   # cnn / unet / resunet / transunet / sam
out_root    = './FeEcho4D-Results/unet_2'
S           = 37       # number of radial slices
```
Run training:
```bash
python train.py
```









