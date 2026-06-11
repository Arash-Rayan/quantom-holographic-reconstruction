# Quantom Project

Deep learning pipeline for **quantum holographic image reconstruction**. The project simulates interference-based holograms from object images (QIUP / QHUP physics models), trains **QHUPnet** to recover the original scene from noisy holograms, and includes simulators and notebooks for experimentation and evaluation.

## Overview

Quantum imaging techniques such as **Quantum Holography with Undetected Photons (QHUP)** and related **QIUP** models encode object information in an intensity hologram. Recovering the underlying image from a single noisy measurement is ill-posed.

This repository provides:

- **Physics-based hologram simulators** — from a lightweight interference model to a full EMCCD-style camera chain
- **QHUPnet** — encoder–decoder CNN with Dense blocks and ASPP for hologram → image reconstruction
- **Training workflow** on MNIST and Fashion-MNIST via `QHUPnet.ipynb`
- **Standalone training script** (`marateb.py`) with metrics, visual exports, and robustness sweeps

```
Object image  →  Hologram simulator  →  Noisy hologram [512×512]
                                              ↓
                                         QHUPnet
                                              ↓
                              Reconstructed image [256×256]
```

## Project Structure

| File | Description |
|------|-------------|
| `QHUPnet.ipynb` | Main workflow: model, `HologramDataset`, training, validation, testing |
| `marateb.py` | End-to-end script: QHUP simulator, `QHUPNet`, MNIST training, exports |
| `noise_preprocess.py` | Lightweight QIUP hologram simulator (used by the training pipeline) |
| `noise_preprocess.ipynb` | Visual exploration of hologram noise levels |
| `transform_save.py` | `QIUPImageSimulator` — photon statistics, EM gain, read noise, phase disturbance |
| `transform_Rand_output.py` | Random image transforms (hologram, blur, contrast, gamma, noise) |
| `dataset_simulator2.ipynb` | `QIUPSingleImageDataset` — detailed single-image hologram simulation |
| `marateb_preprocess.ipynb` | Single-image QIUP hologram generation and visualization |
| `test.ipynb` | Simple image rotation / alignment utility |

### Runtime directories (gitignored)

Created when you run training or simulation — not tracked in git:

- `data/` — MNIST / Fashion-MNIST raw IDX files
- `train_images/`, `processed_images/`, `augmented_images/`, `quantom_dataset/`
- `checkpoint/`, `test_results/`, `exports/`, `simulated_output/`

## Physics Models

### Simple QIUP interference model

Defined in `noise_preprocess.py` and used by `HologramDataset` in `QHUPnet.ipynb`:

```
I = r_p² + r_s² + 2 · r_p · r_s · r_i · γ · cos(φ) + noise
```

- `r_s` — object amplitude (from the input image)
- `φ` — random phase per pixel
- `γ` — visibility / coherence
- Gaussian read noise with a randomly chosen standard deviation

### Detailed QIUP camera model

`transform_save.py` (`QIUPImageSimulator`) models the full detection chain:

```
P = 0.5 · (1 + visibility · T · cos(γ + phase_disturbance)) · mean_photons
```

Followed by quantum efficiency, Poisson shot noise, clock-induced charge (CIC), electron-multiplying gain, read noise, and 16-bit ADU conversion.

### QHUP model (amplitude + phase)

`marateb.py` simulates holograms from separate amplitude and phase maps with tunable visibility, background, and read noise. It trains a dual-head `QHUPNet` that predicts both amplitude and phase.

## QHUPnet Architecture

**Quantum Hologram Undetected Photon Network** — encoder–decoder CNN defined in `QHUPnet.ipynb`:

| Component | Role |
|-----------|------|
| `Conv_pooling` | Initial stem: 1→32 channels, spatial downsample 512→256 |
| `ASPP` (×5) | Atrous spatial pyramid pooling at multiple scales |
| `CatConv_block` | Skip-connection fusion between encoder levels |
| `Dense_block` + `TransitionLayer` | Feature reuse and channel compression in the decoder |
| Output head | 1×1 conv → single-channel reconstruction at 256×256 |

**I/O shapes:**

- Input: hologram `[B, 1, 512, 512]`
- Output: reconstruction `[B, 1, 256, 256]`

Activations use **Hardswish**; TF32 is disabled for numerical stability.

## Training

### Loss function

Combined objective used in `QHUPnet.ipynb`:

```
loss = 4 · MSE + (1 − MS-SSIM) + NPCC_loss
```

`NPCC_loss` maximizes normalized Pearson correlation between prediction and ground truth.

### Default hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (`lr=1e-3`, `weight_decay=5e-4`, `amsgrad=True`) |
| Batch size | 4 |
| Epochs | 3 (notebook default; increase for better results) |
| Early stopping | Patience 20 on validation loss |

### Dataset pipeline

1. Load grayscale images from MNIST / Fashion-MNIST IDX files.
2. Resize to 512×512.
3. Simulate a hologram with `simulate_qiup_hologram` from `noise_preprocess.py`.
4. Use the original image resized to 256×256 as ground truth.
5. Feed `(gt, hologram)` pairs to the DataLoader.

Checkpoints are saved to `checkpoint/model_info.pth`; loss history to `checkpoint/loss.pkl`.

## Requirements

```
torch
torchvision
pytorch-msssim
torchmetrics
numpy
matplotlib
Pillow
scipy
tqdm
imageio          # marateb.py exports only
```

**Recommended:** Python 3.10+, CUDA-capable GPU for training.

```bash
pip install torch torchvision pytorch-msssim torchmetrics numpy matplotlib Pillow scipy tqdm imageio
```

## Quick Start

### 1. Install dependencies

```bash
git clone <repository-url>
cd quantom_project
pip install torch torchvision pytorch-msssim torchmetrics numpy matplotlib Pillow scipy tqdm imageio
```

### 2. Prepare data

MNIST and Fashion-MNIST are pulled in by `torchvision` when you run `QHUPnet.ipynb`, or place IDX files manually:

```
data/MNIST/raw/train-images-idx3-ubyte
data/MNIST/raw/t10k-images-idx3-ubyte
data/FashionMNIST/raw/...
```

### 3. Simulate a hologram

**Lightweight model:**

```python
from noise_preprocess import simulate_qiup_hologram
from torchvision import transforms
from PIL import Image

img = transforms.ToTensor()(Image.open("your_image.png").convert("L").resize((512, 512)))
hologram = simulate_qiup_hologram(img.unsqueeze(0))
```

**Full camera simulator:**

```python
from transform_save import QIUPImageSimulator

sim = QIUPImageSimulator()
results = sim.process_image("your_image.png", save_dir="simulated_output", show=True)
```

### 4. Train QHUPnet

Open `QHUPnet.ipynb` and run cells in order:

1. Model architecture
2. `HologramDataset` and data loaders
3. Training loop
4. Test evaluation and loss plots

### 5. Full automated pipeline

`marateb.py` runs simulation, trains `QHUPNet` on synthetic MNIST holograms, runs visibility/resolution sweeps, and zips artifacts to `exports/`.

```bash
python marateb.py
```

## Outputs

| Location | Contents |
|----------|----------|
| `checkpoint/model_info.pth` | Best model weights + optimizer state |
| `checkpoint/loss.pkl` | Train / val / test loss history |
| `test_results/` | Ground-truth vs prediction PNGs |
| `exports/` (`marateb.py`) | Metrics CSVs, sample galleries, montages, weights |
| `simulated_output/` | Target, hologram, transmittance `T`, and phase `γ` maps |

## Notebooks Guide

| Notebook | Purpose |
|----------|---------|
| `QHUPnet.ipynb` | Primary training and evaluation workflow |
| `noise_preprocess.ipynb` | Tune and visualize hologram noise parameters |
| `marateb_preprocess.ipynb` | Preview QIUP holograms from a single image |
| `dataset_simulator2.ipynb` | High-fidelity single-image simulation with signal/background thresholds |
| `test.ipynb` | Rotate an image by a predicted angle |

## Development Notes

- Image datasets and `.pth` checkpoints are gitignored — regenerate locally after cloning.
- Some notebook cells use hard-coded Windows paths; update `train_path` / `test_path` for your machine.
- `marateb.py` was exported from Google Colab; increase `EPOCHS` (e.g. 50–100) for stronger results.
- Repo name: **Quantom**; model names: **QHUP** / **QIUP**.

## License

No license file is included yet. Add one before public distribution.
