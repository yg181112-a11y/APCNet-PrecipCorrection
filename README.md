# APCNet: Physics-Constrained Deep Learning for Extreme Precipitation Correction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/1242451481.svg)](https://doi.org/10.5281/zenodo.20271614) This repository contains the official PyTorch implementation of **APCNet (AdvancedPrecipCorrectionNet)**, as presented in our paper submitted to *Artificial Intelligence for the Earth Systems (AIES)*.

## 📖 Overview

Numerical weather prediction (NWP) models, such as the Global Forecast System (GFS), often suffer from systematic biases and spatial displacement errors over complex terrain, particularly when predicting extreme precipitation driven by the Northeast China Cold Vortex. Traditional pure data-driven deep learning post-processing frequently encounters the "over-smoothing" trap and feature collapse.

**APCNet** resolves these issues by introducing a **Kinematic-Thermodynamic Decoupled Architecture**. 
* **Kinematic Backbone:** Preserves spatial topology using large-scale wind and precipitation fields.
* **Thermodynamic Gating (SE-Hardsigmoid):** Adaptively filters out high-frequency parameterized noise (e.g., CAPE) while dynamically amplifying high-confidence macro-scale priors (e.g., PWAT).
* **Optimization:** Integrates a Smooth Asymmetric Loss with the spatial Fractions Skill Score (FSS) to mitigate the double-penalty effect.

## 🗂️ Repository Structure

```text
.
├── data/
│   └── sample/                 # A small sample dataset (1-day GFS & ERA5) for quick testing
│       ├── GFS/
│       └── ERA5/
├── train_apcnet.py             # Main training and evaluation script (APCNet & U-Net Baseline)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

```

## ⚙️ Installation & Environment Setup

We recommend using [Anaconda](https://www.anaconda.com/) to manage your environment.

```bash
# 1. Create a new conda environment
conda create -n apcnet_env python=3.9 -y
conda activate apcnet_env

# 2. Install PyTorch (Please select the version matching your CUDA Toolkit)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. Install other required dependencies
pip install -r requirements.txt

```

## 🚀 Quick Start (Testing with Sample Data)

To facilitate reproducibility and allow reviewers to quickly verify our architecture, we provide a minimal sample dataset.

You can execute the training pipeline directly using the sample data. The script will automatically perform data matching, training, and evaluation.

```bash
python train_apcnet.py \
    --gfs_dir ./data/sample/GFS \
    --era5_dir ./data/sample/ERA5 \
    --save_dir ./output

```
*(Note: If you are using the provided hardcoded script, please ensure you update the `GFS_DIR` and `ERA5_DIR` variables inside `train_apcnet.py` to point to the correct relative paths).*

### Running the Baseline (Standard U-Net)

To reproduce the pure data-driven baseline (Standard U-Net) discussed in the ablation study of our paper, simply toggle the baseline flag in the script:

1. Open `train_apcnet.py`.
2. Locate the line `RUN_BASELINE_UNET = False`.
3. Change it to `RUN_BASELINE_UNET = True` and run the script again.

## 📊 Feature Attribution Analysis

Our code includes a perturbation-based attribution module (`analyze_feature_importance`). During the evaluation phase, the script will automatically output the relative contribution of each physical variable to the terminal, demonstrating APCNet's intelligent noise-immunization capability.

## 📝 Citation

If you find this code or our framework useful in your research, please consider citing our paper:

```bibtex
@article{tian2026apcnet,
  title={Physics-Constrained and Neighborhood-Aware Deep Learning for Extreme Precipitation Correction over Complex Terrain},
  author={Tian, Lin and Cui, Feifan and Huang, Xin and Jiang, Yuhan and Yu, Jintao and Wang, Yaoxi and Ni, Jiacheng},
  journal={Artificial Intelligence for the Earth Systems (Under Review)},
  year={2026}
}

```
## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
