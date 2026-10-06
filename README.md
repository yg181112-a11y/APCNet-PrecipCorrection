# APCNet Precipitation Correction — Reanalysis-Trained DL Post-Processing over Northeast China
**Code & data pipeline for the manuscript:**
> *Skill limits and transferability of deep-learning precipitation post-processing trained on reanalysis targets: scale-dependent skill across ERA5, CHM, and GPM verification over Northeast China*
> (Revision 3, submitted to *Weather and Forecasting*, AMS)
---
## Overview
This repository contains the full experimental pipeline for a **counter-example / skill-limit study** of deep-learning (DL) precipitation post-processing trained on reanalysis targets. Using 11 years (2015–2025) of operational GFS f003 forecasts over the Liaohe basin (Northeast China), we show that:
- After correcting a **target-data error** (ERA5 hourly `tp` was paired with GFS 3-h accumulations as if they shared the same accumulation window), a correction network that learns a near-perfect, noise-free mapping **degrades** forecasts whenever the reanalysis target is noisy;
- On two independent verification products (CHM gauge-merged daily rates; GPM IMERG satellite 3-h accumulations), the DL networks (APCNet and a standard U-Net) are the **only methods that make forecasts worse on every reference**, while simple quantile mapping (QM) and grid-point linear regression (OLS) **improve** them on every reference;
- The ranking between DL and simple baselines is **scale-dependent**: at 24–120 h accumulations the same networks gain observationally confirmed skill while the simple baselines collapse.
The manuscript re-frames the original "better network" story as an evidence-based boundary study: **reanalysis-trained DL post-processing inherits the smoothness and noise of its training reference; its operational value is scale-dependent and, at fine 3-h scales, negative.**
---
## Repository layout
```
01_data_reconstruction/   ERA5 hourly re-download, true 3-h reconstruction, six-gate acceptance
02_training_inference/    APCNet / U-Net training & inference (all targets, all leads)
03_baselines/             QM, OLS, bin-conditional (BinCM), probabilistic baselines
04_evaluation/            Block bootstrap, probabilistic, diurnal, terrain-stratified, event-level diagnostics
05_figures/               Publication-quality figure scripts (Times New Roman, 300 dpi)
06_control_experiments/   Controlled experiments isolating loss / pooling / gating effects
```
### 01 — Data reconstruction (the key fix)
- `download_era5_tp_hourly.py` / `era5_download_yearly.py` — CDS download of ERA5 **hourly** `tp` (24 steps/day) for 2015–2025.
- `aggregate_era5_tp.py` — reconstructs true 3-h accumulations as `TP_3h(t) = tp(t−2h) + tp(t−1h) + tp(t)` for t ∈ {03, 09, 15, 21}Z, replacing the old monthly (3-hourly-subsampled) files.
- `verify_era5_target.py` — six-gate acceptance (metadata; grid; GFS/ERA5 domain-mean ratio 0.9–1.1; annual precipitation 600–900 mm; ≥20 mm grid-point parity; time-series correlation r > 0.8). **Training is only permitted after all six gates pass.**
### 02 — Training & inference
- `13.0_main.py` — main APCNet/U-Net training & evaluation script (seeds 42/40/41; input: 8 channels + 6 prior times at 6-h intervals; residual learning; asymmetric intensity-weighted loss and symmetric variant; model selection on the validation composite score S = 3.0·ETS20 + 1.8·POD20 − 0.45·FAR20 + 1.4·ETS15 + 0.9·POD15 − 0.20·FAR15 − 0.003·L_val − 2.0·max(0, 0.20 − POD20), with early stopping when S has not improved for eight epochs or POD20 for sixteen).
- Multi-lead training (`train_24h.py`, `train_24h_seed.py`, `train_unet_lead.py`, …) and GPM-target training (`gpm_train_dataset.py`, `train_gpm3h_apcnet.py`, `train_gpm_unet.py`, `eval_gpm_trained.py`).
### 03 — Baselines
- `qm_baseline.py` — quantile mapping (climatological CDF, training-period calibration, test-period blind application).
- `quick_ols_baseline.py` / `build_gpm3h_ols.py` — grid-point linear regression (OLS).
- `bin_conditional_baseline.py` / `bincm_split_sens.py` — bin-conditional mean correction.
- `prob_*.py` / `probabilistic_apcnet*.py` — probabilistic (ZIG) variants with CRPS/Brier/reliability diagnostics.
### 04 — Evaluation
- Block-bootstrap significance: `run13_bootstrap.py`, `era5_ref_bootstrap.py`, `ets_bootstrap_era5.py`, `ets_bootstrap_gpm.py`, `chm_bootstrap.py`, `chm_bootstrap_blocklen.py`, `gpm3h_bootstrap.py`, `lead_bootstrap.py`, `obs_lead_bootstrap.py`.
- Independent verification: `verify_gpm_independent.py`, `verify_gpm_lead*.py`, `chm_*eval*.py`, `eval_gpm3h_authoritative.py`.
- Event-level / object-based / synoptic diagnostics: `event_verify_gpm3h.py`, `object_based_verification.py`, `storm_synoptic*.py`, `extreme_events.py`.
- Diurnal (`season_3h_24h.py`, `chm_seasonal_run13.py`), terrain-stratified (`terra_phys_eval.py`), probabilistic (`gpm_prob_eval.py`), scale attribution (`scale_attribution.py`).
### 05 — Figures
`make_figs_pub.py` and the `draw_*_run13.py` scripts orchestrate all journal figures (Times New Roman, panel labels, 300 dpi PNG + PDF). The final publication versions of all 14 figures (PNG + PDF, 300 dpi) are in `figures_300dpi/`.
### 06 — Controlled experiments
- `controlled_experiment.py` / `controlled_exp_grid.py` isolate the effect of each architectural/loss component (kinematics pooling, gating, hard-threshold loss) with matched seeds.
- `controlled_experiment_mseonly.py` + `run_mseonly_all.py` — **pure-MSE attribution experiment** (identity / σ = 0.5 / 1.16 / 2.0 mm·3h⁻¹ × seeds 42/40/41). Under target noise, the full asymmetric multi-term loss degrades skill by −28.5% at the observed residual level, whereas the same architecture with a pure MSE loss degrades by only ≈ −1% — isolating the **loss design** as the dominant driver of the reported negative skill, not the target noise alone.
- `train_24h_mseonly.py` + `mseonly_24h_results.json` — **24-h pure-MSE single-seed recheck**. With the composite loss replaced by plain MSE (everything else unchanged), the 24-h APCNet correction reaches **+26.9% MSE improvement** (12.80 vs raw GFS 17.52, CC 0.828) against +20.4% under the composite loss, confirming that the positive 24–120-h skill is **not** an artifact of the composite loss.
---
## Data sources
| Data | Source | Period | Role |
|---|---|---|---|
| GFS f003 (0.25°) | NOAA NCEI / NOMADS (`gfs.0p25.*`) | 2015–2025 | Input |
| ERA5 hourly `tp` | ECMWF CDS | 2015–2025 | Training target (reconstructed 3-h) |
| CHM daily (0.1°) | CMA multisource merging (Shen et al. 2014) | 2024–2025 | Independent verification |
| GPM IMERG Final (0.1°, 30-min) | NASA GES DISC | 2024-01-01 – 2025-09-30 | Independent verification |
Raw data are too large to host here; scripts in `01_data_reconstruction` reproduce the exact download and processing. Key processed artifacts and evaluation results are archived on Zenodo (see manuscript Data Availability).
---
## Reproducibility notes
- Data are split **strictly by year**: train 2015–2021, validation 2022–2023, test 2024–2025.
- Normalization statistics are estimated on the training period only; validation/test are never oversampled.
- GFS precipitation input is clipped to [0,100] mm, capped at the 99.9th percentile, and lightly smoothed (σ=0.6) when the field maximum < 10 mm. **The ERA5 target is not cleaned.**
- All significance tests use monthly **block bootstrap** (autocorrelation-aware); the 3-h test series are not treated as i.i.d.
- Paths in the scripts are local (Windows) and must be adapted to the user's data layout; all data files are regenerable from the scripts in `01_data_reconstruction`.
---
## License
Code: MIT. Data: see original provider terms (ECMWF CDS, NOAA, CMA, NASA GES DISC).
Zenodo archive: see manuscript Data Availability statement.
