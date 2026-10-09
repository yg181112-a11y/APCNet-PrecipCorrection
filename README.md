# APCNet Precipitation Correction — Loss Design, Not the Reanalysis Target, Sets the 3-h Skill Ceiling of DL Precipitation Post-Processing

**Code & data pipeline for the manuscript:**

> *Loss design, not the reanalysis target, sets the 3-h skill ceiling of deep-learning precipitation post-processing: scale-dependent evidence across ERA5, CHM, and GPM verification over Northeast China*
> (Revision 3, resubmitted to *Weather and Forecasting*, AMS)

---

## Overview

This repository contains the full experimental pipeline for a **controlled, evidence-based boundary study** of deep-learning (DL) precipitation post-processing trained on reanalysis targets, using 11 years (2015–2025) of operational GFS f003 forecasts over the Liaohe basin (Northeast China). The three central results are:

1. **The failure is the loss design, not the target noise per se.** After correcting a target-data error (ERA5 hourly `tp` had been paired with GFS 3-h accumulations as if they shared the same accumulation window), the same real-data protocol with an intensity-weighted composite loss gives three-seed mean 3-h skill of −48.0% against the training reference, whereas replacing only the loss with plain MSE yields **−2.5%** (seed 42 turns positive, +8.1%, CI [+1.8, +12.3]). A decomposition over the two interventions (selection criterion ≈ 7%; loss ≈ 95% of the degradation) is reported in the manuscript (Tables S3–S4).
2. **On independent verification** (CHM gauge-merged daily rates; GPM IMERG satellite 3-h accumulations), the DL networks are not systematically worse than simple baselines once the loss design is controlled; simple quantile mapping (QM) and grid-point linear regression (OLS) remain strong competitors.
3. **The DL-vs-baseline ranking is scale-dependent**: at 24–120 h accumulations the same networks gain observationally confirmed skill, and a 24-h plain-MSE recheck confirms the 24-h skill is **not** an artifact of the composite loss (+26.9% vs +20.4% under the composite loss, one seed).

The manuscript re-frames the original "better network" story as a reproducible boundary study: **an intensity-weighted composite loss, widely used in precipitation DL, can manufacture a failure that looks like a property of the noisy reanalysis target; the value of reanalysis-trained DL post-processing is scale-dependent.**

---

## Repository layout

```
01_data_reconstruction/   ERA5 hourly re-download, true 3-h reconstruction, six-gate acceptance
02_training_inference/    APCNet / U-Net training & inference (all targets, all leads)
03_baselines/             QM, OLS, bin-conditional (BinCM), probabilistic baselines
04_evaluation/            Block bootstrap, probabilistic, diurnal, terrain-stratified, event-level diagnostics
05_figures/               Publication-quality figure scripts (Times New Roman, 300 dpi)
06_control_experiments/   Controlled experiments isolating loss / selection / pooling / gating effects (incl. M1, M5)
```

### 01 — Data reconstruction (the key fix)

- `download_era5_tp_hourly.py` / `era5_download_yearly.py` — CDS download of ERA5 **hourly** `tp` (24 steps/day) for 2015–2025.
- `aggregate_era5_tp.py` — reconstructs true 3-h accumulations as `TP_3h(t) = tp(t−2h) + tp(t−1h) + tp(t)` for t ∈ {03, 09, 15, 21}Z, replacing the old monthly (3-hourly-subsampled) files.
- `verify_era5_target.py` — six-gate acceptance (metadata; grid; GFS/ERA5 domain-mean ratio 0.9–1.1; annual precipitation 600–900 mm; ≥20 mm grid-point parity; time-series correlation r > 0.8). **Training is only permitted after all six gates pass.**

### 02 — Training & inference

- `13.0_main.py` — main APCNet/U-Net training & evaluation script (seeds 42/40/41; input: 8 channels + 6 prior times at 6-h intervals; residual learning; asymmetric intensity-weighted loss and symmetric variant).
- Multi-lead training (`train_24h.py`, `train_24h_seed.py`, `train_unet_lead.py`, …) and GPM-target training (`gpm_train_dataset.py`, `train_gpm3h_apcnet.py`, `train_gpm_unet.py`, `eval_gpm_trained.py`).

### 03 — Baselines

- `qm_baseline.py` — quantile mapping (climatological CDF, training-period calibration, test-period blind application).
- `quick_ols_baseline.py` / `build_gpm3h_ols.py` — grid-point linear regression (OLS).
- `bin_conditional_baseline.py` / `bincm_split_sens.py` — bin-conditional mean correction.
- `prob_*.py` / `probabilistic_apcnet*.py` — probabilistic (ZIG) variants with CRPS/Brier/reliability diagnostics.

### 04 — Evaluation

- Block-bootstrap significance: `run13_bootstrap.py`, `era5_ref_bootstrap.py`, `ets_bootstrap_era5.py`, `ets_bootstrap_gpm.py`, `chm_bootstrap.py`, `chm_bootstrap_blocklen.py`, `gpm3h_bootstrap.py`, `lead_bootstrap.py`.
- Independent verification: `verify_gpm_independent.py`, `verify_gpm_lead*.py`, `chm_*eval*.py`, `eval_gpm3h_authoritative.py`.
- Event-level / object-based / synoptic diagnostics: `event_verify_gpm3h.py`, `object_based_verification.py`, `storm_synoptic*.py`, `extreme_events.py`.
- Diurnal (`season_3h_24h.py`, `chm_seasonal_run13.py`), terrain-stratified (`terra_phys_eval.py`), probabilistic (`gpm_prob_eval.py`), scale attribution (`scale_attribution.py`).

### 05 — Figures

`make_figs_pub.py` orchestrates all journal figures (Times New Roman, panel labels, 300 dpi PNG + PDF).

### 06 — Controlled experiments

- `controlled_experiment.py` / `controlled_exp_grid.py` — early ablation of architectural/loss components (kinematics pooling, gating, hard-threshold loss) with matched seeds.
- **M1 (manuscript Table S3)** — real-data reselection: `train_mse_select.py` / `eval_mse_select.py` rerun the main protocol with composite loss but checkpoint selection by validation-residual **MSE** (three seeds).
- **M1 (manuscript Table S4)** — real-data loss swap: `run_mseonly_all.py` / `train_mseonly_select.py` / `eval_mseonly_select.py` rerun the identical protocol with **plain MSE** loss and MSE selection (three seeds). This is the decisive control behind the −48.0% → −2.5% result.
- **M5 (24-h recheck)** — `train_24h_mseonly.py` reruns the 24-h experiment with plain MSE (one seed) to confirm the 24-h skill gain is not an artifact of the composite loss (+26.9% vs +20.4%).

Note: the `train_mse*_select.py` scripts import `main13`, which is the same module as `13.0_main.py` (copy `13.0_main.py` to `main13.py` in the same directory if needed, or adapt the import). Data paths inside the scripts point to the local layout used for the study; raw GFS/ERA5/CHM/GPM data are too large to host here and are reproduced by the scripts in `01_data_reconstruction`.

---

## Data sources

| Data | Source | Period | Role |
|---|---|---|---|
| GFS f003 (0.25°) | NOAA NCEI / NOMADS (`gfs.0p25.*`) | 2015–2025 | Input |
| ERA5 hourly `tp` | ECMWF CDS | 2015–2025 | Training target (reconstructed 3-h) |
| CHM daily (0.1°) | CMA multisource merging (Shen et al. 2014) | 2024–2025 | Independent verification |
| GPM IMERG Final (0.1°, 30-min) | NASA GES DISC | 2024-01-15 – 2025-09-30 | Independent verification |

Raw data are too large to host here; scripts in `01_data_reconstruction` reproduce the exact download and processing. Key processed artifacts and evaluation results are archived on Zenodo (see manuscript Data Availability).

---

## Reproducibility notes

- Data are split **strictly by year**: train 2015–2021, validation 2022–2023, test 2024–2025.
- Normalization statistics are estimated on the training period only; validation/test are never oversampled.
- GFS precipitation input is clipped to [0,100] mm, capped at the 99.9th percentile, and lightly smoothed (σ=0.6) when the field maximum < 10 mm. **The ERA5 target is not cleaned.**
- All significance tests use monthly **block bootstrap** (autocorrelation-aware); the 3-h test series are not treated as i.i.d.
- The composite-loss checkpoint selection criterion used in the main experiment is disclosed verbatim in the manuscript (Section 2a; the Data partitioning paragraph); the M1 controls switch selection to validation-residual MSE, isolating the two interventions.

---

## License

Code: MIT. Data: see original provider terms (ECMWF CDS, NOAA, CMA, NASA GES DISC).

Zenodo archive: see manuscript Data Availability statement.
