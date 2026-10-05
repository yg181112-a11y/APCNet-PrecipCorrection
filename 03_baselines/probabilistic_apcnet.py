# -*- coding: utf-8 -*-
"""
APCNet probabilistic output via Gaussian dressing of residuals.
Computes CRPS and compares with deterministic MAE and QM probabilistic baseline.
Uses temporal split: first half of test set for sigma estimation, second half for evaluation.
"""
import numpy as np
from scipy import stats
import pickle, os, json

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'

pred = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
target = np.load(os.path.join(WORK, 'targets_test.npy'))
gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)

n = len(times)
split = n // 2
print(f"Total samples: {n}, sigma estimation: first {split}, evaluation: last {n-split}")

# === Sigma estimation from first half ===
resid_cal = pred[:split] - target[:split]
# Global sigma
sigma_global = float(np.std(resid_cal))
# Per-gridpoint sigma (spatially varying)
sigma_grid = np.std(resid_cal, axis=0)  # (25, 37)
# Per-gridpoint sigma with floor to avoid zero
sigma_grid = np.maximum(sigma_grid, 0.01)

print(f"Global residual sigma: {sigma_global:.4f}")
print(f"Per-grid sigma: mean={sigma_grid.mean():.4f}, min={sigma_grid.min():.4f}, max={sigma_grid.max():.4f}")

# === Evaluation on second half ===
pred_eval = pred[split:]
target_eval = target[split:]
gfs_eval = gfs[split:]

def gaussian_crps(mu, sigma, y):
    """CRPS for Gaussian distribution N(mu, sigma^2) vs observation y.
    Analytical formula: sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
    """
    z = (y - mu) / sigma
    Phi = stats.norm.cdf(z)
    phi = stats.norm.pdf(z)
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
    return crps

# Global sigma CRPS
crps_global = gaussian_crps(pred_eval, sigma_global, target_eval)
mean_crps_global = float(np.mean(crps_global))

# Per-grid sigma CRPS
sigma_grid_broad = np.broadcast_to(sigma_grid, pred_eval.shape)
crps_grid = gaussian_crps(pred_eval, sigma_grid_broad, target_eval)
mean_crps_grid = float(np.mean(crps_grid))

# Deterministic MAE (APCNet)
mae_apc = float(np.mean(np.abs(pred_eval - target_eval)))
# GFS MAE
mae_gfs = float(np.mean(np.abs(gfs_eval - target_eval)))

print(f"\n=== Probabilistic Evaluation (second half, n={n-split}) ===")
print(f"APCNet deterministic MAE: {mae_apc:.4f}")
print(f"GFS deterministic MAE: {mae_gfs:.4f}")
print(f"APCNet+GaussianDress (global sigma={sigma_global:.4f}) CRPS: {mean_crps_global:.4f}")
print(f"APCNet+GaussianDress (per-grid sigma) CRPS: {mean_crps_grid:.4f}")
print(f"QM+GaussianDress CRPS (from prior run): 0.1540")

# === Precipitation occurrence probabilistic metrics ===
# For threshold >= 0.1mm, compute probability of exceedance from Gaussian
threshold = 0.1
# P(precip >= threshold) = 1 - Phi((threshold - mu)/sigma)
z_thresh = (threshold - pred_eval) / sigma_global
prob_exceed = 1.0 - stats.norm.cdf(z_thresh)
obs_exceed = (target_eval >= threshold).astype(float)

# Brier score
brier = float(np.mean((prob_exceed - obs_exceed)**2))
# Brier skill score vs climatology
clim_freq = float(obs_exceed.mean())
brier_clim = float(np.mean((clim_freq - obs_exceed)**2))
bss = 1.0 - brier / brier_clim if brier_clim > 0 else 0.0

print(f"\n=== Precipitation Occurrence (>=0.1mm) ===")
print(f"Climatological frequency: {clim_freq:.4f}")
print(f"Brier score: {brier:.4f}")
print(f"Brier skill score (vs climatology): {bss:.4f}")

# === Reliability (bin the probabilities) ===
nbins = 10
bin_edges = np.linspace(0, 1, nbins + 1)
reliability = []
for i in range(nbins):
    mask = (prob_exceed >= bin_edges[i]) & (prob_exceed < bin_edges[i+1])
    if mask.sum() > 0:
        avg_prob = float(prob_exceed[mask].mean())
        avg_obs = float(obs_exceed[mask].mean())
        reliability.append({'bin': i, 'avg_prob': avg_prob, 'avg_obs': avg_obs, 'count': int(mask.sum())})

print(f"\n=== Reliability Diagram Data ===")
for r in reliability:
    print(f"  Bin {r['bin']}: prob={r['avg_prob']:.3f}, obs_freq={r['avg_obs']:.3f}, n={r['count']}")

# === Extreme precipitation probabilistic metrics ===
for thresh_name, thresh in [('>=10mm', 10.0), ('>=20mm', 20.0)]:
    z_t = (thresh - pred_eval) / sigma_global
    p_exceed = 1.0 - stats.norm.cdf(z_t)
    o_exceed = (target_eval >= thresh).astype(float)
    n_obs = int(o_exceed.sum())
    if n_obs > 0:
        # CRPS for extreme (using threshold-exceedance)
        # Simplified: use continuous CRPS already computed
        # Brier for extreme
        brier_ext = float(np.mean((p_exceed - o_exceed)**2))
        clim_ext = float(o_exceed.mean())
        brier_clim_ext = float(np.mean((clim_ext - o_exceed)**2))
        bss_ext = 1.0 - brier_ext / brier_clim_ext if brier_clim_ext > 0 else 0.0
        # Reliability (only bins with samples)
        print(f"\n=== {thresh_name} (n_events={n_obs}) ===")
        print(f"  Mean predicted prob: {p_exceed.mean():.4f}")
        print(f"  Observed frequency: {clim_ext:.4f}")
        print(f"  Brier score: {brier_ext:.6f}")
        print(f"  BSS vs climatology: {bss_ext:.4f}")

# Save results
results = {
    'sigma_global': sigma_global,
    'sigma_grid_mean': float(sigma_grid.mean()),
    'evaluation_n': n - split,
    'mae_apcnet': mae_apc,
    'mae_gfs': mae_gfs,
    'crps_global_sigma': mean_crps_global,
    'crps_pergrid_sigma': mean_crps_grid,
    'crps_qm_baseline': 0.154,
    'brier_occurrence': brier,
    'bss_occurrence': bss,
    'clim_freq_occurrence': clim_freq,
    'reliability': reliability,
}
with open(os.path.join(WORK, 'probabilistic_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to probabilistic_results.json")
