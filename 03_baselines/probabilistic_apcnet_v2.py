# -*- coding: utf-8 -*-
"""
APCNet probabilistic output v2: Zero-inflated Gaussian dressing.
- Dry probability estimated per-gridpoint from calibration period
- Wet condition: truncated Gaussian N(mu, sigma^2) for mu > 0
- CRPS computed analytically for zero-inflated Gaussian
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
print(f"Total: {n}, calibration: first {split}, eval: last {n-split}")

# === Calibration: per-gridpoint dry frequency and wet sigma ===
pred_cal = pred[:split]
target_cal = target[:split]

# Dry frequency per gridpoint (target < 0.1mm)
dry_freq = np.mean(target_cal < 0.1, axis=0)  # (25, 37)
# Wet residual sigma per gridpoint (only where target >= 0.1)
wet_mask_cal = target_cal >= 0.1
wet_resid = np.where(wet_mask_cal, pred_cal - target_cal, np.nan)
sigma_wet = np.nanstd(wet_resid, axis=0)
sigma_wet = np.maximum(sigma_wet, 0.05)  # floor

print(f"Dry freq: mean={dry_freq.mean():.4f}, range=[{dry_freq.min():.4f}, {dry_freq.max():.4f}]")
print(f"Wet sigma: mean={sigma_wet.mean():.4f}, range=[{sigma_wet.min():.4f}, {sigma_wet.max():.4f}]")

# === Evaluation ===
pred_eval = pred[split:]
target_eval = target[split:]
gfs_eval = gfs[split:]
n_eval = len(pred_eval)

def zero_inflated_gaussian_crps(mu, sigma, p0, y):
    """CRPS for zero-inflated Gaussian:
    P(Y=0) = p0, P(Y>0) = (1-p0) * TruncatedNormal(mu, sigma^2, lower=0)
    Simplified: use mixture of point mass at 0 and Gaussian.
    CRPS = p0 * |y| + (1-p0) * CRPS(Gaussian(mu,sigma), y)  [approx]
    More accurate: integrate over the mixture.
    """
    # Gaussian CRPS
    z = (y - mu) / sigma
    Phi = stats.norm.cdf(z)
    phi = stats.norm.pdf(z)
    crps_gauss = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
    
    # Point mass at 0 CRPS = |y|
    crps_point = np.abs(y)
    
    # Mixture CRPS (linear combination is approximation; exact requires cross-term)
    # For simplicity and conservativeness, use weighted average
    crps = p0 * crps_point + (1 - p0) * crps_gauss
    return crps

# Broadcast sigma and p0 to eval shape
sigma_broad = np.broadcast_to(sigma_wet, pred_eval.shape)
p0_broad = np.broadcast_to(dry_freq, pred_eval.shape)

# For dry predictions (mu ~ 0), increase p0; for wet predictions, decrease p0
# Adaptive: p0 = dry_freq * exp(-mu/scale) to make wet predictions less likely dry
scale = 1.0  # mm
p0_adaptive = dry_freq[np.newaxis, :, :] * np.exp(-pred_eval / scale)
p0_adaptive = np.clip(p0_adaptive, 0.01, 0.99)

crps_zig = zero_inflated_gaussian_crps(pred_eval, sigma_broad, p0_adaptive, target_eval)
mean_crps_zig = float(np.mean(crps_zig))

# Also compute with constant p0
crps_const = zero_inflated_gaussian_crps(pred_eval, sigma_broad, p0_broad, target_eval)
mean_crps_const = float(np.mean(crps_const))

# Deterministic metrics
mae_apc = float(np.mean(np.abs(pred_eval - target_eval)))
mae_gfs = float(np.mean(np.abs(gfs_eval - target_eval)))

print(f"\n=== Probabilistic Evaluation (n={n_eval}) ===")
print(f"APCNet deterministic MAE: {mae_apc:.4f}")
print(f"GFS deterministic MAE: {mae_gfs:.4f}")
print(f"APCNet+ZIGauss (const p0) CRPS: {mean_crps_const:.4f}")
print(f"APCNet+ZIGauss (adaptive p0) CRPS: {mean_crps_zig:.4f}")
print(f"QM+GaussianDress CRPS: 0.1540")
print(f"QM deterministic MAE: 0.0927")

# === Precipitation occurrence probability ===
# P(precip >= 0.1) = (1-p0) * P(N(mu,sigma) >= 0.1)
threshold = 0.1
z_thresh = (threshold - pred_eval) / sigma_broad
p_wet_exceed = (1 - p0_adaptive) * (1 - stats.norm.cdf(z_thresh))
obs_exceed = (target_eval >= threshold).astype(float)

brier = float(np.mean((p_wet_exceed - obs_exceed)**2))
clim_freq = float(obs_exceed.mean())
brier_clim = float(np.mean((clim_freq - obs_exceed)**2))
bss = 1.0 - brier / brier_clim if brier_clim > 0 else 0.0

print(f"\n=== Occurrence (>=0.1mm) ===")
print(f"Clim freq: {clim_freq:.4f}, Brier: {brier:.4f}, BSS: {bss:.4f}")

# Reliability
nbins = 10
bin_edges = np.linspace(0, 1, nbins + 1)
reliability = []
for i in range(nbins):
    mask = (p_wet_exceed >= bin_edges[i]) & (p_wet_exceed < bin_edges[i+1])
    if mask.sum() > 100:
        reliability.append({
            'bin': i, 'avg_prob': float(p_wet_exceed[mask].mean()),
            'avg_obs': float(obs_exceed[mask].mean()), 'count': int(mask.sum())
        })

print(f"\n=== Reliability ===")
for r in reliability:
    print(f"  Bin {r['bin']}: prob={r['avg_prob']:.3f}, obs={r['avg_obs']:.3f}, n={r['count']}")

# === Extreme thresholds ===
for tname, tval in [('>=10mm', 10.0), ('>=20mm', 20.0)]:
    z_t = (tval - pred_eval) / sigma_broad
    p_ext = (1 - p0_adaptive) * (1 - stats.norm.cdf(z_t))
    o_ext = (target_eval >= tval).astype(float)
    n_evt = int(o_ext.sum())
    if n_evt > 0:
        brier_e = float(np.mean((p_ext - o_ext)**2))
        clim_e = float(o_ext.mean())
        brier_clim_e = float(np.mean((clim_e - o_ext)**2))
        bss_e = 1 - brier_e / brier_clim_e if brier_clim_e > 0 else 0
        print(f"\n{tname} (n={n_evt}): mean_prob={p_ext.mean():.5f}, obs_freq={clim_e:.5f}, Brier={brier_e:.6f}, BSS={bss_e:.2f}")

# Save
results = {
    'dry_freq_mean': float(dry_freq.mean()),
    'sigma_wet_mean': float(sigma_wet.mean()),
    'crps_zig_const': mean_crps_const,
    'crps_zig_adaptive': mean_crps_zig,
    'crps_qm_baseline': 0.154,
    'mae_apcnet': mae_apc,
    'mae_gfs': mae_gfs,
    'mae_qm': 0.0927,
    'brier_occurrence': brier,
    'bss_occurrence': bss,
    'reliability': reliability,
}
with open(os.path.join(WORK, 'probabilistic_results_v2.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to probabilistic_results_v2.json")
