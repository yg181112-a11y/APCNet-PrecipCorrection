# -*- coding: utf-8 -*-
"""GPM 概率 run13：dressing 参数在 ERA5 cal 半标定，GPM IMERG 3h 全 2475 验证。"""
import numpy as np, os, json, sys, pickle
from scipy.stats import norm
from datetime import timezone

OUT = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
SYM = OUT  # run13 S42
WINDOW_END_HOURS = [3, 9, 15, 21]
utc = timezone.utc
sys.path.insert(0, OUT)
from qm_baseline import build_qm, apply_qm

def gaussian_crps(mu, sigma, y):
    sigma = np.maximum(sigma, 1e-4)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))

def zi_crps(mu, sigma, p0, y):
    return p0 * np.abs(y) + (1 - p0) * gaussian_crps(mu, sigma, y)

tr_g = np.load(os.path.join(OUT, 'train_gfs.npy')).astype(np.float32)
tr_e = np.load(os.path.join(OUT, 'train_era5.npy')).astype(np.float32)
te_g = np.load(os.path.join(OUT, 'gfs_test.npy')).astype(np.float32)
te_e = np.load(os.path.join(OUT, 'targets_test.npy')).astype(np.float32)
apc_all = np.load(os.path.join(SYM, 'predictions_apcnet.npy')).astype(np.float32)
gpm3 = np.load(os.path.join(OUT, 'gpm3h_obs.npy')).astype(np.float32)
with open(os.path.join(OUT, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)
sel = np.array([i for i, t in enumerate(times) if (t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)).hour in WINDOW_END_HOURS])
sel = sel[:len(gpm3)]
n = len(te_g); split = n // 2

# QM 映射 + dress（训练全量标定，同 prob_v3）
qm_gfs, qm_era5, p_dry_g, p_dry_e, qs = build_qm(tr_g, tr_e)
qm_tr = apply_qm(tr_g, qm_gfs, qm_era5, p_dry_g, p_dry_e, qs)
sigma_qm = np.maximum((tr_e - qm_tr).std(axis=0), 1e-4)
qm_te = apply_qm(te_g, qm_gfs, qm_era5, p_dry_g, p_dry_e, qs)

# APCNet ZIG 校准（ERA5 cal 前半，protocol 保持）
apc_cal = apc_all[:split]; tgt_cal = te_e[:split]
dry_freq = np.mean(tgt_cal < 0.1, axis=0)
wm = tgt_cal >= 0.1
resid = np.where(wm, apc_cal - tgt_cal, np.nan)
sigma_apc = np.maximum(np.nanstd(resid, axis=0), 0.05)

apc = apc_all[sel]; gfs_e = te_g[sel]; qm_e = qm_te[sel]; tgt_e = gpm3
sigma_qm_b = np.broadcast_to(sigma_qm, apc.shape)
sigma_apc_b = np.broadcast_to(sigma_apc, apc.shape)
dry_b = np.broadcast_to(dry_freq, apc.shape)
scale = 1.0
p0_adapt = np.clip(dry_freq[np.newaxis] * np.exp(-apc / scale), 0.01, 0.99)

res = {}
res['n_eval'] = int(len(apc))
res['protocol'] = 'dressing params calibrated on ERA5 reference (cal half); observations = GPM IMERG 3h'
res['crps_det_gfs'] = float(np.mean(np.abs(gfs_e - tgt_e)))
res['crps_det_qm'] = float(np.mean(np.abs(qm_e - tgt_e)))
res['crps_det_apcnet'] = float(np.mean(np.abs(apc - tgt_e)))
res['crps_qm_dress'] = float(np.mean(gaussian_crps(qm_e, sigma_qm_b, tgt_e)))
res['crps_zig_const'] = float(np.mean(zi_crps(apc, sigma_apc_b, dry_b, tgt_e)))
res['crps_zig_adaptive'] = float(np.mean(zi_crps(apc, sigma_apc_b, p0_adapt, tgt_e)))
print('=== GPM CRPS run13 ===')
for k in ['crps_det_gfs', 'crps_det_qm', 'crps_det_apcnet', 'crps_qm_dress', 'crps_zig_const', 'crps_zig_adaptive']:
    print('  %-18s %.4f' % (k, res[k]))

th = 0.1
z_t = (th - apc) / sigma_apc_b
p_wet = (1 - p0_adapt) * (1 - norm.cdf(z_t))
o = (tgt_e >= th).astype(float)
clim = o.mean()
brier = np.mean((p_wet - o) ** 2)
res['brier_occurrence_0.1mm'] = float(brier)
res['bss_occurrence_0.1mm'] = float(1 - brier / np.mean((clim - o) ** 2))
print('occurrence(>=0.1): Brier=%.4f BSS=%.3f' % (brier, res['bss_occurrence_0.1mm']))

for tname, tval in [('10mm', 10.0), ('20mm', 20.0)]:
    z_t = (tval - apc) / sigma_apc_b
    p_ext = (1 - p0_adapt) * (1 - norm.cdf(z_t))
    o_ext = (tgt_e >= tval).astype(float)
    ne = int(o_ext.sum())
    if ne > 0:
        clim_e = o_ext.mean()
        be = np.mean((p_ext - o_ext) ** 2)
        res[f'bss_{tname}'] = float(1 - be / np.mean((clim_e - o_ext) ** 2))
        res[f'n_{tname}'] = ne
        print('%s (n=%d): BSS=%.3f' % (tname, ne, res[f'bss_{tname}']))

with open(os.path.join(OUT, 'gpm_prob_run13.json'), 'w') as f:
    json.dump(res, f, indent=1, default=str)
print('gpm_prob_run13 saved')
