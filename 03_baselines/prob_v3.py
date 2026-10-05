# -*- coding: utf-8 -*-
"""P2-② 概率评估 v3：全链新数据 + APCNet 对称 S42。
确定性 CRPS(=MAE)：GFS / QM / APCNet；概率化：QM+GaussianDress / APCNet+ZIG(const/adaptive)。
Brier/BSS occurrence(≥0.1mm) + reliability + 极端阈值概率 BSS。
"""
import numpy as np, os, json, sys
from scipy.stats import norm

OUT = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
SYM = os.path.join(OUT, '..', 'manuscript_work_sym', 'seed42')
sys.path.insert(0, OUT)
from qm_baseline import build_qm, apply_qm

def gaussian_crps(mu, sigma, y):
    sigma = np.maximum(sigma, 1e-4)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))

def zi_crps(mu, sigma, p0, y):
    """零膨胀高斯 CRPS 近似：p0*|y| + (1-p0)*高斯CRPS"""
    crps_g = gaussian_crps(mu, sigma, y)
    return p0 * np.abs(y) + (1 - p0) * crps_g

# 数据
tr_g = np.load(os.path.join(OUT, 'train_gfs.npy')).astype(np.float32)
tr_e = np.load(os.path.join(OUT, 'train_era5.npy')).astype(np.float32)
te_g = np.load(os.path.join(OUT, 'gfs_test.npy')).astype(np.float32)
te_e = np.load(os.path.join(OUT, 'targets_test.npy')).astype(np.float32)
apc = np.load(os.path.join(SYM, 'predictions_apcnet.npy')).astype(np.float32)

n = len(te_g)
split = n // 2
print('cal 前 %d / eval 后 %d' % (split, n - split))

# QM 映射 + dress
qm_gfs, qm_era5, p_dry_g, p_dry_e, qs = build_qm(tr_g, tr_e)
qm_tr = apply_qm(tr_g, qm_gfs, qm_era5, p_dry_g, p_dry_e, qs)
sigma_qm = np.maximum((tr_e - qm_tr).std(axis=0), 1e-4)
qm_te = apply_qm(te_g, qm_gfs, qm_era5, p_dry_g, p_dry_e, qs)

# APCNet ZIG 校准（前半）
apc_cal = apc[:split]; tgt_cal = te_e[:split]
dry_freq = np.mean(tgt_cal < 0.1, axis=0)
wm = tgt_cal >= 0.1
resid = np.where(wm, apc_cal - tgt_cal, np.nan)
sigma_apc = np.maximum(np.nanstd(resid, axis=0), 0.05)

apc_e = apc[split:]; tgt_e = te_e[split:]; gfs_e = te_g[split:]; qm_e = qm_te[split:]
sigma_qm_b = np.broadcast_to(sigma_qm, apc_e.shape)
sigma_apc_b = np.broadcast_to(sigma_apc, apc_e.shape)
dry_b = np.broadcast_to(dry_freq, apc_e.shape)

# adaptive p0
scale = 1.0
p0_adapt = np.clip(dry_freq[np.newaxis] * np.exp(-apc_e / scale), 0.01, 0.99)

res = {}
res['n_eval'] = int(n - split)
res['crps_det_gfs'] = float(np.mean(np.abs(gfs_e - tgt_e)))
res['crps_det_qm'] = float(np.mean(np.abs(qm_e - tgt_e)))
res['crps_det_apcnet'] = float(np.mean(np.abs(apc_e - tgt_e)))
res['crps_qm_dress'] = float(np.mean(gaussian_crps(qm_e, sigma_qm_b, tgt_e)))
res['crps_zig_const'] = float(np.mean(zi_crps(apc_e, sigma_apc_b, dry_b, tgt_e)))
res['crps_zig_adaptive'] = float(np.mean(zi_crps(apc_e, sigma_apc_b, p0_adapt, tgt_e)))

print('=== 确定性 CRPS(=MAE) ===')
for k in ['crps_det_gfs', 'crps_det_qm', 'crps_det_apcnet']:
    print('  %-18s %.4f' % (k, res[k]))
print('=== 概率化 CRPS ===')
for k in ['crps_qm_dress', 'crps_zig_const', 'crps_zig_adaptive']:
    print('  %-18s %.4f' % (k, res[k]))

# occurrence Brier/BSS (adaptive ZIG)
th = 0.1
z_t = (th - apc_e) / sigma_apc_b
p_wet = (1 - p0_adapt) * (1 - norm.cdf(z_t))
o = (tgt_e >= th).astype(float)
clim = o.mean()
brier = np.mean((p_wet - o) ** 2)
res['brier_occurrence'] = float(brier)
res['bss_occurrence'] = float(1 - brier / np.mean((clim - o) ** 2))
print('occurrence(>=0.1): Brier=%.4f BSS=%.3f clim=%.4f' % (brier, res['bss_occurrence'], clim))

# reliability
nb = 10
edges = np.linspace(0, 1, nb + 1)
rel = []
for i in range(nb):
    m = (p_wet >= edges[i]) & (p_wet < edges[i + 1])
    if m.sum() > 100:
        rel.append({'bin': i, 'avg_prob': float(p_wet[m].mean()), 'avg_obs': float(o[m].mean()), 'count': int(m.sum())})
res['reliability'] = rel
print('=== reliability ===')
for r in rel:
    print('  bin%d: prob=%.3f obs=%.3f n=%d' % (r['bin'], r['avg_prob'], r['avg_obs'], r['count']))

# 极端阈值
for tname, tval in [('>=10mm', 10.0), ('>=20mm', 20.0)]:
    z_t = (tval - apc_e) / sigma_apc_b
    p_ext = (1 - p0_adapt) * (1 - norm.cdf(z_t))
    o_ext = (tgt_e >= tval).astype(float)
    ne = int(o_ext.sum())
    if ne > 0:
        clim_e = o_ext.mean()
        be = np.mean((p_ext - o_ext) ** 2)
        bss_e = 1 - be / np.mean((clim_e - o_ext) ** 2)
        res[f'bss_{tname}'] = float(bss_e)
        res[f'n_{tname}'] = ne
        print('%s (n=%d): BSS=%.3f' % (tname, ne, bss_e))

with open(os.path.join(OUT, 'prob_v3_newdata.json'), 'w') as f:
    json.dump(res, f, indent=1, default=str)
print('\n✅ prob_v3_newdata.json 已保存')
