# -*- coding: utf-8 -*-
"""P3-Q3: 概率节增强——reliability 曲线 + sharpness + 图（fig7_reliability.png）。
复用 prob_v3 的校准/评估逻辑（确定性 CRPS、QM dress、APCNet ZIG const/adaptive、occurrence Brier/BSS、极端 BSS），
新增：reliability 曲线数据与图、sharpness（预报概率直方图 + 平均 spread）、可靠性分档观测频率。
"""
import numpy as np, os, json, sys
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
FIG = r'D:\liaohe\校正优化过程\第三阶段\12优化\fig_p3'
SYM = os.path.join(OUT, '..', 'manuscript_work_sym', 'seed42')
sys.path.insert(0, OUT)
from qm_baseline import build_qm, apply_qm

def gaussian_crps(mu, sigma, y):
    sigma = np.maximum(sigma, 1e-4)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))

def zi_crps(mu, sigma, p0, y):
    crps_g = gaussian_crps(mu, sigma, y)
    return p0 * np.abs(y) + (1 - p0) * crps_g

tr_g = np.load(os.path.join(OUT, 'train_gfs.npy')).astype(np.float32)
tr_e = np.load(os.path.join(OUT, 'train_era5.npy')).astype(np.float32)
te_g = np.load(os.path.join(OUT, 'gfs_test.npy')).astype(np.float32)
te_e = np.load(os.path.join(OUT, 'targets_test.npy')).astype(np.float32)
apc = np.load(os.path.join(SYM, 'predictions_apcnet.npy')).astype(np.float32)

n = len(te_g)
split = n // 2
print('cal 前 %d / eval 后 %d' % (split, n - split))

qm_gfs, qm_era5, p_dry_g, p_dry_e, qs = build_qm(tr_g, tr_e)
qm_tr = apply_qm(tr_g, qm_gfs, qm_era5, p_dry_g, p_dry_e, qs)
sigma_qm = np.maximum((tr_e - qm_tr).std(axis=0), 1e-4)
qm_te = apply_qm(te_g, qm_gfs, qm_era5, p_dry_g, p_dry_e, qs)

apc_cal = apc[:split]; tgt_cal = te_e[:split]
dry_freq = np.mean(tgt_cal < 0.1, axis=0)
wm = tgt_cal >= 0.1
resid = np.where(wm, apc_cal - tgt_cal, np.nan)
sigma_apc = np.maximum(np.nanstd(resid, axis=0), 0.05)

apc_e = apc[split:]; tgt_e = te_e[split:]; gfs_e = te_g[split:]; qm_e = qm_te[split:]
sigma_qm_b = np.broadcast_to(sigma_qm, apc_e.shape)
sigma_apc_b = np.broadcast_to(sigma_apc, apc_e.shape)
dry_b = np.broadcast_to(dry_freq, apc_e.shape)
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

th = 0.1
z_t = (th - apc_e) / sigma_apc_b
p_wet = (1 - p0_adapt) * (1 - norm.cdf(z_t))
o = (tgt_e >= th).astype(float)
clim = o.mean()
brier = np.mean((p_wet - o) ** 2)
res['brier_occurrence'] = float(brier)
res['bss_occurrence'] = float(1 - brier / np.mean((clim - o) ** 2))
print('occurrence(>=0.1): Brier=%.4f BSS=%.3f clim=%.4f' % (brier, res['bss_occurrence'], clim))

# reliability（10 档，含空档跳过）
nb = 10
edges = np.linspace(0, 1, nb + 1)
rel = []
for i in range(nb):
    m = (p_wet >= edges[i]) & (p_wet < edges[i + 1])
    if m.sum() > 100:
        rel.append({'bin_low': float(edges[i]), 'bin_high': float(edges[i + 1]),
                    'avg_prob': float(p_wet[m].mean()), 'avg_obs': float(o[m].mean()),
                    'count': int(m.sum())})
res['reliability'] = rel
print('=== reliability ===')
for r in rel:
    print('  [%.2f,%.2f) prob=%.3f obs=%.3f n=%d' % (r['bin_low'], r['bin_high'], r['avg_prob'], r['avg_obs'], r['count']))

# sharpness
res['sharpness_p_wet_std'] = float(np.std(p_wet))
res['sharpness_p_wet_q10'] = float(np.percentile(p_wet, 10))
res['sharpness_p_wet_q90'] = float(np.percentile(p_wet, 90))
res['mean_spread_apc_sigma'] = float(np.mean(sigma_apc))
res['mean_spread_qm_sigma'] = float(np.mean(sigma_qm))
# 多分档 reliability（21 档细粒度，用于图）
nb2 = 21
edges2 = np.linspace(0, 1, nb2 + 1)
rel2 = []
for i in range(nb2):
    m = (p_wet >= edges2[i]) & (p_wet < edges2[i + 1])
    if m.sum() > 30:
        rel2.append({'avg_prob': float(p_wet[m].mean()), 'avg_obs': float(o[m].mean()), 'count': int(m.sum())})
res['reliability_fine'] = rel2
print('sharpness std=%.4f q10=%.3f q90=%.3f spread_apc=%.4f spread_qm=%.4f' % (
    res['sharpness_p_wet_std'], res['sharpness_p_wet_q10'], res['sharpness_p_wet_q90'],
    res['mean_spread_apc_sigma'], res['mean_spread_qm_sigma']))

# 极端阈值 BSS
for tname, tval in [('>=10mm', 10.0), ('>=20mm', 20.0)]:
    z_t = (tval - apc_e) / sigma_apc_b
    p_ext = (1 - p0_adapt) * (1 - norm.cdf(z_t))
    o_ext = (tgt_e >= tval).astype(float)
    ne = int(o_ext.sum())
    if ne > 0:
        clim_e = o_ext.mean()
        be = np.mean((p_ext - o_ext) ** 2)
        res[f'bss_{tname}'] = float(1 - be / np.mean((clim_e - o_ext) ** 2))
        res[f'n_{tname}'] = ne
        print('%s (n=%d): BSS=%.3f' % (tname, ne, res[f'bss_{tname}']))

with open(os.path.join(OUT, 'prob_v3_ext.json'), 'w') as f:
    json.dump(res, f, indent=1, default=str)
print('prob_v3_ext.json saved')

# ===== 图：reliability + sharpness =====
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=300)
# (a) reliability
ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfect')
xf = [r['avg_prob'] for r in rel]
yf = [r['avg_obs'] for r in rel]
cf = [r['count'] for r in rel]
ax.scatter(xf, yf, s=np.sqrt(np.array(cf)) * 1.2, c='#d62728', alpha=0.85, edgecolors='k', linewidths=0.5)
for x, y, c in zip(xf, yf, cf):
    ax.annotate(str(c), (x, y), textcoords='offset points', xytext=(5, -9), fontsize=6)
ax.plot([0, 1], [clim, clim], ':', color='gray', lw=0.8)
ax.set_xlabel('Forecast probability (wet, >=0.1 mm)')
ax.set_ylabel('Observed relative frequency')
ax.set_title('(a) Reliability: APCNet occurrence probability', fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(fontsize=7, loc='lower right')
ax.grid(alpha=0.3)
# (b) sharpness histogram
ax = axes[1]
ax.hist(p_wet.flatten(), bins=21, range=(0, 1), color='#1f77b4', alpha=0.85, edgecolor='white', lw=0.4)
ax.set_xlabel('Forecast probability (wet)')
ax.set_ylabel('Sample count')
ax.set_title('(b) Sharpness: forecast probability distribution', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig7_reliability.png'), dpi=300, bbox_inches='tight')
print('fig7_reliability.png saved')
