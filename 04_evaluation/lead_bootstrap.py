# -*- coding: utf-8 -*-
"""三时效（24/72/120h）月块 bootstrap 置信区间
- 测试样本按 (年,月) 分块，月块 i.i.d. 重采样 1000 次
- 指标：APCNet / ERA5 参照 的 MSE 改进率 vs GFS（%）
"""
import numpy as np, os, json
from datetime import datetime

BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
RNG = np.random.default_rng(42)
NBOOT = 1000


def load_lead(fhr):
    if fhr == 24:
        d = os.path.join(BASE, '24h_exp')
        g = np.load(os.path.join(d, 'gfs_24h_accum.npy'))
        t = np.load(os.path.join(d, 'era5_24h_accum.npy'))
        p = np.load(os.path.join(d, 'pred_apcnet_24h.npy'))
        split = np.load(os.path.join(d, 'split_mask.npy'))
        inits = np.load(os.path.join(d, 'init_times.npy'))
        valid = ~(np.isnan(g).any((1, 2)) | np.isnan(t).any((1, 2)))
        tm = (split == 2) & valid
        init_s = inits[tm]
    else:
        d = os.path.join(BASE, 'multi_lead_exp', f'{fhr}h')
        g = np.load(os.path.join(d, f'gfs_{fhr}h_accum.npy'))
        t = np.load(os.path.join(d, f'era5_{fhr}h_accum.npy'))
        p = np.load(os.path.join(d, f'pred_apcnet_{fhr}h.npy'))
        split = np.load(os.path.join(d, 'split_mask.npy'))
        inits = np.load(os.path.join(d, 'init_times.npy'))
        valid = ~(np.isnan(g).any((1, 2)) | np.isnan(t).any((1, 2)))
        tm = (split == 2) & valid
        init_s = inits[tm]
    g, t, p = g[tm], t[tm], p[:tm.sum()]
    init_dt = np.array([np.datetime64(s, 'h') for s in init_s], dtype='datetime64[h]').astype(datetime)
    months = np.array([(d.year, d.month) for d in init_dt])
    return g, t, p, months


out = {}
for fhr in (24, 72, 120):
    g, t, p, months = load_lead(fhr)
    months_l = [tuple(m) for m in months]
    um = sorted(set(months_l))
    idx = {m: np.where(np.array(months_l) == m)[0] for m in um}
    ns = len(g)
    mse_g = np.mean((g - t) ** 2)
    mse_p = np.mean((p - t) ** 2)
    imp_p = (mse_p / mse_g - 1) * 100
    d_imp_p, d_imp_e = [], []
    for _ in range(NBOOT):
        bs_months = [tuple(m) for m in RNG.choice(um, size=len(um), replace=True)]
        sel = np.concatenate([idx[m] for m in bs_months])
        gg, tt, pp = g[sel], t[sel], p[sel]
        d_imp_p.append((np.mean((pp - tt) ** 2) / np.mean((gg - tt) ** 2) - 1) * 100)
        d_imp_e.append((np.mean((tt - tt) ** 2) / np.mean((gg - tt) ** 2) - 1) * 100)
    d_imp_p = np.array(d_imp_p); d_imp_e = np.array(d_imp_e)
    out[f'{fhr}h'] = {
        'n_test': int(ns), 'n_months': len(um),
        'mse_gfs': float(mse_g), 'mse_apc': float(mse_p),
        'apc_improve_pct': float(imp_p),
        'apc_ci95': [float(np.quantile(d_imp_p, 0.025)), float(np.quantile(d_imp_p, 0.975))],
        'era5_improve_pct': -100.0,
        'era5_ci95': [float(np.quantile(d_imp_e, 0.025)), float(np.quantile(d_imp_e, 0.975))],
    }
    print(f"{fhr}h: APCNet 改进 {imp_p:+.2f}%  CI95 [{d_imp_p.min():.1f},{d_imp_p.max():.1f}]  "
          f"分位 [{out[f'{fhr}h']['apc_ci95'][0]:+.2f},{out[f'{fhr}h']['apc_ci95'][1]:+.2f}]")

json.dump(out, open(os.path.join(BASE, 'multi_lead_exp', 'bootstrap_summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('saved multi_lead_exp/bootstrap_summary.json')
