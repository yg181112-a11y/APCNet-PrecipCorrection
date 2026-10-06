# -*- coding: utf-8 -*-
"""gpm_lead_baseline_ci.py — 为表 17 的 QM/BinCM/OLS 基线补月块 bootstrap 95% CI
- 样本集与 obs_lead_bootstrap.gpm_bootstrap 完全一致（剔除 NaN 行后 24h=638/72h=636/120h=633）
- 基线标定 = verify_gpm_lead.py 同实现（QM 逐格点分位数；BinCM 域级分箱比率；OLS 逐格点线性）
- 块 = (年,月)，i.i.d. 重采样 2000 次（rng 42），2.5%/97.5% 分位
输出：D:\\liaohe\\论文三\\03_重建成稿代_2026_R3全链主实验\\gpm_lead_baseline_ci.json
"""
import os, json
import numpy as np
from datetime import datetime, timedelta

BASE = r'D:\liaohe\论文三\03_重建成稿代_2026_R3全链主实验'
NBOOT = 2000
RNG = np.random.default_rng(42)


def fit_qm(g, e, nb=200):
    qa = np.linspace(0.001, 0.999, nb)
    gq = np.zeros((25, 37, nb)); eq = np.zeros((25, 37, nb))
    for i in range(25):
        for j in range(37):
            gq[i, j] = np.nanquantile(g[:, i, j], qa)
            eq[i, j] = np.nanquantile(e[:, i, j], qa)
    return gq, eq


def apply_qm_cov(g, gq, eq, cov_idx):
    out = np.zeros_like(g)
    for k, (i, j) in enumerate(cov_idx):
        out[:, k] = np.interp(g[:, k], gq[i, j], eq[i, j],
                              left=eq[i, j, 0], right=eq[i, j, -1])
    return out


def fit_bm(g, e, nb=20):
    gmin, gmax = np.nanmin(g), np.nanmax(g)
    edges = np.linspace(gmin, gmax, nb + 1)
    ratio = np.zeros(nb)
    for b in range(nb):
        m = (g >= edges[b]) & (g < edges[b + 1]) & np.isfinite(g) & np.isfinite(e)
        gm = g[m].mean() if m.sum() else np.nan
        em = e[m].mean() if m.sum() else np.nan
        ratio[b] = em / gm if gm and gm > 0 else 1.0
    return edges, ratio


def apply_bm(g, edges, ratio):
    idx = np.clip(np.searchsorted(edges, g, side="right") - 1, 0, len(ratio) - 1)
    return g * ratio[idx]


def fit_ols(g, e):
    a = np.zeros((25, 37)); b = np.zeros((25, 37))
    for i in range(25):
        for j in range(37):
            X = g[:, i, j]; Y = e[:, i, j]
            m = np.isfinite(X) & np.isfinite(Y)
            if m.sum() > 5:
                aa, bb = np.polyfit(X[m], Y[m], 1)
                a[i, j], b[i, j] = aa, bb
            else:
                a[i, j], b[i, j] = 1.0, 0.0
    return a, b


def apply_ols_cov(g, a, b, cov_idx):
    out = np.zeros_like(g)
    for k, (i, j) in enumerate(cov_idx):
        out[:, k] = g[:, k] * a[i, j] + b[i, j]
    return out


def bootstrap_ci(gfs, obs, pred, months):
    um = sorted({tuple(m) for m in months})
    idx = {m: np.where((months == m).all(axis=1))[0] for m in um}
    def mse(a, b): return float(np.mean((a - b) ** 2))
    gfs_mse = mse(gfs, obs)
    obs_imp = 100.0 * (gfs_mse - mse(pred, obs)) / gfs_mse
    boots = []
    for _ in range(NBOOT):
        bs = [tuple(m) for m in RNG.choice(um, size=len(um), replace=True)]
        s = np.concatenate([idx[m] for m in bs])
        boots.append(100.0 * (mse(gfs[s], obs[s]) - mse(pred[s], obs[s])) / mse(gfs[s], obs[s]))
    boots = np.array(boots)
    return obs_imp, [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))], float(np.mean(boots > 0))


out = {}
for fhr in (24, 72, 120):
    z = np.load(os.path.join(BASE, 'gpm_lead_arrays', f'gpm_{fhr}h.npz'))
    obs, gfs = z['gpm'], z['gfs']
    apc, unet = z['apc'], z['unet']
    months = z['months']
    bad = np.isnan(gfs).any(axis=1) | np.isnan(obs).any(axis=1) \
        | np.isnan(apc).any(axis=1) | np.isnan(unet).any(axis=1)
    n_drop = int(bad.sum())
    obs, gfs, months = obs[~bad], gfs[~bad], months[~bad]
    n = len(obs)
    cov = z['cov_mask']
    cov_idx = np.argwhere(cov)
    n_cov = int(cov.sum())

    # 训练期标定（全网格）
    if fhr == 24:
        d = os.path.join(BASE, '24h_exp')
    else:
        d = os.path.join(BASE, 'multi_lead_exp', f'{fhr}h')
    gfs_all = np.load(os.path.join(d, f'gfs_{fhr}h_accum.npy'))
    era5_all = np.load(os.path.join(d, f'era5_{fhr}h_accum.npy'))
    split = np.load(os.path.join(d, 'split_mask.npy'))
    valid = ~(np.isnan(gfs_all).any((1, 2)) | np.isnan(era5_all).any((1, 2)))
    tr = np.where((split == 0) & valid)[0]
    gfs_tr, era5_tr = gfs_all[tr], era5_all[tr]

    gq, eq = fit_qm(gfs_tr, era5_tr)
    qm = apply_qm_cov(gfs, gq, eq, cov_idx)
    edges, ratio = fit_bm(gfs_tr, era5_tr)
    bm = apply_bm(gfs, edges, ratio)
    a, b = fit_ols(gfs_tr, era5_tr)
    ols = apply_ols_cov(gfs, a, b, cov_idx)

    row = {'n_samples': int(n), 'n_months': len(set(map(tuple, months))), 'cov_gridpoints': n_cov, 'dropped_nan_rows': int(n_drop)}
    for tag, arr in [('QM', qm), ('BinCM', bm), ('OLS', ols)]:
        imp, ci, ppos = bootstrap_ci(gfs, obs, arr, months)
        row[tag] = {'imp': imp, 'ci95': ci, 'p_pos': ppos}
        print(f"GPM {fhr}h {tag}: {imp:+.2f}% CI [{ci[0]:+.2f},{ci[1]:+.2f}] P>0={ppos:.3f} n={n}")
    out[f'gpm_{fhr}h'] = row

p = os.path.join(BASE, 'gpm_lead_baseline_ci.json')
json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('saved', p)
