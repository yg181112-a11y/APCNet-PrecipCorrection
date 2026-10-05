# -*- coding: utf-8 -*-
"""
eval_gpm3h_authoritative.py — GPM 3h 独立验证权威表（统一口径重算）
=====================================================================
口径：GPM 有效时次（times_test_gpm.pkl, 2475）∩ manuscript_work 测试数组（2839 时次）
真值：gpm3h_obs.npy（2475×25×37，6 半小时 bin -> 3h 累积，行序=times_test_gpm）
系统：GFS / ERA5(target) / BinCM / QM / OLS / APCNet_ERA5 / U-Net_ERA5 / APCNet_GPM(s42)
覆盖：共同覆盖格点（与主评估一致的 828 格点）
输出：gpm_trained_exp/eval_authoritative_s42.json + 控制台表
"""
import os, sys, json, pickle
import numpy as np

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
OUT = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\gpm_trained_exp'
sys.path.insert(0, r'C:\Users\yg181\Desktop\论文三\实验代码优化过程\12.8修')
from verify_gpm_independent import cont_metrics, cat_metrics, PRECIP_THRESHOLDS

# ---------- 1) 时次对齐 ----------
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    ref_times = pickle.load(f)
ref_times = [t.replace(tzinfo=None) if t.tzinfo else t for t in ref_times]
with open(os.path.join(OUT, 'times_test_gpm.pkl'), 'rb') as f:
    my_times = pickle.load(f)
my_times = [t.replace(tzinfo=None) if t.tzinfo else t for t in my_times]
ref_idx = {t: i for i, t in enumerate(ref_times)}
i_arr = np.array([ref_idx[t] for t in my_times])
print('GPM 有效时次 %d，全部对齐主实验数组（%d 时次）' % (len(my_times), len(ref_times)))

# ---------- 2) 加载数组（当前权威版）----------
gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))[i_arr]
era5 = np.load(os.path.join(WORK, 'targets_test.npy'))[i_arr]
bincm = np.load(os.path.join(WORK, 'bin_cm_pred.npy'))[i_arr]
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))[i_arr]
ols = np.load(os.path.join(WORK, 'predictions_ols.npy'))[i_arr]
apc = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))[i_arr]
unet = np.load(os.path.join(WORK, 'predictions_unet.npy'))[i_arr]
apc_gpm = np.load(os.path.join(OUT, 'pred_apcnet_gpm3h_s42.npy'))
assert apc_gpm.shape == (len(my_times), 25, 37), apc_gpm.shape

# ---------- 3) GPM 真值（逐时次重算，与 verify_gpm_independent 同口径）----------
sys.path.insert(0, r'C:\Users\yg181\Desktop\论文三\实验代码优化过程\12.8修')
import verify_gpm_independent as V
GPM_ROOT_D = r'D:\liaohe\GPM_IMERG'
_orig = V.gpm_file_for
V.gpm_file_for = lambda dt, root=GPM_ROOT_D: _orig(dt, root=root)
cache = {}
gpm_all = np.zeros((len(my_times), 25, 37)); mask_all = np.zeros((len(my_times), 25, 37), dtype=bool)
for i, t0 in enumerate(my_times):
    acc, lon, lat, ok = V.gpm_3h_accum(t0, cache)
    if not ok:
        continue
    field, mask = V.regrid_to_main(acc, lon, lat)
    gpm_all[i] = field; mask_all[i] = mask
ok_rows = np.array([i for i in range(len(my_times)) if mask_all[i].any()])
print('GPM 真值有效样本 %d/%d' % (len(ok_rows), len(my_times)))

# ---------- 4) 共同覆盖格点 ----------
mask = mask_all[ok_rows]
cov = mask.all(axis=0)
print('共同覆盖格点 %d/925' % int(cov.sum()))
obs_v = gpm_all[ok_rows][:, cov]
def R(a):
    return a[ok_rows][:, cov]

# OLS(GPM-trained)：复用 3h_exp 训练期拟合系数，应用统一口径
E3 = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\3h_exp'
gfs_tr = np.load(os.path.join(E3, 'gfs3h_train.npy'))
gpm_tr = np.load(os.path.join(E3, 'gpm3h_train.npy'))
H, W = 25, 37
b_fit = np.zeros((H, W)); a_fit = np.zeros((H, W))
for i in range(H):
    for j in range(W):
        if not cov[i, j]:
            continue
        x = gfs_tr[:, i, j]; y = gpm_tr[:, i, j]
        if not (np.isfinite(x).all() and np.isfinite(y).all() and x.std() > 0):
            continue
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        b_fit[i, j], a_fit[i, j] = coef
ols_gpm = np.clip(b_fit * gfs + a_fit, 0, 500)
print('OLS(GPM) 斜率域均 %.3f 截距域均 %.3f（训练期 %d 样本）' % (b_fit[cov].mean(), a_fit[cov].mean(), len(gfs_tr)))

systems = [('GFS', gfs), ('ERA5', era5), ('BinCM', bincm), ('QM', qm), ('OLS', ols),
           ('OLS_GPM', ols_gpm), ('APCNet_ERA5', apc), ('U-Net_ERA5', unet), ('APCNet_GPM', apc_gpm)]
names = [s[0] for s in systems]

# ---------- 5) 指标 ----------
res = {'n_ok': int(len(my_times)), 'n_cov': int(cov.sum()), 'thresholds': [float(x) for x in PRECIP_THRESHOLDS],
       'protocol': 'GPM IMERG V07B 6x0.5h bins -> 3h; 当前 manuscript_work 数组; times_test_gpm 对齐'}
for nm, arr in systems:
    f = R(arr)
    c = cont_metrics(obs_v, f)
    cat = {th: cat_metrics(obs_v, f, th) for th in PRECIP_THRESHOLDS}
    ccs = []
    for r in range(obs_v.shape[0]):
        o, p = obs_v[r], f[r]
        if o.std() > 0 and p.std() > 0:
            ccs.append(float(np.corrcoef(o, p)[0, 1]))
    res[nm] = {'cont': {k: float(v) for k, v in c.items()},
               'cat': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in cat.items()},
               'spatial_cc': float(np.mean(ccs)) if ccs else None}
    mse_gfs = res['GFS']['cont']['MSE']
    imp = 100.0 * (mse_gfs - c['MSE']) / mse_gfs
    res[nm]['mse_improve_vs_GFS_pct'] = imp
    print('%s  MSE %.4f RMSE %.4f MAE %.4f CC %.4f | imp_vs_GFS %+8.2f%% | spCC %.3f' %
          (nm, c['MSE'], c['RMSE'], c['MAE'], c['CC'], imp, res[nm]['spatial_cc']))

# ---------- 6) block bootstrap：APCNet_GPM vs GFS / vs APCNet_ERA5 ----------
def block_boot(obs2, f1, f2, block_days=90, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = obs2.shape[0]
    n_blocks = max(1, n // (4 * block_days))
    bidx = np.array_split(np.arange(n), n_blocks)
    m1 = np.mean((obs2 - f1) ** 2); m2 = np.mean((obs2 - f2) ** 2)
    d0 = (m2 - m1) / m2
    cnt = 0
    for _ in range(n_boot):
        idx = np.concatenate([bidx[rng.integers(0, len(bidx))] for _ in range(len(bidx))])[:n]
        b1 = np.mean((obs2[idx] - f1[idx]) ** 2); b2 = np.mean((obs2[idx] - f2[idx]) ** 2)
        if (b2 - b1) / b2 > 0:
            cnt += 1
    return {'mse_improve_pct': float(100.0 * d0), 'p_positive': float(cnt / n_boot)}

res['bootstrap_APCNet_GPM_vs_GFS'] = block_boot(obs_v, R(apc_gpm), R(gfs))
res['bootstrap_APCNet_GPM_vs_APCNet_ERA5'] = block_boot(obs_v, R(apc_gpm), R(apc))

out_p = os.path.join(OUT, 'eval_authoritative_s42.json')
json.dump(res, open(out_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('\nsaved:', out_p)
print('bootstrap GPM vs GFS:', res['bootstrap_APCNet_GPM_vs_GFS'])
print('bootstrap GPM vs ERA5-train:', res['bootstrap_APCNet_GPM_vs_APCNet_ERA5'])
