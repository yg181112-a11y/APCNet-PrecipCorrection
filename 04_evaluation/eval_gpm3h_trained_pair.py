# -*- coding: utf-8 -*-
"""eval_gpm3h_trained_pair.py — GPM-trained APCNet 与 ERA5-trained 各模型的 GPM 观测对齐评估
样本：GPM 有效时次（correction_test.sample_times，2024-2025）
真值：GPM IMERG 3h 累积（6 bin 累加，同 verify_gpm_independent 口径）
对比：GFS / QM / OLS / APCNet(ERA5) / U-Net(ERA5) / APCNet(GPM-train)
指标：cont + cat + 逐样本空间 CC + block bootstrap（MSE 改进 vs GFS 的显著性）
输出：gpm_trained_exp/eval_pair.json
"""
import sys, os, json, pickle, argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, r"C:\Users\yg181\Desktop\论文三\实验代码优化过程\12.8修")
import verify_gpm_independent as V
from verify_gpm_independent import regrid_to_main, cont_metrics, cat_metrics, PRECIP_THRESHOLDS

BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
OUT = os.path.join(BASE, 'gpm_trained_exp')
WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
GPM_ROOT = r"D:\liaohe\GPM_IMERG"   # 2024-01~2025-09 测试期
_orig = V.gpm_file_for
V.gpm_file_for = lambda dt, root=GPM_ROOT: _orig(dt, root=root)
BLOCK_DAYS = 90

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()

# ---------- 1) 我的预测 + 时次 ----------
pred_gpm = np.load(os.path.join(OUT, 'pred_apcnet_gpm3h_s%d.npy' % args.seed))
with open(os.path.join(OUT, 'times_test_gpm.pkl'), 'rb') as f:
    my_times = pickle.load(f)
my_times = [t.replace(tzinfo=None) if t.tzinfo else t for t in my_times]
N = len(my_times)
print('my pred', pred_gpm.shape, 'times', N)

# ---------- 2) manuscript_work 各模型预测（按主实验测试时次）----------
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    ref_times = pickle.load(f)
ref_times = [t.replace(tzinfo=None) if t.tzinfo else t for t in ref_times]
ref_idx = {t: i for i, t in enumerate(ref_times)}
gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
apc_era5 = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
ols = np.load(os.path.join(WORK, 'predictions_ols.npy'))
unet = np.load(os.path.join(WORK, 'predictions_unet.npy'))
missing = [t for t in my_times if t not in ref_idx]
print('对齐：GPM 有效时次 %d，主实验可匹配 %d，缺失 %d' % (N, N - len(missing), len(missing)))
if missing:
    raise RuntimeError('时次无法与主实验对齐: %s ...' % missing[:5])
i_arr = np.array([ref_idx[t] for t in my_times])
gfs = gfs[i_arr]; apc_era5 = apc_era5[i_arr]; qm = qm[i_arr]; ols = ols[i_arr]; unet = unet[i_arr]

# ---------- 3) GPM 真值（逐时次）----------
gpm_all = np.zeros((N, 25, 37)); mask_all = np.zeros((N, 25, 37), dtype=bool)
cache = {}
for i, t0 in enumerate(my_times):
    acc, lon, lat, ok = V.gpm_3h_accum(t0, cache)
    if not ok:
        continue
    field, mask = regrid_to_main(acc, lon, lat)
    gpm_all[i] = field; mask_all[i] = mask
ok_rows = np.array([i for i in range(N) if mask_all[i].any()])
print('GPM 真值有效样本 %d/%d' % (len(ok_rows), N))
cov = mask_all[ok_rows].all(axis=0)
print('共同覆盖格点 %d/925' % int(cov.sum()))

def restrict(a):
    return a[ok_rows][:, cov]

names = ['GFS', 'QM', 'OLS', 'APCNet_ERA5', 'U-Net_ERA5', 'APCNet_GPM']
arrs = {'GFS': gfs, 'QM': qm, 'OLS': ols, 'APCNet_ERA5': apc_era5,
        'U-Net_ERA5': unet, 'APCNet_GPM': pred_gpm}
obs = restrict(gpm_all)

# ---------- 4) 指标 ----------
res = {'n_ok': int(len(ok_rows)), 'n_cov': int(cov.sum()), 'thresholds': [float(x) for x in PRECIP_THRESHOLDS]}
for nm in names:
    f = restrict(arrs[nm])
    c = cont_metrics(obs, f)
    cat = {th: cat_metrics(obs, f, th) for th in PRECIP_THRESHOLDS}
    # 逐样本空间 CC
    cc_s = []
    for r in ok_rows:
        o = gpm_all[r][cov]; p = arrs[nm][r][cov]
        if o.std() > 0 and p.std() > 0:
            cc_s.append(float(np.corrcoef(o, p)[0, 1]))
    res[nm] = {'cont': {k: float(v) for k, v in c.items()},
               'cat': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in cat.items()},
               'spatial_cc': float(np.mean(cc_s)) if cc_s else None}
    imp = 100.0 * (c['MSE'] - res['GFS']['cont']['MSE']) / res['GFS']['cont']['MSE']
    res[nm]['mse_improve_vs_gfs_pct'] = imp
    print('%s  MSE %.4f RMSE %.4f MAE %.4f CC %.4f | imp_vs_GFS %+.2f%% | spCC %.3f' %
          (nm, c['MSE'], c['RMSE'], c['MAE'], c['CC'], imp, res[nm]['spatial_cc']))

# ---------- 5) block bootstrap（APCNet_GPM vs GFS 的改进显著性）----------
def block_boot(obs2, f1, f2, block_days=BLOCK_DAYS, n_boot=1000, seed=42):
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
        if (b2 - b1) / b2 > 0: cnt += 1
    return {'mse_improve_pct': float(100.0 * d0), 'p_positive': float(cnt / n_boot)}

res['bootstrap_APCNet_GPM_vs_GFS'] = block_boot(obs, arrs['APCNet_GPM'][ok_rows][:, cov], gfs[ok_rows][:, cov])
res['bootstrap_APCNet_GPM_vs_APCNet_ERA5'] = block_boot(obs, arrs['APCNet_GPM'][ok_rows][:, cov], apc_era5[ok_rows][:, cov])

json.dump(res, open(os.path.join(OUT, 'eval_pair_s%d.json' % args.seed), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('\nsaved:', os.path.join(OUT, 'eval_pair_s%d.json' % args.seed))
print('bootstrap GPM-train vs GFS:', res['bootstrap_APCNet_GPM_vs_GFS'])
print('bootstrap GPM-train vs ERA5-train:', res['bootstrap_APCNet_GPM_vs_APCNet_ERA5'])
