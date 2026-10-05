# -*- coding: utf-8 -*-
"""
eval_gpm3h_run13_pair.py — run13 GPM 3h 口径权威配对表（含 APCNet_GPM / OLS_GPM）
=================================================================================
口径与 eval_run13_3h.py / docx Table 8 完全一致：
  - GPM 真值 = gpm3h_obs.npy（2475 时次，6 半小时 bin -> 3h）
  - 时次对齐 sel：按 GPM 文件存在性筛选 sample_times_test（与 eval_run13_3h 相同）
  - 评估：全部 925 格点（仅过滤 NaN）
  - U-Net 权威预测 = manuscript_work_unet/predictions_apcnet.npy
新增：BinCM / OLS_GPM / APCNet_GPM(S42) 行 + block bootstrap 显著性
输出：gpm_trained_exp/eval_run13_pair_s42.json
"""
import os, pickle, json
import numpy as np
from datetime import timedelta

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
UNET_DIR = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work_unet'
GPM_ROOT = r'D:\liaohe\GPM_IMERG'
OUT = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\gpm_trained_exp'
E3 = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\3h_exp'

gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
tgt = np.load(os.path.join(WORK, 'targets_test.npy'))
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
ols = np.load(os.path.join(WORK, 'predictions_ols.npy'))
bincm = np.load(os.path.join(WORK, 'bin_cm_pred.npy'))
unet = np.load(os.path.join(UNET_DIR, 'predictions_apcnet.npy'))
apc = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
gpm3 = np.load(os.path.join(WORK, 'gpm3h_obs.npy')).astype(np.float64)

# ---- 时次对齐（与 eval_run13_3h.py 相同：GPM 文件存在性筛选）----
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)
utc = __import__('datetime').timezone.utc
WINDOW_END_HOURS = [3, 9, 15, 21]
vt = []
for t in times:
    tt = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    if tt.hour not in WINDOW_END_HOURS:
        continue
    fs = []
    for i in range(6):
        e = tt - timedelta(minutes=30 * (5 - i))
        fs.append(os.path.join(GPM_ROOT, f'imerg_{e.year}{e.month:02d}',
                               f'imerg_{e.year}{e.month:02d}{e.day:02d}_{e.hour:02d}{e.minute:02d}00.nc4'))
    if any(not os.path.exists(f) for f in fs):
        continue
    vt.append(tt)
vt = np.array(vt)
idx = {t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc): i for i, t in enumerate(times)}
sel = np.array([idx[t] for t in vt])
assert len(sel) == len(gpm3), (len(sel), len(gpm3))
print('时次对齐：%d（=gpm3h_obs 行数）' % len(sel))

# ---- OLS(GPM-trained)：3h_exp 训练期拟合系数，应用到 gfs[sel] ----
gfs_tr = np.load(os.path.join(E3, 'gfs3h_train.npy'))
gpm_tr = np.load(os.path.join(E3, 'gpm3h_train.npy'))
H, W = 25, 37
b_fit = np.zeros((H, W)); a_fit = np.zeros((H, W))
for i in range(H):
    for j in range(W):
        x = gfs_tr[:, i, j]; y = gpm_tr[:, i, j]
        if not (np.isfinite(x).all() and np.isfinite(y).all() and x.std() > 0):
            continue
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        b_fit[i, j], a_fit[i, j] = coef
ols_gpm = np.clip(b_fit * gfs[sel] + a_fit, 0, 500)
print('OLS(GPM) 斜率域均 %.3f 截距域均 %.3f' % (np.nanmean(b_fit), np.nanmean(a_fit)))

# ---- APCNet_GPM (S42) ----
apc_gpm = np.load(os.path.join(OUT, 'pred_apcnet_gpm3h_s42.npy'))
assert apc_gpm.shape == (len(sel), 25, 37), apc_gpm.shape

def cont(o, f):
    o, f = o.flatten(), f.flatten()
    v = np.isfinite(o) & np.isfinite(f)
    o, f = o[v], f[v]
    mse = np.mean((o - f) ** 2)
    return {'MSE': float(mse), 'RMSE': float(np.sqrt(mse)),
            'MAE': float(np.mean(np.abs(o - f))),
            'CC': float(np.corrcoef(o, f)[0, 1] if len(o) > 1 else 0.0),
            'bias': float(np.mean(f - o)), 'n': int(len(o))}

def cat(o, f, th):
    oo = (o >= th); ff = (f >= th)
    tp = np.sum(oo & ff); fn = np.sum(oo & ~ff); fp = np.sum(~oo & ff); tn = np.sum(~oo & ~ff)
    pod = tp / (tp + fn) if tp + fn > 0 else 0.0
    far = fp / (fp + tp) if fp + tp > 0 else 0.0
    e = (tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)
    ets = (tp * tn - fp * fn) / e if e > 0 else 0.0
    return {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
            'POD': float(pod), 'FAR': float(far), 'ETS': float(ets)}

systems = [('GFS', gfs[sel]), ('ERA5', tgt[sel]), ('BinCM', bincm[sel]), ('QM', qm[sel]),
           ('OLS', ols[sel]), ('OLS_GPM', ols_gpm),
           ('APCNet_ERA5', apc[sel]), ('U-Net_ERA5', unet[sel]), ('APCNet_GPM', apc_gpm)]
TH = [0.1, 3.0, 10.0, 20.0]
res = {'n_samples': int(len(sel)), 'n_gridpoints': 925, 'protocol': 'run13 GPM 口径（gpm3h_obs.npy, 925 格点）'}
for nm, a in systems:
    c = cont(gpm3, a)
    res[nm] = {'cont': c, 'cat': {str(th): cat(gpm3, a, th) for th in TH}}
    imp = 100.0 * (res['GFS']['cont']['MSE'] - c['MSE']) / res['GFS']['cont']['MSE']
    res[nm]['mse_improve_vs_GFS_pct'] = imp
    print('%s  MSE %.4f RMSE %.4f MAE %.4f CC %.4f bias %+.4f | imp %+8.2f%%' %
          (nm, c['MSE'], c['RMSE'], c['MAE'], c['CC'], c['bias'], imp))

# ---- block bootstrap ----
def block_boot(obs2, f1, f2, block_days=90, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = obs2.shape[0]
    n_blocks = max(1, n // (4 * block_days))
    bidx = np.array_split(np.arange(n), n_blocks)
    m1 = np.mean((obs2 - f1) ** 2); m2 = np.mean((obs2 - f2) ** 2)
    d0 = (m2 - m1) / m2
    cnt = 0
    for _ in range(n_boot):
        ii = np.concatenate([bidx[rng.integers(0, len(bidx))] for _ in range(len(bidx))])[:n]
        b1 = np.mean((obs2[ii] - f1[ii]) ** 2); b2 = np.mean((obs2[ii] - f2[ii]) ** 2)
        if (b2 - b1) / b2 > 0:
            cnt += 1
    return {'mse_improve_pct': float(100.0 * d0), 'p_positive': float(cnt / n_boot)}

res['bootstrap_APCNet_GPM_vs_GFS'] = block_boot(gpm3, apc_gpm, gfs[sel])
res['bootstrap_APCNet_GPM_vs_APCNet_ERA5'] = block_boot(gpm3, apc_gpm, apc[sel])
out_p = os.path.join(OUT, 'eval_run13_pair_s42.json')
json.dump(res, open(out_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('\nsaved:', out_p)
print('bootstrap GPM vs GFS:', res['bootstrap_APCNet_GPM_vs_GFS'])
print('bootstrap GPM vs ERA5-train:', res['bootstrap_APCNet_GPM_vs_APCNet_ERA5'])
