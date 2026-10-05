# -*- coding: utf-8 -*-
"""eval_run13_3h.py — run13 三种子在 ERA5 3h 与 GPM 3h 上的权威数字（925 口径对齐 Table 8/15）。"""
import os, pickle, json
import numpy as np
from datetime import datetime, timedelta

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
GPM_ROOT = r'D:\liaohe\GPM_IMERG'

gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
tgt = np.load(os.path.join(WORK, 'targets_test.npy'))
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
ols = np.load(os.path.join(WORK, 'predictions_ols.npy'))
unet = np.load(os.path.join(WORK, '..', 'manuscript_work_unet', 'predictions_apcnet.npy'))
gpm3 = np.load(os.path.join(WORK, 'gpm3h_obs.npy'))

def apc_for(seed):
    if seed == 42:
        p = os.path.join(WORK, 'predictions_apcnet.npy')
    else:
        p = os.path.join(WORK, f'seed{seed}', 'predictions_apcnet.npy')
    if os.path.exists(p):
        return np.load(p)
    return None

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
assert len(sel) == len(gpm3)

def cont(o, f):
    o, f = o.flatten(), f.flatten()
    v = np.isfinite(o) & np.isfinite(f)
    o, f = o[v], f[v]
    mse = np.mean((o - f) ** 2)
    return mse, float(np.sqrt(mse)), float(np.corrcoef(o, f)[0, 1]), float(np.mean(f - o))

# ---- ERA5 参考 3h（2839 全样本）----
print('=== ERA5 参考 3h (n=2839, 925 格点) ===')
mg = cont(tgt, gfs)
print('GFS MSE=%.4f RMSE=%.4f CC=%.4f' % mg[:3])
for nm, a in [('QM', qm), ('OLS', ols), ('U-Net', unet)]:
    mse, rmse, cc, b = cont(tgt, a)
    print('%-6s MSE=%.4f imp=%+.1f%% CC=%.4f' % (nm, mse, 100 * (mg[0] - mse) / mg[0], cc))
for s in [42, 40, 41]:
    a = apc_for(s)
    if a is None:
        print(f'SEED={s}: 预测缺失'); continue
    mse, rmse, cc, b = cont(tgt, a)
    print('APCNet S%-2d MSE=%.4f imp=%+.1f%% CC=%.4f per-samp-CC=%.3f' % (s, mse, 100 * (mg[0] - mse) / mg[0], cc, np.mean([np.corrcoef(tgt[i], a[i])[0, 1] for i in range(len(tgt)) if np.std(tgt[i]) > 0 and np.std(a[i]) > 0])))

# ---- GPM 3h（2475 样本, 925 格点）----
print('\n=== GPM 3h (n=2475, 925 格点, 对齐 Table 8) ===')
mg = cont(gpm3, gfs[sel])
print('GFS MSE=%.4f RMSE=%.4f CC=%.4f' % mg[:3])
for nm, a in [('QM', qm[sel]), ('OLS', ols[sel]), ('U-Net', unet[sel]), ('ERA5', tgt[sel])]:
    mse, rmse, cc, b = cont(gpm3, a)
    print('%-5s MSE=%.4f imp=%+.1f%% CC=%.4f' % (nm, mse, 100 * (mg[0] - mse) / mg[0], cc))
for s in [42, 40, 41]:
    a = apc_for(s)
    if a is None:
        print(f'SEED={s}: 预测缺失'); continue
    mse, rmse, cc, b = cont(gpm3, a[sel])
    print('APCNet S%-2d MSE=%.4f RMSE=%.4f imp=%+.1f%% CC=%.4f bias=%+.4f' % (s, mse, rmse, 100 * (mg[0] - mse) / mg[0], cc, b))
