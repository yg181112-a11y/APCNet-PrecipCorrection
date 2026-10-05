# -*- coding: utf-8 -*-
"""time_cc run13：每格点时间序列 CC 域平均（GPM/ERA5）。"""
import os, pickle
import numpy as np
from datetime import timezone

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
WINDOW_END_HOURS = [3, 9, 15, 21]
utc = timezone.utc
gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
tgt = np.load(os.path.join(WORK, 'targets_test.npy'))
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
ols = np.load(os.path.join(WORK, 'predictions_ols.npy'))
apc = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
unet = np.load(os.path.join(WORK, '..', 'manuscript_work_unet', 'predictions_apcnet.npy'))
gpm3 = np.load(os.path.join(WORK, 'gpm3h_obs.npy'))
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)
sel = np.array([i for i, t in enumerate(times) if (t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)).hour in WINDOW_END_HOURS])
sel = sel[:len(gpm3)]

def tcc(o, f):
    cs = []
    o = o.reshape(len(o), -1); f = f.reshape(len(f), -1)
    for j in range(o.shape[1]):
        v = np.isfinite(o[:, j]) & np.isfinite(f[:, j])
        if v.sum() < 5:
            continue
        if np.std(o[v, j]) > 1e-8 and np.std(f[v, j]) > 1e-8:
            cs.append(np.corrcoef(o[v, j], f[v, j])[0, 1])
    return float(np.mean(cs))

# GPM（2475）
o = gpm3
for nm, arr in [('GFS', gfs[sel]), ('QM', qm[sel]), ('OLS', ols[sel]), ('APCNet', apc[sel]), ('U-Net', unet[sel])]:
    print(f'GPM time-CC {nm}: {tcc(o, arr):.3f}')
# ERA5（2839）
for nm, arr in [('GFS', gfs), ('QM', qm), ('OLS', ols), ('APCNet', apc), ('U-Net', unet)]:
    print(f'ERA5 time-CC {nm}: {tcc(tgt, arr):.3f}')
