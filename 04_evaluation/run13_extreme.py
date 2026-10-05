# -*- coding: utf-8 -*-
"""run13 [P90] 极端样本 per-sample RMSE + ERA5/GPM ETS20 bootstrap。"""
import os, pickle, json
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

o = gpm3; g = gfs[sel]; a = apc[sel]; q = qm[sel]; l = ols[sel]; u = unet[sel]
out = {}
print('===== [P90] GPM 极端样本 per-sample RMSE 改进 =====')
for th, name in [(10, 'ge10'), (20, 'ge20')]:
    m = o.max(axis=(1, 2)) >= th
    print(f'>={th} mm: n={m.sum()} samples')
    obs_rate = o[m].mean()
    peak = o[m].max(axis=(1, 2)).mean()
    print(f'  obs domain rate={obs_rate:.2f} mm/3h, mean peak={peak:.1f}')
    for nm, arr in [('GFS', g[m]), ('QM', q[m]), ('OLS', l[m]), ('U-Net', u[m]), ('APCNet', a[m])]:
        oo, ff = o[m], arr
        v = np.isfinite(oo) & np.isfinite(ff)
        rmse_g = np.sqrt(np.mean((oo[v] - g[m][v]) ** 2))
        rmse_f = np.sqrt(np.mean((oo[v] - ff[v]) ** 2))
        imp = 100 * (rmse_g - rmse_f) / rmse_g
        rate_f = np.nanmean(ff)
        print(f'  {nm}: per-sample RMSE imp={imp:+.1f}% (rate={rate_f:.2f})')
        out[f'{name}_{nm}'] = {'imp': float(imp), 'rate': float(rate_f), 'obs_rate': float(obs_rate), 'peak': float(peak), 'n': int(m.sum())}
json.dump(out, open(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\run13_extreme.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved')
