# -*- coding: utf-8 -*-
"""A1-GPM：验证期 GPM 观测侧季节分解（JJA vs 全年）。"""
import os, pickle, json
import numpy as np
from datetime import timezone

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
utc = timezone.utc
WINDOW_END_HOURS = [3, 9, 15, 21]

gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
obs = np.load(os.path.join(WORK, 'gpm3h_obs.npy'))
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
ols = np.load(os.path.join(WORK, 'predictions_ols.npy'))
apc = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
unet = np.load(os.path.join(WORK, '..', 'manuscript_work_unet', 'predictions_apcnet.npy'))
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)
sel = np.array([i for i, t in enumerate(times)
                if ((t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)).hour in WINDOW_END_HOURS)])
sel = sel[:len(obs)]
vt = [(times[i].replace(tzinfo=utc) if times[i].tzinfo is None else times[i].astimezone(utc)) for i in sel]

G, O, Q, L, A, U = gfs[sel], obs, qm[sel], ols[sel], apc[sel], unet[sel]
N = len(O)
mon = np.array([t.month for t in vt])
jja = mon >= 6  # 6-8 月暖季

def stats(arr_obs, arr_fc, name):
    mse = float(((arr_fc - arr_obs) ** 2).mean())
    cc = float(np.corrcoef(arr_obs.ravel(), arr_fc.ravel())[0, 1])
    rate = float(arr_fc.mean())
    return {'mse': mse, 'cc': cc, 'rate': rate}

methods = {'GFS': G, 'QM': Q, 'OLS': L, 'APCNet': A, 'U-Net': U}
out = {}
for tag, mask in [('all', np.ones(N, bool)), ('JJA', jja), ('nonJJA', ~jja)]:
    n = int(mask.sum())
    row = {'n_samples': n, 'obs_rate': float(O[mask].mean()), 'methods': {}}
    ref_mse = float(((G[mask] - O[mask]) ** 2).mean())
    row['gfs_mse'] = ref_mse
    for nm, arr in methods.items():
        st = stats(O[mask], arr[mask], nm)
        st['imp_pct'] = 100 * (ref_mse - st['mse']) / ref_mse
        row['methods'][nm] = st
    out[tag] = row
    print(f'== {tag} n={n} obs_rate={row["obs_rate"]:.4f}')
    for nm in methods:
        m = row['methods'][nm]
        print(f'   {nm:6s} MSE={m["mse"]:.4f} imp={m["imp_pct"]:+.2f}% CC={m["cc"]:.4f} rate={m["rate"]:.4f}')

json.dump(out, open(os.path.join(WORK, 'gpm_seasonal_run13.json'), 'w', encoding='utf-8'),
          indent=1, ensure_ascii=False, default=float)
print('saved gpm_seasonal_run13.json')
