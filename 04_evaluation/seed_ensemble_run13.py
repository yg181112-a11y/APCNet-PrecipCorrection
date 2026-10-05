# -*- coding: utf-8 -*-
"""A3：三 seed ensemble mean（S40/S41/S42 平均）在 ERA5 参照与 GPM 观测上的表现。"""
import os, pickle, json
import numpy as np

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
WINDOW_END_HOURS = [3, 9, 15, 21]

# ---- 载入 ----
gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
tgt = np.load(os.path.join(WORK, 'targets_test.npy'))       # ERA5 3h 参照
obs = np.load(os.path.join(WORK, 'gpm3h_obs.npy'))          # GPM 3h 观测
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
ols = np.load(os.path.join(WORK, 'predictions_ols.npy'))
s40 = np.load(os.path.join(WORK, 'seed40', 'predictions_apcnet.npy'))
s41 = np.load(os.path.join(WORK, 'seed41', 'predictions_apcnet.npy'))
s42 = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)

sel = np.array([i for i, t in enumerate(times)
                if ((t.replace(tzinfo=None) if t.tzinfo is None else t).hour in WINDOW_END_HOURS)])
sel = sel[:len(obs)]
print('sel length:', len(sel), 'obs length:', len(obs))

G, T, O, Q, L = gfs[sel], tgt[sel], obs, qm[sel], ols[sel]
S40, S41, S42 = s40[sel], s41[sel], s42[sel]
ENS = (S40 + S41 + S42) / 3.0

def metr(o, f):
    o, f = o.ravel(), f.ravel()
    mse = float(np.mean((o - f) ** 2))
    cc = float(np.corrcoef(o, f)[0, 1])
    return mse, cc

out = {'era5_ref': {}, 'gpm_obs': {}}
for tag, o in [('era5_ref', T), ('gpm_obs', O)]:
    ref_mse, ref_cc = metr(o, G)
    print(f'== {tag}  GFS MSE={ref_mse:.4f} CC={ref_cc:.4f}')
    out[tag]['GFS'] = {'mse': ref_mse, 'imp': 0.0, 'cc': ref_cc, 'rate': float(G.mean())}
    for nm, arr in [('QM', Q), ('OLS', L), ('APCNet_S42', S42), ('APCNet_S40', S40),
                    ('APCNet_S41', S41), ('Ensemble_S40-42', ENS)]:
        m, c = metr(o, arr)
        imp = 100 * (ref_mse - m) / ref_mse
        out[tag][nm] = {'mse': m, 'imp': imp, 'cc': c, 'rate': float(arr.mean())}
        print(f'   {nm:16s} MSE={m:.4f} imp={imp:+.2f}% CC={c:.4f} rate={float(arr.mean()):.4f}')
    # 域均率 bias vs 观测率
    print(f'   obs_rate={float(o.mean()):.4f}')

json.dump(out, open(os.path.join(WORK, 'seed_ensemble_run13.json'), 'w', encoding='utf-8'),
          indent=1, ensure_ascii=False, default=float)
print('saved seed_ensemble_run13.json')
