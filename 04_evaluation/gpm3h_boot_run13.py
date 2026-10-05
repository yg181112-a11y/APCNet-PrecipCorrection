# -*- coding: utf-8 -*-
"""GPM QM/OLS/APCNet block bootstrap，口径严格对齐 run13_boot2（固定切块+常数基线+seed2026）。"""
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
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)
sel = np.array([i for i, t in enumerate(times)
                if ((t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)).hour in WINDOW_END_HOURS)])
sel = sel[:len(obs)]
n = len(sel)
O = obs.reshape(n, -1)
G = gfs[sel].reshape(n, -1)
Q = qm[sel].reshape(n, -1)
L = ols[sel].reshape(n, -1)
A = apc[sel].reshape(n, -1)

def pms(o, f):
    return np.mean((o - f) ** 2, axis=1)

def block_boot(mse_g, mse_f, bl, n_boot=2000, seed=2026):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / bl))
    bid = np.minimum(np.arange(n) // bl, nb - 1)
    dist = np.empty(n_boot)
    base = mse_g.mean()
    for b in range(n_boot):
        blocks = rng.choice(nb, size=nb, replace=True)
        s = np.concatenate([np.where(bid == m)[0] for m in blocks])
        dist[b] = 100.0 * (base - mse_f[s].mean()) / base
    return dist

mg = pms(O, G)
out = {}
for nm, arr in [('qm', Q), ('ols', L), ('apc', A)]:
    ma = pms(O, arr)
    imp = 100 * (mg.mean() - ma.mean()) / mg.mean()
    for bl in [8, 16, 32]:
        dist = block_boot(mg, ma, bl)
        out[f'gpm_{nm}_bl{bl}'] = {'obs_imp': float(imp),
                                   'ci95': [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
                                   'p_neg': float(np.mean(dist < 0))}
    dist16 = block_boot(mg, ma, 16)
    print(f'GPM {nm}: imp={imp:+.2f}% bl16 CI=[{np.percentile(dist16,2.5):+.2f},{np.percentile(dist16,97.5):+.2f}] P(neg)={np.mean(dist16<0):.3f}')

with open(os.path.join(WORK, 'gpm3h_boot_run13.json'), 'w', encoding='utf-8') as fo:
    json.dump(out, fo, indent=1, ensure_ascii=False, default=float)
print('saved gpm3h_boot_run13.json')
