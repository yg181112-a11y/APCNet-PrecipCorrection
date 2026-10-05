# -*- coding: utf-8 -*-
"""Table5 补算 v2：逐格点 CC（忽略NaN）、域均 rate、>=10mm bias。"""
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
apc42 = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
unet = np.load(os.path.join(WORK, '..', 'manuscript_work_unet', 'predictions_apcnet.npy'))
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)

day_map = {}
for i, t in enumerate(times):
    tt = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    day_map.setdefault(tt.date(), []).append(i)

def daily_acc(arr):
    dates, accs = [], []
    for day in sorted(day_map.keys()):
        keep = [i for i in day_map[day] if (times[i].replace(tzinfo=utc) if times[i].tzinfo is None else t.astimezone(utc)).hour in WINDOW_END_HOURS]
        if not keep: continue
        dates.append(day); accs.append(arr[keep].sum(axis=0))
    return dates, np.stack(accs)

D = {nm: daily_acc(a) for nm, a in [('gfs', gfs), ('tgt', tgt), ('apc42', apc42), ('qm', qm), ('ols', ols), ('unet', unet)]}
common = sorted(set.intersection(*[set(D[nm][0]) for nm in D]))
T = np.stack([D['tgt'][1][D['tgt'][0].index(x)] for x in common])
G = np.stack([D['gfs'][1][D['gfs'][0].index(x)] for x in common])
mg = np.mean((T - G) ** 2)
out = {}
for nm in ['gfs', 'apc42', 'qm', 'ols', 'unet']:
    A = np.stack([D[nm][1][D[nm][0].index(x)] for x in common])
    mse = np.mean((T - A) ** 2)
    cs = []
    for j in range(T.shape[1]):
        tv, av = T[:, j], A[:, j]
        if np.std(tv) > 1e-9 and np.std(av) > 1e-9:
            cs.append(np.corrcoef(tv, av)[0, 1])
    cc = float(np.mean(cs)) if cs else float('nan')
    rate = float(A.mean())
    tdm = T.mean(axis=1); adm = A.mean(axis=1)
    big = tdm >= 10
    bias10 = float(adm[big].mean() / 10.0) if big.sum() >= 3 else float('nan')
    imp = 100 * (mg - mse) / mg
    print(f'{nm}: MSE={mse:.4f} imp={imp:+.1f}% CC={cc:.4f} rate12h={rate:.3f} bias10={bias10:.2f}')
    out[nm] = {'mse': float(mse), 'imp': float(imp), 'cc': cc, 'rate12h': rate, 'bias10': bias10}
json.dump(out, open(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\table5_run13_full.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved')
