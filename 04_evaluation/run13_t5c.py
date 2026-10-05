# -*- coding: utf-8 -*-
"""Table5 精确重算：逐格点 12h 累计 MSE/CC + 域均 rate + >=10mm bias（713 天）。"""
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
        keep = [i for i in day_map[day] if (times[i].replace(tzinfo=utc) if times[i].tzinfo is None else times[i].astimezone(utc)).hour in WINDOW_END_HOURS]
        if not keep: continue
        dates.append(day); accs.append(arr[keep].sum(axis=0))
    return dates, np.stack(accs)  # (days, 925)

D = {nm: daily_acc(a) for nm, a in [('gfs', gfs), ('tgt', tgt), ('apc42', apc42), ('qm', qm), ('ols', ols), ('unet', unet)]}
common = sorted(set.intersection(*[set(D[nm][0]) for nm in D]))
idx = {nm: [D[nm][0].index(x) for x in common] for nm in D}
T = D['tgt'][1][idx['tgt']]  # (days, 925)
print('days:', len(common))
mg = np.mean((T - D['gfs'][1][idx['gfs']]) ** 2)
out = {}
for nm in ['gfs', 'apc42', 'qm', 'ols', 'unet']:
    A = D[nm][1][idx[nm]]
    mse = np.mean((T - A) ** 2)
    imp = 100 * (mg - mse) / mg
    cs = []
    for j in range(T.shape[1]):
        if np.std(A[:, j]) > 1e-8 and np.std(T[:, j]) > 1e-8:
            cs.append(np.corrcoef(T[:, j], A[:, j])[0, 1])
    cc = float(np.mean(cs))
    rate = float(A.mean())
    big = T.mean(axis=1) >= 10  # 域均 >=10 mm 的日子
    bias10 = float(np.mean(A[big])) / 10.0 if big.sum() > 3 else float('nan')
    print(f'{nm}: MSE={mse:.4f} imp={imp:+.1f}% CC={cc:.4f} rate12h={rate:.3f} bias10={bias10:.2f}')
    out[nm] = {'mse': float(mse), 'imp': float(imp), 'cc': cc, 'rate12h': rate, 'bias10': bias10}
json.dump(out, open(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\table5_run13_full.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved')
