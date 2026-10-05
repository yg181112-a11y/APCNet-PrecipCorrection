# -*- coding: utf-8 -*-
"""验证 Table 5(ERA5 日12h) run13 + CHM S40/S41。"""
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
unet = np.load(os.path.join(WORK, '..', 'manuscript_work_unet', 'predictions_apcnet.npy'))
apc42 = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
apc40 = np.load(os.path.join(WORK, 'seed40', 'predictions_apcnet.npy'))
apc41 = np.load(os.path.join(WORK, 'seed41', 'predictions_apcnet.npy'))
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
        acc = sum(arr[i] for i in keep)
        dates.append(day); accs.append(acc)
    return dates, np.array(accs)

dts = {nm: daily_acc(a) for nm, a in [('gfs', gfs), ('tgt', tgt), ('apc42', apc42), ('apc40', apc40), ('apc41', apc41),
                                      ('qm', qm), ('ols', ols), ('unet', unet)]}
common = sorted(set.intersection(*[set(dts[nm][0]) for nm in dts]))
print('ERA5 日12h days', len(common))
rates = {nm: np.array([dts[nm][1][dts[nm][0].index(d)] / 12.0 for d in common]) for nm in dts}
T = rates['tgt']; G = rates['gfs']
print('GFS mean=%.4f TGT mean=%.4f' % (G.mean(), T.mean()))
mg = np.mean((T - G) ** 2)
out = {}
for nm in ['apc42', 'apc40', 'apc41', 'qm', 'ols', 'unet']:
    ma = np.mean((T - rates[nm]) ** 2)
    out[nm] = {'mse': float(ma), 'imp': float(100*(mg-ma)/mg), 'mean': float(rates[nm].mean())}
    print(f"{nm}: MSE={ma:.4f} imp={100*(mg-ma)/mg:+.1f}% mean={rates[nm].mean():.3f}")
# 对比 daily_eval 旧值：APCNet mse 11.506（-101%）
json.dump(out, open(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\table5_run13.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
