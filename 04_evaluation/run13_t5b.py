# -*- coding: utf-8 -*-
"""补算：Table5 CC/>=10bias run13、CHM S40/S41 全指标、GPM 域均率、asym 文件检查。"""
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
apc40 = np.load(os.path.join(WORK, 'seed40', 'predictions_apcnet.npy'))
apc41 = np.load(os.path.join(WORK, 'seed41', 'predictions_apcnet.npy'))
unet = np.load(os.path.join(WORK, '..', 'manuscript_work_unet', 'predictions_apcnet.npy'))
gpm3 = np.load(os.path.join(WORK, 'gpm3h_obs.npy'))
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)
sel = np.array([i for i, t in enumerate(times) if (t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)).hour in WINDOW_END_HOURS])
sel = sel[:len(gpm3)]

day_map = {}
for i, t in enumerate(times):
    tt = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    day_map.setdefault(tt.date(), []).append(i)
def daily_acc(arr):
    dates, accs = [], []
    for day in sorted(day_map.keys()):
        keep = [i for i in day_map[day] if (times[i].replace(tzinfo=utc) if times[i].tzinfo is None else times[i].astimezone(utc)).hour in WINDOW_END_HOURS]
        if not keep: continue
        dates.append(day); accs.append(arr[keep].sum(axis=0).mean())
    return dates, np.array(accs)

d = {nm: daily_acc(a) for nm, a in [('gfs', gfs), ('tgt', tgt), ('apc42', apc42), ('qm', qm), ('ols', ols), ('unet', unet)]}
common = sorted(set.intersection(*[set(d[nm][0]) for nm in d]))
T = np.array([d['tgt'][1][d['tgt'][0].index(x)] for x in common])
out = {}
print('== Table5 run13 补算（12h 累计口径）==')
for nm in ['gfs', 'apc42', 'qm', 'ols', 'unet']:
    A = np.array([d[nm][1][d[nm][0].index(x)] for x in common])
    mse = np.mean((T - A) ** 2)
    cc = np.corrcoef(T, A)[0, 1] if np.std(A) > 0 else float('nan')
    big = (T >= 10)
    bias10 = np.mean(A[big]) / 10.0 if big.sum() > 5 else float('nan')
    imp = 100 * (np.mean((T - d['gfs'][1]) ** 2) - mse) / np.mean((T - d['gfs'][1]) ** 2)
    print(f'  {nm}: MSE={mse:.4f} imp={imp:+.1f}% CC={cc:.4f} mean12h={A.mean():.3f} bias10={bias10:.2f}')
    out[nm] = {'mse12h': float(mse), 'imp': float(imp), 'cc': float(cc), 'mean12h': float(A.mean()), 'bias10': float(bias10)}
json.dump(out, open(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\table5_run13_full.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# GPM 域均率（[P84]）
o = gpm3.mean(axis=(1, 2)); a = apc42[sel].mean(axis=(1, 2)); g = gfs[sel].mean(axis=(1, 2))
print('== GPM 域均率 ==')
print(f'  obs={o.mean():.3f} GFS={g.mean():.3f} APCNet={a.mean():.3f} ratio={a.mean()/o.mean():.1f}x')
# CHM S40/S41 全指标：用 chm eval 脚本逻辑（读 CHM 2024/2025 日数据重算较繁；改从已有 json 补）
print('== 检查 asym 预测文件 ==')
for p in ['predictions_asym.npy', 'asym', 'manuscript_work_asym']:
    fp = os.path.join(WORK, p)
    print(f'  {p}: exists={os.path.exists(fp)}')
print('saved')
