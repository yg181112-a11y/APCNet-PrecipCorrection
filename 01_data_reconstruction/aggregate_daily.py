# -*- coding: utf-8 -*-
"""P1-② 日累积（12h 覆盖）口径评估。
日 D = D日 03Z+09Z+15Z+21Z 四个 3h 窗口之和（GFS 仅有此 4 时次；与 CHM 24h 比较时注明覆盖差）。
对比 GFS / ERA5(target) / APCNet(对称S42) / QM / OLS / UNet。
"""
import pickle, numpy as np, os, json

W = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
WS = os.path.join(W, '..', 'manuscript_work_sym', 'seed42')

def load(name, base):
    p = os.path.join(base, name)
    if not os.path.exists(p):
        return None
    return np.load(p).astype(np.float32)

gfs = load('gfs_test.npy', W)
tgt = load('targets_test.npy', W)
apc = load('predictions_apcnet.npy', WS)   # 对称 S42
qm  = load('predictions_qm.npy', W)
ols = load('predictions_ols.npy', W)
unet= load('predictions_apcnet.npy', os.path.join(W, '..', 'manuscript_work_unet'))  # 新数据重训 UNet (sym S42)
with open(os.path.join(W, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)

assert len(times) == gfs.shape[0]
# 按 (date) 分组，取当日 03/09/15/21 四窗口；丢弃不完整日
from collections import defaultdict
groups = defaultdict(list)
for i, tt in enumerate(times):
    groups[tt.date()].append(i)
day_ids = []
for d, idxs in groups.items():
    if len(idxs) == 4:
        day_ids.append((d, idxs))
day_ids.sort()
print('完整日数:', len(day_ids), '覆盖:', day_ids[0][0], '→', day_ids[-1][0])

def daily(arr):
    out = np.zeros((len(day_ids), arr.shape[1], arr.shape[2]), dtype=np.float32)
    for k, (d, idxs) in enumerate(day_ids):
        out[k] = arr[idxs].sum(axis=0)
    return out

DG = daily(gfs); DT = daily(tgt); DA = daily(apc); DQ = daily(qm); DO = daily(ols); DU = daily(unet)

def metrics(pred, ref, name):
    mse_p = np.mean((pred - ref) ** 2)
    mse_g = np.mean((DG - ref) ** 2)
    imp = (1 - mse_p / mse_g) * 100
    cc = np.corrcoef(pred.flatten(), ref.flatten())[0, 1]
    cc_g = np.corrcoef(DG.flatten(), ref.flatten())[0, 1]
    bias = {}
    for th in [0.1, 3.0, 10.0, 20.0, 50.0]:
        o = np.mean(ref >= th); p = np.mean(pred >= th)
        bias[th] = p / o if o > 0 else float('nan')
    print('%-6s | MSE=%.4f 改进=%.2f%% | CC=%.4f (GFS %.4f) | 均值 %.3f/%.3f | bias%s' % (
        name, mse_p, imp, cc, cc_g, pred.mean(), ref.mean(),
        {k: round(v, 2) for k, v in bias.items() if k in (0.1, 10.0, 20.0, 50.0)}))
    return dict(mse=mse_p, imp=imp, cc=cc, cc_g=cc_g, mean=float(pred.mean()), bias={str(k): float(v) for k, v in bias.items()})

res = {}
print('=== 日累积（12h 覆盖）对 ERA5 target ===')
for nm, arr in [('GFS', DG), ('APCNet', DA), ('QM', DQ), ('OLS', DO), ('UNet', DU)]:
    res[nm] = metrics(arr, DT, nm)

np.save(os.path.join(W, 'daily_gfs.npy'), DG)
np.save(os.path.join(W, 'daily_targets.npy'), DT)
np.save(os.path.join(W, 'daily_apcnet.npy'), DA)
np.save(os.path.join(W, 'daily_qm.npy'), DQ)
np.save(os.path.join(W, 'daily_ols.npy'), DO)
np.save(os.path.join(W, 'daily_unet.npy'), DU)
with open(os.path.join(W, 'daily_eval.json'), 'w') as f:
    json.dump({'n_days': int(len(day_ids)), 'coverage': '12h (03/09/15/21Z 4x3h)', **res}, f, indent=1, default=float)
print('daily_eval.json 已保存')
