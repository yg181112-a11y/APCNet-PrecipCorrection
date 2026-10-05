# -*- coding: utf-8 -*-
"""eval_run13_all.py — run13 三种子在 ERA5/GPM 3h 上的完整指标（分类表对齐 Table 9/10、per-sample CC、bootstrap）。"""
import os, pickle, json
import numpy as np
from datetime import timezone, timedelta

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
GPM_ROOT = r'D:\liaohe\GPM_IMERG'
SEEDS = {'42': r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work\predictions_apcnet.npy',
         '40': r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work\seed40\predictions_apcnet.npy',
         '41': r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work\seed41\predictions_apcnet.npy'}

gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
tgt = np.load(os.path.join(WORK, 'targets_test.npy'))
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
ols = np.load(os.path.join(WORK, 'predictions_ols.npy'))
unet = np.load(os.path.join(WORK, '..', 'manuscript_work_unet', 'predictions_apcnet.npy'))
gpm3 = np.load(os.path.join(WORK, 'gpm3h_obs.npy'))
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)

utc = timezone.utc
WINDOW_END_HOURS = [3, 9, 15, 21]
vt, sel = [], []
idx_map = {}
for i, t in enumerate(times):
    tt = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    idx_map.setdefault(tt, []).append(i)
for i, t in enumerate(times):
    tt = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    if tt.hour not in WINDOW_END_HOURS:
        continue
    fs = []
    for k in range(6):
        e = tt - timedelta(minutes=30 * (5 - k))
        fs.append(os.path.join(GPM_ROOT, f'imerg_{e.year}{e.month:02d}',
                               f'imerg_{e.year}{e.month:02d}{e.day:02d}_{e.hour:02d}{e.minute:02d}00.nc4'))
    if any(not os.path.exists(f) for f in fs):
        continue
    vt.append(tt)
    sel.append(i)
sel = np.array(sel)
print('GPM 样本', len(sel), '期望 2475')

def cat(o, f, th):
    o, f = np.asarray(o), np.asarray(f)
    v = np.isfinite(o) & np.isfinite(f)
    o, f = o[v], f[v]
    a = o >= th; b = f >= th
    hits = np.sum(a & b); fa = np.sum(~a & b); miss = np.sum(a & ~b); cn = np.sum(~a & ~b)
    pod = hits / max(hits + miss, 1); far = fa / max(hits + fa, 1)
    ph = (hits + fa) / max(hits + fa + cn + miss, 1); po = (hits + miss) / max(hits + miss + cn + fa, 1)
    ets = (hits - ph * po * (hits + fa + cn + miss)) / max(hits + fa + miss - ph * po * (hits + fa + cn + miss), 1)
    return pod, far, ets

def per_sample_cc(o, f):
    cs = []
    for i in range(len(o)):
        a, b = o[i], f[i]
        if np.std(a) > 0 and np.std(b) > 0:
            cs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(cs))

out = {}
for seed, apc_path in SEEDS.items():
    apc = np.load(apc_path)
    # ERA5 3h
    mse_g = np.mean((tgt - gfs) ** 2)
    mse_a = np.mean((tgt - apc) ** 2)
    era5 = {'gfs_mse': float(mse_g), 'apc_mse': float(mse_a),
            'apc_imp': float(100 * (mse_g - mse_a) / mse_g),
            'apc_cc': float(np.corrcoef(tgt.flatten(), apc.flatten())[0, 1]),
            'per_sample_cc': per_sample_cc(tgt, apc)}
    for th in [10, 15, 20]:
        era5[f'ets{th}'] = cat(tgt, apc, th)[2]
        era5[f'gfs_ets{th}'] = cat(tgt, gfs, th)[2]
    out[f'era5_s{seed}'] = era5
    # GPM 3h
    o = gpm3; g = gfs[sel]; a = apc[sel]
    mg = np.mean((o - g) ** 2); ma = np.mean((o - a) ** 2)
    gpm = {'gfs_mse': float(mg), 'apc_mse': float(ma),
           'apc_imp': float(100 * (mg - ma) / mg),
           'apc_cc': float(np.corrcoef(o.flatten(), a.flatten())[0, 1]),
           'apc_bias': float(np.mean(a - o)),
           'gfs_rmse': float(np.sqrt(mg)), 'apc_rmse': float(np.sqrt(ma))}
    for th in [0.1, 1, 3, 10, 20]:
        gpm[f'ets{th}'] = cat(o, a, th)[2]
        gpm[f'gfs_ets{th}'] = cat(o, g, th)[2]
    out[f'gpm_s{seed}'] = gpm

json.dump(out, open(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\run13_3h_full.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for seed in SEEDS:
    e, g = out[f'era5_s{seed}'], out[f'gpm_s{seed}']
    print(f"S{seed}: ERA5 imp {e['apc_imp']:+.1f}% per-samp-CC {e['per_sample_cc']:.3f} | "
          f"GPM imp {g['apc_imp']:+.1f}% RMSE {g['apc_rmse']:.4f} bias {g['apc_bias']:+.4f} | "
          f"ETS20 ERA5 {e['ets20']:.3f}(GFS {e['gfs_ets20']:.3f}) GPM {g['ets20']:.3f}(GFS {g['gfs_ets20']:.3f})")
# U-Net 与基线也归档
o = gpm3
for nm, arr in [('QM', qm), ('OLS', ols), ('U-Net', unet), ('ERA5', tgt)]:
    a = arr[sel] if arr is tgt or arr is qm or arr is ols or arr is unet else arr
    ma = np.mean((o - a) ** 2)
    print(f"GPM {nm}: imp {100*(np.mean((o-gfs[sel])**2)-ma)/np.mean((o-gfs[sel])**2):+.1f}%")
