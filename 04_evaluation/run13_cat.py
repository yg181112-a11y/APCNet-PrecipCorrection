# -*- coding: utf-8 -*-
"""正确对齐：Table 9 (GPM, 阈值1/3/10/20) + Table 10 (ERA5, 阈值0.1/3/10/20) run13 S42。"""
import os, pickle, json
import numpy as np
from datetime import timezone

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
WINDOW_END_HOURS = [3, 9, 15, 21]
utc = timezone.utc
gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
tgt = np.load(os.path.join(WORK, 'targets_test.npy'))
apc = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
gpm3 = np.load(os.path.join(WORK, 'gpm3h_obs.npy'))
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)
sel = np.array([i for i, t in enumerate(times) if (t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)).hour in WINDOW_END_HOURS])
sel = sel[:len(gpm3)]

def cat(o, f, th):
    o, f = np.asarray(o), np.asarray(f)
    v = np.isfinite(o) & np.isfinite(f)
    o, f = o[v], f[v]
    a = o >= th; b = f >= th
    hits = np.sum(a & b); fa = np.sum(~a & b); miss = np.sum(a & ~b); cn = np.sum(~a & ~b)
    pod = hits / max(hits + miss, 1); far = fa / max(hits + fa, 1)
    tot = hits + fa + cn + miss
    ph = (hits + fa) / max(tot, 1); po = (hits + miss) / max(tot, 1)
    exp = ph * po * tot
    ets = (hits - exp) / max(hits + fa + miss - exp, 1)
    area = (hits + fa) / max(hits + miss, 1)
    return pod, far, ets, area

out = {}
print('===== Table 9 (GPM 分类, 阈值 1/3/10/20) =====')
o = gpm3; g = gfs[sel]; a = apc[sel]
for th in [1, 3, 10, 20]:
    pg, fg, eg, _ = cat(o, g, th)
    pa, fa_, ea, _ = cat(o, a, th)
    out[f'gpm_{th}'] = {'gfs': [pg, fg, eg], 'apc': [pa, fa_, ea]}
    print(f">={th:>2}: POD {pg:.3f}/{pa:.3f} FAR {fg:.3f}/{fa_:.3f} ETS {eg:.3f}/{ea:.3f}")
print('===== Table 10 (ERA5 分类, 阈值 0.1/3/10/20) =====')
for th in [0.1, 3, 10, 20]:
    pg, fg, eg, ag = cat(tgt, gfs, th)
    pa, fa_, ea, aa = cat(tgt, apc, th)
    out[f'era5_{th}'] = {'gfs': [pg, fg, eg, ag], 'apc': [pa, fa_, ea, aa]}
    print(f">={th:>2}: POD {pg:.3f}/{pa:.3f} FAR {fg:.3f}/{fa_:.3f} ETS {eg:.3f}/{ea:.3f} area {ag:.2f}/{aa:.2f}")
json.dump(out, open(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\run13_cat.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved')
