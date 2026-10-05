# -*- coding: utf-8 -*-
"""ETS20 delta block bootstrap（ERA5 + GPM，run13 S42）。"""
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

def ets20(o, f):
    v = np.isfinite(o) & np.isfinite(f)
    o, f = o[v], f[v]
    a = o >= 20; b = f >= 20
    hits = np.sum(a & b); fa = np.sum(~a & b); miss = np.sum(a & ~b); cn = np.sum(~a & ~b)
    tot = hits + fa + cn + miss
    ph = (hits + fa) / tot; po = (hits + miss) / tot
    exp = ph * po * tot
    return (hits - exp) / max(hits + fa + miss - exp, 1)

def block_boot_delta(o, g, a, bl, n_boot=2000, seed=2026):
    n = len(o)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / bl))
    bid = np.minimum(np.arange(n) // bl, nb - 1)
    dist = np.empty(n_boot)
    for b in range(n_boot):
        blocks = rng.choice(nb, size=nb, replace=True)
        s = np.concatenate([np.where(bid == m)[0] for m in blocks])
        e_g = ets20(o[s], g[s]); e_a = ets20(o[s], a[s])
        dist[b] = e_a - e_g
    return dist, nb

out = {}
for ref, o, g, a, n in [('ERA5', tgt, gfs, apc, len(tgt)), ('GPM', gpm3, gfs[sel], apc[sel], len(sel))]:
    d_obs = ets20(o, a) - ets20(o, g)
    print(f'{ref}: ETS20 GFS={ets20(o,g):.4f} APC={ets20(o,a):.4f} delta={d_obs:+.4f}')
    for bl in ([16] if ref == 'GPM' else [28]):
        dist, nb = block_boot_delta(o, g, a, bl)
        p = float(np.mean(dist > 0))
        ci = [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))]
        out[f'{ref}_bl{bl}'] = {'delta_obs': float(d_obs), 'p_pos': p, 'ci95': ci, 'n_blocks': nb}
        print(f'  bl={bl}: 95%CI=[{ci[0]:+.4f},{ci[1]:+.4f}] P(delta>0)={p:.3f}')
json.dump(out, open(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\run13_ets_boot.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved')
