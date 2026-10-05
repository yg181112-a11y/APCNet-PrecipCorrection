# -*- coding: utf-8 -*-
"""事件级验证：GPM 3h 尺度 5 个最强夏季强降水事件，逐事件比较各模型。"""
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
unet = np.load(os.path.join(WORK, '..', 'manuscript_work_unet', 'predictions_apcnet.npy'))
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)
sel = np.array([i for i, t in enumerate(times)
                if ((t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)).hour in WINDOW_END_HOURS)])
sel = sel[:len(obs)]
vt = [(times[i].replace(tzinfo=utc) if times[i].tzinfo is None else times[i].astimezone(utc)) for i in sel]

G, O, Q, L, A, U = gfs[sel], obs, qm[sel], ols[sel], apc[sel], unet[sel]
N = len(O)
print(f'samples {N}, obs shape {O.shape}')

# ---- 事件检测：JJA 域均 >= 0.8 mm/3h，连续 >= 3 窗 ----
dm = O.mean(axis=(1, 2))
mon = np.array([t.month for t in vt])
summer = (mon >= 6) & (mon <= 8)
cand = np.where(summer & (dm >= 0.8))[0]
runs, cur = [], [cand[0]] if len(cand) else []
for k in range(1, len(cand)):
    if cand[k] == cand[k-1] + 1:
        cur.append(cand[k])
    else:
        if len(cur) >= 3:
            runs.append(cur)
        cur = [cand[k]]
if len(cur) >= 3:
    runs.append(cur)
runs.sort(key=lambda r: float(dm[r].max()), reverse=True)
print('runs:', len(runs))
top5 = runs[:5]

methods = {'GFS': G, 'APCNet': A, 'U-Net': U, 'QM': Q, 'OLS': L}

out = []
for k, run in enumerate(top5):
    pk = int(run[np.argmax(dm[run])])
    r0, r1 = max(0, pk - 2), min(N, pk + 3)   # 事件窗 ±2 窗
    ev = {}
    ev['event'] = k + 1
    ev['peak_time'] = vt[pk].strftime('%Y-%m-%d %HZ')
    ev['n_windows'] = int(r1 - r0)
    ev['obs_peak_dm'] = float(dm[pk])
    ev['obs_event_total'] = float(O[r0:r1].mean())
    ev['obs_peak_grid'] = float(O[pk].max())
    ev['models'] = {}
    for nm, arr in methods.items():
        m = arr[r0:r1]
        ev['models'][nm] = {
            'peak_dm': float(arr[pk].mean()),
            'event_total': float(m.mean()),
            'peak_grid': float(arr[pk].max()),
            'spatial_cc': float(np.corrcoef(O[pk].ravel(), arr[pk].ravel())[0, 1]) if (np.std(O[pk]) > 1e-6 and np.std(arr[pk]) > 1e-6) else float('nan'),
            'total_bias_pct': 100 * (m.mean() - O[r0:r1].mean()) / max(O[r0:r1].mean(), 1e-9),
        }
    out.append(ev)
    print(f'Ev{k+1}: {ev["peak_time"]} peak_obs={ev["obs_peak_dm"]:.2f} total={ev["obs_event_total"]:.2f} max_grid={ev["obs_peak_grid"]:.1f}')

json.dump(out, open(os.path.join(WORK, 'event_verification_gpm3h.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=float)
print('saved event_verification_gpm3h.json')
