# -*- coding: utf-8 -*-
"""season_3h_24h.py — 3h/24h 季节分解表（warm JJAS vs cool）。
3h: 2475 样本 925 格点（Table 8 口径）；24h: 00Z 测试期 828 格点。
输出 JSON + 控制台表格。
"""
import os, pickle, json
import numpy as np
from datetime import datetime, timedelta

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
SYM42 = os.path.join(WORK, '..', 'manuscript_work_sym', 'seed42', 'predictions_apcnet.npy')
UNET = os.path.join(WORK, '..', 'manuscript_work_unet', 'predictions_apcnet.npy')
E3 = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\3h_exp'
E24 = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\24h_exp'
GPM_ROOT = r'D:\liaohe\GPM_IMERG'

def cont(o, f):
    o, f = o.flatten(), f.flatten()
    v = np.isfinite(o) & np.isfinite(f)
    o, f = o[v], f[v]
    return np.mean((o - f) ** 2), float(np.sqrt(np.mean((o - f) ** 2)))

# ---------------- 3h ----------------
gpm3 = np.load(os.path.join(WORK, 'gpm3h_obs.npy'))
gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
ols = np.load(os.path.join(WORK, 'predictions_ols.npy'))
apc = np.load(SYM42)
unet = np.load(UNET)
init3h = np.load(os.path.join(E3, 'init3h_test.npy'))
pred_ols_gpm = np.load(os.path.join(E3, 'pred_ols_gpm3h_test.npy'))

with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)
utc = __import__('datetime').timezone.utc
WINDOW_END_HOURS = [3, 9, 15, 21]
vt = []
for t in times:
    tt = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    if tt.hour not in WINDOW_END_HOURS:
        continue
    fs = []
    for i in range(6):
        e = tt - timedelta(minutes=30 * (5 - i))
        fs.append(os.path.join(GPM_ROOT, f'imerg_{e.year}{e.month:02d}',
                               f'imerg_{e.year}{e.month:02d}{e.day:02d}_{e.hour:02d}{e.minute:02d}00.nc4'))
    if any(not os.path.exists(f) for f in fs):
        continue
    vt.append(tt)
vt = np.array(vt)
idx = {t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc): i for i, t in enumerate(times)}
sel = np.array([idx[t] for t in vt])
init_s = np.array([(t - timedelta(hours=3)).strftime('%Y%m%d%H') for t in vt])
idx_new = np.array([int(np.where(init3h == s)[0][0]) for s in init_s])

methods3 = {'GFS': gfs[sel], 'APCNet': apc[sel], 'QM': qm[sel], 'OLS-ERA5': ols[sel],
            'U-Net-ERA5': unet[sel], 'OLS-GPM': pred_ols_gpm[idx_new]}
months3 = np.array([t.month for t in vt])
warm3 = np.isin(months3, [6, 7, 8, 9])

print('=== 3h 季节分解（925 格点, 2475 样本）===')
out = {'3h': {}}
for nm, msk in [('warm_JJAS', warm3), ('cool', ~warm3)]:
    T, n = gpm3[msk], int(msk.sum())
    m_g = cont(T, gfs[sel][msk])[0]
    row = {'n_samples': n}
    parts = [f'{nm} n={n}: GFS RMSE={np.sqrt(m_g):.2f}']
    for k, a in methods3.items():
        mse, rmse = cont(T, a[msk])
        row[k] = {'RMSE': rmse, 'mse_improve_pct': 100.0 * (m_g - mse) / m_g}
        parts.append(f'{k} {100.0*(m_g-mse)/m_g:+.1f}%')
    print('  ' + ' | '.join(parts))
    out['3h'][nm] = row

# ---------------- 24h（00Z，828 格点，从已有 gpm_24h_eval.json 的 seasonal）----------------
d24 = json.load(open(os.path.join(E24, 'gpm_24h_eval.json'), encoding='utf-8'))
out['24h'] = {}
for nm in ['warm_JJAS', 'cool_others']:
    s = d24['seasonal'][nm]
    row = {'n_samples': s['n_samples']}
    m_g = s['cont_GFS']['MSE']
    for k in ['APCNet', 'U-Net', 'QM']:
        key = f'cont_{k}'
        if key in s:
            mse = s[key]['MSE']
            row[k] = {'RMSE': s[key]['RMSE'], 'mse_improve_pct': 100.0 * (m_g - mse) / m_g}
    out['24h'][nm] = row
print('\n24h 季节（来自 gpm_24h_eval.json seasonal）:')
for nm, row in out['24h'].items():
    print(' ', nm, row)

def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o
json.dump(clean(out), open(os.path.join(E24, 'seasonal_table.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('✅ 已保存 seasonal_table.json')
