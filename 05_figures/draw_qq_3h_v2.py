# -*- coding: utf-8 -*-
"""draw_qq_3h_v2.py — 3h QQ 图，口径与 Table 8 完全一致。
观测 = gpm3h_obs.npy (2475 x 25 x 37, 925 全格点, gpm3h_eval.py 产物)
方法 = GFS/QM/OLS/U-Net (manuscript_work) + APCNet(SYM42) + OLS(GPM目标, init 对齐)
"""
import os, pickle
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
SYM42 = os.path.join(WORK, '..', 'manuscript_work_sym', 'seed42', 'predictions_apcnet.npy')
E = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\3h_exp'
MEDIA = r'C:\Users\yg181\Desktop\论文三\WAF\r3_media'

gpm3 = np.load(os.path.join(WORK, 'gpm3h_obs.npy'))
gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
ols = np.load(os.path.join(WORK, 'predictions_ols.npy'))
unet = np.load(os.path.join(WORK, '..', 'manuscript_work_unet', 'predictions_apcnet.npy'))
apc = np.load(SYM42)
print('gpm3h_obs', gpm3.shape, 'gfs', gfs.shape, 'apc', apc.shape)

# 3h 数据集行序的 init 字符串
init3h = np.load(os.path.join(E, 'init3h_test.npy'))
pred_ols_gpm = np.load(os.path.join(E, 'pred_ols_gpm3h_test.npy'))
print('pred_ols_gpm', pred_ols_gpm.shape, 'init3h', len(init3h))

# 2475 样本的 t0 -> init（复刻 gpm3h_bin_eval 行序：hour 过滤 + GPM 文件齐全）
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)
GPM_ROOT = r'D:\liaohe\GPM_IMERG'
WINDOW_END_HOURS = [3, 9, 15, 21]
utc = __import__('datetime').timezone.utc
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
print('重建 vt:', len(vt), '== gpm3h_obs 行数:', len(vt) == len(gpm3))
assert len(vt) == len(gpm3)

# gpm3h_eval 的 sel 索引（在 times 中的位置）
idx_map = {}
for i, t in enumerate(times):
    tt = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    idx_map[tt] = i
sel = [idx_map[t] for t in vt]

# OLS(GPM) 对齐：init = t0 - 3h
init_main = np.array([t - timedelta(hours=3) for t in vt])
init_main_s = np.array([t.strftime('%Y%m%d%H') for t in init_main])
new_set = set(init3h)
idx_new = np.array([int(np.where(init3h == s)[0][0]) for s in init_main_s if s in new_set])
assert len(idx_new) == len(gpm3), f'对齐失败 {len(idx_new)} vs {len(gpm3)}'

methods = {
    'GFS': gfs[sel],
    'APCNet': apc[sel],
    'QM': qm[sel],
    'OLS (ERA5 target)': ols[sel],
    'U-Net (ERA5 target)': unet[sel],
    'OLS (GPM target)': pred_ols_gpm[idx_new],
}
T = gpm3
G = gfs[sel]

def cont(o, f):
    o, f = o.flatten(), f.flatten()
    v = np.isfinite(o) & np.isfinite(f)
    o, f = o[v], f[v]
    mse = np.mean((o - f) ** 2)
    return mse, float(np.sqrt(mse)), float(np.corrcoef(o, f)[0, 1]), float(np.mean(f - o))

mg = cont(T, G)
print('\n=== 3h GPM 验证（925 格点, n=%d, 与 Table 8 同口径）===' % len(sel))
print('%-22s %8s %8s %8s %8s' % ('method', 'RMSE', 'CC', 'bias', 'MSEimp%'))
for k, a in methods.items():
    mse, rmse, cc, bias = cont(T, a)
    print('%-22s %8.3f %8.3f %8.3f %+7.1f' % (k, rmse, cc, bias, 100.0 * (mg[0] - mse) / mg[0]))

# QQ 图（0-20mm）
qs = np.linspace(0.001, 0.999, 200)
obs_q = np.quantile(T, qs)
colors = {'GFS': '#555555', 'APCNet': '#E69F00', 'QM': '#D55E00',
          'OLS (ERA5 target)': '#CC79A7', 'U-Net (ERA5 target)': '#0072B2',
          'OLS (GPM target)': '#009E73'}
fig, ax = plt.subplots(figsize=(5.2, 5.2))
for k, a in methods.items():
    ax.plot(obs_q, np.quantile(a, qs), color=colors[k], lw=1.6, label=k)
ax.plot(obs_q, obs_q, 'k--', lw=1.0, label='1:1')
ax.set_xlabel('GPM observed 3-h accumulation (mm)')
ax.set_ylabel('Predicted 3-h accumulation (mm)')
ax.set_title('Quantile-quantile, 3-h test period (2024-2025)')
ax.set_xlim(0, 20); ax.set_ylim(0, 20)
ax.legend(frameon=False, fontsize=7.5, loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(MEDIA, 'fig_qq_3h.pdf'))
fig.savefig(os.path.join(MEDIA, 'fig_qq_3h.png'), dpi=300)
print('✅ 已保存 fig_qq_3h.pdf/.png')
