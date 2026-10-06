# -*- coding: utf-8 -*-
"""Fig.9 日循环 run13：GPM 验证样本按 03/09/15/21Z 域均率重算并重绘。"""
import os, pickle, json
import numpy as np
from datetime import timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
})

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
vts = [(times[i].replace(tzinfo=utc) if times[i].tzinfo is None else times[i].astimezone(utc)) for i in sel]
hours = np.array([t.hour for t in vts])

G, O, Q, L, A, U = gfs[sel], obs, qm[sel], ols[sel], apc[sel], unet[sel]

def dom_mean(arr, h):
    m = hours == h
    return float(np.nanmean(arr[m]))

data = {}
for label, arr in [('GPM IMERG (obs)', O), ('GFS', G), ('QM', Q), ('OLS', L), ('APCNet', A), ('U-Net', U)]:
    data[label] = [dom_mean(arr, h) for h in WINDOW_END_HOURS]
print(json.dumps(data, indent=1))
with open(os.path.join(WORK, 'diurnal_run13.json'), 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)

colors = {'GPM IMERG (obs)': 'k', 'GFS': '#333333', 'QM': '#E69F00', 'OLS': '#0072B2', 'APCNet': '#D55E00', 'U-Net': '#CC79A7'}
styles = {'GPM IMERG (obs)': '-o', 'GFS': '--s', 'QM': '--^', 'OLS': '--D', 'APCNet': '-s', 'U-Net': '--v'}
fig, ax = plt.subplots(figsize=(7.2, 4.6))
x = list(range(4))
for name, y in data.items():
    ax.plot(x, y, styles[name], color=colors[name], label=name,
            lw=2 if name in ('GPM IMERG (obs)', 'APCNet') else 1.5,
            ms=6 if name in ('GPM IMERG (obs)', 'APCNet') else 4)
ax.set_xticks(x); ax.set_xticklabels(['03Z', '09Z', '15Z', '21Z'])
ax.set_xlabel('Valid time (UTC)', fontsize=11)
ax.set_ylabel('Domain-mean precipitation rate (mm/3h)', fontsize=11)
ax.set_title('Diurnal cycle over the GPM-verification samples (2024-2025)', fontsize=12)
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.set_ylim(0, 0.72)
ax.text(2.0, 0.15, 'DL networks overestimate by ~2x vs GPM', fontsize=8.5, color='0.35', ha='center',
        bbox=dict(fc='white', ec='none', alpha=0.85))
ax.grid(alpha=0.25)
fig.tight_layout()
out = r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\figures_300dpi\Fig11.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
print('saved', out)
