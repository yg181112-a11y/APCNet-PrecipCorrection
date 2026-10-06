# -*- coding: utf-8 -*-
"""Fig10 可靠性+锐度（手稿编号）：修复 1:1 图例 / N 标注 / 0.9-1.0 标签"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 7.5, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_APC = '#D55E00'
C_GFS = '#0072B2'

OUT = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
prob = json.load(open(os.path.join(OUT, 'gpm_prob_eval.json')))
rel = prob['reliability']
bins_x = [r['avg_prob'] for r in rel]
bins_y = [r['avg_obs'] for r in rel]
counts = [r['count'] for r in rel]
tot = sum(counts)

fig = plt.figure(figsize=(6.9, 3.0))
gs = GridSpec(1, 2, width_ratios=[1.1, 1.0], wspace=0.32, left=0.09, right=0.985, bottom=0.36, top=0.90)
ax = fig.add_subplot(gs[0])
ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfect reliability')
ax.plot(bins_x, bins_y, 'o-', color=C_APC, ms=4, lw=1.2, label='APCNet occurrence')
ax.set_xlabel('Forecast probability (wet)')
ax.set_ylabel('Observed frequency')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title('Reliability diagram (0.1 mm threshold)', fontsize=9)
ax.legend(frameon=False, loc='upper left', fontsize=6.8)
ax.text(0.02, 0.05, 'N = %s grid-point samples' % f'{tot:,}',
        transform=ax.transAxes, fontsize=6.8, va='bottom', ha='left', color='0.25')
ax.text(0.5, -0.34, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='center')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax = fig.add_subplot(gs[1])
ax.bar(np.arange(len(counts)), [100.0 * c / tot for c in counts], color=C_GFS, alpha=0.85,
       edgecolor='k', linewidth=0.4, width=0.75)
ax.set_xticks(np.arange(len(counts)))
ax.set_xticklabels(['0.0–0.1', '0.1–0.2', '0.2–0.3', '0.3–0.4', '0.4–0.5',
                    '0.5–0.6', '0.6–0.7', '0.7–0.8', '0.8–0.9', '0.9–1.0'], rotation=45, ha='right', fontsize=5.5)
ax.set_xlabel('Forecast probability bin')
ax.set_ylabel('Share of grid-point samples (%)')
ax.set_title('Sharpness: P(wet) distribution', fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.text(0.5, -0.34, '(b)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='center')

fig.suptitle('Probabilistic calibration of APCNet precipitation occurrence (GPM verification)', fontsize=10, y=0.99)
out = r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\figures_300dpi\Fig10.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
print('saved', out, 'N =', tot)
