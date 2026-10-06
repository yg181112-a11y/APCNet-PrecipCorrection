# -*- coding: utf-8 -*-
"""Fig06 重绘（T1+F3+M1视觉化）：
- (a) S42 复合损失 vs GFS（新数字 −3.2/−28.5/−9.5，真实任务区间带 −82.8~−14.5）
- (b) vs zero-correction：复合(S42 实线) + 纯MSE(S42 虚线) → 可视化 loss design 决定量级
- 标题改为中性归因（F3）
- wspace 增大避免 y 轴标签压图（F3）
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
CTL = r'D:\liaohe\校正优化过程\第三阶段\12优化\controlled_exp'
FIG = r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\figures_300dpi'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 7.5,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
C_APC = '#D55E00'
C_MSE = '#0072B2'


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def panel_label(ax, label):
    ax.text(0.5, -0.34, label, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top', ha='center')


# ---- 数据（表4口径，S42）----
sigmas = [0, 0.5, 1.16, 2.0]
imp_full = [0.0, -3.17, -28.54, -9.47]        # 复合损失 S42（表4: −3.2/−28.5/−9.5）
sig2 = [0.5, 1.16, 2.0]
imp_full2 = [-3.17, -28.54, -9.47]
imp_mse2 = [-2.44, -1.18, -0.79]              # 纯MSE S42（表4: −2.4/−1.2/−0.8）
cc_id = 0.9984

fig = plt.figure(figsize=(7.1, 3.05))
gs = GridSpec(1, 2, width_ratios=[1.1, 1.0], wspace=0.5,
              left=0.095, right=0.97, bottom=0.27, top=0.72)

# (a) vs GFS
ax = fig.add_subplot(gs[0])
ax.plot(sigmas, imp_full, 'o-', color=C_APC, lw=1.5, ms=5,
        label='Controlled experiment (S42)')
ax.axhspan(-82.8, -14.5, color='0.75', alpha=0.28,
           label='Real task (3 seeds, sym)')
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.axvline(1.16, color='0.45', lw=0.8, ls=':', label=r'$\sigma$ = observed residual')
ax.set_xlabel(r'Target noise $\sigma$ (mm/3h)')
ax.set_ylabel('MSE improvement vs GFS (%)')
ax.set_ylim(-95, 8)
ax.set_xticks(sigmas)
ax.legend(frameon=False, fontsize=6.3, loc='lower center', bbox_to_anchor=(0.5, 1.10), ncol=3)
panel_label(ax, '(a)')
despine(ax)

# (b) vs zero-correction：复合 + 纯MSE
ax = fig.add_subplot(gs[1])
ax.plot(sig2, imp_full2, 'o-', color=C_APC, lw=1.5, ms=5, label='Composite loss (S42)')
ax.plot(sig2, imp_mse2, 's--', color=C_MSE, lw=1.5, ms=4.5, label='Pure MSE (S42)')
for s, v in zip(sig2, imp_full2):
    dy = -7.5 if s == 0.5 else 2.2
    ax.text(s, v + dy, '%.1f%%' % v, ha='center', fontsize=7.5)
for s, v in zip(sig2, imp_mse2):
    if s == 0.5:
        continue
    ax.text(s, v + 1.8, '%.1f%%' % v, ha='center', fontsize=7.2, color=C_MSE)
ax.text(0.14, 6, 'identity: CC=%.3f' % cc_id, fontsize=8)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'$\sigma$ (mm/3h)')
ax.set_ylabel('MSE improvement vs zero-correction (%)')
ax.set_ylim(-40, 11)
ax.set_xticks(sigmas)
ax.legend(frameon=False, fontsize=6.3, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2)
panel_label(ax, '(b)')
despine(ax)

fig.suptitle('Controlled experiment: target noise drives degradation, '
             'and the composite loss sets its magnitude',
             fontsize=9.6, y=0.99)
fig.savefig(os.path.join(FIG, 'Fig06_new.png'), dpi=300, bbox_inches='tight')
print('saved Fig06_new.png')
