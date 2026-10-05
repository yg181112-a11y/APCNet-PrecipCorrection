# -*- coding: utf-8 -*-
"""尺度反转归因图：3h vs 24h 的条件偏差结构、信噪比、DL 行为"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, json, os

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.labelsize': 10,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'axes.linewidth': 0.8})

d = json.load(open(r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work\scale_attribution_diag.json'))
bins = [0, 0.1, 1, 3, 5, 10, 20, 50, 1e9]
labels = ['0-0.1', '0.1-1', '1-3', '3-5', '5-10', '10-20', '20-50', '50+']

fig = plt.figure(figsize=(10, 3.6))
gs = fig.add_gridspec(1, 3, wspace=0.35)

# Panel A: 条件偏差结构 E[resid | GFS bin]
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(labels))
for tag, color, mk in [('3h', '#C0392B', 'o'), ('24h', '#1A7F37', 's')]:
    ce = d[tag]['cond_exp']
    ax.plot(x, ce, color=color, marker=mk, ms=5, lw=1.6, label=f'{tag}')
ax.axhline(0, color='0.5', lw=0.8, ls='--')
ax.set_yscale('symlog', linthresh=0.1)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_xlabel('GFS precipitation bin (mm per period)')
ax.set_ylabel('E[ERA5 − GFS | GFS bin]  (mm)')
ax.set_title('(a) Conditional residual structure', fontsize=10)
ax.legend(frameon=False, loc='upper right')

# Panel B: 信噪比、偏差、CC 改进
ax = fig.add_subplot(gs[0, 1])
vars_ = ['SNR$^{-1}$\n(noise/signal)', 'APC bias\n(% of mean)', 'CC gain\n(APC−GFS)']
vals_3 = [d['3h']['snr_inv'], d['3h']['bias_apc'] / d['3h']['mean_target'] * 100, (d['3h']['cc_apc'] - d['3h']['cc_gfs']) * 100]
vals_24 = [d['24h']['snr_inv'], d['24h']['bias_apc'] / d['24h']['mean_target'] * 100, (d['24h']['cc_apc'] - d['24h']['cc_gfs']) * 100]
x = np.arange(3); w = 0.32
ax.bar(x - w / 2, vals_3, w, color='#C0392B', label='3h')
ax.bar(x + w / 2, vals_24, w, color='#1A7F37', label='24h')
ax.axhline(0, color='0.5', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(vars_, fontsize=8)
ax.set_title('(b) Noise & correction behaviour', fontsize=10)
ax.legend(frameon=False)
for xi, (v3, v24) in enumerate(zip(vals_3, vals_24)):
    ax.text(xi - w / 2, v3 + 0.15 * abs(v3) + 0.1, f'{v3:.1f}', ha='center', fontsize=7.5)
    ax.text(xi + w / 2, v24 + 0.15 * abs(v24) + 0.1, f'{v24:.1f}', ha='center', fontsize=7.5)

# Panel C: 强降水档 GFS vs ERA5（t20_mean / p20_mean）
ax = fig.add_subplot(gs[0, 2])
tags = ['3h', '24h']
for i, tag in enumerate(tags):
    t20 = d[tag]['t20_mean']; p20 = d[tag]['p20_mean']; g_ = 20.0
    ax.bar([i * 3 + 0], [g_], 0.7, color='0.65', label='GFS (bin floor)' if i == 0 else None)
    ax.bar([i * 3 + 1], [t20], 0.7, color='#2E86C1', label='ERA5 mean' if i == 0 else None)
    ax.bar([i * 3 + 2], [p20], 0.7, color='#8E44AD', label='APCNet mean' if i == 0 else None)
    ax.text(i * 3 + 0, g_ + 1, f'{g_:.0f}', ha='center', fontsize=8)
    ax.text(i * 3 + 1, t20 + 1, f'{t20:.1f}', ha='center', fontsize=8)
    ax.text(i * 3 + 2, p20 + 1, f'{p20:.1f}', ha='center', fontsize=8)
ax.set_xticks([1, 4]); ax.set_xticklabels(['3h', '24h'])
ax.set_ylabel('Precip. when GFS ≥ 20 mm (mm)')
ax.set_title('(c) Heavy-precip. behaviour', fontsize=10)
ax.legend(frameon=False, fontsize=7)

plt.tight_layout()
os.makedirs(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\r3_media', exist_ok=True)
fig.savefig(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\r3_media\fig_scale_attribution.pdf')
fig.savefig(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\r3_media\fig_scale_attribution.png', dpi=300)
print('saved r3_media/fig_scale_attribution.pdf/.png')
