# -*- coding: utf-8 -*-
"""Fig.2 数据修复证据：2024-07 域均值比对。
(a) 修复前：GFS(3h) vs ERA5 旧(1h 抽稀) —— ~3 倍错位
(b) 修复后：GFS(3h) vs ERA5 新(3h 重建) —— 重叠
(c) 散点+月总量：修复后相关 + 月度总量柱状
"""
import sys, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, r'C:\Users\yg181\Desktop\论文三\13.0修复重跑')
from verify_era5_target import load_era5_monthly, load_gfs_f003

GFS = r'D:/liaohe/GFS-data/gfs.0p25.2015-2025.f003'
NEW = r'D:\liaohe\ERA5-data\monthly'
OLD = r'D:\liaohe\ERA5-data\monthly_old_b1'
FIG = r'D:\liaohe\校正优化过程\第三阶段\12优化\fig_p3'
os.makedirs(FIG, exist_ok=True)

Y, M = 2024, 7
t_g, g = load_gfs_f003(GFS, Y, M)
t_e_new, e_new = load_era5_monthly(NEW, Y, M)
t_e_old, e_old = load_era5_monthly(OLD, Y, M)

# 对齐共同时次
g_t = {t.timestamp(): v for t, v in zip(t_g, g)}
e_n = {t.timestamp(): float(np.mean(v)) for t, v in zip(t_e_new, e_new)}
e_o = {t.timestamp(): float(np.mean(v)) for t, v in zip(t_e_old, e_old)}
common = sorted(set(g_t) & set(e_n) & set(e_o))
print('共同时次:', len(common))
gv = np.array([g_t[t] for t in common])
env = np.array([e_n[t] for t in common])
eov = np.array([e_o[t] for t in common])

fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.6))
x = np.arange(len(common))
ax = axes[0, 0]
ax.plot(x, gv, lw=1.0, color='#1f77b4', label='GFS f003 (true 3-h)')
ax.plot(x, eov * 3, lw=1.0, color='#d62728', ls='--', label='ERA5 old (1-h) ×3')
ax.plot(x, eov, lw=1.0, color='#d62728', ls=':', label='ERA5 old (1-h) raw')
ax.set_title('(a) Before fix: 1-h target vs 3-h GFS\nratio=%.3f' % (np.mean(gv) / np.mean(eov)))
ax.legend(frameon=False, fontsize=7, loc='upper left')
ax.set_ylabel('mm per window')

ax = axes[0, 1]
ax.plot(x, gv, lw=1.0, color='#1f77b4', label='GFS f003')
ax.plot(x, env, lw=1.0, color='#2ca02c', label='ERA5 new (true 3-h)')
ax.set_title('(b) After fix: aligned 3-h windows\nratio=%.3f  r=%.3f' % (np.mean(gv) / np.mean(env), np.corrcoef(gv, env)[0, 1]))
ax.legend(frameon=False, fontsize=7, loc='upper left')
ax.set_ylabel('mm/3h')

ax = axes[1, 0]
ax.scatter(env, gv, s=6, alpha=0.5, color='#2ca02c')
ax.plot([0, 3], [0, 3], 'k--', lw=0.8)
ax.set_xlabel('ERA5 new (mm/3h)'); ax.set_ylabel('GFS (mm/3h)')
ax.set_title('(c) Scatter (common %d times)\ncorrelation=%.3f' % (len(common), np.corrcoef(gv, env)[0, 1]))
lim = max(gv.max(), env.max()) * 1.05
ax.set_xlim(0, lim); ax.set_ylim(0, lim)

ax = axes[1, 1]
monthly = [np.sum(gv), np.sum(env), np.sum(eov)]
labels = ['GFS f003', 'ERA5 new', 'ERA5 old']
colors = ['#1f77b4', '#2ca02c', '#d62728']
bars = ax.bar(labels, monthly, color=colors, alpha=0.85)
for b, v in zip(bars, monthly):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, '%.1f' % v, ha='center', fontsize=9)
ax.set_ylabel('July 2024 total (mm)')
ax.set_title('(d) Monthly totals\nGFS vs ERA5-new: +%.1f%%' % (100 * (np.sum(gv) - np.sum(env)) / np.sum(env)))
fig.suptitle('Target-correction evidence: July 2024, domain mean (Liaohe 25x37)', y=1.0, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig2_target_evidence.png'), bbox_inches='tight', dpi=300)
print('✅ fig2_target_evidence.png')
