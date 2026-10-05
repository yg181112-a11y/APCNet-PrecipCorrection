# -*- coding: utf-8 -*-
"""Fig.8：GPM 验证样本的域均降水率日循环（4 个 UTC 时次）。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hours = ['03Z', '09Z', '15Z', '21Z']
data = {
    'GPM IMERG (obs)': [0.2102, 0.2607, 0.2336, 0.2781],
    'GFS':             [0.2349, 0.2833, 0.2457, 0.2608],
    'QM':              [0.2228, 0.2719, 0.2390, 0.2498],
    'OLS':             [0.2258, 0.2546, 0.2319, 0.2427],
    'APCNet':          [0.5556, 0.6171, 0.5765, 0.5800],
    'U-Net':           [0.4446, 0.5397, 0.4497, 0.4456],
}
colors = {'GPM IMERG (obs)': 'k', 'GFS': '#1f77b4', 'QM': '#2ca02c',
          'OLS': '#ff7f0e', 'APCNet': '#d62728', 'U-Net': '#9467bd'}
styles = {'GPM IMERG (obs)': '-o', 'GFS': '--s', 'QM': '--^', 'OLS': '--D',
          'APCNet': '-o', 'U-Net': '--v'}

fig, ax = plt.subplots(figsize=(7.2, 4.6))
x = list(range(4))
for name, y in data.items():
    ax.plot(x, y, styles[name], color=colors[name], label=name, lw=2 if name in ('GPM IMERG (obs)', 'APCNet') else 1.5,
            ms=6 if name in ('GPM IMERG (obs)', 'APCNet') else 4)
ax.set_xticks(x)
ax.set_xticklabels(hours)
ax.set_xlabel('Valid time (UTC)', fontsize=11)
ax.set_ylabel('Domain-mean precipitation rate (mm/3h)', fontsize=11)
ax.set_title('Diurnal cycle over the GPM-verification samples (2024-2025)', fontsize=12)
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.set_ylim(0, 0.72)
ax.grid(alpha=0.25)
fig.tight_layout()
out = r"D:\liaohe\校正优化过程\第三阶段\12优化\fig_p3\fig8_diurnal.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
