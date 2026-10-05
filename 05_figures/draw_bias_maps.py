# -*- coding: utf-8 -*-
"""draw_bias_maps.py — 多年平均偏差空间分布图（审稿人点名图）。
行 = 尺度（3h / 24h），列 = GFS−ERA5 / APCNet−ERA5 / QM−ERA5
显示：多年平均域偏差场（mm），右上角标注域均值偏差；RdBu_r 发散配色。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np, os

OUTD = r'C:\Users\yg181\Desktop\论文三\WAF\r3_media'
os.makedirs(OUTD, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
})
LATS = np.linspace(46.0, 40.0, 25)   # 降序
LONS = np.linspace(117.0, 126.0, 37)

# ---- 3h 数据（测试期 2024-2025, 2839 样本）----
D3 = r'D:\liaohe\校正优化过程\第三阶段\12优化\12.8修\output\predictions'
g3 = np.load(os.path.join(D3, 'gfs_test.npy'))
t3 = np.load(os.path.join(D3, 'targets_test.npy'))
a3 = np.load(os.path.join(D3, 'predictions_apcnet.npy'))
q3 = np.load(os.path.join(D3, 'predictions_qm.npy'))
print('3h 形状:', g3.shape, '域均 GFS', g3.mean(), 'ERA5', t3.mean())

# ---- 24h 数据（测试期 2024-2025 有效样本）----
E = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\24h_exp'
g24_all = np.load(os.path.join(E, 'gfs_24h_accum.npy'))
e24_all = np.load(os.path.join(E, 'era5_24h_accum.npy'))
split = np.load(os.path.join(E, 'split_mask.npy'))
valid = ~(np.isnan(g24_all).any((1,2)) | np.isnan(e24_all).any((1,2)))
te = (split == 2) & valid
g24 = g24_all[te]; e24 = e24_all[te]
a24 = np.load(os.path.join(E, 'pred_apcnet_24h.npy'))
q24 = np.load(os.path.join(E, 'qm_24h.npy'))
print('24h 形状:', g24.shape, '域均 GFS', g24.mean(), 'ERA5', e24.mean())

# ---- 多年平均偏差场 ----
def mean_bias(pred, target):
    return np.nanmean(pred - target, axis=0)   # [25,37]

rows = [
    ('3-h scale', [
        ('GFS − ERA5', mean_bias(g3, t3)),
        ('APCNet − ERA5', mean_bias(a3, t3)),
        ('QM − ERA5', mean_bias(q3, t3)),
    ]),
    ('24-h scale', [
        ('GFS − ERA5', mean_bias(g24, e24)),
        ('APCNet − ERA5', mean_bias(a24, e24)),
        ('QM − ERA5', mean_bias(q24, e24)),
    ]),
]

# 统一对称色限（取最大绝对值）
vmax = max(np.nanmax(np.abs(b)) for _, cols in rows for _, b in cols)
print('vmax =', vmax)

fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.6),
                         gridspec_kw={'hspace': 0.28, 'wspace': 0.12})
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
cmap = plt.get_cmap('RdBu_r')

for ri, (row_title, cols) in enumerate(rows):
    for ci, (name, bias) in enumerate(cols):
        ax = axes[ri, ci]
        im = ax.imshow(bias, cmap=cmap, norm=norm, origin='upper',
                       extent=[LONS[0], LONS[-1], LATS[-1], LATS[0]], aspect='auto')
        ax.set_title(name, fontsize=9)
        ax.set_ylabel('Lat (°N)' if ci == 0 else '')
        ax.set_xlabel('Lon (°E)' if ri == 1 else '')
        ax.set_xticks([118, 120, 122, 124, 126])
        ax.set_yticks([41, 43, 45])
        if ri == 1:
            ax.set_xticklabels(['118', '120', '122', '124', '126'])
        else:
            ax.set_xticklabels([])
        if ci == 0:
            ax.set_yticklabels(['41', '43', '45'])
        else:
            ax.set_yticklabels([])
        dm = np.nanmean(bias)
        ax.text(0.02, 0.02, f'mean {dm:+.3f} mm', transform=ax.transAxes,
                fontsize=7.5, color='black', ha='left', va='bottom',
                bbox=dict(fc='white', ec='0.5', alpha=0.85, lw=0.5, pad=1.5))
        # 域界（模拟 Liaohe 域）
        ax.add_patch(plt.Rectangle((117.0, 40.0), 9.0, 6.0, fill=False,
                                   ec='0.4', lw=0.6, ls='--'))

# 行标签
axes[0, 0].annotate('3-h scale', xy=(-0.32, 0.5), xycoords='axes fraction',
                    rotation=90, ha='center', va='center', fontsize=10, fontweight='bold')
axes[1, 0].annotate('24-h scale', xy=(-0.32, 0.5), xycoords='axes fraction',
                    rotation=90, ha='center', va='center', fontsize=10, fontweight='bold')

cbar_ax = fig.add_axes([0.93, 0.12, 0.025, 0.76])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('Mean bias vs ERA5 (mm)', fontsize=9)

fig.savefig(os.path.join(OUTD, 'fig_bias_maps.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTD, 'fig_bias_maps.png'), dpi=300, bbox_inches='tight')
print('✅ 已保存 fig_bias_maps.pdf/.png')
print('域均值: 3h GFS', np.nanmean(g3-t3), 'APCNet', np.nanmean(a3-t3), 'QM', np.nanmean(q3-t3))
print('24h GFS', np.nanmean(g24-e24), 'APCNet', np.nanmean(a24-e24), 'QM', np.nanmean(q24-e24))
