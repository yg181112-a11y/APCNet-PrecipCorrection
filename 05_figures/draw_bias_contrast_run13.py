# -*- coding: utf-8 -*-
"""A4：偏差场直接对比图（GFS-ERA5 vs APCNet-ERA5 双列 + 差场 + 逐格点散点）。
v5 修复：
- (a)(b)(c)(d) 面板编号全部移到对应小图下方（图注已按 abcd 描述）；
- colorbar 改为底部横向，不再跨第一行右侧压住右上图 x 轴标签；
- 四图统一去掉 set_aspect，2x2 网格等分，四图尺寸均匀。"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
})
from matplotlib.colors import TwoSlopeNorm

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
MEDIA = r'C:\Users\yg181\Desktop\论文三\WAF\r3_media'
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)

gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
tgt = np.load(os.path.join(WORK, 'targets_test.npy'))
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
apc = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
mask = np.load(os.path.join(WORK, 'bias_field_rainmask.npy')).astype(bool)

bg = np.nanmean(gfs - tgt, axis=0)
ba = np.nanmean(apc - tgt, axis=0)
bq = np.nanmean(qm - tgt, axis=0)
bd = np.nanmean(apc - gfs, axis=0)   # 模型对输入的净修改

print(f'GFS bias {bg.mean():+.3f} std {np.nanstd(bg[mask]):.3f}')
print(f'QM  bias {bq.mean():+.3f} std {np.nanstd(bq[mask]):.3f}')
print(f'APC bias {ba.mean():+.3f} std {np.nanstd(ba[mask]):.3f}')
print(f'APC-GFS delta {bd.mean():+.3f} std {np.nanstd(bd[mask]):.3f}')

vmax = float(np.ceil(max(np.nanmax(np.abs(bg[mask])), np.nanmax(np.abs(ba[mask])),
                         np.nanmax(np.abs(bq[mask])), np.nanmax(np.abs(bd[mask]))) * 10) / 10)

def panel_label(ax, label):
    ax.text(0.5, -0.15, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='center')

fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.8),
                         gridspec_kw={'left': 0.07, 'right': 0.97, 'top': 0.93,
                                      'bottom': 0.12, 'hspace': 0.50, 'wspace': 0.14})
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
pans = [(axes[0, 0], bg, r'GFS $-$ ERA5', 'Longitude (°E)', '(a)'),
        (axes[0, 1], ba, r'APCNet $-$ ERA5', 'Longitude (°E)', '(b)'),
        (axes[1, 0], bd, r'APCNet $-$ GFS (net modification)', 'Longitude (°E)', '(c)')]
im = None
for ax, d, t, xl, lab in pans:
    d2 = np.where(mask, d, np.nan)
    im = ax.pcolormesh(GLOBAL_LONS, GLOBAL_LATS, d2, cmap='RdBu_r', norm=norm, shading='auto')
    ax.set_title(t, fontsize=12)
    ax.set_xlabel(xl, fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    panel_label(ax, lab)

# (d) 逐格点散点：GFS 偏差 vs APCNet 偏差（全域均匀缩减 -> 落在对角线上）
ax = axes[1, 1]
xg = bg[mask]
ya = ba[mask]
ax.scatter(xg, ya, s=14, c='#0072B2', alpha=0.65, edgecolors='none')
lim = max(np.nanmax(np.abs(xg)), np.nanmax(np.abs(ya)))
ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1.2, label='1:1 (unchanged structure)')
ok = np.isfinite(xg) & np.isfinite(ya)
b, a = np.polyfit(xg[ok], ya[ok], 1)
xs = np.linspace(-lim, lim, 50)
ax.plot(xs, a + b * xs, color='#D55E00', lw=1.6, label=f'OLS fit (slope {b:.2f})')
ax.axhline(0, color='0.7', lw=0.7)
ax.axvline(0, color='0.7', lw=0.7)
ax.text(0.04, 0.93, f'N = {int(ok.sum())} grid points\n'
        f'GFS bias {np.nanmean(xg):+.3f} / APC bias {np.nanmean(ya):+.3f} mm/3h\n'
        f'corr = {np.corrcoef(xg[ok], ya[ok])[0,1]:.2f}',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(fc='white', ec='0.6', alpha=0.85))
ax.set_xlabel(r'GFS $-$ ERA5 mean bias (mm/3h)', fontsize=10)
ax.set_ylabel(r'APCNet $-$ ERA5 mean bias (mm/3h)', fontsize=10)
ax.legend(frameon=False, fontsize=9, loc='upper right')
ax.set_title('Grid-point bias relation', fontsize=12)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.grid(alpha=0.25)
panel_label(ax, '(d)')

# 底部横向 colorbar（cax 精确定位在面板下方，不再压住任何子图）
cax = fig.add_axes([0.16, 0.022, 0.68, 0.020])
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Mean bias vs ERA5 target (mm/3h), test period 2024-2025', fontsize=10)
cbar.ax.tick_params(labelsize=8.5)
fig.suptitle('What the DL correction actually changes: mean-bias fields and their grid-point relationship',
             fontsize=13, y=0.985)
out = os.path.join(MEDIA, 'fig_bias_contrast_run13.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
print('saved', out)
print('RATIO', 12.5 / 9.8)
