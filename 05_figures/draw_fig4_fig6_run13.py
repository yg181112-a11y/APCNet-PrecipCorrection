# -*- coding: utf-8 -*-
"""重绘 Fig.4（run13 bias 场）与 Fig.6（run13 CHM）+ 替换 docx blip。"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Times New Roman 统一字体
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
})

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
RUN13 = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
MEDIA = r'C:\Users\yg181\Desktop\论文三\WAF\r3_media'
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
OKABE = ['#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7','#000000']

# ============ Fig.4: run13 bias fields ============
print('== Fig.4 ==')
gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
tgt = np.load(os.path.join(WORK, 'targets_test.npy'))
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
apc = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
mask = np.load(os.path.join(WORK, 'bias_field_rainmask.npy')).astype(bool)

bg = np.nanmean(gfs - tgt, axis=0)
bq = np.nanmean(qm - tgt, axis=0)
ba = np.nanmean(apc - tgt, axis=0)
print(f'GFS bias {bg.mean():+.3f} std {np.nanstd(bg[mask]):.3f}')
print(f'QM  bias {bq.mean():+.3f} std {np.nanstd(bq[mask]):.3f}')
print(f'APC bias {ba.mean():+.3f} std {np.nanstd(ba[mask]):.3f} local max {np.nanmax(ba[mask]):+.2f}')

vmax = float(np.ceil(max(np.nanmax(np.abs(bg[mask])), np.nanmax(np.abs(bq[mask])), np.nanmax(np.abs(ba[mask]))) * 10) / 10)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={'wspace': 0.10})
titles = [r'GFS $-$ ERA5', r'QM $-$ ERA5', r'APCNet $-$ ERA5']
datas = [bg, bq, ba]
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
for ax, d, t in zip(axes, datas, titles):
    d2 = np.where(mask, d, np.nan)
    im = ax.pcolormesh(GLOBAL_LONS, GLOBAL_LATS, d2, cmap='RdBu_r', norm=norm, shading='auto')
    ax.set_title(t, fontsize=12)
    ax.set_xlabel('Longitude (°E)', fontsize=11)
    ax.set_ylabel('Latitude (°N)', fontsize=11)
    ax.set_aspect(1.6)
for ax, lab in zip(axes, ['(a)', '(b)', '(c)']):
    ax.text(0.5, -0.20, lab, transform=ax.transAxes, fontsize=13, fontweight='bold',
            va='top', ha='center')
cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04, shrink=0.85)
cbar.set_label('Mean bias vs ERA5 target (mm/3h), test period 2024-2025', fontsize=11)
fig.suptitle('Spatial structure of the corrections on the training reference', fontsize=14, y=1.02)
fig.tight_layout(rect=[0, 0, 0.985, 1])
fig4 = os.path.join(MEDIA, 'fig_bias_maps_run13.png')
fig.savefig(fig4, dpi=300, bbox_inches='tight')
print('saved', fig4)

# ============ Fig.6: CHM run13 ============
print('== Fig.6 ==')
d = json.load(open(os.path.join(WORK, 'chm_run13_s42_eval.json'), encoding='utf-8'))
boot = json.load(open(os.path.join(RUN13, 'run13_boot2.json'), encoding='utf-8'))
series = d['monthly_series']
series = [s for s in series if s['n_days'] >= 10]
months = [f"{s['year']%100:02d}-{s['month']:02d}" for s in series]
chm_m = [s['chm'] for s in series]
gfs_m = [s['gfs'] for s in series]
apc_m = [s['apc'] for s in series]

fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.95))
ax = axes[0]
ax.plot(months, chm_m, '-o', color='k', lw=1.8, ms=3, label='CHM obs')
ax.plot(months, gfs_m, '-s', color=OKABE[1], lw=1.6, ms=3, label='GFS')
ax.plot(months, apc_m, '-^', color=OKABE[5], lw=1.6, ms=3, label='APCNet')
ax.set_ylabel('Monthly mean rate (mm h$^{-1}$)')
ax.tick_params(axis='x', rotation=60, labelsize=7)
ax.legend(frameon=False, fontsize=9)
ax.set_title('Monthly mean rate vs CHM (713 days)', fontsize=11)
ax.text(0.5, -0.15, '(a)', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top', ha='center')
ax.grid(alpha=0.3)

ax = axes[1]
methods = ['APCNet', 'QM', 'OLS']
obs = [boot['chm_apc']['obs_imp'], 3.5795242607746784, 12.15065411105823]
ci = [[boot['chm_apc']['ci95'][0], boot['chm_apc']['ci95'][1]],
      [1.47, 5.12],
      [9.80, 15.86]]
yerr = [[o - lo, hi - o] for o, (lo, hi) in zip(obs, ci)]
bars = ax.bar(methods, obs, yerr=np.array(yerr).T, capsize=4, color=[OKABE[5], OKABE[2], OKABE[4]], alpha=0.9, width=0.5)
ax.axhline(0, color='k', lw=0.8)
for i, (o, (lo, hi)) in enumerate(zip(obs, ci)):
    ax.text(i, hi + 1.5, f'{o:+.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.text(i, lo - 3.5, f'[{lo:+.1f}, {hi:+.1f}]', ha='center', fontsize=7.5, color='0.35')
ax.set_ylabel('RMSE improvement vs GFS (%)')
ax.set_title('Monthly block bootstrap, 95% CI', fontsize=11)
ax.text(0.5, -0.15, '(b)', transform=ax.transAxes, fontsize=13, fontweight='bold', va='top', ha='center')
ax.set_ylim(-40, 30)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig6 = os.path.join(MEDIA, 'fig_chm_run13.png')
fig.savefig(fig6, dpi=300, bbox_inches='tight')
print('saved', fig6)
print('FIG4_RATIO', 15 / 4.6)
print('FIG6_RATIO', 12.0 / 5.95)
