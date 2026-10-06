# -*- coding: utf-8 -*-
"""R3 稿 Fig.2-8 期刊风格统一重绘（scientific-visualization 规范）。
输出: D:\\liaohe\\校正优化过程\\第三阶段\\12优化\\fig_p3_pub\\  (*.png 300dpi + *.pdf 矢量)
规范: Arial / Okabe-Ito 色盲安全 / despine / 面板标签 bold / 冗余线型编码
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

OUT = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
FIG = r'D:\liaohe\校正优化过程\第三阶段\12优化\fig_p3_pub'
os.makedirs(FIG, exist_ok=True)

# ---------- 期刊统一风格 ----------
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
# Okabe-Ito
C_GFS = '#0072B2'   # blue
C_QM = '#009E73'    # green
C_OLS = '#56B4E9'   # light blue
C_APC = '#D55E00'   # vermillion
C_UNET = '#CC79A7'  # purple
C_BIN = '#E69F00'   # orange
C_OBS = '#000000'   # black
LS = {'GFS': '-', 'QM': '--', 'OLS': '-.', 'APCNet': '-', 'U-Net': ':', 'BinCM': '--', 'obs': '-'}
MK = {'GFS': 'o', 'QM': '^', 'OLS': 'D', 'APCNet': 's', 'U-Net': 'v', 'BinCM': 'P', 'obs': 'o'}


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def panel_label(ax, label):
    ax.text(0.5, -0.34, label, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='center')


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('OK', name)


# ================= Fig.2: 目标订正证据 =================
sys.path.insert(0, r'C:\Users\yg181\Desktop\论文三\13.0修复重跑')
from verify_era5_target import load_era5_monthly, load_gfs_f003

GFSD = r'D:/liaohe/GFS-data/gfs.0p25.2015-2025.f003'
NEW = r'D:\liaohe\ERA5-data\monthly'
OLD = r'D:\liaohe\ERA5-data\monthly_old_b1'
Y, M = 2024, 7
t_g, g = load_gfs_f003(GFSD, Y, M)
t_e_new, e_new = load_era5_monthly(NEW, Y, M)
t_e_old, e_old = load_era5_monthly(OLD, Y, M)
g_t = {t.timestamp(): v for t, v in zip(t_g, g)}
e_n = {t.timestamp(): float(np.mean(v)) for t, v in zip(t_e_new, e_new)}
e_o = {t.timestamp(): float(np.mean(v)) for t, v in zip(t_e_old, e_old)}
common = sorted(set(g_t) & set(e_n) & set(e_o))
gv = np.array([g_t[t] for t in common])
env = np.array([e_n[t] for t in common])
eov = np.array([e_o[t] for t in common])
x = np.arange(len(common))

fig = plt.figure(figsize=(6.9, 5.4))
gs = GridSpec(2, 2, hspace=0.55, wspace=0.35,
              left=0.09, right=0.97, top=0.80, bottom=0.10)

ax = fig.add_subplot(gs[0, 0])
ax.plot(x, gv, lw=1.0, color=C_GFS, label='GFS f003 (true 3-h)')
ax.plot(x, eov * 3, lw=1.0, color=C_APC, ls='--', label='ERA5 old (1-h) $\\times$3')
ax.plot(x, eov, lw=0.9, color=C_APC, ls=':', label='ERA5 old (1-h) raw')
ax.set_title('Before fix: 1-h target vs 3-h GFS  ratio=%.3f' % (np.mean(gv) / np.mean(eov)), pad=18)
ax.legend(frameon=False, fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, 1.32), ncol=3)
ax.set_ylabel('mm per window')
ax.set_xlabel('Common 3-h time step (July 2024)')
ax.set_ylim(0, max(gv.max(), eov.max() * 3) * 1.25)
panel_label(ax, '(a)')
despine(ax)

ax = fig.add_subplot(gs[0, 1])
ax.plot(x, gv, lw=1.0, color=C_GFS, label='GFS f003')
ax.plot(x, env, lw=1.0, color=C_QM, label='ERA5 new (true 3-h)')
ax.set_title('After fix: aligned windows  ratio=%.3f  r=%.3f'
             % (np.mean(gv) / np.mean(env), np.corrcoef(gv, env)[0, 1]), pad=18)
ax.legend(frameon=False, fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, 1.32), ncol=2)
ax.set_ylabel('mm/3h')
ax.set_xlabel('Common 3-h time step (July 2024)')
ax.set_ylim(0, max(gv.max(), env.max()) * 1.25)
panel_label(ax, '(b)')
despine(ax)

ax = fig.add_subplot(gs[1, 0])
ax.scatter(env, gv, s=6, alpha=0.55, color=C_QM, edgecolors='none')
lim = max(gv.max(), env.max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', lw=0.8)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('ERA5 new (mm/3h)'); ax.set_ylabel('GFS (mm/3h)')
ax.set_title('Scatter (%d common times)  r=%.3f' % (len(common), np.corrcoef(gv, env)[0, 1]))
panel_label(ax, '(c)')
despine(ax)

ax = fig.add_subplot(gs[1, 1])
monthly = [np.sum(gv), np.sum(env), np.sum(eov)]
labels = ['GFS f003', 'ERA5 new', 'ERA5 old (1-h)']
colors = [C_GFS, C_QM, C_APC]
bars = ax.bar(labels, monthly, color=colors, alpha=0.9, width=0.62, edgecolor='k', linewidth=0.5)
for b, v in zip(bars, monthly):
    ax.text(b.get_x() + b.get_width() / 2, v + 2.5, '%.1f' % v, ha='center', fontsize=8)
ax.set_ylabel('July 2024 total (mm)')
ax.set_title('Monthly totals  GFS vs new: +%.1f%%' % (100 * (np.sum(gv) - np.sum(env)) / np.sum(env)))
panel_label(ax, '(d)')
despine(ax)
fig.suptitle('Target-correction evidence: July 2024, domain mean (Liaohe basin, 25$\\times$37)', fontsize=10.5, y=0.995)
save(fig, 'fig2_target_evidence')

# ================= Fig.3: 空间偏差场 =================
b_g = np.load(os.path.join(OUT, 'bias_field_gfs.npy'))
b_q = np.load(os.path.join(OUT, 'bias_field_qm.npy'))
b_a = np.load(os.path.join(OUT, 'bias_field_apcnet.npy'))
mask = np.load(os.path.join(OUT, 'bias_field_rainmask.npy')).astype(bool)
LATS = np.linspace(46.0, 40.0, 25)
LONS = np.linspace(117.0, 126.0, 37)
vmax = float(np.ceil(max(abs(b_g[mask]).max(), abs(b_q[mask]).max(), abs(b_a[mask]).max()) * 10) / 10)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9))
titles = [r'GFS $-$ ERA5', r'QM $-$ ERA5', r'APCNet $-$ ERA5']
for ax, d, t in zip(axes, [b_g, b_q, b_a], titles):
    d2 = np.where(mask, d, np.nan)
    im = ax.pcolormesh(LONS, LATS, d2, cmap='RdBu_r', norm=norm, shading='auto')
    ax.set_title(t + '  mean=%.3f' % np.nanmean(d2), fontsize=9)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_aspect(1.4)
    ax.set_xticks(np.arange(118, 127, 2))
    ax.set_yticks(np.arange(41, 47, 1))
    ax.tick_params(length=2.5)
cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.055, pad=0.10, aspect=45)
cbar.set_label('Mean bias vs ERA5 target (mm/3h), test period 2024–2025', fontsize=8)
cbar.ax.tick_params(labelsize=7)
fig.suptitle('Spatial structure of corrections on the training reference', fontsize=10, y=1.01)
save(fig, 'fig3_bias')

# ================= Fig.4: 受控实验 =================
grids = {}
for tag, fn in [('ns0.5', 'grid_0.5_s42_ep24.json'), ('ns1.16', 'grid_1.16_s42_ep24.json'),
                ('ns2.0', 'grid_2.0_s42_ep24.json')]:
    p = os.path.join(OUT, '..', 'controlled_exp', fn)
    if os.path.exists(p):
        grids[tag] = json.load(open(p))
identity = json.load(open(os.path.join(OUT, '..', 'controlled_exp', 'res_identity_s42.json')))
cc_id = identity.get('cc_pred_gfs', np.nan)

sigmas = [0, 0.5, 1.16, 2.0]
imp = [0.0]
for s in [0.5, 1.16, 2.0]:
    tag = {0.5: 'ns0.5', 1.16: 'ns1.16', 2.0: 'ns2.0'}[s]
    r = grids.get(tag)
    imp.append(r['improvement'] * 100 if r else np.nan)

fig = plt.figure(figsize=(6.9, 3.0))
gs = GridSpec(1, 2, width_ratios=[1.1, 1.0], wspace=0.32, left=0.09, right=0.97, bottom=0.26, top=0.74)

ax = fig.add_subplot(gs[0])
ax.plot(sigmas, imp, 'o-', color=C_APC, lw=1.5, ms=5, label='Controlled experiment (S42)')
ax.axhspan(-63.1, -32.6, color=C_APC, alpha=0.12, label='Real task (3 seeds, sym)')
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.axvline(1.16, color='0.45', lw=0.8, ls=':', label=r'$\sigma$ = observed residual')
ax.set_xlabel(r'Target noise $\sigma$ (mm/3h)')
ax.set_ylabel('MSE improvement vs GFS (%)')
ax.set_ylim(-75, 5)
ax.set_xticks(sigmas)
ax.legend(frameon=False, fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, 1.12), ncol=3)
panel_label(ax, '(a)')
despine(ax)

ax = fig.add_subplot(gs[1])
sig2 = [0.5, 1.16, 2.0]
imp2 = [imp[1], imp[2], imp[3]]
ax.plot(sig2, imp2, 'o-', color=C_APC, lw=1.5, ms=5)
for s, v in zip(sig2, imp2):
    ax.text(s, v + 2, '%.1f%%' % v, ha='center', fontsize=7.5)
ax.text(0.13, 5, 'identity: CC=%.3f' % cc_id, fontsize=8)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'$\sigma$ (mm/3h)')
ax.set_ylabel('MSE improvement vs zero-correction (%)')
ax.set_ylim(-45, 10)
ax.set_xticks(sigmas)
panel_label(ax, '(b)')
despine(ax)
fig.suptitle('Controlled experiment: target noise drives degradation, and the composite loss sets its magnitude', fontsize=10, y=0.99)
save(fig, 'fig4_snr')

# ================= Fig.5: CHM 验证 =================
chm = json.load(open(os.path.join(OUT, 'chm_sym_s42_eval.json')))
boot = json.load(open(os.path.join(OUT, 'chm_bootstrap.json')))
fig = plt.figure(figsize=(6.9, 3.4))
gs = GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32, left=0.09, right=0.97, bottom=0.16, top=0.90)

ax = fig.add_subplot(gs[0])
rows = chm['monthly_series']
months = ['%s-%02d' % (r['year'], int(r['month'])) for r in rows]
g = [float(r['gfs']) for r in rows]; a = [float(r['apc']) for r in rows]; c = [float(r['chm']) for r in rows]
x = np.arange(len(months))
ax.plot(x, g, ls=LS['GFS'], marker=MK['GFS'], ms=3, lw=1.1, color=C_GFS, label='GFS (%.3f mm/h)' % chm['rate_gfs']['fcst_rate'])
ax.plot(x, a, ls=LS['APCNet'], marker=MK['APCNet'], ms=3, lw=1.1, color=C_APC, label='APCNet (%.3f mm/h)' % chm['rate_apc']['fcst_rate'])
ax.plot(x, c, ls='-', marker=MK['obs'], ms=3, lw=1.1, color=C_OBS, label='CHM obs (%.3f mm/h)' % chm['rate_gfs']['obs_rate'])
ax.set_xticks(x[::3]); ax.set_xticklabels([months[i] for i in x[::3]], rotation=30, fontsize=6.5)
ax.set_ylabel('Monthly mean rate (mm/h)')
ax.set_title('(a) Monthly mean rate vs CHM (%d days)' % chm['n_common_days'], fontsize=9)
ax.legend(frameon=False, loc='upper left', fontsize=6.5)
ax.set_ylim(0, max(max(g), max(a)) * 1.15)
despine(ax)

ax = fig.add_subplot(gs[1])
labels = ['GFS', 'APCNet', 'QM', 'OLS']
impr = [0.0, chm['rmse_improve_apc'], chm['rmse_improve_qm'], chm['rmse_improve_ols']]
ci_lo = [0] + [boot['month_block_' + k]['ci95'][0] for k in ('apcnet', 'qm', 'ols')]
ci_hi = [0] + [boot['month_block_' + k]['ci95'][1] for k in ('apcnet', 'qm', 'ols')]
colors = [C_GFS, C_APC, C_QM, C_OLS]
xpos = np.arange(len(labels))
for xp, lab, v, lo, hi, col in zip(xpos, labels, impr, ci_lo, ci_hi, colors):
    ax.errorbar([xp], [v], yerr=[[v - lo], [hi - v]], fmt='o', ms=6, color=col, capsize=4, lw=1.2)
    ax.text(xp, v + (3 if v >= 0 else -8), '%.1f%%' % v, ha='center', fontsize=8, color=col)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_ylabel('RMSE improvement vs GFS (%)')
ax.set_title('(b) Monthly block bootstrap, 95% CI', fontsize=9)
ax.set_ylim(-45, 20)
despine(ax)
fig.suptitle('Independent verification: CHM gauge-merged daily product (rate basis, %d days)' % chm['n_common_days'],
             fontsize=10, y=0.985)
save(fig, 'fig5_chm')

# ================= Fig.6: 三参照对比 =================
gpm = json.load(open(os.path.join(OUT, 'gpm3h_eval.json')))
era5 = {'BinCM': 27.3, 'OLS': 22.2, 'QM': 11.1, 'APCNet': -32.6}
chm6 = {'BinCM': None, 'OLS': 12.15, 'QM': 3.58, 'APCNet': -31.94}
gpm6 = {'BinCM': 22.6, 'OLS': gpm['mse_improve_OLS'], 'QM': gpm['mse_improve_QM'], 'APCNet': gpm['mse_improve_APCNet']}

fig, ax = plt.subplots(figsize=(6.6, 3.4))
methods = ['BinCM', 'OLS', 'QM', 'APCNet']
refs = ['ERA5 (training ref)', 'CHM (obs, daily)', 'GPM IMERG (obs, 3h)']
mcolors = [C_BIN, C_OLS, C_QM, C_APC]
x6 = np.arange(len(methods)); w = 0.26
for i, ref in enumerate(refs):
    src = era5 if i == 0 else (chm6 if i == 1 else gpm6)
    labeled = False
    for j, m in enumerate(methods):
        v = src[m]
        if v is None:
            continue
        ax.bar(x6[j] + (i - 1) * w, v, w, color=mcolors[j], edgecolor='k', linewidth=0.4,
               label=ref if not labeled else None)
        labeled = True
        dy = 3.0 if v >= 0 else -7.0
        ax.text(x6[j] + (i - 1) * w, v + dy, '%.1f' % v, ha='center', fontsize=7)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xticks(x6); ax.set_xticklabels(methods)
ax.set_ylabel('MSE/RMSE improvement vs GFS (%)')
ax.set_ylim(-45, 33)
ax.legend(frameon=False, loc='upper right', fontsize=6.8, ncol=1)
ax.set_title('Consistent ranking: BinCM > OLS > QM > GFS > APCNet (3-h scales)', fontsize=9.5)
despine(ax)
save(fig, 'fig6_three_refs')

# ================= Fig.7: 可靠性 + 锐度 =================
prob = json.load(open(os.path.join(OUT, 'gpm_prob_eval.json')))
rel = prob['reliability']
bins_x = [r['avg_prob'] for r in rel]
bins_y = [r['avg_obs'] for r in rel]
counts = [r['count'] for r in rel]
fig = plt.figure(figsize=(6.9, 3.0))
gs = GridSpec(1, 2, width_ratios=[1.1, 1.0], wspace=0.32, left=0.09, right=0.985, bottom=0.36, top=0.90)
ax = fig.add_subplot(gs[0])
ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
ax.plot(bins_x, bins_y, 'o-', color=C_APC, ms=4, lw=1.2, label='APCNet occurrence')
ax.set_xlabel('Forecast probability (wet)')
ax.set_ylabel('Observed frequency')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title('Reliability diagram (0.1 mm threshold)', fontsize=9)
ax.legend(frameon=False, loc='upper left', fontsize=6.8)
panel_label(ax, '(a)')
despine(ax)
ax = fig.add_subplot(gs[1])
tot = sum(counts)
ax.bar(np.arange(len(counts)), [100.0 * c / tot for c in counts], color=C_GFS, alpha=0.85,
       edgecolor='k', linewidth=0.4, width=0.75)
ax.set_xticks(np.arange(len(counts)))
ax.set_xticklabels(['0–0.1', '0.1–0.2', '0.2–0.3', '0.3–0.4', '0.4–0.5',
                    '0.5–0.6', '0.6–0.7', '0.7–0.8', '0.8–0.9', '0.9–1'], rotation=45, ha='right', fontsize=5.5)
ax.set_xlabel('Forecast probability bin')
ax.set_ylabel('Share of grid-point samples (%)')
ax.set_title('Sharpness: P(wet) distribution', fontsize=9)
despine(ax)
panel_label(ax, '(b)')
fig.suptitle('Probabilistic calibration of APCNet precipitation occurrence (GPM verification)', fontsize=10, y=0.99)
save(fig, 'fig7_reliability')

# ================= Fig.8: 日循环 =================
hours = ['03Z', '09Z', '15Z', '21Z']
data = {
    'GPM IMERG (obs)': [0.2102, 0.2607, 0.2336, 0.2781],
    'GFS': [0.2349, 0.2833, 0.2457, 0.2608],
    'QM': [0.2228, 0.2719, 0.2390, 0.2498],
    'OLS': [0.2258, 0.2546, 0.2319, 0.2427],
    'APCNet': [0.5556, 0.6171, 0.5765, 0.5800],
    'U-Net': [0.4446, 0.5397, 0.4497, 0.4456],
}
colors = {'GPM IMERG (obs)': C_OBS, 'GFS': C_GFS, 'QM': C_QM, 'OLS': C_OLS, 'APCNet': C_APC, 'U-Net': C_UNET}
fig, ax = plt.subplots(figsize=(6.6, 4.35))
x = list(range(4))
keymap = {'GPM IMERG (obs)': 'obs', 'GFS': 'GFS', 'QM': 'QM', 'OLS': 'OLS', 'APCNet': 'APCNet', 'U-Net': 'U-Net'}
for name, y in data.items():
    k = keymap[name]
    ax.plot(x, y, ls=LS[k], marker=MK[k], color=colors[name],
            label=name, lw=1.6 if name in ('GPM IMERG (obs)', 'APCNet') else 1.2,
            ms=5.5 if name in ('GPM IMERG (obs)', 'APCNet') else 4)
ax.set_xticks(x)
ax.set_xticklabels(hours)
ax.set_xlabel('Valid time (UTC)')
ax.set_ylabel('Domain-mean precipitation rate (mm/3h)')
ax.set_title('Diurnal cycle over the GPM-verification samples (2024–2025)', fontsize=9.5)
ax.legend(ncol=6, frameon=False, fontsize=7.0, loc='upper center', bbox_to_anchor=(0.5, -0.155))
ax.set_ylim(0, 0.72)
despine(ax)
save(fig, 'fig8_diurnal')

print('\n全部完成')

