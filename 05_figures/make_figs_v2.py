# -*- coding: utf-8 -*-
"""P3 图件 v2：Fig.5 CHM 修正口径重画 + Fig.6 三参照对比主图。
matplotlib 学术图，300dpi PNG 输出到 fig_p3/。论文图用英文标注。
"""
import numpy as np, os, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
FIG = r'D:\liaohe\校正优化过程\第三阶段\12优化\fig_p3'
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10.5,
                     'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans'})

C_GFS = '#1f77b4'; C_APC = '#d62728'; C_QM = '#2ca02c'; C_OLS = '#ff7f0e'

# ---------- Fig.5: CHM 修正口径验证 ----------
chm = json.load(open(os.path.join(OUT, 'chm_sym_s42_eval.json')))
boot = json.load(open(os.path.join(OUT, 'chm_bootstrap.json')))   # 统一口径（全 925 格点 + 月块 + 60/90 天块）

fig5 = plt.figure(figsize=(6.6, 3.4))
gs5 = fig5.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.4)

# (a) 月度平均率时间序列
axl = fig5.add_subplot(gs5[0])
rows = chm.get('monthly_series', [])
if rows:
    months = ['%s-%02d' % (r['year'], int(r['month'])) for r in rows]
    g = [float(r['gfs']) for r in rows]; a = [float(r['apc']) for r in rows]; c = [float(r['chm']) for r in rows]
    x = np.arange(len(months))
    axl.plot(x, g, 'o-', ms=3, lw=1.2, color=C_GFS, label='GFS (0.076 mm/h)')
    axl.plot(x, a, 's-', ms=3, lw=1.2, color=C_APC, label='APCNet (0.176 mm/h)')
    axl.plot(x, c, '^-', ms=3, lw=1.2, color='#333333', label='CHM obs (0.071 mm/h)')
    axl.set_xticks(x[::3]); axl.set_xticklabels([months[i] for i in x[::3]], rotation=30, fontsize=7)
    axl.set_ylabel('Monthly mean rate (mm/h)')
    axl.set_title('(a) Monthly mean rate vs CHM (713 days)')
    axl.legend(frameon=False, loc='upper left')
    axl.set_ylim(0, max(max(g), max(a)) * 1.15)

# (b) RMSE 改进率 + bootstrap CI（统一口径月块，四方法）
axr = fig5.add_subplot(gs5[1])
labels = ['GFS', 'APCNet', 'QM', 'OLS']
impr = [0.0, chm['rmse_improve_apc'], chm['rmse_improve_qm'], chm['rmse_improve_ols']]
# CI 统一从 chm_bootstrap.json 的 month_block_* 读取（全部 925 格点口径）
ci_lo = [0, boot['month_block_apcnet']['ci95'][0], boot['month_block_qm']['ci95'][0], boot['month_block_ols']['ci95'][0]]
ci_hi = [0, boot['month_block_apcnet']['ci95'][1], boot['month_block_qm']['ci95'][1], boot['month_block_ols']['ci95'][1]]
colors = [C_GFS, C_APC, C_QM, C_OLS]
xpos = np.arange(len(labels))
for x, lab, v, lo, hi, col in zip(xpos, labels, impr, ci_lo, ci_hi, colors):
    axr.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt='o', ms=8, color=col,
                 capsize=5, lw=1.5)
    axr.text(x, v + (4 if v >= 0 else -7), '%.1f%%' % v, ha='center', fontsize=9, color=col)
axr.axhline(0, color='k', lw=0.9, ls='--')
axr.set_xticks(xpos); axr.set_xticklabels(labels)
axr.set_ylabel('RMSE improvement vs GFS (%)')
axr.set_ylim(-45, 20)
axr.set_title('(b) RMSE improvement\n(monthly block bootstrap, 95% CI)')
fig5.suptitle('Independent verification: CHM gauge-merged daily product (rate basis, 713 days)',
              y=1.02, fontsize=11)
fig5.savefig(os.path.join(FIG, 'fig5_chm.png'), bbox_inches='tight')
print('OK fig5_chm.png')

# ---------- Fig.6: 三参照对比主图（含 BinCM） ----------
gpm = json.load(open(os.path.join(OUT, 'gpm3h_eval.json')))
era5 = {'BinCM': 27.3, 'OLS': 22.2, 'QM': 11.1, 'APCNet': -32.6}
chm6 = {'BinCM': None, 'OLS': 12.15, 'QM': 3.58, 'APCNet': -31.94}
gpm6 = {'BinCM': 22.6, 'OLS': gpm['mse_improve_OLS'], 'QM': gpm['mse_improve_QM'], 'APCNet': gpm['mse_improve_APCNet']}

fig6 = plt.figure(figsize=(6.2, 3.6))
ax6 = fig6.add_subplot(111)
methods = ['BinCM', 'OLS', 'QM', 'APCNet']
refs = ['ERA5 (training ref)', 'CHM (obs, daily)', 'GPM IMERG (obs, 3h)']
x6 = np.arange(len(methods)); w = 0.26
for i, ref in enumerate(refs):
    src = era5 if i == 0 else (chm6 if i == 1 else gpm6)
    vals = [src[m] for m in methods]
    labeled = False
    for j, v in enumerate(vals):
        if v is None:
            continue
        ax6.bar(x6[j] + (i - 1) * w, v, w, color=['#9467bd', C_OLS, C_QM, C_APC][j],
                edgecolor='k', linewidth=0.4, label=ref if not labeled else None)
        labeled = True
        ax6.text(x6[j] + (i - 1) * w, v + (2.5 if v >= 0 else -6), '%.1f' % v, ha='center', fontsize=8)
ax6.axhline(0, color='k', lw=0.9, ls='--')
ax6.set_xticks(x6); ax6.set_xticklabels(methods)
ax6.set_ylabel('MSE/RMSE improvement vs GFS (%)')
ax6.set_ylim(-45, 32)
ax6.legend(frameon=False, loc='upper left', fontsize=7.5)
ax6.set_title('Consistent ranking: BinCM > OLS > QM > GFS > APCNet (3-h scales)')
fig6.savefig(os.path.join(FIG, 'fig6_three_refs.png'), bbox_inches='tight')
print('OK fig6_three_refs.png')
print('done')
