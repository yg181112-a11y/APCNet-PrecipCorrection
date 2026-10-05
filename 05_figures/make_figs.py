# -*- coding: utf-8 -*-
"""P3 论文图件 v1：Fig.3 SNR曲线 / Fig.4 空间偏差场 / Fig.5 CHM验证。
matplotlib 学术图，300dpi PNG 输出到 fig_p3/。
"""
import numpy as np, os, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
FIG = r'D:\liaohe\校正优化过程\第三阶段\12优化\fig_p3'
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10.5,
                     'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans'})

# ---------- Fig.3: SNR -> 可学性 ----------
res = {}
for tag, fn in [('identity', 'res_identity_s42.json'), ('ns0.5', 'grid_0.5_s42_ep24.json'),
                ('ns1.16', 'grid_1.16_s42_ep24.json'), ('ns2.0', 'grid_2.0_s42_ep24.json')]:
    p = os.path.join(OUT, '..', 'controlled_exp', fn)
    if os.path.exists(p):
        res[tag] = json.load(open(p))
print('受控实验 keys:', list(res.keys()))

fig = plt.figure(figsize=(6.6, 3.2))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.35)

# 左：改进率 vs sigma（含主实验区间带）
ax = fig.add_subplot(gs[0])
sigmas = [0, 0.5, 1.16, 2.0]
imp = [0.0, -2.67, -27.18, -9.62]
ax.plot(sigmas, imp, 'o-', color='#d62728', lw=1.6, ms=5, label='Controlled experiment (S42)')
ax.axhspan(-63.1, -32.6, color='#d62728', alpha=0.10, label='Real task (3 seeds, sym)')
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'Target noise $\sigma$ (mm/3h)')
ax.set_ylabel('MSE improvement vs GFS (%)')
ax.set_ylim(-75, 5)
ax.set_xticks(sigmas)
ax.axvline(1.16, color='gray', lw=0.8, ls=':', label=r'$\sigma$ = observed residual')
ax.legend(loc='lower left', frameon=False)
ax.set_title('(a) Skill vs target SNR')

# 右：改进率 vs sigma（对 zero-correction 基线）+ identity 标注
ax2 = fig.add_subplot(gs[1])
sig2 = [0.5, 1.16, 2.0]
improve = []
for s in sig2:
    tag = {0.5: 'ns0.5', 1.16: 'ns1.16', 2.0: 'ns2.0'}[s]
    r = res.get(tag)
    improve.append(r['improvement'] * 100 if r else np.nan)
ax2.plot(sig2, improve, 'o-', color='#d62728', lw=1.6, ms=6)
for s, v in zip(sig2, improve):
    ax2.text(s, v + 2, '%.1f%%' % v, ha='center', fontsize=8)
r = res.get('identity')
if r:
    ax2.text(0.15, 5, 'identity: CC=%.3f' % r.get('cc_pred_gfs', 0), fontsize=8)
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.set_xlabel(r'$\sigma$ (mm/3h)')
ax2.set_ylabel('MSE improvement vs zero-correction (%)')
ax2.set_ylim(-45, 10)
ax2.set_xticks(sigmas)
ax2.set_title('(b) Negative skill is causal')
fig.savefig(os.path.join(FIG, 'fig3_snr.png'), bbox_inches='tight')
print('✅ fig3_snr.png')

# ---------- Fig.4: 空间偏差场 ----------
def loadb(name):
    p = os.path.join(OUT, name)
    return np.load(p) if os.path.exists(p) else None
b_g = loadb('bias_field_gfs.npy'); b_a = loadb('bias_field_apcnet.npy'); b_q = loadb('bias_field_qm.npy')
mask = loadb('bias_field_rainmask.npy')
print('bias shapes:', None if b_g is None else b_g.shape, None if b_a is None else b_a.shape, None if b_q is None else b_q.shape)

if b_g is not None and b_a is not None:
    fig2, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    fields = [('(a) GFS', b_g, [-0.3, 0.3]), ('(b) APCNet', b_a, [-0.6, 1.5]), ('(c) QM', b_q, [-0.3, 0.3])]
    for ax, (ttl, fld, vrg) in zip(axes, fields):
        if fld is None:
            ax.set_title(ttl); ax.axis('off'); continue
        if mask is not None:
            fld = np.where(mask, fld, np.nan)
        im = ax.imshow(fld, origin='lower', cmap='RdBu_r', vmin=vrg[0], vmax=vrg[1],
                       extent=[117, 126, 40, 46])
        ax.set_title(ttl + '  mean=%.3f' % np.nanmean(fld))
        ax.set_xlabel('Lon'); ax.set_ylabel('Lat')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig2.suptitle('Multi-year mean spatial bias (mm/3h)', y=1.02, fontsize=11)
    fig2.savefig(os.path.join(FIG, 'fig4_bias.png'), bbox_inches='tight')
    print('✅ fig4_bias.png')

# ---------- Fig.5: CHM 验证 ----------
fig5 = plt.figure(figsize=(6.6, 3.0))
gs5 = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.35)
chm_path = os.path.join(OUT, 'chm_newdata_eval.json')
chm = json.load(open(chm_path)) if os.path.exists(chm_path) else None
if chm:
    print('CHM: imp_apc=%.3f%% imp_qm=%.3f%% imp_ols=%.3f%% n_days=%s' % (
        chm.get('rmse_improve_apc', 0), chm.get('rmse_improve_qm', 0), chm.get('rmse_improve_ols', 0),
        chm.get('n_common_days')))
boot = json.load(open(os.path.join(OUT, 'chm_bootstrap.json'))) if os.path.exists(os.path.join(OUT, 'chm_bootstrap.json')) else None
if boot:
    mb = boot.get('month_block_apcnet', {})
    ci = mb.get('ci95', [0, 0])
    obs = boot.get('obs_improve_pct', {}).get('apcnet', 0)
    print('bootstrap: obs=%.3f%% CI=[%.3f,%.3f] Pneg=%.4f' % (obs, ci[0], ci[1], mb.get('p_negative', 0)))

if chm and chm.get('monthly_series'):
    axl = fig5.add_subplot(gs5[0])
    rows = chm['monthly_series']
    months = ['%s-%02d' % (r['year'], int(r['month'])) for r in rows]
    g = [float(r['gfs']) for r in rows]; a = [float(r['apc']) for r in rows]
    c = [float(r['chm']) for r in rows]
    x = np.arange(len(months))
    axl.plot(x, g, 'o-', ms=3, lw=1.2, color='#1f77b4', label='GFS')
    axl.plot(x, a, 's-', ms=3, lw=1.2, color='#d62728', label='APCNet (sym S42)')
    axl.plot(x, c, '^-', ms=3, lw=1.2, color='#2ca02c', label='CHM (obs)')
    axl.set_xticks(x[::3]); axl.set_xticklabels([months[i] for i in x[::3]], rotation=30, fontsize=7)
    axl.set_ylabel('Monthly mean rate (mm/h)')
    axl.set_title('(a) Monthly mean rate vs CHM (713 days)')
    axl.legend(frameon=False, loc='upper left')

    axr = fig5.add_subplot(gs5[1])
    if boot:
        ci = boot.get('month_block_apcnet', {}).get('ci95', [0, 0]); lo, hi = ci
        obs = boot.get('obs_improve_pct', {}).get('apcnet', 0)
        axr.errorbar(['GFS'], [0], fmt='o', color='#1f77b4', ms=6)
        axr.errorbar(['APCNet'], [obs], yerr=[[obs - lo], [hi - obs]], fmt='s', color='#d62728',
                     ms=7, capsize=5, lw=1.5)
        axr.axhline(0, color='k', lw=0.8, ls='--')
        axr.set_ylabel('RMSE improvement vs GFS (%)')
        axr.set_title('(b) Block-bootstrap 95%% CI\n[%.2f, %.2f]%%  P(neg)<0.001' % (lo, hi))
        axr.set_ylim(-4, 10)
        axr.text(0.5, obs + 2, '+%.2f%%' % obs, ha='center', fontsize=9, color='#d62728')
    fig5.suptitle('Independent verification: CHM gauge-merged daily product', y=1.02, fontsize=11)
    fig5.savefig(os.path.join(FIG, 'fig5_chm.png'), bbox_inches='tight')
    print('✅ fig5_chm.png')

print('done')
