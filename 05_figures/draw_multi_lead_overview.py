# -*- coding: utf-8 -*-
"""R3 多时效总览图：尺度×方法 MSE 改进 + CHM 独立验证。
Panel A: ERA5 参照下 MSE 改进 vs 累积尺度（APCNet/U-Net/QM/BM/OLS）
Panel B: CHM 独立验证 RMSE 改进 vs 尺度（APCNet/U-Net/ERA5 参照）
Panel C: ERA5 参照下 CC（GFS/APCNet/U-Net）随尺度
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTD = r'D:\liaohe\论文三\03_重建成稿代_2026_R3全链主实验\r3_media\r3_media'
os.makedirs(OUTD, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False,
})
# 全稿统一"方法—颜色"映射（v4 N3：跨图同方法必须同色）
C_APC = '#D55E00'; C_UNET = '#CC79A7'; C_QM = '#E69F00'; C_OLS = '#0072B2'; C_BIN = '#009E73'
C_GFS = '#000000'

# 数据（% vs GFS；本脚本统一为正=改进，故取反 MSE 改进符号以与正文一致）
scales = ['3h', '24h', '72h', '120h']
xs = np.arange(len(scales))
apc = [0.0]*4; unet = [0.0]*4; qm = [0.0]*4; bm = [0.0]*4; ols = [0.0]*4

# 从 summary.json 取全部时效（era5_3h 键含 bincm_imp，24/72/120 键含 bm_imp）
S = json_load = __import__('json').load(open(os.path.join(OUTD, 'multi_lead_summary.json'), encoding='utf-8'))
for i, fhr in enumerate([3, 24, 72, 120]):
    v = S[f'era5_{fhr}h']
    if fhr == 3:
        # 注意：summary 的 3h 键中 apcnet/unet 存的是退化率（正=退化），qm/ols/bincm 为改进率（负=改进）
        apc[i] = -v['apcnet_imp']          # -14.48 ≈ Table 3 的 -14.5%
        unet[i] = v['unet_imp']            # -55.1（直接为退化）
        qm[i] = -v['qm_imp']; ols[i] = -v['ols_imp']; bm[i] = -v['bincm_imp']
    else:
        # 24-120h 键统一为负=改进
        apc[i] = -v['apcnet_imp']; unet[i] = -v['unet_imp']
        qm[i] = -v['qm_imp']; ols[i] = -v['ols_imp']
        bm[i] = -v['bm_imp'] if 'bm_imp' in v else bm[i]
# 统一为正=改进
# apc/unet/qm/bm/ols 已按正文口径（正=改进）

# CHM（RMSE 改进 %）
chm_apc = [S['chm_24h']['apcnet_imp'], S['chm_72h']['apcnet_imp'], S['chm_120h']['apcnet_imp']]
chm_unet = [S['chm_24h']['unet_imp'], S['chm_72h']['unet_imp'], S['chm_120h']['unet_imp']]
chm_e5 = [S['chm_24h']['era5_ref_imp'], S['chm_72h']['era5_ref_imp'], S['chm_120h']['era5_ref_imp']]

# GPM（MSE 改进 % = RMSE 改进 % 近似；脚本内用 summary 字段）
G = S['gpm_multi_lead']
gpm_apc = [G['24']['mse_improve_pct']['APCNet'], G['72']['mse_improve_pct']['APCNet'], G['120']['mse_improve_pct']['APCNet']]
gpm_unet = [G['24']['mse_improve_pct']['U-Net'], G['72']['mse_improve_pct']['U-Net'], G['120']['mse_improve_pct']['U-Net']]
gpm_e5 = [G['24']['mse_improve_pct']['ERA5'], G['72']['mse_improve_pct']['ERA5'], G['120']['mse_improve_pct']['ERA5']]

# 月块 bootstrap 95% CI（obs_lead_bootstrap.json：点估计+CI，正=改进）
BS = __import__('json').load(open(os.path.join(OUTD, 'obs_lead_bootstrap.json'), encoding='utf-8'))
def _ci(key, model):
    out = []
    for fhr in (24, 72, 120):
        d = BS[f'{key}_{fhr}h']
        out.append((d[f'{model}_ci95'][0], d[f'{model}_ci95'][1]))
    return out
chm_apc_ci = _ci('chm', 'apc'); chm_unet_ci = _ci('chm', 'unet')
gpm_apc_ci = _ci('gpm', 'apc'); gpm_unet_ci = _ci('gpm', 'unet')
def _err(points, cis):
    return np.array([[p - lo, hi - p] for (p, (lo, hi)) in zip(points, cis)]).T  # (2, n)
chm_apc_err = _err(chm_apc, chm_apc_ci); chm_unet_err = _err(chm_unet, chm_unet_ci)
gpm_apc_err = _err(gpm_apc, gpm_apc_ci); gpm_unet_err = _err(gpm_unet, gpm_unet_ci)

fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), gridspec_kw={'wspace': 0.45})

# Panel A
ax = axes[0]
ax.axhline(0, color='0.55', lw=0.8, zorder=1)
ax.plot(xs, apc, 'o-', color=C_APC, lw=1.6, ms=4.5, label='APCNet', zorder=3)
ax.plot(xs, unet, 's-', color=C_UNET, lw=1.6, ms=4.5, label='U-Net', zorder=3)
ax.plot(xs, qm, '^-', color=C_QM, lw=1.6, ms=4.5, label='QM', zorder=3)
ax.plot(xs, bm, 'v-', color=C_BIN, lw=1.4, ms=4, label='BinCM', zorder=2)
ax.plot(xs, ols, 'd-', color=C_OLS, lw=1.4, ms=4, label='OLS', zorder=2)
ax.set_xticks(xs); ax.set_xticklabels(scales)
ax.set_xlabel('Accumulation scale')
ax.set_ylabel('MSE improvement vs GFS (%)')
ax.set_title('Reanalysis (ERA5) reference')
ax.text(0.5, -0.28, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='center')
ax.legend(frameon=True, loc='upper right', ncol=1, facecolor='white', framealpha=0.85)
ax.set_ylim(-70, 62)
ax.annotate('positive = improvement', xy=(0.02, 0.30), xycoords='axes fraction', fontsize=7, color='0.35')

# Panel B
ax = axes[1]
ax.axhline(0, color='0.55', lw=0.8, zorder=1)
x2 = np.arange(3)
# Panel B：CHM 用方法原色实线；GPM 换浅蓝/浅黄虚线（色相区分参考，不与其他方法色冲突）
C_GPM_APC = '#56B4E9'   # 浅蓝（Okabe-Ito，全稿未用）
C_GPM_UNET = '#F0E442'  # 黄（Okabe-Ito，全稿未用）
ax.plot(x2, chm_apc, 'o-', color=C_APC, lw=1.6, ms=4.5, label='APCNet (CHM)', zorder=3)
ax.plot(x2, chm_unet, 's-', color=C_UNET, lw=1.6, ms=4.5, label='U-Net (CHM)', zorder=3)
ax.plot(x2, gpm_apc, 'o--', color=C_GPM_APC, lw=1.6, ms=4.5, label='APCNet (GPM)', zorder=3)
ax.plot(x2, gpm_unet, 's--', color=C_GPM_UNET, lw=1.6, ms=4.5, label='U-Net (GPM)', zorder=3)
ax.errorbar(x2, chm_apc, yerr=chm_apc_err, fmt='none', ecolor=C_APC, elinewidth=1.0, capsize=2.5, zorder=2)
ax.errorbar(x2, chm_unet, yerr=chm_unet_err, fmt='none', ecolor=C_UNET, elinewidth=1.0, capsize=2.5, zorder=2)
ax.errorbar(x2, gpm_apc, yerr=gpm_apc_err, fmt='none', ecolor=C_GPM_APC, elinewidth=1.0, capsize=2.5, zorder=2)
ax.errorbar(x2, gpm_unet, yerr=gpm_unet_err, fmt='none', ecolor=C_GPM_UNET, elinewidth=1.0, capsize=2.5, zorder=2)
ax.set_xticks(x2); ax.set_xticklabels(['24h', '72h', '120h'])
ax.set_xlabel('Accumulation scale')
ax.set_ylabel('Skill improvement vs GFS (%)')
ax.set_title('Independent observations (CHM & GPM)')
ax.text(0.5, -0.28, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='center')
ax.legend(frameon=False, loc='lower left', ncol=2, fontsize=6.5)
ax.set_ylim(-2, 22)

# Panel C
ax = axes[2]
S_ = S
cc_g = [S_['era5_24h']['gfs_cc'], S_['era5_72h']['gfs_cc'], S_['era5_120h']['gfs_cc']]
cc_a = [S_['era5_24h']['apcnet_cc'], S_['era5_72h']['apcnet_cc'], S_['era5_120h']['apcnet_cc']]
cc_u = [S_['era5_24h']['unet_cc'], S_['era5_72h']['unet_cc'], S_['era5_120h']['unet_cc']]
ax.plot(x2, cc_g, 'o--', color='0.45', lw=1.2, ms=4, label='GFS', zorder=2)
ax.plot(x2, cc_a, 'o-', color=C_APC, lw=1.6, ms=4.5, label='APCNet', zorder=3)
ax.plot(x2, cc_u, 's-', color=C_UNET, lw=1.6, ms=4.5, label='U-Net', zorder=3)
ax.set_xticks(x2); ax.set_xticklabels(['24h', '72h', '120h'])
ax.set_xlabel('Accumulation scale')
ax.set_ylabel('Spatial CC (vs ERA5)')
ax.set_title('Correlation, ERA5 ref.')
ax.text(0.5, -0.28, '(c)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='center')
# 三条曲线末端标注（无图例框，避免与 0.70-0.85 数据区重叠）
for xv, yv, lab, col in [(x2[-1], cc_g[-1], 'GFS', '0.45'),
                          (x2[-1], cc_a[-1], 'APCNet', C_APC),
                          (x2[-1], cc_u[-1], 'U-Net', C_UNET)]:
    ax.annotate(lab, xy=(xv, yv), xytext=(6, 0), textcoords='offset points',
                fontsize=7, color=col, va='center', ha='left')
ax.set_xlim(-0.35, 2.55)
ax.set_ylim(0.7, 0.85)

fig.savefig(r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\figures_300dpi\Fig12.png', dpi=300, bbox_inches='tight')
print('saved Fig12.png')
print('bm =', bm)
print('ols =', ols)
