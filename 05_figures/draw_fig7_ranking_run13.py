# -*- coding: utf-8 -*-
"""重绘 Fig.7 排名图（run13 权威数字，含 U-Net），并替换 docx 中旧图。"""
import sys, os
sys.path.insert(0, r'C:\Users\yg181\AppData\Roaming\Python\Python313\site-packages')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
})
import numpy as np

FIG = r'D:\liaohe\论文三\WAF\r3_media'
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito 色
C_BIN = '#009E73'; C_OLS = '#0072B2'; C_QM = '#E69F00'; C_APC = '#D55E00'; C_UNET = '#CC79A7'
C_GFS = '#000000'

# ===== run13 权威数字 =====
# 口径：ERA5 与 GPM 参照为 MSE improvement（Tables 4 & 8），CHM 为 RMSE improvement（Table 14）。
# 为统一绘图轴，ERA5/GPM 换算为 RMSE 等价改进：RMSE_imp = 1 - sqrt(1 - MSE_imp/100)。
def mse2rmse(x):
    return round(100.0 * (1.0 - (1.0 - x / 100.0) ** 0.5), 2)

era5_mse = {'BinCM': 27.3, 'OLS': 22.2, 'QM': 11.1, 'APCNet': -14.5, 'U-Net': -55.1}
gpm_mse = {'BinCM': 22.6, 'OLS': 19.6, 'QM': 8.3, 'APCNet': -16.5, 'U-Net': -47.3}
era5 = {k: mse2rmse(v) for k, v in era5_mse.items()}
gpm6 = {k: mse2rmse(v) for k, v in gpm_mse.items()}
chm6 = {'BinCM': 12.5, 'OLS': 12.15, 'QM': 3.58, 'APCNet': -20.82, 'U-Net': -36.84}

fig, ax = plt.subplots(figsize=(6.8, 3.5))
methods = ['BinCM', 'OLS', 'QM', 'APCNet', 'U-Net']
refs = ['ERA5 (training ref)', 'CHM (obs, daily)', 'GPM IMERG (obs, 3h)']
# 柱按参照着色（图例与柱严格对应）；方法由 x 位置区分
# 换配色：ERA5=蓝, CHM=绿, GPM=橙（Okabe-Ito 高区分度）
ref_colors = ['#0072B2', '#009E73', '#E69F00']
x6 = np.arange(len(methods)); w = 0.26
for i, ref in enumerate(refs):
    src = era5 if i == 0 else (chm6 if i == 1 else gpm6)
    labeled = False
    for j, m in enumerate(methods):
        v = src[m]
        if v is None:
            continue
        ax.bar(x6[j] + (i - 1) * w, v, w, color=ref_colors[i], edgecolor='k', linewidth=0.4,
               label=ref if not labeled else None)
        labeled = True
        dy = 2.2 if v >= 0 else -4.6
        ax.text(x6[j] + (i - 1) * w, v + dy, '%.1f' % v, ha='center', fontsize=7)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xticks(x6); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel('RMSE improvement vs GFS (%)', fontsize=9)
ax.set_ylim(-42, 22)
ax.set_title('Consistent ranking across three references: BinCM > OLS > QM > APCNet > U-Net (GFS = 0 baseline)', fontsize=9)
ax.legend(frameon=False, loc='upper right', fontsize=6.8, ncol=1)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
out = r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\figures_300dpi\Fig08.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print('saved', out)

# ===== docx 回填由统一脚本处理（本脚本只渲染，不再直接改旧手稿） =====
print('render only; docx replacement handled centrally')
