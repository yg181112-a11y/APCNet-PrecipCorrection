# -*- coding: utf-8 -*-
"""R3 稿新增 Fig 11：ERA5 参考确定性技能的 Taylor 图（期刊风格）"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.projections import PolarAxes
import os

MW = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
OUT = r'D:\liaohe\论文三\03_重建成稿代_2026_R3全链主实验'
os.makedirs(OUT, exist_ok=True)

def load(n):
    return np.load(os.path.join(MW, n))

ref = load('targets_test.npy')
methods = {
    'GFS': load('gfs_test.npy'),
    'QM': load('predictions_qm.npy'),
    'OLS': load('predictions_ols.npy'),
    'BinCM': load('bin_cm_pred.npy'),
    'APCNet': load('predictions_apcnet.npy'),
    'U-Net': load('predictions_unet.npy'),
}

# 全格点-时次展平
ref_f = ref.ravel().astype(np.float64)
data = {}
for name, arr in methods.items():
    f = arr.ravel().astype(np.float64)
    m = np.isfinite(ref_f) & np.isfinite(f)
    r_, p_ = ref_f[m], f[m]
    cc = np.corrcoef(r_, p_)[0, 1]
    std_ratio = p_.std() / r_.std()
    data[name] = (cc, std_ratio)
    print(f'{name}: corr={cc:.4f} std_ratio={std_ratio:.4f}')

# ---------------- 绘制 ----------------
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8

# 全稿统一"方法—颜色"映射（v4 N3）：GFS 深灰、QM 橙、OLS 蓝、BinCM 绿、APCNet 朱红、U-Net 粉
std_colors = {'GFS': '#333333', 'QM': '#E69F00', 'OLS': '#0072B2', 'BinCM': '#009E73',
              'APCNet': '#D55E00', 'U-Net': '#CC79A7'}
markers = ['o', 's', '^', 'D', 'v', 'P', 'p']
colors = {k: std_colors[k] for k in methods}
marks = dict(zip(methods.keys(), markers[:len(methods)]))

fig = plt.figure(figsize=(6.4, 4.6), dpi=150)
ax = fig.add_axes([0.44, 0.08, 0.54, 0.85], projection='polar')

# 参考弧
std_max = 1.6
theta_max = 90.0
theta_ticks = np.arange(0, 100, 10)
r_ticks = [0.5, 1.0, 1.5]

for r in r_ticks:
    ax.plot(np.radians(np.linspace(0, theta_max, 100)), np.full(100, r),
            color='0.6', lw=0.5, zorder=0)
for t in theta_ticks:
    ax.plot(np.full(100, np.radians(t)), np.linspace(0, std_max, 100),
            color='0.75', lw=0.4, zorder=0)

# 参考点
ax.plot(np.radians(0), 1.0, 'k*', ms=12, label='ERA5 reference', zorder=5)

for name, (cc, sr) in data.items():
    th = np.degrees(np.arccos(cc))
    # 白色 halo 分离重叠标记（加大，保证打印尺寸下可分辨）
    ax.plot(np.radians(th), sr, marker=marks[name], color='white', ms=18,
            mec='none', lw=0, zorder=6.2)
    ax.plot(np.radians(th), sr, marker=marks[name], color=colors[name],
            ms=10, mec='k', mew=0.7, lw=1.2, label=name, zorder=6.5)
    # 每个方法加名字标注：polar 轴上 annotate 的 text 坐标有已知缺陷，改用 ax.text + 手动 leader 线段
    # 偏移量按实测坐标（corr≈0.60-0.63 → θ≈51°-53°，std 0.65-1.46 分层）手工布局：
    # 低/中半径点左列（θ-24°），高半径点右列（θ+20°），避开 45° r 标签与 0° 参考星
    offs = {'GFS': (-0.42, 0.16), 'QM': (-0.42, 0.10),
            'OLS': (-0.42, 0.08), 'BinCM': (-0.42, -0.25),
            'APCNet': (0.35, 0.04), 'U-Net': (0.35, -0.06)}
    dx, dy = offs.get(name, (0.08, 0.04))
    tth = np.radians(th) + dx
    tr = sr + dy
    ax.plot([np.radians(th), tth], [sr, tr], color='0.4', lw=0.5, zorder=6.8)
    ax.text(tth, tr, name, fontsize=8, ha='center', va='center',
            color='0.05', zorder=7)

# 径向刻度：去掉 set_rgrids 的全圆网格线，改在 45° 弧上手动放 r 标签（90°-180° 保持空白，供说明文字使用）
for r in r_ticks:
    ax.text(np.radians(45), r + 0.04, f'{r:.1f}', fontsize=8,
            ha='center', va='bottom', color='0.3')
# 关键：不得使用 set_rorigin 负值——它会把径向轴原点偏移，压缩 r∈[0.65,1.46] 的全部标记到同一半径带（Fig14 标记堆叠的根源）
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_thetagrids(theta_ticks, labels=[f'{t:.0f}°' if t > 0 else '0°' for t in theta_ticks], fontsize=7.5)

ax.set_title('Deterministic skill on the ERA5 reference (test period 2024-2025)', fontsize=9.5, pad=16)
# 说明文字：全部放右下空扇区（90°-180°，无网格线、无弧），避开 45° r 标签与 0°-90° 刻度标签区
ax.text(np.radians(120), 1.45, 'Correlation', fontsize=8, color='0.25',
        ha='center', va='bottom')
ax.text(np.radians(140), 1.00, 'Normalized std (Ref = 1.0)', fontsize=8, color='0.25',
        ha='center', va='center')

lg = fig.legend(loc='center left', bbox_to_anchor=(0.02, 0.50), frameon=False,
                handlelength=2.0, ncol=1, fontsize=8)
ax.set_ylim(0, std_max)
ax.set_rmax(std_max)
ax.set_rgrids([])  # 关闭 set_rmax 触发的默认径向刻度线与标签（手动 45° 标签与 0-90° 弧已自行绘制）

fig.savefig(r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\figures_300dpi\Fig14.png',
            dpi=300, bbox_inches='tight')
print('saved Fig14.png')
