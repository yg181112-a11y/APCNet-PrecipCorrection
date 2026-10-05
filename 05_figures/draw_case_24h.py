# -*- coding: utf-8 -*-
"""draw_case_24h.py — 24h 强降水个例六面板对比图（Fig.13 候选）。
面板: GFS / APCNet / U-Net / QM / ERA5 / GPM 观测（24h 累积 mm）。
个例: 测试期(2024-2025)域均 GPM 24h 累积最大的 00Z 日期。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np, os
from datetime import datetime, timedelta

BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
EXP = os.path.join(BASE, '24h_exp')
OUTD = r'C:\Users\yg181\Desktop\论文三\WAF\r3_media'
os.makedirs(OUTD, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
})
LATS = np.linspace(46.0, 40.0, 25)
LONS = np.linspace(117.0, 126.0, 37)

# ---- 数据 ----
gpm = np.load(os.path.join(EXP, 'gpm24_accum_00z.npy'))     # [N00,25,37]
init00 = np.load(os.path.join(EXP, 'init_00z.npy'), allow_pickle=True)
split00 = np.load(os.path.join(EXP, 'split_00z.npy'))
cov = np.load(os.path.join(EXP, 'gpm24_mask_00z.npy'))
gfs00 = np.load(os.path.join(EXP, 'gfs24_accum_00z.npy'))
era500 = np.load(os.path.join(EXP, 'era5_24h_accum_00z.npy'))

# 全 init 数组（预测索引映射）
gfs_all = np.load(os.path.join(EXP, 'gfs_24h_accum.npy'))
mask_all = np.load(os.path.join(EXP, 'split_mask.npy'))
init_all = np.load(os.path.join(EXP, 'init_times.npy'), allow_pickle=True)
apc_all = np.load(os.path.join(EXP, 'pred_apcnet_24h.npy'))
unet_all = np.load(os.path.join(EXP, 'unet_24h.npy'))
qm_all = np.load(os.path.join(EXP, 'qm_24h.npy'))

# 测试期有效全局索引（pred 数组行序）
te_all = np.where(mask_all == 2)[0]
te_nan = np.isnan(gfs_all[te_all]).any(axis=(1, 2))
valid_te_global = te_all[~te_nan]

# 00Z 测试期域均 GPM（仅 cov 格点）
te00 = split00 == 2
te00_idx = np.where(te00)[0]
gpm_te00 = gpm[te00_idx]
gpm_dm = gpm_te00[:, cov].mean(axis=1)
dates00 = [datetime(1970, 1, 1) + timedelta(hours=int(t)) for t in init00[te00].astype('datetime64[h]').astype(np.int64)]
i_best = int(np.argmax(gpm_dm))
i_best_global = int(te00_idx[i_best])
best_date = dates00[i_best]
print(f'个例: {best_date}  域均 GPM 24h = {gpm_dm[i_best]:.1f} mm  最大格点 = {gpm[i_best_global][cov].max():.1f} mm')

# 该日期全局索引（init_all 中 00Z 位置）
target_dt = best_date
glob_idx = int(np.where((init_all.astype('datetime64[h]').astype(np.int64) == (target_dt - datetime(1970,1,1)).total_seconds()//3600))[0][0])
pos = int(np.searchsorted(valid_te_global, glob_idx))
print('全局索引', glob_idx, 'pred 位置', pos, 'init 校验', datetime(1970,1,1)+timedelta(hours=int(init_all[glob_idx].astype('datetime64[h]').astype(np.int64))))

fields = {
    'GFS': gfs_all[glob_idx],
    'APCNet': apc_all[pos],
    'U-Net': unet_all[pos],
    'QM': qm_all[pos],
    'ERA5': era500[i_best_global],
    'GPM (obs)': gpm[i_best_global],
}
print('ERA5 域均', era500[i_best_global][cov].mean().round(2), 'GFS', fields['GFS'][cov].mean().round(2),
      'APCNet', fields['APCNet'][cov].mean().round(2), 'U-Net', fields['U-Net'][cov].mean().round(2))

vmax = max(np.nanmax(f[cov]) for f in fields.values())
norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=5.0, vmax=min(vmax, 80))
cmap = plt.get_cmap('turbo')

fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4),
                         gridspec_kw={'hspace': 0.30, 'wspace': 0.10})
titles = list(fields.keys())
for k, (ax, (name, f)) in enumerate(zip(axes.ravel(), fields.items())):
    im = ax.imshow(f, cmap=cmap, norm=norm, origin='upper',
                   extent=[LONS[0], LONS[-1], LATS[-1], LATS[0]], aspect='auto')
    ax.set_title(f'({chr(97+k)}) {name}', fontsize=9.5)
    if k % 3 == 0:
        ax.set_ylabel('Lat (°N)')
    if k >= 3:
        ax.set_xlabel('Lon (°E)')
        ax.set_xticks([118, 120, 122, 124, 126])
    else:
        ax.set_xticks([])
    ax.set_yticks([41, 43, 45])
    if k % 3 != 0:
        ax.set_yticklabels([])
    dm = np.nanmean(f[cov])
    ax.text(0.02, 0.03, f'{dm:.1f} mm', transform=ax.transAxes, fontsize=8,
            color='white', ha='left', va='bottom',
            bbox=dict(fc='black', alpha=0.45, lw=0, pad=1))

cbar_ax = fig.add_axes([0.93, 0.12, 0.025, 0.76])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('24-h accumulated precipitation (mm)', fontsize=9)

fig.suptitle(f'{best_date.strftime("%Y-%m-%d")} 00Z, 24-h accumulation (init-00Z window)', y=0.98, fontsize=11)
fig.savefig(os.path.join(OUTD, 'fig_case_24h.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTD, 'fig_case_24h.png'), dpi=300, bbox_inches='tight')
print('✅ 已保存 fig_case_24h.pdf/.png')
