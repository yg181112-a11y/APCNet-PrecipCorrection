# -*- coding: utf-8 -*-
"""draw_qq_season.py — 24h 分位数-分位数诊断图 + 季节分解表（00Z 测试期）。
QQ: GFS / APCNet(ERA5) / U-Net(ERA5) / U-Net(GPM) / QM vs GPM 观测（200 分位点）。
季节: warm JJAS vs cool，含全部方法 MSE 改进。
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
})

E = r'D:\liaohe\论文三\03_重建成稿代_2026_R3全链主实验\24h_exp'
MEDIA = r'D:\liaohe\论文三\WAF\r3_media'
os.makedirs(MEDIA, exist_ok=True)

gpm = np.load(os.path.join(E, 'gpm24_accum_00z.npy'))
gfs = np.load(os.path.join(E, 'gfs24_accum_00z.npy'))
era5 = np.load(os.path.join(E, 'era5_24h_accum_00z.npy'))
split = np.load(os.path.join(E, 'split_00z.npy'))
cov = np.load(os.path.join(E, 'gpm24_mask_00z.npy'))
init00 = np.load(os.path.join(E, 'init_00z.npy'), allow_pickle=True)
te = split == 2

t = gpm[te][:, cov]
g = gfs[te][:, cov]

def slice_allinit(pred_all, m):
    """把全体 init 预测切到 00Z 测试期（与 _tmp_cmp 同逻辑）。"""
    return pred_all[m][:, cov]

# 00Z 切片（全体 init 模型）
apc = np.load(os.path.join(E, 'pred_apcnet_24h.npy'))       # 全体 init
unet_e = np.load(os.path.join(E, 'unet_24h.npy'))           # 全体 init
qm = np.load(os.path.join(E, 'qm_24h.npy'))                 # 全体 init
mask_all = np.load(os.path.join(E, 'split_mask.npy'))
init_all = np.load(os.path.join(E, 'init_times.npy'), allow_pickle=True)
gfs_all = np.load(os.path.join(E, 'gfs_24h_accum.npy'))
te_all = np.where(mask_all == 2)[0]
te_nan = np.isnan(gfs_all[te_all]).any(axis=(1, 2))
valid_te_global = te_all[~te_nan]
sel_global = np.where(te)[0]
init_te00 = init00[sel_global]
idx_all = []
for it in init_te00:
    t0 = it.astype('datetime64[h]').astype(np.int64)
    j = int(np.where(init_all.astype('datetime64[h]').astype(np.int64) == t0)[0][0])
    idx_all.append(int(np.searchsorted(valid_te_global, j)))
idx_all = np.array(idx_all)

methods = {
    'GFS': g,
    'APCNet (ERA5 target)': apc[idx_all][:, cov],
    'U-Net (ERA5 target)': unet_e[idx_all][:, cov],
    'U-Net (GPM target)': np.load(os.path.join(E, 'pred_unet_gpm_24h_00z_nowt.npy'))[:, cov],
    'QM': qm[idx_all][:, cov],
}

def cont(o, f):
    o, f = o.flatten(), f.flatten()
    v = np.isfinite(o) & np.isfinite(f)
    o, f = o[v], f[v]
    mse = np.mean((o - f) ** 2)
    return mse, float(np.sqrt(mse)), float(np.corrcoef(o, f)[0, 1]), float(np.mean(f - o))

m_g, _, _, _ = cont(t, g)
print('=== 24h 00Z 测试期 GPM 验证（QQ 图用）===')
print('%-24s %8s %8s %8s %8s' % ('method', 'RMSE', 'CC', 'bias', 'MSEimp%'))
for k, a in methods.items():
    mse, rmse, cc, bias = cont(t, a)
    print('%-24s %8.3f %8.3f %8.2f %+7.1f' % (k, rmse, cc, bias, 100.0 * (m_g - mse) / m_g))

# ---- QQ 图 ----
qs = np.linspace(0.001, 0.999, 200)
obs_q = np.quantile(t, qs)
fig, ax = plt.subplots(figsize=(5.2, 5.2))
colors = {'GFS': '#333333', 'APCNet (ERA5 target)': '#D55E00',
          'U-Net (ERA5 target)': '#CC79A7', 'U-Net (GPM target)': '#CC79A7',
          'QM': '#E69F00'}
for k, a in methods.items():
    ls = '--' if k == 'U-Net (GPM target)' else '-'
    ax.plot(obs_q, np.quantile(a, qs), color=colors[k], lw=1.6, ls=ls, label=k)
ax.plot(obs_q, obs_q, 'k--', lw=1.0, label='1:1')
ax.set_xlabel('GPM observed 24-h accumulation (mm)')
ax.set_ylabel('Predicted 24-h accumulation (mm)')
ax.set_title('Quantile–quantile, 00Z test period (2024–2025)')
ax.set_xlim(0, 60); ax.set_ylim(0, 60)
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(r'D:\liaohe\论文三\04_定稿投稿代_2026_R3投稿包与归档\投稿系统上传\figures_300dpi\Fig13.png', dpi=300)
print('\n✅ 已保存 Fig13.png')

# ---- 季节分解 ----
months = [datetime_m for datetime_m in []]
from datetime import datetime, timedelta
tstr = [datetime(1970, 1, 1) + timedelta(hours=int(x)) for x in init00[te].astype('datetime64[h]').astype(np.int64)]
tstr = [x.month for x in tstr]
warm = np.array([m in [6, 7, 8, 9] for m in tstr])
print('\n=== 季节分解（24h, 00Z 测试期）===')
for name, msk in [('warm JJAS', warm), ('cool others', ~warm)]:
    tw, gw = t[msk], g[msk]
    m_gw, _, _, _ = cont(tw, gw)
    line = f'{name}  n={msk.sum()}:  GFS RMSE={np.sqrt(m_gw):.2f}'
    parts = [line]
    for k, a in methods.items():
        aw = a[msk]
        mse, rmse, cc, bias = cont(tw, aw)
        parts.append('  %s %+5.1f%%(RMSE %.2f)' % (k.split(' (')[0], 100.0 * (m_gw - mse) / m_gw, rmse))
    print('  '.join(parts))
