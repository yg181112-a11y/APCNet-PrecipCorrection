# -*- coding: utf-8 -*-
"""draw_qq_3h.py — 3h 分位数-分位数图（GPM 真值）。
对齐主实验预测(sample_times_test, 2839) 与 3h GPM OLS 数据集(2552) 的公共 init 时次。
曲线: GFS / APCNet / QM / OLS(ERA5目标) / U-Net / OLS(GPM目标) vs GPM obs。
"""
import os, pickle
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
E = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\3h_exp'
MEDIA = r'C:\Users\yg181\Desktop\论文三\WAF\r3_media'

# 主实验测试期（2839 时次）
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    stimes = pickle.load(f)
apc = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
ols = np.load(os.path.join(WORK, 'predictions_ols.npy'))
unet = np.load(os.path.join(WORK, 'predictions_unet.npy'))
gfs_ref = np.load(os.path.join(WORK, 'gfs_test.npy'))
print('主实验: samples', len(stimes), apc.shape)

# 我的 3h 数据集（2552）
gfs3h = np.load(os.path.join(E, 'gfs3h_test.npy'))
gpm3h = np.load(os.path.join(E, 'gpm3h_test.npy'))
init3h = np.load(os.path.join(E, 'init3h_test.npy'))
pred_ols_gpm = np.load(os.path.join(E, 'pred_ols_gpm3h_test.npy'))
cov = np.load(os.path.join(E, 'gpm3h_cov.npy'))
print('3h 数据集: samples', len(init3h))

# 主实验 t0(valid) -> init = t0 - 3h
init_main = []
for t0 in stimes:
    init_main.append((t0 - timedelta(hours=3)).strftime('%Y%m%d%H'))
init_main = np.array(init_main)

# 匹配
idx_main = []
idx_new = []
new_set = set(init3h)
for k, it in enumerate(init_main):
    if it in new_set:
        j = int(np.where(init3h == it)[0][0])
        idx_main.append(k); idx_new.append(j)
idx_main = np.array(idx_main); idx_new = np.array(idx_new)
print('匹配样本:', len(idx_main))

# 评估（cov 格点）
def cont(o, f):
    o, f = o.flatten(), f.flatten()
    v = np.isfinite(o) & np.isfinite(f)
    o, f = o[v], f[v]
    return np.mean((o - f) ** 2), float(np.sqrt(np.mean((o - f) ** 2))), float(np.corrcoef(o, f)[0, 1]), float(np.mean(f - o))

T = gpm3h[idx_new][:, cov]
G = gfs3h[idx_new][:, cov]
methods = {
    'GFS': G,
    'APCNet': apc[idx_main][:, cov],
    'QM': qm[idx_main][:, cov],
    'OLS (ERA5 target)': ols[idx_main][:, cov],
    'U-Net (ERA5 target)': unet[idx_main][:, cov],
    'OLS (GPM target)': pred_ols_gpm[idx_new][:, cov],
}
mg = cont(T, G)
print('\n=== 3h GPM 验证（QQ 对齐子集, n=%d）===' % len(idx_new))
print('%-22s %8s %8s %8s %8s' % ('method', 'RMSE', 'CC', 'bias', 'MSEimp%'))
for k, a in methods.items():
    mse, rmse, cc, bias = cont(T, a)
    print('%-22s %8.3f %8.3f %8.3f %+7.1f' % (k, rmse, cc, bias, 100.0 * (mg[0] - mse) / mg[0]))

# QQ 图
qs = np.linspace(0.001, 0.999, 200)
obs_q = np.quantile(T, qs)
colors = {'GFS': '#555555', 'APCNet': '#E69F00', 'QM': '#D55E00',
          'OLS (ERA5 target)': '#CC79A7', 'U-Net (ERA5 target)': '#0072B2',
          'OLS (GPM target)': '#009E73'}
fig, ax = plt.subplots(figsize=(5.2, 5.2))
for k, a in methods.items():
    ax.plot(obs_q, np.quantile(a, qs), color=colors[k], lw=1.6, label=k)
ax.plot(obs_q, obs_q, 'k--', lw=1.0, label='1:1')
ax.set_xlabel('GPM observed 3-h accumulation (mm)')
ax.set_ylabel('Predicted 3-h accumulation (mm)')
ax.set_title('Quantile-quantile, 3-h test period (2024-2025)')
ax.set_xlim(0, 20); ax.set_ylim(0, 20)
ax.legend(frameon=False, fontsize=7.5, loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(MEDIA, 'fig_qq_3h.pdf'))
fig.savefig(os.path.join(MEDIA, 'fig_qq_3h.png'), dpi=300)
print('✅ 已保存 fig_qq_3h.pdf/.png')
