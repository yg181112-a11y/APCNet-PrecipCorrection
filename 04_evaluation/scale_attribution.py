# -*- coding: utf-8 -*-
"""
3h vs 24h APCNet 归因诊断
对比尺度依赖的：信噪比、残差结构、条件可学性、DL 行为、特征可预测性
"""
import numpy as np, json, os

MW = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
E24 = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\24h_exp'

g3 = np.load(os.path.join(MW, 'gfs_test.npy'))          # (2839,25,37) mm/3h
t3 = np.load(os.path.join(MW, 'targets_test.npy'))
p3 = np.load(os.path.join(MW, 'predictions_apcnet.npy'))
q3 = np.load(os.path.join(MW, 'predictions_qm.npy'))

g24 = np.load(os.path.join(E24, 'gfs_24h_accum.npy'))
t24 = np.load(os.path.join(E24, 'era5_24h_accum.npy'))
p24 = np.load(os.path.join(E24, 'pred_apcnet_24h.npy'))
split = np.load(os.path.join(E24, 'split_mask.npy'))
valid = ~(np.isnan(g24).any((1, 2)) | np.isnan(t24).any((1, 2)))
tm = (split == 2) & valid
g24, t24 = g24[tm], t24[tm]
# pred_apcnet_24h.npy 已是测试集过滤后数组（2916）
p24 = p24[:len(g24)]
print('shapes: 3h', g3.shape, '24h', g24.shape)

def mse(a, b): return float(np.mean((a - b) ** 2))
def cc(a, b): return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

out = {}
for tag, g, t, p in [('3h', g3, t3, p3), ('24h', g24, t24, p24)]:
    r = t - g  # 残差
    d = {}
    d['mse_gfs'] = mse(g, t); d['rmse_gfs'] = np.sqrt(d['mse_gfs'])
    d['mse_apc'] = mse(p, t); d['dmse_pct'] = (d['mse_apc'] / d['mse_gfs'] - 1) * 100
    d['cc_gfs'] = cc(g, t); d['cc_apc'] = cc(p, t)
    d['mean_target'] = float(t.mean()); d['mean_gfs'] = float(g.mean()); d['mean_apc'] = float(p.mean())
    d['bias_gfs'] = float((g - t).mean()); d['bias_apc'] = float((p - t).mean())
    d['resid_std'] = float(r.std()); d['snr_inv'] = float(r.std() / max(t.mean(), 1e-6))  # 噪声/信号
    d['rain_frac_t'] = float((t >= 0.1).mean()); d['rain_frac_g'] = float((g >= 0.1).mean())
    # 残差分布特征（非零残差）
    rz = r[r != 0]
    d['resid_p90'] = float(np.quantile(rz, 0.9)); d['resid_p99'] = float(np.quantile(rz, 0.99))
    d['resid_pos_frac'] = float((r > 0).mean())
    # 条件可学性：按 GFS 分箱的 E[residual|gfs]（BinCM 结构）形状
    bins = [0, 0.1, 1, 3, 5, 10, 20, 50, 1e9]
    ce, cnt = [], []
    for k in range(len(bins) - 1):
        m = (g >= bins[k]) & (g < bins[k + 1])
        ce.append(float(r[m].mean()) if m.sum() > 50 else None)
        cnt.append(int(m.sum()))
    d['cond_exp'] = ce; d['cond_cnt'] = cnt
    # 条件期望的离散度（可学信号强度）：加权 std of cond_exp
    vals = [c for c in ce if c is not None]
    d['cond_range'] = float(max(vals) - min(vals)) if vals else None
    d['cond_sd'] = float(np.std(vals)) if vals else None
    # 样本级（时次）可预测性：每样本格点相关分布
    ns = len(g)
    per = np.array([np.corrcoef(g[i].ravel(), t[i].ravel())[0, 1] for i in range(0, ns, 7)])
    d['per_sample_cc_med'] = float(np.nanmedian(per)); d['per_sample_cc_iqr'] = float(np.nanpercentile(per, 75) - np.nanpercentile(per, 25))
    # 残差与 GFS 的逐格点相关（条件结构的线性代理）
    d['corr_resid_gfs'] = float(np.corrcoef(r.ravel(), g.ravel())[0, 1])
    # 极端误差：GFS>=20mm 时的目标分布
    m20 = g >= 20
    d['n_gfs20'] = int(m20.sum())
    d['t20_mean'] = float(t[m20].mean()) if m20.any() else None
    d['p20_mean'] = float(p[m20].mean()) if m20.any() else None
    out[tag] = d

print(json.dumps(out, ensure_ascii=False, indent=2))
json.dump(out, open(os.path.join(MW, 'scale_attribution_diag.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# QM 3h 也在场，直接算 24h QM 无数组 → 用 summary
print('3h QM MSE:', mse(q3, t3), 'pct:', (mse(q3, t3) / mse(g3, t3) - 1) * 100)
