# -*- coding: utf-8 -*-
"""P3-Q6: BinCM 分时段敏感性 + 训练/测试期样本代表性。
- 全期 bin 表 vs 前半期(2015-2018) vs 后半期(2019-2021) 拟合的 bin 表，在测试期(2024-2025)盲测 MSE
- bin 表分时段最大差异（量化气候漂移）
- 训练期 vs 测试期 GFS/ERA5 域均值（回应 2024 样本代表性）
"""
import numpy as np, os, json

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
def L(name):
    return np.load(os.path.join(WORK, name)).astype(np.float32)

tr_g = L("train_gfs.npy"); tr_e = L("train_era5.npy")
te_g = L("gfs_test.npy");  te_e = L("targets_test.npy")

edges = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 100.0]
def bin_of(x):
    b = np.zeros_like(x, dtype=np.int16)
    for i in range(1, len(edges)):
        b[x > edges[i]] = i
    return b

def fit_bin_table(g, e):
    tg, te_ = g.ravel(), e.ravel()
    b = bin_of(tg); n_bins = len(edges)
    cnt = np.bincount(b, minlength=n_bins)
    s = np.bincount(b, weights=te_, minlength=n_bins)
    return np.divide(s, np.maximum(cnt, 1), out=np.zeros(n_bins), where=cnt > 0)

def apply_bin(cm, g):
    return cm[bin_of(g.ravel())].reshape(g.shape)

def mse(o, f): return float(np.mean((o - f) ** 2))

res = {}
# 训练期拆分（按时间顺序，train_gfs 为 2015-2021 展平时次）
ntr = len(tr_g)
half = ntr // 2
cm_full = fit_bin_table(tr_g, tr_e)
cm_h1 = fit_bin_table(tr_g[:half], tr_e[:half])
cm_h2 = fit_bin_table(tr_g[half:], tr_e[half:])
print('train 时次 = %d, 前半 %d / 后半 %d' % (ntr, half, ntr - half))

m_g = mse(te_e, te_g)
res['gfs_mse'] = m_g
res['test_n'] = int(len(te_g))
for nm, cm in [('full', cm_full), ('first_half', cm_h1), ('second_half', cm_h2)]:
    p = apply_bin(cm, te_g)
    m = mse(te_e, p)
    res[nm] = {'mse': m, 'improve_pct': 100 * (m_g - m) / m_g}
    print('BinCM[%s] MSE=%.4f 改进=%+.2f%%' % (nm, m, res[nm]['improve_pct']))

# bin 表分时段差异
diff_h12 = np.abs(cm_h1 - cm_h2)
diff_full = np.abs(cm_full - np.mean([cm_h1, cm_h2], axis=0))
res['bin_table'] = {
    'bins': edges,
    'cm_full': cm_full.tolist(),
    'cm_first_half': cm_h1.tolist(),
    'cm_second_half': cm_h2.tolist(),
    'max_abs_diff_h1_h2': float(np.max(diff_h12)),
    'mean_abs_diff_h1_h2': float(np.mean(diff_h12)),
    'max_abs_diff_full_vs_mean': float(np.max(diff_full)),
}
print('bin 表 H1 vs H2: max abs diff = %.4f, mean = %.4f' % (
    res['bin_table']['max_abs_diff_h1_h2'], res['bin_table']['mean_abs_diff_h1_h2']))

# 样本代表性：域均强度
def domain_mean(arr): return float(arr.mean())
res['representativeness'] = {
    'train_gfs_mean': domain_mean(tr_g), 'train_era5_mean': domain_mean(tr_e),
    'test_gfs_mean': domain_mean(te_g), 'test_era5_mean': domain_mean(te_e),
    'test_over_train_gfs': float(te_g.mean() / tr_g.mean()),
    'test_over_train_era5': float(te_e.mean() / tr_e.mean()),
    'train_gfs_std': float(tr_g.std()), 'test_gfs_std': float(te_g.std()),
    'train_era5_std': float(tr_e.std()), 'test_era5_std': float(te_e.std()),
}
print('代表性: train GFS %.4f/ERA5 %.4f | test GFS %.4f/ERA5 %.4f | 比值 GFS %.3f ERA5 %.3f' % (
    res['representativeness']['train_gfs_mean'], res['representativeness']['train_era5_mean'],
    res['representativeness']['test_gfs_mean'], res['representativeness']['test_era5_mean'],
    res['representativeness']['test_over_train_gfs'], res['representativeness']['test_over_train_era5']))

with open(os.path.join(WORK, 'bincm_split_sens.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print('✅ bincm_split_sens.json 已保存')
