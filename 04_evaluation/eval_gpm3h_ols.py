# -*- coding: utf-8 -*-
"""eval_gpm3h_ols.py — 3h OLS 目标对照（GPM 目标 vs 已有 ERA5 目标 +19.63%）。
逐格点线性: gpm_3h = a + b*gfs_3h（训练期拟合），测试期应用。
评估 vs GFS（GPM 真值，cov 828 格点）。另加残差 OLS 变体（y = gfs + c + d*(gfs-gpm)_train）。
"""
import os, json
import numpy as np

E = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\3h_exp'
gfs_tr = np.load(os.path.join(E, 'gfs3h_train.npy'))
gpm_tr = np.load(os.path.join(E, 'gpm3h_train.npy'))
gfs_te = np.load(os.path.join(E, 'gfs3h_test.npy'))
gpm_te = np.load(os.path.join(E, 'gpm3h_test.npy'))
cov = np.load(os.path.join(E, 'gpm3h_cov.npy'))
print('train', gfs_tr.shape, 'test', gfs_te.shape, 'cov', int(cov.sum()))

def cont(o, f):
    o, f = o.flatten(), f.flatten()
    v = np.isfinite(o) & np.isfinite(f)
    o, f = o[v], f[v]
    mse = np.mean((o - f) ** 2)
    return mse, float(np.sqrt(mse)), float(np.corrcoef(o, f)[0, 1]), float(np.mean(f - o))

# 逐格点线性拟合（含截距）
H, W = 25, 37
a_fit = np.zeros((H, W)); b_fit = np.zeros((H, W))
for i in range(H):
    for j in range(W):
        if not cov[i, j]:
            continue
        x = gfs_tr[:, i, j]; y = gpm_tr[:, i, j]
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        b_fit[i, j], a_fit[i, j] = coef
print('斜率 b 域均: %.3f  截距 a 域均: %.3f' % (b_fit[cov].mean(), a_fit[cov].mean()))

pred_ols = b_fit * gfs_te + a_fit
pred_ols = np.clip(pred_ols, 0, 500)

mg = cont(gpm_te[:, cov], gfs_te[:, cov])
mo = cont(gpm_te[:, cov], pred_ols[:, cov])
imp = 100.0 * (mg[0] - mo[0]) / mg[0]
print('\n[3h GPM 目标 OLS — 测试期 GPM 验证]')
print('GFS   : RMSE=%.3f CC=%.3f bias=%+.2f' % (mg[1], mg[2], mg[3]))
print('OLS(GPM目标): RMSE=%.3f CC=%.3f bias=%+.2f  MSE改进=%+.1f%%' % (mo[1], mo[2], mo[3], imp))

# 对比参考：3h ERA5 目标 OLS（verify_gpm_independent 输出）
ref = r'D:\liaohe\校正优化过程\第三阶段\12优化\12.8修\output\gpm_independent_eval.json'
try:
    d = json.load(open(ref, encoding='utf-8'))
    ols = d.get('OLS') or {}
    print('(参考) 3h ERA5 目标 OLS: RMSE=%.3f MSE改进=%+.1f%%' % (ols.get('RMSE', float('nan')), 100.0 * ols.get('mse_improve_pct', 0)))
except Exception as e:
    print('(参考) 读取失败:', e)

out = {'n_train': len(gfs_tr), 'n_test': len(gfs_te), 'cov': int(cov.sum()),
       'GFS': {'RMSE': mg[1], 'CC': mg[2], 'bias': mg[3], 'MSE': mg[0]},
       'OLS_GPM_target': {'RMSE': mo[1], 'CC': mo[2], 'bias': mo[3], 'MSE': mo[0], 'mse_improve_pct': imp},
       'slope_mean': float(b_fit[cov].mean()), 'intercept_mean': float(a_fit[cov].mean())}
json.dump(out, open(os.path.join(E, 'gpm3h_ols_eval.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
np.save(os.path.join(E, 'pred_ols_gpm3h_test.npy'), pred_ols)
print('✅ 已保存 gpm3h_ols_eval.json / pred_ols_gpm3h_test.npy')
