# -*- coding: utf-8 -*-
"""P1-③ 经典基线：逐格点 OLS 线性回归（GFS→ERA5），作为 QM 之外的对照。
y_hat(g,t) = a_g * gfs(g,t) + b_g，系数用训练期（10157 样本）拟合。
评估测试期 MSE 改进率 / CC / 雨区面积 bias。
"""
import numpy as np
import os, json

W = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
tr_g = np.load(os.path.join(W, 'train_gfs.npy')).astype(np.float32)   # [N,25,37]
tr_e = np.load(os.path.join(W, 'train_era5.npy')).astype(np.float32)
te_g = np.load(os.path.join(W, 'gfs_test.npy')).astype(np.float32)
te_e = np.load(os.path.join(W, 'targets_test.npy')).astype(np.float32)

N, H, Wd = tr_g.shape
# 逐格点最小二乘: [a, b] = inv(XtX) XtY ; X = [gfs, 1]
G = tr_g.reshape(N, -1)
E = tr_e.reshape(N, -1)
cols = G.shape[1]
A = np.zeros((cols, 2)); B = np.zeros(cols)
for j in range(cols):
    x = G[:, j]; y = E[:, j]
    X = np.stack([x, np.ones_like(x)], axis=1)  # [N,2]
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        coef = [0.0, y.mean()]
    A[j] = coef

# 测试期预测
T = te_g.shape[0]
Gt = te_g.reshape(T, -1)
pred = np.zeros_like(Gt)
for j in range(cols):
    pred[:, j] = A[j, 0] * Gt[:, j] + A[j, 1]
pred = pred.reshape(T, H, Wd)

mse_p = np.mean((pred - te_e) ** 2)
mse_g = np.mean((te_g - te_e) ** 2)
cc = np.corrcoef(pred.flatten(), te_e.flatten())[0, 1]
cc_g = np.corrcoef(te_g.flatten(), te_e.flatten())[0, 1]
print('OLS 基线: MSE=%.4f 改进=%.2f%% | CC=%.4f (GFS CC=%.4f)' % (mse_p, (1 - mse_p / mse_g) * 100, cc, cc_g))
print('预测均值 %.4f vs 目标 %.4f (GFS %.4f)' % (pred.mean(), te_e.mean(), te_g.mean()))
for th in [0.1, 3.0, 10.0, 20.0]:
    o = np.mean(te_e >= th); p = np.mean(pred >= th); g = np.mean(te_g >= th)
    print(' >=%5.1f | obs=%.3f%% OLS=%.3f%% (bias=%.2f) GFS=%.3f%% (bias=%.2f)' % (th, o * 100, p * 100, p / o, g * 100, g / o))
np.save(os.path.join(W, 'predictions_ols.npy'), pred.astype(np.float32))
print('predictions_ols.npy 已保存')
