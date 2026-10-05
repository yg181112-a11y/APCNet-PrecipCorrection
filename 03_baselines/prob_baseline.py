# -*- coding: utf-8 -*-
"""
简单概率基线：QM 确定性预测 + 训练期残差高斯 dress（Gaussian dressing）
→ 逐格点高斯概率预测 N(μ=QM_pred, σ²=训练期残差方差)
→ CRPS（解析公式）对比 APCNet/GFS 确定性预测的 CRPS(=MAE)
审稿人 2.3 要求 "compare probabilistic models to their deterministic framework"
"""
import numpy as np
from scipy.stats import norm
import os, sys

OUT = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
sys.path.insert(0, OUT)
from qm_baseline import build_qm, apply_qm  # 复用 QM 映射


def gaussian_crps(mu, sigma, y):
    """逐元素高斯 CRPS 解析公式: σ[z(2Φ(z)-1)+2φ(z)-1/√π], z=(y-μ)/σ"""
    sigma = np.maximum(sigma, 1e-4)
    z = (y - mu) / sigma
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    return sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / np.sqrt(np.pi))


def main():
    # 加载数据
    gfs_train = np.load(os.path.join(OUT, 'train_gfs.npy'))
    era5_train = np.load(os.path.join(OUT, 'train_era5.npy'))
    gfs_test = np.load(os.path.join(OUT, 'gfs_test.npy'))
    tgt = np.load(os.path.join(OUT, 'targets_test.npy'))
    print(f'train {gfs_train.shape} | test {gfs_test.shape}')

    # 1. 构建 QM 映射
    print('构建 QM 映射...')
    qm_gfs, qm_era5, p_dry_g, p_dry_e, qs = build_qm(gfs_train, era5_train)

    # 2. 训练期 QM 预测 → 残差 std（逐格点）
    print('计算训练期 QM 预测与残差 std...')
    qm_train_pred = apply_qm(gfs_train, qm_gfs, qm_era5, p_dry_g, p_dry_e, qs)
    resid_train = era5_train - qm_train_pred
    sigma = resid_train.std(axis=0)  # [H,W] 逐格点
    print(f'残差 std: mean={sigma.mean():.4f}, max={sigma.max():.4f}, '
          f'p95={np.percentile(sigma,95):.4f}')

    # 3. 测试期 QM 预测（μ）
    print('应用 QM 到测试期...')
    qm_test_pred = apply_qm(gfs_test, qm_gfs, qm_era5, p_dry_g, p_dry_e, qs)

    # 4. 高斯概率预测 CRPS
    print('计算高斯概率预测 CRPS...')
    mu = qm_test_pred
    sigma_bc = np.broadcast_to(sigma, mu.shape)
    crps_prob = gaussian_crps(mu, sigma_bc, tgt)
    crps_prob_mean = float(crps_prob.mean())

    # 5. 确定性基线 CRPS = MAE
    # APCNet 预测需要加载——12.6 重跑轮没有存 APCNet npy，用日志 MSE/MAE
    # 但可以从 gfs_test 算 GFS CRPS；APCNet MAE=0.1482（日志）
    crps_gfs = float(np.mean(np.abs(gfs_test - tgt)))
    crps_apcnet = 0.1482  # 12.6 日志 MAE（确定性点预测 CRPS=MAE）
    crps_qm_det = float(np.mean(np.abs(qm_test_pred - tgt)))  # QM 确定性 MAE

    print('\n=== 概率基线 vs 确定性框架（CRPS，越低越好）===')
    print(f'GFS (deterministic):     CRPS = {crps_gfs:.4f} (=MAE)')
    print(f'QM (deterministic):      CRPS = {crps_qm_det:.4f} (=MAE)')
    print(f'QM+GaussianDress (prob): CRPS = {crps_prob_mean:.4f}')
    print(f'APCNet (deterministic):  CRPS = {crps_apcnet:.4f} (=MAE, 日志)')
    print(f'\n概率基线相对 GFS 改进: {(1-crps_prob_mean/crps_gfs)*100:+.2f}%')
    print(f'概率基线相对 APCNet:     {(1-crps_prob_mean/crps_apcnet)*100:+.2f}%')

    # 6. 分级 CRPS（按 ERA5 强度分档）
    print('\n=== 分级 CRPS（按 ERA5 强度）===')
    bins = [(0, 0.1, '无雨'), (0.1, 3.0, '小雨'), (3.0, 10.0, '中雨'),
            (10.0, 20.0, '大雨'), (20.0, 999, '暴雨')]
    for lo, hi, name in bins:
        mask = (tgt >= lo) & (tgt < hi)
        if mask.sum() > 0:
            cp = float(crps_prob[mask].mean())
            cg = float(np.abs(gfs_test - tgt)[mask].mean())
            ca = None  # APCNet 分级 MAE 无直接数据
            print(f'  {name:6s} (N={mask.sum():7d}): prob_CRPS={cp:.4f} | GFS_MAE={cg:.4f}')

    np.save(os.path.join(OUT, 'crps_prob.npy'), crps_prob.astype(np.float32))
    np.save(os.path.join(OUT, 'qm_test_pred.npy'), qm_test_pred.astype(np.float32))
    print('\n已保存 crps_prob.npy, qm_test_pred.npy')


if __name__ == '__main__':
    main()
