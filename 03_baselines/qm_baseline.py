# -*- coding: utf-8 -*-
"""
Quantile Mapping (QM) 基线：审稿人 2.2 要求 "hierarchy of existing bias correction
baseline, from simple quantile-mapping ..."。本脚本实现逐格点 1D QM：
  - 训练期(2015-2021) 逐格点构建 GFS→ERA5 湿降水分位数映射 + 干湿频率校正
  - 应用到测试期(2024-2025) GFS
  - 评估同口径指标（MSE/RMSE/MAE/CC/FSS + 4 阈值分级 + bootstrap CI n=1000）

依赖：12.6 评估副本提供 --extract-train 输出的 train_gfs.npy / train_era5.npy
（逐样本逐格点 GFS/ERA5 降水，[N, 25, 37]）
用法：python qm_baseline.py
"""
import numpy as np
import os

OUT = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'


def build_qm(gfs_train, era5_train, wet_thresh=0.1, n_quantiles=1000):
    """逐格点构建 QM 映射。
    返回: qm_era5 (wet CDF 分位数数组 [H,W,nq]), qm_gfs ([H,W,nq]),
          p_dry_era5 ([H,W]), p_dry_gfs ([H,W])
    """
    H, W = gfs_train.shape[1], gfs_train.shape[2]
    qs = np.linspace(0, 1, n_quantiles + 2)[1:-1]  # 避开 0/1
    qm_gfs = np.zeros((H, W, n_quantiles), dtype=np.float32)
    qm_era5 = np.zeros((H, W, n_quantiles), dtype=np.float32)
    p_dry_gfs = np.zeros((H, W), dtype=np.float32)
    p_dry_era5 = np.zeros((H, W), dtype=np.float32)
    g = gfs_train.reshape(gfs_train.shape[0], -1)
    e = era5_train.reshape(era5_train.shape[0], -1)
    for idx in range(H * W):
        i, j = divmod(idx, W)
        gg = g[:, idx]
        ee = e[:, idx]
        wet_g = gg >= wet_thresh
        wet_e = ee >= wet_thresh
        p_dry_gfs[i, j] = 1.0 - wet_g.mean()
        p_dry_era5[i, j] = 1.0 - wet_e.mean()
        if wet_g.sum() > 5 and wet_e.sum() > 5:
            qm_gfs[i, j] = np.quantile(gg[wet_g], qs)
            qm_era5[i, j] = np.quantile(ee[wet_e], qs)
        else:
            # 样本太少：退化用全局
            qm_gfs[i, j] = np.quantile(gg, qs)
            qm_era5[i, j] = np.quantile(ee, qs)
    return qm_gfs, qm_era5, p_dry_gfs, p_dry_era5, qs


def apply_qm(gfs_test, qm_gfs, qm_era5, p_dry_gfs, p_dry_era5, qs, wet_thresh=0.1):
    """应用 QM。先做干湿频率校正（随机化，seed 固定），再映射湿值。"""
    rng = np.random.default_rng(42)
    out = np.zeros_like(gfs_test)
    N = gfs_test.shape[0]
    g = gfs_test.reshape(N, -1)
    o = out.reshape(N, -1)
    H, W = qm_gfs.shape[:2]
    for idx in range(H * W):
        i, j = divmod(idx, W)
        gg = g[:, idx]
        dry_g = gg < wet_thresh
        # 干湿频率校正：GFS 干概率 -> ERA5 干概率
        # 对 GFS 湿值，按 ERA5 湿概率随机置干
        p_keep_wet = np.clip((1 - p_dry_era5[i, j]) / max(1 - p_dry_gfs[i, j], 1e-6), 0, 1)
        wet = ~dry_g
        keep = wet.copy()
        if p_keep_wet < 1.0 and keep.any():
            rand = rng.random(keep.sum())
            keep[keep] = rand < p_keep_wet
        # 映射保留的湿值
        mapped = np.interp(gg[keep], qm_gfs[i, j], qm_era5[i, j])
        # 下限保护
        mapped = np.maximum(mapped, wet_thresh)
        o[keep, idx] = mapped
    return out


def compute_fss(pred, obs, threshold=10.0, window=3):
    import torch
    import torch.nn.functional as F
    p = (torch.from_numpy(pred) > threshold).float().unsqueeze(1)
    o = (torch.from_numpy(obs) > threshold).float().unsqueeze(1)
    k = window
    pf = F.avg_pool2d(p, k, stride=1, padding=k // 2)
    of = F.avg_pool2d(o, k, stride=1, padding=k // 2)
    num = torch.mean((pf - of) ** 2).item()
    den = torch.mean(pf ** 2 + of ** 2).item() + 1e-12
    return 1.0 - num / den


def categorical(pred, obs, th):
    hits = ((pred >= th) & (obs >= th)).sum()
    misses = ((pred < th) & (obs >= th)).sum()
    fa = ((pred >= th) & (obs < th)).sum()
    cn = ((pred < th) & (obs < th)).sum()
    pod = hits / (hits + misses + 1e-12)
    far = fa / (hits + fa + 1e-12)
    n = hits + misses + fa + cn
    hrand = (hits + misses) * (hits + fa) / (n + 1e-12)
    ets = (hits - hrand) / (hits + misses + fa - hrand + 1e-12)
    return pod, far, ets


def bootstrap_ci(pred, obs, th, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    ns = len(pred)
    fp, fo = pred.reshape(ns, -1), obs.reshape(ns, -1)
    pods, etss = [], []
    for _ in range(n):
        idx = rng.integers(0, ns, ns)
        pod, far, ets = categorical(fp[idx], fo[idx], th)
        pods.append(pod); etss.append(ets)
    return np.percentile(pods, [2.5, 97.5]), np.percentile(etss, [2.5, 97.5])


def main():
    gfs_test = np.load(os.path.join(OUT, 'gfs_test.npy'))
    tgt = np.load(os.path.join(OUT, 'targets_test.npy'))
    gfs_train = np.load(os.path.join(OUT, 'train_gfs.npy'))
    era5_train = np.load(os.path.join(OUT, 'train_era5.npy'))
    print(f'train {gfs_train.shape} | test {gfs_test.shape}')

    print('构建逐格点 QM 映射...')
    qm_gfs, qm_era5, p_dry_g, p_dry_e, qs = build_qm(gfs_train, era5_train)
    print('应用 QM 到测试集...')
    qm_pred = apply_qm(gfs_test, qm_gfs, qm_era5, p_dry_g, p_dry_e, qs)

    # 连续指标
    m = np.mean((qm_pred - tgt) ** 2)
    r = np.sqrt(m)
    a = np.mean(np.abs(qm_pred - tgt))
    c = np.corrcoef(qm_pred.flatten(), tgt.flatten())[0, 1]
    f = compute_fss(qm_pred, tgt)
    m_g = np.mean((gfs_test - tgt) ** 2)
    print('\n=== QM 基线 (2024-2025 测试) ===')
    print(f'MSE: {m:.4f} (GFS {m_g:.4f}, 改进 {(1-m/m_g)*100:+.1f}%)')
    print(f'RMSE: {r:.4f} | MAE: {a:.4f} | CC: {c:.4f} | FSS(th10): {f:.4f}')

    print('\n=== 分级指标 (POD/FAR/ETS + 95%CI) ===')
    for th in [0.1, 3.0, 10.0, 20.0]:
        pod, far, ets = categorical(qm_pred, tgt, th)
        ci_p, ci_e = bootstrap_ci(qm_pred, tgt, th)
        print(f'  th={th:5.1f} | POD={pod:.3f} ({ci_p[0]:.3f}-{ci_p[1]:.3f}) | '
              f'FAR={far:.3f} | ETS={ets:.3f} ({ci_e[0]:.3f}-{ci_e[1]:.3f})')

    np.save(os.path.join(OUT, 'predictions_qm.npy'), qm_pred)
    print('\n已保存 predictions_qm.npy')


if __name__ == '__main__':
    main()
