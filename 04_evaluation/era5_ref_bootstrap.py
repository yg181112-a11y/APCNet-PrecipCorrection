# -*- coding: utf-8 -*-
"""P1-6 ERA5 参照各方法改进率的时次块 bootstrap 显著性。
2839 时次按月块（2024-01..2025-12）block bootstrap 2000 次，
对 QM/OLS/BinCM/APCNet(S42)/UNet 计算 MSE 改进率分布与 95% CI。
"""
import numpy as np, os, json, pickle
from datetime import timezone

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
SYM = os.path.join(WORK, "..", "manuscript_work_sym", "seed42")
UNET = os.path.join(WORK, "..", "manuscript_work_unet", "predictions_apcnet.npy")

with open(os.path.join(WORK, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
tgt = np.load(os.path.join(WORK, "targets_test.npy")).astype(np.float32)
gfs = np.load(os.path.join(WORK, "gfs_test.npy")).astype(np.float32)
models = {
    "QM": np.load(os.path.join(WORK, "predictions_qm.npy")).astype(np.float32),
    "OLS": np.load(os.path.join(WORK, "predictions_ols.npy")).astype(np.float32),
    "BinCM": np.load(os.path.join(WORK, "bin_cm_pred.npy")).astype(np.float32),
    "APCNet": np.load(os.path.join(WORK, "..", "manuscript_work_sym", "seed42", "predictions_apcnet.npy")).astype(np.float32),
    "UNet": np.load(UNET).astype(np.float32),
}
N = len(times)
utc = timezone.utc
month_of = np.array([(t.year, t.month) for t in times])
months = sorted(set(map(tuple, month_of)))
print("时次:", N, "月块:", len(months))

# 逐时次 MSE
err_gfs = np.array([np.mean((tgt[i] - gfs[i]) ** 2) for i in range(N)])
err_mod = {k: np.array([np.mean((tgt[i] - p[i]) ** 2) for i in range(N)]) for k, p in models.items()}

rng = np.random.default_rng(2026)
n_boot = 2000
res = {}
for k in models:
    dist = []
    for b in range(n_boot):
        sel_blocks = rng.choice(len(months), size=len(months), replace=True)
        idx = np.concatenate([np.where((month_of[:, 0] == months[m][0]) & (month_of[:, 1] == months[m][1]))[0]
                              for m in sel_blocks])
        imp = 100 * (np.mean(err_gfs[idx]) - np.mean(err_mod[k][idx])) / np.mean(err_gfs[idx])
        dist.append(imp)
    dist = np.array(dist)
    obs_imp = 100 * (err_gfs.mean() - err_mod[k].mean()) / err_gfs.mean()
    res[k] = {"obs_improve_pct": float(obs_imp), "boot_mean": float(dist.mean()),
              "ci95": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
              "p_negative": float(np.mean(dist < 0))}
    print(f"{k:<8} 观测改进={obs_imp:+.2f}%  boot均值={dist.mean():+.2f}%  95%CI=[{np.percentile(dist,2.5):+.2f},{np.percentile(dist,97.5):+.2f}]%  P(负)={np.mean(dist<0):.3f}")

with open(os.path.join(WORK, "era5_ref_bootstrap.json"), "w", encoding="utf-8") as f:
    json.dump({"n_samples": N, "n_month_blocks": len(months), "n_bootstrap": n_boot, "models": res},
              f, indent=1, ensure_ascii=False)
print("\n✅ era5_ref_bootstrap.json 已保存")
