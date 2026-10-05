# -*- coding: utf-8 -*-
"""P0 空间相关统一口径重算。
per-sample space CC = mean_i corr(pred[i].flatten(), target[i].flatten())，全部样本。
all-point CC = corr(所有样本全部格点展平)。
对所有方法输出权威值，供 v1 Table 2 / 正文唯一引用。
"""
import numpy as np, os, json

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
SYM = os.path.join(WORK, "..", "manuscript_work_sym")
UNET = os.path.join(WORK, "..", "manuscript_work_unet", "predictions_apcnet.npy")

def L(name, base=WORK):
    return np.load(os.path.join(base, name)).astype(np.float64)

te_g = L("gfs_test.npy"); te_e = L("targets_test.npy")
qm = L("predictions_qm.npy"); ols = L("predictions_ols.npy")
bincm = L("bin_cm_pred.npy"); pergrid = L("bin_cm_pergrid_pred.npy")
apc = L("predictions_apcnet.npy", os.path.join(SYM, "seed42"))
unet = np.load(UNET).astype(np.float64)

def allpoint_cc(o, f):
    return float(np.corrcoef(o.ravel(), f.ravel())[0, 1])

def per_sample_space_cc(o, f):
    cs = []
    for i in range(len(o)):
        p, r = o[i].ravel(), f[i].ravel()
        if p.std() > 0 and r.std() > 0:
            cs.append(float(np.corrcoef(p, r)[0, 1]))
    return float(np.mean(cs)), len(cs)

rows = [("GFS", te_g), ("QM", qm), ("OLS", ols), ("BinCM", bincm),
        ("PerGridBin", pergrid), ("APCNet_sym_S42", apc), ("UNet", unet)]
out = {}
for nm, P in rows:
    apc_all = allpoint_cc(te_e, P)
    ps, n = per_sample_space_cc(te_e, P)
    out[nm] = {"allpoint_cc": apc_all, "per_sample_space_cc": ps, "n_samples_used": n}
    print(f"{nm:<16} all-point CC={apc_all:.4f}  per-sample space CC={ps:.4f} (n={n})")

with open(os.path.join(WORK, "space_cc_unified.json"), "w", encoding="utf-8") as f:
    json.dump({"reference": "ERA5 targets_test (2839 samples)", "methods": out}, f, indent=1, ensure_ascii=False)
print("\n✅ space_cc_unified.json 已保存")
