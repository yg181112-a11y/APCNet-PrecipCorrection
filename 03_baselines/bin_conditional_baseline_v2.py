# -*- coding: utf-8 -*-
"""P1-5b 分箱条件期望基准 v2：
- 全域分箱 BinCM（权威版，保存预测供 GPM/CHM 复用）
- 逐格点分箱 PerGrid-BinCM（稳健性对照；样本<30 的箱回退全域箱）
- UNet 用权威重训版 manuscript_work_unet\predictions_apcnet.npy
"""
import numpy as np, os, json

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
SYM = os.path.join(WORK, "..", "manuscript_work_sym", "seed42")
UNET = os.path.join(WORK, "..", "manuscript_work_unet", "predictions_apcnet.npy")

def L(name, base=WORK):
    return np.load(os.path.join(base, name)).astype(np.float32)

print("载入数据 ...")
tr_g = L("train_gfs.npy"); tr_e = L("train_era5.npy")
te_g = L("gfs_test.npy");  te_e = L("targets_test.npy")
apc = L("predictions_apcnet.npy", SYM)
qm = L("predictions_qm.npy"); ols = L("predictions_ols.npy")
unet = np.load(UNET).astype(np.float32)   # 权威重训 UNet

edges = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 100.0]
def bin_of(x):
    b = np.zeros_like(x, dtype=np.int16)
    for i in range(1, len(edges)):
        b[x > edges[i]] = i
    return b

# ---- 全域分箱 ----
tg, te_ = tr_g.ravel(), tr_e.ravel()
b_tr = bin_of(tg); n_bins = len(edges)
cnt = np.bincount(b_tr, minlength=n_bins)
s_xy = np.bincount(b_tr, weights=te_, minlength=n_bins)
cm_glob = np.divide(s_xy, np.maximum(cnt, 1), out=np.zeros(n_bins), where=cnt > 0)

# ---- 逐格点分箱（25x37 各自拟合；样本<30 回退全域） ----
Npix = tr_g.shape[1] * tr_g.shape[2]
cm_grid = np.empty((Npix, n_bins), dtype=np.float32)
for p in range(Npix):
    gx = tr_g[:, p // 37, p % 37].ravel()
    ex = tr_e[:, p // 37, p % 37].ravel()
    bx = bin_of(gx)
    c = np.bincount(bx, minlength=n_bins)
    s = np.bincount(bx, weights=ex, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        cm = np.where(c >= 30, s / np.maximum(c, 1), cm_glob)
    cm_grid[p] = cm

# ---- 测试期应用 ----
b_te = bin_of(te_g.ravel())
pred_glob = cm_glob[b_te].reshape(te_g.shape)
pred_grid = cm_grid[np.arange(Npix)[None, :], b_te.reshape(-1, Npix)].T.reshape(te_g.shape) if False else None
# 逐格点映射：b_te (N,Npix) → cm_grid[格点idx, bin]
b2 = b_te.reshape(-1, Npix)
pidx = np.arange(Npix)[None, :].repeat(b2.shape[0], axis=0)
pred_grid = cm_grid[pidx, b2].reshape(te_g.shape)

np.save(os.path.join(WORK, "bin_cm_pred.npy"), pred_glob)
np.save(os.path.join(WORK, "bin_cm_pergrid_pred.npy"), pred_grid)
print("已保存 bin_cm_pred.npy / bin_cm_pergrid_pred.npy")

def mse(o, f): return float(np.mean((o - f) ** 2))
def rmse(o, f): return float(np.sqrt(np.mean((o - f) ** 2)))
def cc(o, f): return float(np.corrcoef(o.ravel(), f.ravel())[0, 1])
def space_cc(pred, ref):
    cs = []
    for i in range(0, len(pred), 200):
        p, r = pred[i], ref[i]
        if p.std() > 0 and r.std() > 0:
            cs.append(np.corrcoef(p.flatten(), r.flatten())[0, 1])
    return float(np.mean(cs))

m_g = mse(te_e, te_g)
print("\n=== ERA5 参照（测试期 2839 时次）===")
rows = [("GFS", te_g), ("BinCM", pred_glob), ("PerGridBin", pred_grid), ("QM", qm), ("OLS", ols), ("APCNet", apc), ("UNet", unet)]
out = {"gfs_mse": m_g, "models": {}}
for nm, arr in rows:
    m = mse(te_e, arr)
    out["models"][nm] = {"mse": m, "rmse": rmse(te_e, arr), "cc": cc(te_e, arr),
                         "space_cc": space_cc(arr, te_e), "mean": float(arr.mean()),
                         "improve_pct": 100 * (m_g - m) / m_g}
    print(f"{nm:<11} MSE={m:.4f} RMSE={rmse(te_e,arr):.4f} CC={cc(te_e,arr):.4f} spaceCC={space_cc(arr,te_e):.4f} mean={arr.mean():.4f} 改进={100*(m_g-m)/m_g:+.2f}%")

with open(os.path.join(WORK, "bin_conditional_baseline.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("\n✅ bin_conditional_baseline.json 已保存")
