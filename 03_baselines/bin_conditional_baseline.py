# -*- coding: utf-8 -*-
"""P1-5 分箱条件期望基准（bin-conditional-mean baseline）。

问题：最小化 MSE 的回归网络理论上逼近 E[Y|X]；若残差独立，E[Y|X]=X（恒等，零改进），
实验却是 -32.6% 实质恶化。本实验用训练期(2015-2021)分箱拟合 E[Y|X]，测试期盲测，
回答：条件期望本身的"可学习信号"是正是负？网络是差于理论最优，还是任务本身不可解？

- 全域分箱（所有格点 pooled）：bin edges 按 GFS 值对数箱
- 零降水单独一箱
- 评估 MSE/RMSE 改进率 vs GFS、CC、逐样本空间 CC、均值偏差
- 输出 per-bin 条件偏差（E[Y|X]-X），供讨论"条件偏差有结构、分箱能抓住"
"""
import numpy as np, os, json

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
SYM = os.path.join(WORK, "..", "manuscript_work_sym", "seed42")

def L(name, base=WORK):
    return np.load(os.path.join(base, name)).astype(np.float32)

print("载入数据 ...")
tr_g = L("train_gfs.npy")          # (10157, 25, 37)
tr_e = L("train_era5.npy")
te_g = L("gfs_test.npy")           # (2839, 25, 37)
te_e = L("targets_test.npy")
apc  = L("predictions_apcnet.npy", SYM)
qm   = L("predictions_qm.npy")
ols  = L("predictions_ols.npy")
unet = L("predictions_unet.npy")

# ---------- 分箱 ----------
# 对数箱（mm/3h）；零降水单独一箱（bin 0）
edges = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 100.0]
def bin_of(x):
    b = np.zeros_like(x, dtype=np.int16)
    for i in range(1, len(edges)):
        b[x > edges[i]] = i
    return b

print("训练期分箱拟合 ...")
tg, te_ = tr_g.ravel(), tr_e.ravel()
b_tr = bin_of(tg)
n_bins = len(edges)
cnt = np.bincount(b_tr, minlength=n_bins)
s_xy = np.bincount(b_tr, weights=te_, minlength=n_bins)
cm = np.divide(s_xy, np.maximum(cnt, 1), out=np.zeros(n_bins), where=cnt > 0)
# 每箱条件偏差 E[Y|X]-X：用每箱 GFS 均值代表 X
s_x = np.bincount(b_tr, weights=tg, minlength=n_bins)
xbar = np.divide(s_x, np.maximum(cnt, 1), out=np.zeros(n_bins), where=cnt > 0)
cond_dev = cm - xbar
print("每箱: GFS均值, ERA5条件均值, 条件偏差, 样本数")
for i in range(n_bins):
    lo = edges[i-1] if i > 0 else 0.0
    hi = edges[i] if i < len(edges) else "inf"
    print(f"  bin{i} [{lo:>5},{hi:>5}): n={cnt[i]:>9}  xbar={xbar[i]:.4f}  E[Y|X]={cm[i]:.4f}  dev={cond_dev[i]:+.4f}")

# 测试期应用
print("测试期应用 ...")
b_te = bin_of(te_g.ravel())
pred_bin = cm[b_te].reshape(te_g.shape)

def mse(o, f):
    return float(np.mean((o - f) ** 2))

def rmse(o, f):
    return float(np.sqrt(np.mean((o - f) ** 2)))

def cc(o, f):
    return float(np.corrcoef(o.ravel(), f.ravel())[0, 1])

def space_cc(pred, ref):
    cs = []
    for i in range(0, len(pred), 200):
        p, r = pred[i], ref[i]
        if p.std() > 0 and r.std() > 0:
            cs.append(np.corrcoef(p.flatten(), r.flatten())[0, 1])
    return float(np.mean(cs))

m_g, m_b = mse(te_e, te_g), mse(te_e, pred_bin)
print("\n=== ERA5 参照（测试期 2839 时次）===")
print(f"GFS        MSE={m_g:.4f} RMSE={rmse(te_e,te_g):.4f} CC={cc(te_e,te_g):.4f} spaceCC={space_cc(te_g,te_e):.4f} mean={te_g.mean():.4f}")
print(f"BinCM      MSE={m_b:.4f} RMSE={rmse(te_e,pred_bin):.4f} CC={cc(te_e,pred_bin):.4f} spaceCC={space_cc(pred_bin,te_e):.4f} mean={pred_bin.mean():.4f} 改进={100*(m_g-m_b)/m_g:+.2f}%")
for nm, arr in [("APCNet", apc), ("QM", qm), ("OLS", ols), ("UNet", unet)]:
    m = mse(te_e, arr)
    print(f"{nm:<8} MSE={m:.4f} RMSE={rmse(te_e,arr):.4f} CC={cc(te_e,arr):.4f} spaceCC={space_cc(arr,te_e):.4f} mean={arr.mean():.4f} 改进={100*(m_g-m)/m_g:+.2f}%")

res = {
    "bin_edges": edges,
    "n_train_samples": int(tr_g.size),
    "n_bins": int(n_bins),
    "per_bin": [{"count": int(cnt[i]), "gfs_mean": float(xbar[i]), "era5_cond_mean": float(cm[i]),
                 "cond_dev": float(cond_dev[i])} for i in range(n_bins)],
    "test": {
        "gfs": {"mse": m_g, "rmse": rmse(te_e, te_g), "cc": cc(te_e, te_g), "space_cc": space_cc(te_g, te_e), "mean": float(te_g.mean())},
        "bin_cm": {"mse": m_b, "rmse": rmse(te_e, pred_bin), "cc": cc(te_e, pred_bin), "space_cc": space_cc(pred_bin, te_e),
                   "mean": float(pred_bin.mean()), "improve_pct": 100 * (m_g - m_b) / m_g},
    },
}
with open(os.path.join(WORK, "bin_conditional_baseline.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print("\n✅ bin_conditional_baseline.json 已保存")
