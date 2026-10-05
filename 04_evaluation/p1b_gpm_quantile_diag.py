# -*- coding: utf-8 -*-
r"""
p1b_gpm_quantile_diag.py — P1b: GPM 分位数诊断（审稿 R1·2.1 敏感性实验之二）
====================================================================================================
目的：验证"APCNet 以 ERA5 为训练真值所继承的低估偏差是单调可逆的"。
做法：
  1. 复用 verify_gpm_independent 的 GPM 3h 累积 + 重采样 + coverage mask；
  2. 在共同覆盖区上建立 ERA5(target) → GPM 的分位数映射 F(x)（湿区分箱，等频分位）;
  3. 对 APCNet 预测施加订正 pred_corr = F(pred)；
  4. 以 GPM 为真值，评估 GFS / APCNet / APCNet+分位数订正 的连续 + 分级 + FSS；
  5. 可选：--extra-pred 传入其他预测 npy（如 P1a 重训结果）一并评估。

用法：
  python p1b_gpm_quantile_diag.py [--outdir ...] [--savefig] [--extra-pred <npy路径> [--extra-name <名>]]...
"""
import os, sys, glob, json, pickle, argparse
import numpy as np

# 复用 verify 的 GPM 加载/插值/指标函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\yg181\Doubao\chats\2026-09-15\new-chat")  # verify_gpm_independent.py 所在目录
from verify_gpm_independent import (
    load_sample_times, load_npy, gpm_3h_accum, regrid_to_main,
    cont_metrics, cat_metrics, fss_field, PRECIP_THRESHOLDS,
)

WORK_DIR = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"


def build_gpm_fields(times, targets):
    """逐样本构建 GPM 3h 场，返回 (gpm_all, mask_all, ok_rows, cov)。"""
    N = len(times)
    gpm_all = np.zeros_like(targets)
    mask_all = np.zeros((N, 25, 37), dtype=bool)
    # datetime / np.datetime64 统一为 naive datetime
    t0s2 = []
    for t in times:
        if isinstance(t, np.datetime64):
            t0s2.append(t.astype("datetime64[s]").astype(object).replace(tzinfo=None))
        else:
            t0s2.append(t.replace(tzinfo=None) if getattr(t, "tzinfo", None) else t)
    n_ok = 0
    cache = {}
    for i, t0 in enumerate(t0s2):
        acc, lon, lat, ok = gpm_3h_accum(t0, cache)
        if not ok:
            continue
        field, mask = regrid_to_main(acc, lon, lat)
        gpm_all[i] = field
        mask_all[i] = mask
        n_ok += 1
        if (i + 1) % 200 == 0:
            print(f"  GPM 进度 {i+1}/{N}: 有效 {n_ok}")
    ok_rows = np.array([i for i in range(N) if mask_all[i].any()])
    cov = mask_all[ok_rows].all(axis=0)
    return gpm_all, mask_all, ok_rows, cov


def quantile_map_fit(era5_vals, gpm_vals, n_bins=40, wet_th=0.1):
    """分位数映射拟合：对湿区 ERA5 值等频分箱，取各箱 GPM 分位数中值。
    返回 (bin_edges, bin_gpm)，可对任意 x 分段线性插值。"""
    wet = (era5_vals >= wet_th) & (gpm_vals >= wet_th) & np.isfinite(era5_vals) & np.isfinite(gpm_vals)
    e = era5_vals[wet]
    g = gpm_vals[wet]
    if e.size < 100:
        return None, None
    order = np.argsort(e)
    e, g = e[order], g[order]
    n = e.size
    edges = np.zeros(n_bins + 1)
    gpm_mid = np.zeros(n_bins)
    for b in range(n_bins):
        lo = int(n * b / n_bins)
        hi = int(n * (b + 1) / n_bins)
        if hi <= lo:
            hi = lo + 1
        seg_e = e[lo:hi]
        seg_g = g[lo:hi]
        # 用该箱内 ERA5 最大值作为边界（升序）
        edges[b] = float(seg_e[0])
        gpm_mid[b] = float(np.percentile(seg_g, 50))
    edges[-1] = float(e[-1])
    # 单调化：累积最大值保证非降
    for b in range(1, n_bins):
        gpm_mid[b] = max(gpm_mid[b], gpm_mid[b - 1])
    # 用箱中点作为插值节点（与 gpm_mid 等长）
    edges_mid = 0.5 * (edges[:-1] + edges[1:])
    return edges_mid, gpm_mid


def quantile_map_apply(x, edges, gpm_mid):
    """分段线性插值应用。x < wet_th 或 > 边界 → 原样（保守，不强加降水）。"""
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    wet_th = 0.1
    lo = x >= wet_th
    if not lo.any():
        return out
    xe = x[lo]
    # clip 到 [edges[0], edges[-1]]
    xe_c = np.clip(xe, edges[0], edges[-1])
    vals = np.interp(xe_c, edges, gpm_mid)
    out[lo] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=WORK_DIR)
    ap.add_argument("--savefig", action="store_true")
    ap.add_argument("--n-bins", type=int, default=40)
    ap.add_argument("--extra-pred", action="append", default=[], help="额外预测 npy（如 P1a 结果）")
    ap.add_argument("--extra-name", action="append", default=[], help="与 --extra-pred 对应的名称")
    args = ap.parse_args()

    times = load_sample_times()
    targets = load_npy("targets_test.npy")
    gfs = load_npy("gfs_test.npy")
    apcnet = load_npy("predictions_apcnet.npy")
    N = len(times)
    print("=" * 72)
    print("P1b: GPM 分位数诊断（低估可逆性检验）")
    print("=" * 72)
    print(f"样本: {N}, targets {targets.shape}, apcnet {apcnet.shape}")

    gpm_all, mask_all, ok_rows, cov = build_gpm_fields(times, targets)
    n_ok = len(ok_rows)
    print(f"GPM 有效样本 {n_ok}/{N}, 覆盖格点 {int(cov.sum())}/925")

    def restrict(a):
        return a[ok_rows][:, cov]

    gpm_v = restrict(gpm_all)
    tgt_v = restrict(targets)
    gfs_v = restrict(gfs)
    apc_v = restrict(apcnet)

    # ---- 分位数映射（ERA5 → GPM） ----
    print("\n[1] 拟合 ERA5→GPM 分位数映射（湿区等频分箱）...")
    edges, gpm_mid = quantile_map_fit(tgt_v.flatten(), gpm_v.flatten(), n_bins=args.n_bins)
    if edges is None:
        print("!! 湿区样本不足，无法拟合")
        sys.exit(1)
    print(f"  映射 bin 数={args.n_bins}, 湿区像素={len(edges)-1} 个箱")
    # 映射统计：ERA5 10/20mm 对应的 GPM 值
    f10 = float(np.interp(10.0, edges, gpm_mid))
    f20 = float(np.interp(20.0, edges, gpm_mid))
    print(f"  映射: ERA5 10mm → GPM {f10:.1f}mm | ERA5 20mm → GPM {f20:.1f}mm")

    # ---- 订正 ----
    apc_qm = quantile_map_apply(apc_v, edges, gpm_mid)
    # 订正幅度统计
    print(f"  订正后 APCNet 湿区均值: {np.mean(apc_v[apc_v>=0.1]):.3f} → {np.mean(apc_qm[apc_qm>=0.1]):.3f} mm/3h")

    # ---- 评估 ----
    models = [("GFS", gfs_v), ("APCNet", apc_v), ("APCNet+QM", apc_qm)]
    for i, (p, nm) in enumerate(args.extra_pred and zip(args.extra_pred, args.extra_name) or []):
        arr = np.load(p)
        # 与 apcnet 同形状处理
        arr = restrict(arr)
        models.append((f"{nm}", arr))

    result = {
        "protocol": "P1b GPM quantile mapping diagnosis (ERA5->GPM monotonic correction)",
        "n_samples_ok": n_ok, "coverage_gridpoints": int(cov.sum()),
        "quantile_map": {"n_bins": args.n_bins,
                         "era5_10mm_gpm": float(f10), "era5_20mm_gpm": float(f20),
                         "edges_first_last": [float(edges[0]), float(edges[-1])]},
        "unit": "mm/3h",
    }

    print("\n[2] 连续指标（真值=GPM）")
    result["cont"] = {}
    for nm, arr in models:
        m = cont_metrics(gpm_v, arr)
        result["cont"][nm] = m
        print(f"  {nm:12s} MSE={m['MSE']:.4f} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} CC={m['CC']:.4f}")
    # 改进率相对 GFS
    mg = result["cont"]["GFS"]
    for nm, arr in models:
        if nm == "GFS":
            continue
        mm = result["cont"][nm]
        result["cont"][f"{nm}_mse_improve_pct_vs_gfs"] = 100.0 * (mg["MSE"] - mm["MSE"]) / mg["MSE"]

    print("\n[3] 分级指标（真值=GPM）")
    result["categorical"] = {}
    for th in PRECIP_THRESHOLDS:
        row = {}
        for nm, arr in models:
            row[nm] = cat_metrics(gpm_v, arr, th)
        result["categorical"][str(th)] = row
        print(f"  >= {th:4.1f}mm  " + " | ".join(
            f"{nm}: POD={row[nm]['POD']:.3f} ETS={row[nm]['ETS']:.4f} FAR={row[nm]['FAR']:.3f}" for nm, _ in models))

    print("\n[4] FSS（5x5, 真值=GPM）")
    result["fss"] = {}
    for th in [0.1, 10.0, 20.0]:
        row = {}
        for nm, arr in models:
            row[nm] = fss_field(gpm_v, arr, th, w=5)
        result["fss"][str(th)] = row
        print(f"  >= {th:4.1f}mm  " + " | ".join(f"{nm}: {row[nm]:.4f}" for nm, _ in models))

    out_path = os.path.join(args.outdir, "p1b_gpm_quantile_diag.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ 结果已保存: {out_path}")

    # 可选图：分位数映射曲线
    if args.savefig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        # 左：映射曲线
        ax = axes[0]
        xs = np.linspace(edges[0], edges[-1], 200)
        ax.plot(xs, np.interp(xs, edges, gpm_mid), "b-", lw=2, label="ERA5→GPM 映射")
        ax.plot(xs, xs, "k--", lw=1, label="1:1")
        ax.set_xlabel("ERA5 (mm/3h)"); ax.set_ylabel("GPM 分位数中值 (mm/3h)")
        ax.set_title("分位数映射 F(x)"); ax.legend(); ax.grid(alpha=0.3)
        # 右：APCNet 订正前后 vs GPM 散点
        ax = axes[1]
        ax.hexbin(gpm_v.flatten(), apc_v.flatten(), gridsize=60, mincnt=1, cmap="Blues", label="before")
        ax.hexbin(gpm_v.flatten(), apc_qm.flatten(), gridsize=60, mincnt=1, cmap="Reds", alpha=0.6)
        mx = max(gpm_v.max(), apc_qm.max())
        ax.plot([0, mx], [0, mx], "k--", lw=0.8)
        ax.set_xlabel("GPM (mm/3h)"); ax.set_ylabel("APCNet (mm/3h)")
        ax.set_title("APCNet 订正前后（蓝→红）"); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig_path = os.path.join(args.outdir, "p1b_quantile_diag.png")
        fig.savefig(fig_path, dpi=150)
        print(f"✅ 图已保存: {fig_path}")


if __name__ == "__main__":
    main()
