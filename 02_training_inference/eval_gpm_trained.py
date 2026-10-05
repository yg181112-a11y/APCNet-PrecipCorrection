# -*- coding: utf-8 -*-
r"""
eval_gpm_trained.py — GPM 观测训练模型的独立验证评估（敏感性对照实验）
=====================================================================
对照实验口径（与主实验 manuscript_work 产物严格对齐）：
  - 样本时刻 t0 (UTC, 03/09/15/21Z, 6h 步长) 代表 3h 累积窗 [t0-3h, t0]
  - 真值 = GPM IMERG V07B：6 个半小时 bin 累加 -> 3h 累积 mm（口径同 verify_gpm_independent.py）
  - 对比体系：GFS / QM / OLS / APCNet(ERA5-train) / U-Net(ERA5-train) / APCNet(GPM-train)
  - 评估子集：GPM 覆盖的有效样本 + 共同覆盖格点（主网格外边缘剔除）

输出：
  - manuscript_work/eval_gpm_trained.json  （全套指标，含 block bootstrap 与逐样本空间 CC）
  - 可选：散点/日循环图

用法：
  python eval_gpm_trained.py --apc <gpm_trained_pred.npy> [--tag gpm_trained] [--savefig]
"""
import os, sys, json, pickle, argparse
import numpy as np
from datetime import datetime

WORK_DIR = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
GPM_ROOT_D = r"D:\liaohe\GPM_IMERG"   # 2024-01 ~ 2025-09 已就位（测试期）

# 复用 verify_gpm_independent.py 的 GPM 加载与指标函数（同口径，防漂移）
sys.path.insert(0, r"C:\Users\yg181\Desktop\论文三\实验代码优化过程\12.8修")
import verify_gpm_independent as V
from verify_gpm_independent import (
    regrid_to_main, cont_metrics, cat_metrics,
    PRECIP_THRESHOLDS, GLOBAL_LATS, GLOBAL_LONS,
)
# 模块级覆盖 GPM_ROOT 后，gpm_file_for 的默认参数仍绑定 import 时的旧值，
# 故运行时替换 gpm_file_for 使 gpm_3h_accum 内部调用指向 D 盘。
_orig_gpm_file_for = V.gpm_file_for
def _gpm_file_for_d(dt, root=GPM_ROOT_D):
    return _orig_gpm_file_for(dt, root=root)
V.gpm_file_for = _gpm_file_for_d

BLOCK_DAYS = 90  # block bootstrap 块长（审稿人：自相关未计入）

def load_arrays(apc_path):
    with open(os.path.join(WORK_DIR, "sample_times_test.pkl"), "rb") as f:
        times = pickle.load(f)
    t0s = [t.replace(tzinfo=None) if t.tzinfo else t for t in times if isinstance(t, datetime)]
    gfs = np.load(os.path.join(WORK_DIR, "gfs_test.npy"))
    apc_era5 = np.load(os.path.join(WORK_DIR, "predictions_apcnet.npy"))
    qm = np.load(os.path.join(WORK_DIR, "predictions_qm.npy"))
    ols = np.load(os.path.join(WORK_DIR, "predictions_ols.npy"))
    unet = np.load(os.path.join(WORK_DIR, "predictions_unet.npy"))
    apc_gpm = np.load(apc_path)
    assert gfs.shape == apc_era5.shape == qm.shape == ols.shape == unet.shape == apc_gpm.shape == (len(t0s), 25, 37), \
        f"形状不一致: gfs{gfs.shape} apc_era5{apc_era5.shape} qm{qm.shape} ols{ols.shape} unet{unet.shape} apc_gpm{apc_gpm.shape} n={len(t0s)}"
    return t0s, gfs, apc_era5, qm, ols, unet, apc_gpm


def load_gpm_truth(t0s):
    """逐样本 GPM 3h 累积 + 重采样到主网格，返回 (gpm_all, mask_all)。"""
    N = len(t0s)
    gpm_all = np.zeros((N, 25, 37))
    mask_all = np.zeros((N, 25, 37), dtype=bool)
    cache = {}
    n_ok = n_skip = 0
    for i, t0 in enumerate(t0s):
        acc, lon, lat, ok = V.gpm_3h_accum(t0, cache)
        if not ok:
            n_skip += 1
            continue
        field, mask = regrid_to_main(acc, lon, lat)
        gpm_all[i] = field
        mask_all[i] = mask
        n_ok += 1
        if (i + 1) % 500 == 0:
            print(f"  GPM 加载 {i+1}/{N}: 有效 {n_ok} 跳过 {n_skip}")
    ok_rows = np.array([i for i in range(N) if mask_all[i].any()])
    print(f"GPM 有效样本: {len(ok_rows)}/{N}, 跳过: {N - len(ok_rows)}")
    if len(ok_rows) < 50:
        raise RuntimeError("GPM 有效样本过少，检查数据目录")
    cov = mask_all[ok_rows].all(axis=0)
    print(f"GPM 空间覆盖格点: {int(cov.sum())}/925")
    return ok_rows, cov, gpm_all


def restrict(arr, ok_rows, cov):
    return arr[ok_rows][:, cov]


def block_bootstrap_diff(obs, f1, f2, block_days=BLOCK_DAYS, n_boot=1000, seed=42):
    """块长 block_days 天（每天 4 时次）的 block bootstrap：检验 f1 相对 f2 的 MSE 改进显著性。
    返回 (mse_improve_pct, p(改进<=0))。"""
    rng = np.random.default_rng(seed)
    n = obs.shape[0]
    n_blocks = max(1, n // (4 * block_days))
    block_idx = np.array_split(np.arange(n), n_blocks)
    m1 = np.mean((obs - f1) ** 2)
    m2 = np.mean((obs - f2) ** 2)
    d0 = (m2 - m1) / m2  # 相对 f2 的改进
    cnt = 0
    for _ in range(n_boot):
        idx = np.concatenate([block_idx[rng.integers(0, len(block_idx))]
                              for _ in range(len(block_idx))])[:n]
        m1b = np.mean((obs[idx] - f1[idx]) ** 2)
        m2b = np.mean((obs[idx] - f2[idx]) ** 2)
        if (m2b - m1b) / m2b <= 0:
            cnt += 1
    return float(d0 * 100.0), float(cnt / n_boot)


def spatial_cc(obs, fcst):
    """逐样本空间相关（审稿人 R2 关注点：全局展平 CC 不能代表空间结构保持）。"""
    ccs = []
    for i in range(obs.shape[0]):
        o, f = obs[i], fcst[i]
        m = (o > 0) | (f > 0)
        if m.sum() < 5:
            continue
        c = np.corrcoef(o[m], f[m])[0, 1]
        if np.isfinite(c):
            ccs.append(c)
    return float(np.mean(ccs)), int(len(ccs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apc", required=True, help="GPM-训练 APCNet 预测 .npy 路径")
    ap.add_argument("--tag", default="gpm_trained", help="结果标签")
    ap.add_argument("--savefig", action="store_true")
    args = ap.parse_args()

    print("=" * 76)
    print(f"GPM 观测训练模型验证 (2024-2025 测试期, 真值=GPM IMERG) [{args.tag}]")
    print("=" * 76)

    t0s, gfs, apc_era5, qm, ols, unet, apc_gpm = load_arrays(args.apc)
    ok_rows, cov, gpm_all = load_gpm_truth(t0s)

    def R(a):
        return restrict(a, ok_rows, cov)

    gpm_v, gfs_v = R(gpm_all), R(gfs)
    apc5_v, apcG_v = R(apc_era5), R(apc_gpm)
    qm_v, ols_v, unet_v = R(qm), R(ols), R(unet)

    systems = [("GFS", gfs_v), ("QM", qm_v), ("OLS", ols_v),
               ("APCNet_ERA5", apc5_v), ("U-Net_ERA5", unet_v),
               ("APCNet_GPM", apcG_v)]

    result = {
        "protocol": "GPM-trained model independent verification (GPM IMERG truth)",
        "tag": args.tag,
        "n_samples_ok": len(ok_rows), "n_samples_total": len(t0s),
        "coverage_gridpoints": int(cov.sum()),
        "truth": "GPM IMERG V07B (6 half-hourly bins -> 3h accum)",
        "block_bootstrap": {"block_days": BLOCK_DAYS, "n_boot": 1000},
    }

    # 1) 连续指标
    print("\n[1] 连续指标（真值=GPM）")
    result["cont"] = {}
    m_gfs = cont_metrics(gpm_v, gfs_v)
    for name, arr in systems:
        m = cont_metrics(gpm_v, arr)
        imp = 100.0 * (m_gfs["MSE"] - m["MSE"]) / m_gfs["MSE"]
        result["cont"][name] = {**m, "mse_improve_vs_GFS_pct": float(imp)}
        print(f"  {name:12s} MSE={m['MSE']:.4f} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} "
              f"CC={m['CC']:.4f} 改进={imp:+.2f}%")

    # 2) 分级指标
    print("\n[2] 分级指标（真值=GPM）")
    result["categorical"] = {}
    for th in PRECIP_THRESHOLDS:
        result["categorical"][str(th)] = {}
        for name, arr in systems:
            row = cat_metrics(gpm_v, arr, th)
            result["categorical"][str(th)][name] = row
        print(f"  >= {th:4.1f}mm  " + "  ".join(
            f"{n[:6]} POD={result['categorical'][str(th)][n]['POD']:.3f}"
            for n, _ in systems))

    # 3) 逐样本空间 CC
    print("\n[3] 逐样本空间 CC（>0 或预报>0 的格点, 每样本相关后平均）")
    result["spatial_cc"] = {}
    for name, arr in systems:
        cc, nv = spatial_cc(gpm_v, arr)
        result["spatial_cc"][name] = {"mean_spatial_cc": cc, "n_valid_samples": nv}
        print(f"  {name:12s} 空间CC={cc:.4f} (n={nv})")

    # 4) block bootstrap：APCNet_GPM vs GFS / vs APCNet_ERA5（MSE 改进显著性）
    print(f"\n[4] block bootstrap (块长={BLOCK_DAYS}天, n=1000, 真值=GPM)")
    result["bootstrap"] = {}
    for ref_name, ref_arr in [("GFS", gfs_v), ("APCNet_ERA5", apc5_v)]:
        imp, p = block_bootstrap_diff(gpm_v, apcG_v, ref_arr)
        result["bootstrap"][ref_name] = {"mse_improve_pct": imp, "p_no_improve": p}
        print(f"  APCNet_GPM vs {ref_name:12s} MSE改进={imp:+.2f}%  p(无改进)={p:.3f}")

    # 5) 日循环（域均值 @ 4 时次）
    print("\n[5] 日循环（域均值 mm/3h, 03/09/15/21Z）")
    hrs = np.array([(t.hour) for t in t0s])[ok_rows]
    result["diurnal"] = {}
    for name, arr in [("GPM", gpm_all[ok_rows]),
                      ("GFS", gfs[ok_rows]), ("QM", qm[ok_rows]),
                      ("OLS", ols[ok_rows]), ("APCNet_ERA5", apc_era5[ok_rows]),
                      ("APCNet_GPM", apc_gpm[ok_rows])]:
        vals = []
        for h in (3, 9, 15, 21):
            m = hrs == h
            vals.append(float(np.nanmean(arr[m][:, cov]))) if m.any() else vals.append(None)
        result["diurnal"][name] = vals
        print(f"  {name:12s} 03Z={vals[0]:.3f} 09Z={vals[1]:.3f} 15Z={vals[2]:.3f} 21Z={vals[3]:.3f}")

    # 保存
    out_path = os.path.join(WORK_DIR, "eval_gpm_trained.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ 结果已保存: {out_path}")

    if args.savefig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        # 左：日循环
        ax = axes[0]
        colors = {"GPM": "k", "GFS": "#0072B2", "QM": "#009E73", "OLS": "#D55E00",
                  "APCNet_ERA5": "#CC79A7", "APCNet_GPM": "#E69F00"}
        for name, vals in result["diurnal"].items():
            ax.plot([3, 9, 15, 21], vals, "o-", label=name, color=colors.get(name),
                    lw=1.6 if name not in ("GPM",) else 2.2)
        ax.set_xlabel("Valid time (UTC)")
        ax.set_ylabel("Domain-mean precipitation (mm/3h)")
        ax.set_title(f"Diurnal cycle (GPM truth, {args.tag})")
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.3)
        # 右：分级 ETS
        ax = axes[1]
        ths = [float(t) for t in PRECIP_THRESHOLDS]
        for name, _ in systems:
            ets = [result["categorical"][str(t)][name]["ETS"] for t in ths]
            ax.plot(ths, ets, "o-", label=name, color=colors.get(name), lw=1.6)
        ax.set_xlabel("Threshold (mm/3h)")
        ax.set_ylabel("ETS")
        ax.set_title(f"Categorical skill (GPM truth, {args.tag})")
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = os.path.join(WORK_DIR, f"eval_gpm_trained_{args.tag}.png")
        fig.savefig(fig_path, dpi=200)
        print(f"✅ 图已保存: {fig_path}")


if __name__ == "__main__":
    main()
