# -*- coding: utf-8 -*-
r"""
verify_gpm_independent.py — GPM IMERG 独立真值验证管线（回应审稿人 R2·2.1 "ERA5 作真值" P0 质疑）
====================================================================================================
口径（与 12.8修_final.py 主程序严格对齐）：
  - 样本时刻 t0 (UTC, 03/09/15/21Z, 6h 步长) 代表 ERA5 3h 累积降水窗口 [t0-3h, t0]
  - GPM IMERG V07B 半小时降水率 (mm/hr)，6 个 bin 累加 -> 3h 累积 mm
      gpm_3h(t0) = sum_{k=0..5} rate(t0-3h + k*30min) * 0.5h
  - 空间：GPM 0.1° (lon=91 升序 117.05~126.05, lat=60 升序 40.05~45.95)
          主网格 0.25° (GLOBAL_LATS=46.0→40.0 降序 25, GLOBAL_LONS=117.0→126.0 升序 37)
          双线性插值 + coverage mask（GPM 覆盖不到的主网格边缘剔除）

输入：
  - GPM 目录: C:\Users\yg181\Downloads\GPM_IMERG_test\imerg_YYYYMM\imerg_YYYYMMDD_HHMMSS.nc4
  - 主程序产物: D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work\
        targets_test.npy (ERA5 3h 累积 mm, [N,25,37])
        gfs_test.npy     (GFS 3h 累积 mm,   [N,25,37])
        predictions_apcnet.npy ([N,25,37])
        sample_times_test.pkl (list[datetime], UTC)

输出：
  - gpm_independent_eval.json（全套指标 + GPM-vs-ERA5 一致性）
  - 可选图：gpm_vs_era5_scatter.png 等
用法：
  python verify_gpm_independent.py [--outdir ...] [--savefig]
"""
import os, sys, glob, json, pickle, argparse
import numpy as np
from datetime import datetime, timedelta

GPM_ROOT = r"D:\liaohe\GPM_IMERG"   # 测试期 GPM（2024-01~2025-09 已全，2025Q4 补下载中）
WORK_DIR = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"

# 主程序网格（与 12.8修_final.py GLOBAL_LATS/GLOBAL_LONS 一致）
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)   # 降序（北→南）
GLOBAL_LONS = np.linspace(117.0, 126.0, 37) # 升序（西→东）

PRECIP_THRESHOLDS = [0.1, 3.0, 10.0, 20.0]   # mm/3h
BIN_H = 0.5                                   # 半小时
N_BINS = 6                                    # 3h / 0.5h


def load_sample_times():
    with open(os.path.join(WORK_DIR, "sample_times_test.pkl"), "rb") as f:
        return pickle.load(f)


def load_npy(name):
    p = os.path.join(WORK_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"缺少 {p}")
    return np.load(p)


def gpm_file_for(dt, root=None):
    """返回 (path, exists)。dt 为半小时 bin 开始时刻 (UTC)。
    目录按月份组织: imerg_YYYYMM/imerg_YYYYMMDD_HHMMSS.nc4
    [FIX] root=None 时读模块级 GPM_ROOT（默认参数不再绑定旧值）"""
    if root is None:
        root = GPM_ROOT
    d = dt.date()
    ym = d.strftime("%Y%m")
    ymd_hms = d.strftime("%Y%m%d") + "_" + dt.strftime("%H%M%S")
    p = os.path.join(root, f"imerg_{ym}", f"imerg_{ymd_hms}.nc4")
    return p, os.path.exists(p)


def read_gpm_rate(path):
    """读取单文件 GPM 半小时降水率场，返回 (rate_lonlat, lon, lat)。
    rate 形状 (lon=91, lat=60)，lon/lat 升序。"""
    import netCDF4 as nc
    ds = nc.Dataset(path)
    g = ds.groups["Grid"]
    lat = g.variables["lat"][:].astype(np.float64)   # 60 升序
    lon = g.variables["lon"][:].astype(np.float64)   # 91 升序
    rate = g.variables["precipitation"][0, :, :].astype(np.float64)  # (lon, lat)
    ds.close()
    return rate, lon, lat


def gpm_3h_accum(t0, cache=None):
    """对样本时刻 t0（代表 [t0-3h, t0] 3h 累积），累加 6 个半小时 GPM bin。
    返回 (field_lonlat, lon, lat, ok)。若任一 bin 缺失 -> ok=False。"""
    if cache is None:
        cache = {}
    bins = []
    for k in range(N_BINS):
        bt = t0 - timedelta(hours=3) + timedelta(minutes=int(30 * k))
        key = bt.strftime("%Y%m%d_%H%M%S")
        if key in cache:
            r = cache[key]
            if r is None:
                return None, None, None, False
        else:
            p, exists = gpm_file_for(bt)
            if not exists:
                cache[key] = None
                return None, None, None, False
            try:
                r, lon, lat = read_gpm_rate(p)
            except Exception:
                cache[key] = None
                return None, None, None, False
            cache[key] = (r, lon, lat)
        bins.append(cache[key])
    lon, lat = bins[0][1], bins[0][2]
    acc = np.zeros_like(bins[0][0])
    for r, _, _ in bins:
        acc += r * BIN_H  # mm/hr * 0.5h -> mm
    return acc, lon, lat, True


def regrid_to_main(acc_lonlat, lon, lat):
    """双线性插值：GPM (lon=91,lat=60) -> 主网格 (25,37)（lat 降序）。
    返回 (field, mask)，mask=True 表示该主网格点在 GPM 覆盖范围内（内部插值）。"""
    from scipy.interpolate import RegularGridInterpolator
    # GPM lat/lon 均升序；主网格 lat 降序 -> 生成目标点坐标
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)  # (25,37)
    pts = np.stack([lon2d.ravel(), lat2d.ravel()], axis=-1)
    interp = RegularGridInterpolator(
        (lon, lat), acc_lonlat, method="linear", bounds_error=False, fill_value=np.nan
    )
    vals = interp(pts).reshape(25, 37)
    mask = ~np.isnan(vals)
    vals = np.where(mask, vals, 0.0)
    return vals, mask


# ---------------- 指标 ----------------
def cont_metrics(obs, fcst):
    obs_f, fcst_f = obs.flatten(), fcst.flatten()
    valid = np.isfinite(obs_f) & np.isfinite(fcst_f)
    obs_f, fcst_f = obs_f[valid], fcst_f[valid]
    n = obs_f.size
    if n == 0:
        return None
    mse = np.mean((obs_f - fcst_f) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(obs_f - fcst_f))
    cc = np.corrcoef(obs_f, fcst_f)[0, 1] if n > 1 else 0.0
    return {"n": int(n), "MSE": float(mse), "RMSE": float(rmse),
            "MAE": float(mae), "CC": float(cc)}


def cat_metrics(obs, fcst, th):
    o = (obs >= th).astype(int)
    f = (fcst >= th).astype(int)
    tp = np.sum((o == 1) & (f == 1)); fn = np.sum((o == 1) & (f == 0))
    fp = np.sum((o == 0) & (f == 1)); tn = np.sum((o == 0) & (f == 0))
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    c = tp * tn - fp * fn
    e = (tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)
    ets = (c / e) if e > 0 else 0.0
    return {"th": float(th), "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "POD": float(pod), "FAR": float(far), "ETS": float(ets)}


def fss_field(obs, fcst, th, w=5):
    from scipy.ndimage import uniform_filter
    o = (obs >= th).astype(np.float64)
    f = (fcst >= th).astype(np.float64)
    oa = uniform_filter(o, size=w, mode="constant")
    fa = uniform_filter(f, size=w, mode="constant")
    denom = np.sum(oa ** 2) + np.sum(fa ** 2)
    if denom == 0:
        return 1.0
    return float(1.0 - np.sum((oa - fa) ** 2) / denom)


def paired_bootstrap(obs, f1, f2, th, metric="ETS", n_boot=1000, seed=42):
    """逐样本配对 bootstrap。f1=APCNet, f2=GFS。返回 p(改进由抽样误差引起的概率)。"""
    rng = np.random.default_rng(seed)
    n = obs.shape[0]
    idx_all = np.arange(n)
    # 每样本先算块级指标（整场）
    def block_metric(o, f, m):
        if m == "ETS":
            return cat_metrics(o, f, th)["ETS"]
        if m == "POD":
            return cat_metrics(o, f, th)["POD"]
        if m == "FAR":
            return cat_metrics(o, f, th)["FAR"]
        return 0.0
    d_obs = np.array([block_metric(obs[i], f1[i], metric) for i in idx_all])
    d_gfs = np.array([block_metric(obs[i], f2[i], metric) for i in idx_all])
    diff = d_obs - d_gfs
    d0 = np.mean(diff)
    cnt = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.mean(diff[idx]) <= 0:
            cnt += 1
    # 单侧 p：真实差异<=0 的概率
    p = cnt / n_boot
    return float(d0), float(p)


def object_based(obs, fcst, th, min_area=5):
    """Davis 2006 风格 object-based（area >= min_area 的连通域匹配）。
    简化协议：观察与预报对象按质心距离匹配（<=3 格点），指标 POD/FAR/pos_err。"""
    from scipy.ndimage import label, center_of_mass

    def get_objects(field, th):
        m = field >= th
        lab, n = label(m)
        objs = []
        for i in range(1, n + 1):
            idx = np.argwhere(lab == i)
            if len(idx) < min_area:
                continue
            area = len(idx)
            r, c = center_of_mass(lab == i)
            objs.append({"area": area, "r": float(r), "c": float(c),
                         "intensity": float(np.max(field[idx[:, 0], idx[:, 1]]))})
        return objs

    oo = get_objects(obs, th)
    fo = get_objects(fcst, th)
    matched_obs = set()
    matched_fcst = set()
    pos_errs = []
    for j, fo_j in enumerate(fo):
        best_i, best_d = None, 1e9
        for i, oo_i in enumerate(oo):
            if i in matched_obs:
                continue
            d = np.hypot(fo_j["r"] - oo_i["r"], fo_j["c"] - oo_i["c"])
            if d < best_d:
                best_i, best_d = i, d
        if best_i is not None and best_d <= 3.0:
            matched_obs.add(best_i); matched_fcst.add(j)
            pos_errs.append(best_d)
    pod = len(matched_obs) / len(oo) if oo else 0.0
    far = (len(fo) - len(matched_fcst)) / len(fo) if fo else 0.0
    return {"th": float(th), "total_obs": len(oo), "total_fcst": len(fo),
            "total_matched": len(matched_obs), "POD": float(pod), "FAR": float(far),
            "pos_error": float(np.mean(pos_errs)) if pos_errs else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work")
    ap.add_argument("--savefig", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("GPM IMERG 独立真值验证 (2024-2025 测试期)")
    print("=" * 72)

    times = load_sample_times()
    targets = load_npy("targets_test.npy")
    gfs = load_npy("gfs_test.npy")
    apcnet = load_npy("predictions_apcnet.npy")
    N = len(times)
    print(f"样本时刻: {N}, targets {targets.shape}, gfs {gfs.shape}, apcnet {apcnet.shape}")
    assert targets.shape == gfs.shape == apcnet.shape == (N, 25, 37)

    # 逐样本 GPM 3h 累积 + 重采样
    gpm_all = np.zeros_like(targets)
    mask_all = np.zeros((N, 25, 37), dtype=bool)
    t0s = [t for t in times if isinstance(t, datetime)]
    t0s = [t.replace(tzinfo=None) if t.tzinfo else t for t in t0s]
    n_ok = 0
    n_skip = 0
    cache = {}
    for i, t0 in enumerate(t0s):
        acc, lon, lat, ok = gpm_3h_accum(t0, cache)
        if not ok:
            n_skip += 1
            continue
        field, mask = regrid_to_main(acc, lon, lat)
        gpm_all[i] = field
        mask_all[i] = mask
        n_ok += 1
        if (i + 1) % 200 == 0:
            print(f"  进度 {i+1}/{N}: 有效 {n_ok} 跳过 {n_skip}")

    # 只保留有 GPM 数据的样本（缺失样本 gpm_all[i] 是 0，必须剔除否则污染指标）
    ok_rows = np.array([i for i in range(N) if mask_all[i].any()])
    n_ok = len(ok_rows)
    print(f"\nGPM 有效样本: {n_ok}/{N}, 跳过(数据缺失): {N - n_ok}")
    if n_ok < 50:
        print("!! GPM 有效样本过少，检查下载目录")
        sys.exit(1)

    # 空间覆盖掩码（主网格外边缘剔除）
    cov = mask_all[ok_rows].all(axis=0)  # (25,37)，仅统计有 GPM 数据的样本
    n_cov = int(cov.sum())
    print(f"GPM 空间覆盖格点: {n_cov}/925  (主网格外边缘剔除)")

    # 评估仅在有效样本 + 共同覆盖格点上进行
    def restrict(a):
        return a[ok_rows][:, cov]

    gpm_v, tgt_v, gfs_v, apc_v = restrict(gpm_all), restrict(targets), restrict(gfs), restrict(apcnet)

    result = {
        "protocol": "GPM IMERG V07B independent truth validation",
        "n_samples_ok": n_ok, "n_samples_total": N, "n_skipped": n_skip,
        "coverage_gridpoints": n_cov,
        "time_range": [str(t0s[0]), str(t0s[-1])],
        "unit": "mm/3h",
        "aggregation": "6 half-hourly rates * 0.5h",
        "resampling": "bilinear 0.1deg -> 0.25deg + coverage mask",
    }

    # 1) 连续指标（以 GPM 为真值）
    print("\n[1] 连续指标（真值 = GPM）")
    for name, arr in [("GFS", gfs_v), ("APCNet", apc_v)]:
        m = cont_metrics(gpm_v, arr)
        print(f"  {name:8s} MSE={m['MSE']:.4f} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} CC={m['CC']:.4f} (n={m['n']})")
        result[f"cont_{name}"] = m
    # 改进率
    m_gfs = cont_metrics(gpm_v, gfs_v)
    m_apc = cont_metrics(gpm_v, apc_v)
    result["mse_improve_pct"] = 100.0 * (m_gfs["MSE"] - m_apc["MSE"]) / m_gfs["MSE"]
    result["rmse_improve_pct"] = 100.0 * (m_gfs["RMSE"] - m_apc["RMSE"]) / m_gfs["RMSE"]
    print(f"  MSE 改进 {result['mse_improve_pct']:.2f}% | RMSE 改进 {result['rmse_improve_pct']:.2f}%")

    # 2) 分级指标
    print("\n[2] 分级指标（真值 = GPM）")
    result["categorical"] = {}
    for th in PRECIP_THRESHOLDS:
        row = {}
        for name, arr in [("GFS", gfs_v), ("APCNet", apc_v)]:
            row[name] = cat_metrics(gpm_v, arr, th)
            print(f"  >= {th:4.1f}mm  {name:8s} POD={row[name]['POD']:.4f} FAR={row[name]['FAR']:.4f} ETS={row[name]['ETS']:.4f} (TP={row[name]['TP']}, FN={row[name]['FN']}, FP={row[name]['FP']})")
        result["categorical"][str(th)] = row

    # 3) FSS
    print("\n[3] FSS（5x5 邻域, 真值 = GPM）")
    result["fss"] = {}
    for th in [0.1, 10.0, 20.0]:
        row = {}
        for name, arr in [("GFS", gfs_v), ("APCNet", apc_v)]:
            row[name] = fss_field(gpm_v, arr, th, w=5)
        print(f"  >= {th:4.1f}mm  GFS FSS={row['GFS']:.4f}  APCNet FSS={row['APCNet']:.4f}")
        result["fss"][str(th)] = row

    # 4) object-based
    print("\n[4] object-based（Davis 2006, min_area=5, 质心<=3格点）")
    result["object_based"] = {}
    for th in [10.0, 20.0]:
        row = {}
        for name, arr in [("GFS", gfs_v), ("APCNet", apc_v)]:
            row[name] = object_based(gpm_v, arr, th)
        print(f"  >= {th:4.1f}mm  GFS POD={row['GFS']['POD']:.3f} pos_err={row['GFS']['pos_error']:.2f} | "
              f"APCNet POD={row['APCNet']['POD']:.3f} pos_err={row['APCNet']['pos_error']:.2f}")
        result["object_based"][str(th)] = row

    # 5) paired bootstrap（APCNet vs GFS, 以 GPM 为真值）
    print("\n[5] paired bootstrap（APCNet vs GFS, 真值=GPM, n=1000）")
    result["bootstrap"] = {}
    for th in PRECIP_THRESHOLDS:
        row = {}
        for mtr in ["ETS", "POD", "FAR"]:
            d0, p = paired_bootstrap(gpm_v, apc_v, gfs_v, th, metric=mtr)
            row[mtr] = {"diff_mean": d0, "p": p}
        print(f"  >= {th:4.1f}mm  dETS={row['ETS']['diff_mean']:+.4f} p={row['ETS']['p']:.3f} | "
              f"dPOD={row['POD']['diff_mean']:+.4f} p={row['POD']['p']:.3f} | dFAR={row['FAR']['diff_mean']:+.4f} p={row['FAR']['p']:.3f}")
        result["bootstrap"][str(th)] = row

    # 6) GPM vs ERA5 一致性（评估 ERA5 目标是否系统性偏低）
    print("\n[6] GPM vs ERA5 一致性（共同覆盖区, 同口径 3h 累积）")
    m_era5 = cont_metrics(gpm_v, tgt_v)
    result["gpm_vs_era5"] = m_era5
    bias = np.nanmean(tgt_v - gpm_v)
    result["gpm_vs_era5"]["mean_bias_mm3h_ERA5_minus_GPM"] = float(bias)
    print(f"  ERA5(target) vs GPM: RMSE={m_era5['RMSE']:.4f} CC={m_era5['CC']:.4f} 均值偏差(ERA5-GPM)={bias:+.4f} mm/3h")
    # 极端区一致性
    for th in [10.0, 20.0]:
        msk = gpm_v >= th
        nb = int(msk.sum())
        if nb > 0:
            b_ext = float(np.nanmean(tgt_v[msk] - gpm_v[msk]))
            print(f"    GPM>= {th:4.1f}mm 像素 {nb}: ERA5-GPM 均值偏差 {b_ext:+.2f} mm/3h")
            result[f"gpm_vs_era5_bias_ge{int(th)}mm"] = {"n": nb, "mean_bias": b_ext}

    # 7) 季节分层（2024-01/2024-04 dry-run 实测：冷季 GPM/ERA5 比值 0.12-0.49，
    #    IMERG 对固态降水系统性低估；暖季 JJAS 比值 1.03、相关 0.885 -> GPM 可信。
    #    主结论（模型相对 GFS 变化）两组一致，绝对指标以暖季为准）
    print("\n[7] 季节分层（暖季 JJAS: GPM 可信；冷季: IMERG 固态降水局限）")
    result["seasonal"] = {}
    month_arr = np.array([t.month for t in t0s])[ok_rows]
    for season, sel in [("warm_JJAS", (month_arr >= 6) & (month_arr <= 9)),
                        ("cool_others", (month_arr < 6) | (month_arr > 9))]:
        if int(sel.sum()) < 20:
            print(f"  {season}: 样本不足 ({int(sel.sum())}), 跳过")
            continue
        gpm_s, tgt_s, gfs_s, apc_s = gpm_v[sel], tgt_v[sel], gfs_v[sel], apc_v[sel]
        row = {"n_samples": int(sel.sum()), "months": sorted(set(month_arr[sel].tolist()))}
        m_g, m_a = cont_metrics(gpm_s, gfs_s), cont_metrics(gpm_s, apc_s)
        row["cont_GFS"], row["cont_APCNet"] = m_g, m_a
        row["mse_improve_pct"] = 100.0 * (m_g["MSE"] - m_a["MSE"]) / m_g["MSE"]
        row["categorical"] = {}
        for th in PRECIP_THRESHOLDS:
            row["categorical"][str(th)] = {
                "GFS": cat_metrics(gpm_s, gfs_s, th),
                "APCNet": cat_metrics(gpm_s, apc_s, th)}
        print(f"  {season} (n={row['n_samples']}): "
              f"GFS RMSE={m_g['RMSE']:.4f} CC={m_g['CC']:.3f} | "
              f"APCNet RMSE={m_a['RMSE']:.4f} CC={m_a['CC']:.3f} | MSE改进 {row['mse_improve_pct']:+.2f}%")
        result["seasonal"][season] = row

    # 保存
    out_path = os.path.join(args.outdir, "gpm_independent_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ 结果已保存: {out_path}")

    # 可选图：GPM vs ERA5 / GFS / APCNet 散点
    if args.savefig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, (nm, arr), title in zip(
            axes,
            [("ERA5", tgt_v), ("GFS", gfs_v), ("APCNet", apc_v)],
            ["GPM vs ERA5 (target)", "GPM vs GFS", "GPM vs APCNet"],
        ):
            ax.hexbin(gpm_v.flatten(), arr.flatten(), gridsize=80, mincnt=1, cmap="Spectral_r")
            mx = max(gpm_v.max(), arr.max())
            ax.plot([0, mx], [0, mx], "k--", lw=0.8)
            ax.set_xlim(0, mx); ax.set_ylim(0, mx)
            ax.set_xlabel("GPM (mm/3h)"); ax.set_ylabel(title.split(" vs ")[1] + " (mm/3h)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = os.path.join(args.outdir, "gpm_independent_scatter.png")
        fig.savefig(fig_path, dpi=150)
        print(f"✅ 散点图已保存: {fig_path}")


if __name__ == "__main__":
    main()
