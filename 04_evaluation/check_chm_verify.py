# -*- coding: utf-8 -*-
r"""
check_chm_verify.py — CHM 日值降水独立验证（第三路真值，回应 R1·2.1 "ERA5 作真值" P0 质疑）
====================================================================================================
口径：
  - 模型（GFS/APCNet）输出为 3h 累积场 [N,25,37]，样本时刻 t0 ∈ {03,09,15,21}Z，窗口 [t0-3h, t0]。
  - 对每个 UTC 日 d：模型侧 12h 累积 = 当天 4 个 3h 窗口之和；CHM 侧为 24h 日值。
  - 对比口径统一为平均降水率（mm/h）：模型率 = 12h累积/12h，CHM率 = 日值/24h —— 避免覆盖时长差异带来的系统性偏差。
  - CHM 0.1°（lat 18.05-53.95 升序、lon 72.05-135.95 升序）双线性插值到主网格（lat 46→40 降序 25、lon 117→126 升序 37）。
  - 指标：逐日逐格点 CC / MAE / 均值偏差 / RMSE；月度平均日降水序列；全流域平均日序列时间线。
输出：
  - chm_daily_eval.json + chm_daily_series.png
用法：
  python check_chm_verify.py [--outdir ...] [--savefig]
"""
import os, sys, json, pickle, argparse
from datetime import datetime, timedelta, timezone
import numpy as np

WORK_DIR = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
CHM_2024 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2024.nc"
CHM_2025 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2025.nc"

GLOBAL_LATS = np.linspace(46.0, 40.0, 25)   # 降序
GLOBAL_LONS = np.linspace(117.0, 126.0, 37) # 升序

# 3h 窗口结束时刻（UTC 小时）
WINDOW_END_HOURS = [3, 9, 15, 21]


def load_arrays():
    with open(os.path.join(WORK_DIR, "sample_times_test.pkl"), "rb") as f:
        times = pickle.load(f)
    targets = np.load(os.path.join(WORK_DIR, "targets_test.npy"))
    gfs = np.load(os.path.join(WORK_DIR, "gfs_test.npy"))
    apc = np.load(os.path.join(WORK_DIR, "predictions_apcnet.npy"))
    return times, targets, gfs, apc


def read_chm_daily(path):
    """读 CHM 日值场，返回 (prec[time,lat,lon], lat, lon, dates[datetime])。"""
    import netCDF4 as nc
    ds = nc.Dataset(path)
    lat = ds.variables["lat"][:].astype(np.float64)
    lon = ds.variables["lon"][:].astype(np.float64)
    prec = ds.variables["prec"][:]  # [time, lat, lon] or [time, lon, lat]
    # 时间解析
    tv = ds.variables["time"]
    units = tv.units
    dates = nc.num2date(tv[:], units=units, only_use_cftime_datetimes=False)
    ds.close()
    # 维度顺序判断
    if prec.ndim == 3:
        # 尝试 (time, lat, lon)
        if prec.shape[1] == len(lat) and prec.shape[2] == len(lon):
            pass
        elif prec.shape[1] == len(lon) and prec.shape[2] == len(lat):
            prec = prec.transpose(0, 2, 1)
        else:
            raise ValueError(f"CHM 维度不匹配: prec.shape={prec.shape}, lat={len(lat)}, lon={len(lon)}")
    return prec, lat, lon, dates


def regrid_chm_to_main(chm_latlon, lat, lon):
    """CHM (lat 升序, lon 升序) -> 主网格 (25,37)，双线性。chm_latlon 形状 (lat, lon)。"""
    from scipy.interpolate import RegularGridInterpolator
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    # grid=(lat, lon) → pts 列顺序必须是 (lat, lon)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    interp = RegularGridInterpolator(
        (lat, lon), chm_latlon, method="linear", bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    mask = ~np.isnan(vals)
    vals = np.where(mask, vals, 0.0)
    return vals, mask


def group_by_day(sample_times, arrs):
    """按 UTC 日聚合 4 个 3h 窗口 -> 日 12h 累积。返回 (dates, accs, counts)。
    accs[k] 为 (25,37) 场。样本窗口结束时刻 t0；窗口为 [t0-3h, t0]。
    当天 4 窗 = t0 ∈ {03,09,15,21}Z 的样本。"""
    utc = timezone.utc
    day_map = {}  # date -> list of (arr)
    for i, t in enumerate(sample_times):
        if isinstance(t, np.datetime64):
            t = t.astype("datetime64[s]").astype(object)
        t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
        day = t.date()
        day_map.setdefault(day, []).append(i)
    dates = []
    accs = []
    n_windows = []
    for day in sorted(day_map.keys()):
        idx = day_map[day]
        # 只保留 t0 ∈ {03,09,15,21}Z 的样本（防样本缺失导致窗口不全）
        keep = []
        for i in idx:
            t = sample_times[i]
            if isinstance(t, np.datetime64):
                t = t.astype("datetime64[s]").astype(object)
            t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
            if t.hour in WINDOW_END_HOURS:
                keep.append(i)
        if len(keep) == 0:
            continue
        acc = np.zeros_like(arrs[0][0])
        for i in keep:
            for a in arrs:
                acc = acc + a[i]
        dates.append(day)
        accs.append(acc / len(keep))  # 均值窗口（若某日窗口缺，则按实际窗口数平均）
        n_windows.append(len(keep))
    return dates, np.array(accs), n_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=WORK_DIR)
    ap.add_argument("--savefig", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("CHM 日值降水独立验证（模型 3h -> 日聚合）")
    print("=" * 72)

    times, targets, gfs, apc = load_arrays()
    print(f"样本: {len(times)}, targets {targets.shape}, apc {apc.shape}")

    # 读 CHM
    chm_all, chm_mask = [], None
    chm_dates_all = []
    for p in (CHM_2024, CHM_2025):
        prec, lat, lon, dates = read_chm_daily(p)
        print(f"CHM {os.path.basename(p)}: shape={prec.shape}, lat[{lat[0]:.2f},{lat[-1]:.2f}] lon[{lon[0]:.2f},{lon[-1]:.2f}]")
        for k in range(len(prec)):
            field, mask = regrid_chm_to_main(prec[k], lat, lon)
            chm_all.append(field)
            chm_dates_all.append(dates[k])
        chm_mask = mask if chm_mask is None else chm_mask
    chm_arr = np.array(chm_all)  # [T, 25, 37]
    chm_dates_all = [d.replace(tzinfo=None) if getattr(d, "tzinfo", None) else d for d in chm_dates_all]
    chm_day_map = {d.date(): chm_arr[k] for k, d in enumerate(chm_dates_all)}
    print(f"CHM 总天数: {len(chm_day_map)} (2024-2025)")

    # 模型日聚合（12h 覆盖）
    dates, acc_apc, nwin = group_by_day(times, [apc])
    dates2, acc_gfs, _ = group_by_day(times, [gfs])
    # 对齐公共日期
    common = sorted(set(dates) & set(dates2) & set(chm_day_map.keys()))
    print(f"模型-模型-CHM 公共日: {len(common)}")
    if len(common) < 30:
        print("!! 公共日过少，检查样本时间与 CHM 覆盖")
        sys.exit(1)

    # 平均降水率（mm/h）
    apc_rate, gfs_rate, chm_rate = [], [], []
    for d in common:
        apc_rate.append(acc_apc[dates.index(d)] / 12.0)     # 12h 覆盖 / 12h
        gfs_rate.append(acc_gfs[dates2.index(d)] / 12.0)
        chm_rate.append(chm_day_map[d] / 24.0)              # 24h 日值 / 24h
    apc_rate = np.array(apc_rate)
    gfs_rate = np.array(gfs_rate)
    chm_rate = np.array(chm_rate)

    # 仅在 CHM 覆盖格点上评估
    cm = chm_mask
    def flatten_masked(a):
        return a[:, cm].flatten()
    apc_f, gfs_f, chm_f = flatten_masked(apc_rate), flatten_masked(gfs_rate), flatten_masked(chm_rate)
    valid = np.isfinite(apc_f) & np.isfinite(chm_f)
    apc_f, gfs_f, chm_f = apc_f[valid], gfs_f[valid], chm_f[valid]

    def metr(o, f):
        o = np.asarray(o).flatten()
        f = np.asarray(f).flatten()
        valid = np.isfinite(o) & np.isfinite(f)
        o, f = o[valid], f[valid]
        if o.size == 0:
            return {"n": 0}
        mse = np.mean((o - f) ** 2)
        cc = np.corrcoef(o, f)[0, 1] if np.std(o) > 0 and np.std(f) > 0 else 0.0
        return {"n": int(o.size), "RMSE": float(np.sqrt(mse)), "MAE": float(np.mean(np.abs(o - f))),
                "CC": float(cc), "mean_bias_fcst_minus_obs": float(np.mean(f - o)),
                "mean_obs_rate": float(np.mean(o)), "mean_fcst_rate": float(np.mean(f))}

    result = {
        "protocol": "CHM daily precipitation verification (model 3h->daily aggregation)",
        "aggregation": "model: 4 windows(00-03/06-09/12-15/18-21Z) sum /12h; CHM: 24h daily value /24h",
        "unit": "mm/h (rate)", "n_common_days": len(common),
        "coverage_gridpoints": int(cm.sum()),
    }
    print("\n[1] 逐日逐格点降水率指标（真值=CHM）")
    for nm, arr in [("GFS", gfs_rate), ("APCNet", apc_rate)]:
        m = metr(chm_rate, arr)
        result[f"rate_{nm}"] = m
        print(f"  {nm:8s} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} CC={m['CC']:.4f} "
              f"偏差={m['mean_bias_fcst_minus_obs']:+.4f} mm/h (obs均值={m['mean_obs_rate']:.4f})")
    m_g = result["rate_GFS"]; m_a = result["rate_APCNet"]
    result["rmse_improve_pct"] = 100.0 * (m_g["RMSE"] - m_a["RMSE"]) / m_g["RMSE"]
    print(f"  RMSE 改进 {result['rmse_improve_pct']:.2f}%")

    # 日尺度分级（日 12h/24h 量纲不同，不做分级；补充 12h vs CHM*0.5 的敏感说明）
    print("\n[2] 月度平均日降水序列（全流域覆盖区平均, mm/h）")
    months = sorted({(d.year, d.month) for d in common})
    rows = []
    for y, mth in months:
        sel = [k for k, d in enumerate(common) if (d.year, d.month) == (y, mth)]
        def basin_mean(a):
            return float(np.mean(a[sel][:, cm]))
        rows.append({"year": y, "month": mth, "n_days": len(sel),
                     "gfs_rate": basin_mean(gfs_rate), "apcnet_rate": basin_mean(apc_rate),
                     "chm_rate": basin_mean(chm_rate)})
        print(f"  {y}-{mth:02d} ({len(sel)}d)  GFS={rows[-1]['gfs_rate']:.4f} "
              f"APCNet={rows[-1]['apcnet_rate']:.4f} CHM={rows[-1]['chm_rate']:.4f} mm/h")
    result["monthly_series"] = rows

    out_path = os.path.join(args.outdir, "chm_daily_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ 结果已保存: {out_path}")

    if args.savefig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(13, 8))
        # 月度序列
        ax = axes[0]
        xs = range(len(rows))
        ax.plot(xs, [r["gfs_rate"] for r in rows], "o-", lw=1.5, label="GFS (model)")
        ax.plot(xs, [r["apcnet_rate"] for r in rows], "s-", lw=1.5, label="APCNet (model)")
        ax.plot(xs, [r["chm_rate"] for r in rows], "^-", lw=1.5, label="CHM (truth)")
        ax.set_xticks(list(xs), [f"{r['year']}-{r['month']:02d}" for r in rows], rotation=45)
        ax.set_ylabel("basin-mean precip rate (mm/h)")
        ax.set_title("Monthly basin-mean precipitation rate: GFS / APCNet vs CHM")
        ax.legend(); ax.grid(alpha=0.3)
        # 密度散点
        ax = axes[1]
        ax.hexbin(chm_f, apc_f, gridsize=80, mincnt=1, cmap="Spectral_r")
        mx = max(chm_f.max(), apc_f.max())
        ax.plot([0, mx], [0, mx], "k--", lw=0.8)
        ax.set_xlabel("CHM (mm/h)"); ax.set_ylabel("APCNet (mm/h)")
        ax.set_title("Daily precip rate: APCNet vs CHM")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig_path = os.path.join(args.outdir, "chm_daily_series.png")
        fig.savefig(fig_path, dpi=150)
        print(f"✅ 图已保存: {fig_path}")


if __name__ == "__main__":
    main()
