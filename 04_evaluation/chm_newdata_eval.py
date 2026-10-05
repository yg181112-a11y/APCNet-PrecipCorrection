# -*- coding: utf-8 -*-
"""P2-① CHM 独立验证（新数据 + 公平最优 APCNet 对称 S42 重算）。
基于 check_chm_verify.py 口径：模型 12h 日聚合 vs CHM 24h 日值，统一 mm/h 率。
"""
import os, sys, json, pickle, argparse
from datetime import datetime, timedelta, timezone
import numpy as np

WORK_DIR = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
CHM_2024 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2024.nc"
CHM_2025 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2025.nc"

GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]


def load_arrays(apc_path):
    with open(os.path.join(WORK_DIR, "sample_times_test.pkl"), "rb") as f:
        times = pickle.load(f)
    gfs = np.load(os.path.join(WORK_DIR, "gfs_test.npy"))
    apc = np.load(apc_path)
    qm = np.load(os.path.join(WORK_DIR, "predictions_qm.npy"))
    ols = np.load(os.path.join(WORK_DIR, "predictions_ols.npy"))
    return times, gfs, apc, qm, ols


def read_chm_daily(path):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    lat = ds.variables["lat"][:].astype(np.float64)
    lon = ds.variables["lon"][:].astype(np.float64)
    prec = ds.variables["prec"][:]
    tv = ds.variables["time"]
    dates = nc.num2date(tv[:], units=tv.units, only_use_cftime_datetimes=False)
    ds.close()
    if prec.ndim == 3:
        if prec.shape[1] == len(lat) and prec.shape[2] == len(lon):
            pass
        elif prec.shape[1] == len(lon) and prec.shape[2] == len(lat):
            prec = prec.transpose(0, 2, 1)
        else:
            raise ValueError(f"CHM 维度不匹配: prec.shape={prec.shape}, lat={len(lat)}, lon={len(lon)}")
    return prec, lat, lon, dates


def regrid_chm_to_main(chm_latlon, lat, lon):
    from scipy.interpolate import RegularGridInterpolator
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    interp = RegularGridInterpolator((lat, lon), chm_latlon, method="linear", bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    mask = ~np.isnan(vals)
    vals = np.where(mask, vals, 0.0)
    return vals, mask


def group_by_day(sample_times, arr):
    utc = timezone.utc
    day_map = {}
    for i, t in enumerate(sample_times):
        t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
        day_map.setdefault(t.date(), []).append(i)
    dates, accs = [], []
    for day in sorted(day_map.keys()):
        keep = []
        for i in day_map[day]:
            t = sample_times[i]
            t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
            if t.hour in WINDOW_END_HOURS:
                keep.append(i)
        if len(keep) == 0:
            continue
        acc = np.zeros_like(arr[0])
        for i in keep:
            acc = acc + arr[i]
        dates.append(day)
        accs.append(acc)  # FIX B2: 保留 12h 总量（此前 /len(keep) 引入 ×4 率错误）
    return dates, np.array(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apc", default=r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work_sym\seed42\predictions_apcnet.npy")
    ap.add_argument("--tag", default="sym_s42")
    args = ap.parse_args()
    times, gfs, apc, qm, ols = load_arrays(args.apc)
    print(f"tag={args.tag} 样本: {len(times)}, apc {apc.shape}")

    chm_all, chm_mask = [], None
    chm_dates_all = []
    for p in (CHM_2024, CHM_2025):
        prec, lat, lon, dates = read_chm_daily(p)
        print(f"CHM {os.path.basename(p)}: shape={prec.shape}")
        for k in range(len(prec)):
            field, mask = regrid_chm_to_main(prec[k], lat, lon)
            chm_all.append(field)
            chm_dates_all.append(dates[k])
        chm_mask = mask if chm_mask is None else mask
    chm_arr = np.array(chm_all)
    chm_day_map = {d.date(): chm_arr[k] for k, d in enumerate(chm_dates_all)}
    print(f"CHM 总天数: {len(chm_day_map)}")

    dts = {}
    for nm, arr in [("gfs", gfs), ("apc", apc), ("qm", qm), ("ols", ols)]:
        dates, acc = group_by_day(times, arr)
        dts[nm] = (dates, acc)
    common = sorted(set(dts["gfs"][0]) & set(dts["apc"][0]) & set(chm_day_map.keys()))
    print(f"公共日: {len(common)}")
    if len(common) < 30:
        sys.exit(1)

    rates = {}
    for nm in ("gfs", "apc", "qm", "ols"):
        dates, acc = dts[nm]
        rates[nm] = np.array([acc[dates.index(d)] / 12.0 for d in common])
    chm_rate = np.array([chm_day_map[d] / 24.0 for d in common])

    cm = chm_mask
    def flatten_masked(a):
        return a[:, cm].flatten()
    chm_f = flatten_masked(chm_rate)
    valid = np.isfinite(chm_f)

    def metr(o, f):
        o = np.asarray(o).flatten(); f = np.asarray(f).flatten()
        valid = np.isfinite(o) & np.isfinite(f)
        o, f = o[valid], f[valid]
        mse = np.mean((o - f) ** 2)
        cc = np.corrcoef(o, f)[0, 1] if np.std(o) > 0 and np.std(f) > 0 else 0.0
        return {"n": int(o.size), "RMSE": float(np.sqrt(mse)), "MAE": float(np.mean(np.abs(o - f))),
                "CC": float(cc), "bias_fcst_minus_obs": float(np.mean(f - o)),
                "obs_rate": float(np.mean(o)), "fcst_rate": float(np.mean(f))}

    result = {
        "protocol": f"CHM daily verification (NEW DATA + APCNet {args.tag})",
        "aggregation": "model: 4x3h windows 12h coverage /12h; CHM: 24h /24h",
        "unit": "mm/h", "n_common_days": len(common), "coverage_gridpoints": int(cm.sum()),
    }
    print("\n=== 降水率指标（真值=CHM） ===")
    for nm in ("gfs", "apc", "qm", "ols"):
        m = metr(chm_rate, rates[nm])
        result[f"rate_{nm}"] = m
        print(f"  {nm:6s} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} CC={m['CC']:.4f} "
              f"偏差={m['bias_fcst_minus_obs']:+.4f} (obs={m['obs_rate']:.4f}, fcst={m['fcst_rate']:.4f})")
    for nm in ("apc", "qm", "ols"):
        imp = 100.0 * (result["rate_gfs"]["RMSE"] - result[f"rate_{nm}"]["RMSE"]) / result["rate_gfs"]["RMSE"]
        result[f"rmse_improve_{nm}"] = float(imp)
        print(f"  {nm} RMSE 改进 vs GFS: {imp:.2f}%")

    # 月序列
    months = sorted({(d.year, d.month) for d in common})
    rows = []
    for y, mth in months:
        sel = [k for k, d in enumerate(common) if (d.year, d.month) == (y, mth)]
        def basin_mean(a): return float(np.mean(a[sel][:, cm]))
        rows.append({"year": y, "month": mth, "n_days": len(sel),
                     "gfs": basin_mean(rates["gfs"]), "apc": basin_mean(rates["apc"]),
                     "qm": basin_mean(rates["qm"]), "ols": basin_mean(rates["ols"]),
                     "chm": basin_mean(chm_rate)})
    result["monthly_series"] = rows

    out = os.path.join(WORK_DIR, f"chm_{args.tag}_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ 已保存: {out}")


if __name__ == "__main__":
    main()
