# -*- coding: utf-8 -*-
"""A1-CHM：CHM 观测侧季节分解（JJA vs 全年），全部方法（GFS/QM/OLS/APCNet/U-Net）。"""
import os, sys, json, pickle
from datetime import timezone
import numpy as np

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
CHM_2024 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2024.nc"
CHM_2025 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2025.nc"
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]

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
            raise ValueError(f"CHM dim mismatch: {prec.shape}")
    return prec, lat, lon, dates

def regrid_chm_to_main(chm_latlon, lat, lon):
    from scipy.interpolate import RegularGridInterpolator
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    interp = RegularGridInterpolator((lat, lon), chm_latlon, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    return np.where(~np.isnan(vals), vals, 0.0), ~np.isnan(vals)

def group_by_day(sample_times, arrs, full_arrays):
    """按 UTC 日聚合 4 个 3h 窗 -> 日 12h 累积场。"""
    utc = timezone.utc
    day_map = {}
    for i, t in enumerate(sample_times):
        if isinstance(t, np.datetime64):
            t = t.astype("datetime64[s]").astype(object)
        t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
        day_map.setdefault(t.date(), []).append(i)
    dates, accs, nwin = [], [], []
    for day in sorted(day_map.keys()):
        keep = []
        for i in day_map[day]:
            t = sample_times[i]
            if isinstance(t, np.datetime64):
                t = t.astype("datetime64[s]").astype(object)
            t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
            if t.hour in WINDOW_END_HOURS:
                keep.append(i)
        if not keep:
            continue
        acc = np.zeros((len(full_arrays), 25, 37))
        for i in keep:
            for k, a in enumerate(full_arrays):
                acc[k] = acc[k] + a[i]
        # 12h 累积 = 当天 4 个 3h 窗之和（不除以窗数）；rate = 总和 / 12h
        dates.append(day)
        accs.append(acc)
        nwin.append(len(keep))
    return dates, np.array(accs), nwin

def metr(o, f):
    o, f = np.asarray(o).flatten(), np.asarray(f).flatten()
    v = np.isfinite(o) & np.isfinite(f)
    o, f = o[v], f[v]
    if o.size == 0:
        return {"n": 0}
    mse = np.mean((o - f) ** 2)
    cc = np.corrcoef(o, f)[0, 1] if np.std(o) > 0 and np.std(f) > 0 else 0.0
    return {"n": int(o.size), "RMSE": float(np.sqrt(mse)), "CC": float(cc),
            "obs_rate": float(np.mean(o)), "fcst_rate": float(np.mean(f))}

# ---- 载入 ----
with open(os.path.join(WORK, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
names = ["GFS", "QM", "OLS", "APCNet", "U-Net"]
arrs = [np.load(os.path.join(WORK, "gfs_test.npy")),
        np.load(os.path.join(WORK, "predictions_qm.npy")),
        np.load(os.path.join(WORK, "predictions_ols.npy")),
        np.load(os.path.join(WORK, "predictions_apcnet.npy")),
        np.load(os.path.join(WORK, "..", "manuscript_work_unet", "predictions_apcnet.npy"))]
print("arrays:", [a.shape for a in arrs])

# ---- CHM ----
chm_all, chm_mask = [], None
chm_day_map = {}
for p in (CHM_2024, CHM_2025):
    prec, lat, lon, dates = read_chm_daily(p)
    for k in range(len(prec)):
        field, mask = regrid_chm_to_main(prec[k], lat, lon)
        chm_all.append(field)
        chm_mask = mask if chm_mask is None else mask
        d = dates[k]
        d = d.replace(tzinfo=None) if getattr(d, "tzinfo", None) else d
        chm_day_map[d.date()] = field
chm_arr = np.array(chm_all)
print("CHM days:", len(chm_day_map), "mask pts:", int(chm_mask.sum()))

# ---- 日聚合 ----
dates, acc_all, nwin = group_by_day(times, arrs, arrs)
common = sorted(set(dates) & set(chm_day_map.keys()))
print("common days:", len(common))
chm_d = np.array([chm_day_map[d] for d in common])
acc_sel = {nm: acc_all[:, k][[common.index(d) for d in dates if d in common]] for k, nm in enumerate(names)}

# ---- 季节分解 ----
out = {}
for tag, keep in [("all", None), ("JJA", lambda m: 6 <= m <= 8), ("nonJJA", lambda m: m < 6 or m > 8)]:
    idx = list(range(len(common))) if keep is None else [k for k, d in enumerate(common) if keep(d.month)]
    row = {"n_days": len(idx), "methods": {}}
    chm_f = chm_d[idx][:, chm_mask]
    chm_rate = chm_f / 24.0
    ref = metr(chm_rate, chm_rate / 1.0)  # placeholder
    gfs_r = None
    for nm in names:
        acc = acc_sel[nm][idx][:, chm_mask]
        rate = acc / 12.0
        m = metr(chm_rate, rate)
        row["methods"][nm] = m
        if nm == "GFS":
            gfs_r = m["RMSE"]
    for nm in names:
        row["methods"][nm]["rmse_imp_pct"] = 100.0 * (gfs_r - row["methods"][nm]["RMSE"]) / gfs_r
    out[tag] = row
    print(f"== {tag} n_days={len(idx)}")
    for nm in names:
        m = row["methods"][nm]
        print(f"   {nm:6s} RMSE={m['RMSE']:.4f} imp={m['rmse_imp_pct']:+.2f}% CC={m['CC']:.4f} fcst_rate={m['fcst_rate']:.4f} obs_rate={m['obs_rate']:.4f}")

json.dump(out, open(os.path.join(WORK, "chm_seasonal_run13.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False, default=float)
print("saved chm_seasonal_run13.json")
