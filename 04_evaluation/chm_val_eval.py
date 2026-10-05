# -*- coding: utf-8 -*-
"""A 项：CHM 独立验证期（2022-2023）评估。
- 验证期推理数组：gfs_val_precip / predictions_val_apcnet / predictions_val_unet / sample_times_val.pkl
- QM/OLS 基线：训练期(2015-2021)逐格点拟合 + 验证期 GFS 应用（样本外）
- CHM 2022/2023 日数据 regrid 到 GFS 网格，12h 覆盖（03/09/15/21Z），口径与测试期 Table 6 一致
"""
import numpy as np, os, json, pickle
from datetime import timedelta, timezone
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
CHM_FILES = [r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2022.nc",
             r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2023.nc"]
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]
utc = timezone.utc

# ---------- 验证期数组 ----------
with open(os.path.join(WORK, "sample_times_val.pkl"), "rb") as f:
    times = pickle.load(f)
gfs = np.load(os.path.join(WORK, "gfs_val_precip.npy")).astype(np.float32)
apc = np.load(os.path.join(WORK, "predictions_val_apcnet.npy")).astype(np.float32)
unet = np.load(os.path.join(WORK, "predictions_val_unet.npy")).astype(np.float32)
print("val shapes:", gfs.shape, apc.shape, unet.shape)

# ---------- QM/OLS（训练期拟合） ----------
tr_g = np.load(os.path.join(WORK, 'train_gfs.npy')).astype(np.float32)
tr_e = np.load(os.path.join(WORK, 'train_era5.npy')).astype(np.float32)

# OLS 逐格点
N = tr_g.shape[0]
G = tr_g.reshape(N, -1); E = tr_e.reshape(N, -1)
cols = G.shape[1]
A = np.zeros((cols, 2))
for j in range(cols):
    x = G[:, j]; y = E[:, j]
    X = np.stack([x, np.ones_like(x)], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        coef = [0.0, y.mean()]
    A[j] = coef
T = gfs.shape[0]
Gt = gfs.reshape(T, -1)
ols = np.zeros_like(Gt)
for j in range(cols):
    ols[:, j] = A[j, 0] * Gt[:, j] + A[j, 1]
ols = ols.reshape(T, 25, 37).astype(np.float32)
print("OLS val predicted")

# QM 逐格点（与 13.0_main._build_qm/_apply_qm 同逻辑）
qs = np.linspace(0, 1, 1002)[1:-1]
qm_gfs = np.zeros((25, 37, 1000), dtype=np.float32)
qm_era5 = np.zeros((25, 37, 1000), dtype=np.float32)
p_dry_gfs = np.zeros((25, 37), dtype=np.float32)
p_dry_era5 = np.zeros((25, 37), dtype=np.float32)
for idx in range(25 * 37):
    i, j = divmod(idx, 37)
    gg, ee = G[:, idx], E[:, idx]
    wet_g, wet_e = gg >= 0.1, ee >= 0.1
    p_dry_gfs[i, j] = 1.0 - wet_g.mean()
    p_dry_era5[i, j] = 1.0 - wet_e.mean()
    if wet_g.sum() > 5 and wet_e.sum() > 5:
        qm_gfs[i, j] = np.quantile(gg[wet_g], qs)
        qm_era5[i, j] = np.quantile(ee[wet_e], qs)
    else:
        qm_gfs[i, j] = np.quantile(gg, qs)
        qm_era5[i, j] = np.quantile(ee, qs)
rng = np.random.default_rng(42)
qm = np.zeros_like(gfs)
Gv = gfs.reshape(T, -1)
for idx in range(25 * 37):
    i, j = divmod(idx, 37)
    gg = Gv[:, idx]
    wet = gg >= 0.1
    p_keep_wet = np.clip((1 - p_dry_era5[i, j]) / max(1 - p_dry_gfs[i, j], 1e-6), 0, 1)
    keep = wet.copy()
    if p_keep_wet < 1.0 and keep.any():
        keep[keep] = rng.random(keep.sum()) < p_keep_wet
    mapped = np.maximum(np.interp(gg[keep], qm_gfs[i, j], qm_era5[i, j]), 0.1)
    qm.reshape(T, -1)[keep, idx] = mapped
qm = qm.reshape(T, 25, 37).astype(np.float32)
print("QM val predicted")

# ---------- CHM 2022-2023 ----------
def read_chm_daily(path):
    ds = nc.Dataset(path)
    lat = ds.variables["lat"][:].astype(np.float64)
    lon = ds.variables["lon"][:].astype(np.float64)
    prec = ds.variables["prec"][:]
    tv = ds.variables["time"]
    dates = nc.num2date(tv[:], units=tv.units, only_use_cftime_datetimes=False)
    ds.close()
    if prec.ndim == 3:
        if prec.shape[1] == len(lon) and prec.shape[2] == len(lat):
            prec = prec.transpose(0, 2, 1)
    return prec, lat, lon, dates

def regrid_chm_to_main(chm_latlon, lat, lon):
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    interp = RegularGridInterpolator((lat, lon), chm_latlon, method="linear", bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    mask = ~np.isnan(vals)
    return np.where(mask, vals, 0.0), mask

chm_all, chm_dates_all, chm_mask = [], [], None
for p in CHM_FILES:
    prec, lat, lon, dates = read_chm_daily(p)
    for k in range(len(prec)):
        field, mask = regrid_chm_to_main(prec[k], lat, lon)
        chm_all.append(field)
        chm_dates_all.append(dates[k])
    chm_mask = mask if chm_mask is None else mask
chm_arr = np.array(chm_all)
chm_day_map = {d.date(): chm_arr[k] for k, d in enumerate(chm_dates_all)}
print("CHM days:", len(chm_day_map))

def group_by_day(sample_times, arr):
    day_map = {}
    for i, t in enumerate(sample_times):
        t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
        day_map.setdefault(t.date(), []).append(i)
    dates, accs = [], []
    for day in sorted(day_map.keys()):
        keep = [i for i in day_map[day] if (sample_times[i].replace(tzinfo=utc) if sample_times[i].tzinfo is None else sample_times[i].astimezone(utc)).hour in WINDOW_END_HOURS]
        if not keep:
            continue
        acc = np.zeros_like(arr[0])
        for i in keep:
            acc = acc + arr[i]
        dates.append(day)
        accs.append(acc)
    return dates, np.array(accs)

dts = {}
for nm, arr in [("gfs", gfs), ("apc", apc), ("qm", qm), ("ols", ols), ("unet", unet)]:
    dates, acc = group_by_day(times, arr)
    dts[nm] = (dates, acc)
common = sorted(set(dts["gfs"][0]) & set(chm_day_map.keys()))
print("CHM common days:", len(common))

rates = {}
for nm in ("gfs", "apc", "qm", "ols", "unet"):
    dates, acc = dts[nm]
    rates[nm] = np.array([acc[dates.index(d)] / 12.0 for d in common])
chm_rate = np.array([chm_day_map[d] / 24.0 for d in common])

def metr(o, f, mask):
    o = o[:, mask].flatten(); f = f[:, mask].flatten()
    valid = np.isfinite(o) & np.isfinite(f)
    o, f = o[valid], f[valid]
    mse = np.mean((o - f) ** 2)
    cc = np.corrcoef(o, f)[0, 1] if np.std(o) > 0 and np.std(f) > 0 else float('nan')
    return {"n": int(o.size), "RMSE": float(np.sqrt(mse)), "MAE": float(np.mean(np.abs(o - f))),
            "CC": float(cc), "bias": float(np.mean(f - o))}

res = {"chm_val": {"n_days": int(len(common)), "n_grid": int(chm_mask.sum()),
                   "obs_rate": float(np.mean(chm_rate[:, chm_mask]))}}
for nm in rates:
    r = metr(chm_rate, rates[nm], chm_mask)
    res["chm_val"][nm] = r
    print(f"{nm:6s} RMSE={r['RMSE']:.4f} MAE={r['MAE']:.4f} CC={r['CC']:.3f} bias={r['bias']:+.4f} n={r['n']}")

base = res["chm_val"]["gfs"]["RMSE"]
for nm in ("apc", "unet", "qm", "ols"):
    r = res["chm_val"][nm]["RMSE"]
    res["chm_val"][f"{nm}_rmse_pct"] = float((r - base) / base * 100)
    print(f"{nm} RMSE change vs GFS: {(r - base) / base * 100:+.2f}%")

# 域均率
for nm in rates:
    res["chm_val"][f"{nm}_rate"] = float(np.mean(rates[nm][:, chm_mask]))
    print(f"{nm} rate: {np.mean(rates[nm][:, chm_mask]):.4f} mm/h")

with open(os.path.join(WORK, "chm_val_eval.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1, default=str)
print("\n✅ chm_val_eval.json saved")
