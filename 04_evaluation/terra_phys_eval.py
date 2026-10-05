# -*- coding: utf-8 -*-
"""提升项 B+C：分地形验证（CHM/GPM 按高程分层）+ 物理一致性（日循环、湿面积、域均率）。
复用测试期权威数组；DEM = srtm_merged_cropped_gfsgrid.tif（GFS 网格高程）。
"""
import numpy as np, os, json, pickle
from datetime import timedelta, timezone
import rasterio

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
SYM = os.path.join(WORK, "..", "manuscript_work_sym", "seed42")
UNET = os.path.join(WORK, "..", "manuscript_work_unet")
GPM_ROOT = r"D:\liaohe\GPM_IMERG"
CHM_2024 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2024.nc"
CHM_2025 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2025.nc"
WINDOW_END_HOURS = [3, 9, 15, 21]
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)

# ---------- 高程 ----------
with rasterio.open(r'D:\liaohe\DEM\dem_merged\srtm_merged_cropped_gfsgrid.tif') as src:
    elev = src.read(1).astype(np.float32)
print("elev shape:", elev.shape, "min/max:", elev.min(), elev.max())
valid_el = elev > 0
np.save(os.path.join(WORK, 'dem_gfsgrid.npy'), elev)
print("valid elev gridpoints:", valid_el.sum())

# ---------- 测试期数组 ----------
with open(os.path.join(WORK, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
gfs = np.load(os.path.join(WORK, "gfs_test.npy")).astype(np.float32)
qm = np.load(os.path.join(WORK, "predictions_qm.npy")).astype(np.float32)
ols = np.load(os.path.join(WORK, "predictions_ols.npy")).astype(np.float32)
apc = np.load(os.path.join(SYM, "predictions_apcnet.npy")).astype(np.float32)
unet = np.load(os.path.join(UNET, "predictions_apcnet.npy")).astype(np.float32)
print("shapes:", gfs.shape, apc.shape, unet.shape)

# ---------- GPM 对齐 ----------
obs_gpm = np.load(os.path.join(WORK, "gpm3h_obs.npy")).astype(np.float32)
utc = timezone.utc
def gpm_3h_window(dt):
    files = []
    for i in range(6):
        e = dt - timedelta(minutes=30 * (5 - i))
        files.append(os.path.join(GPM_ROOT, f"imerg_{e.year}{e.month:02d}",
                                  f"imerg_{e.year}{e.month:02d}{e.day:02d}_{e.hour:02d}{e.minute:02d}00.nc4"))
    return files
vtimes = []
for t in times:
    t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    if t.hour not in WINDOW_END_HOURS:
        continue
    if all(os.path.exists(f) for f in gpm_3h_window(t)):
        vtimes.append(t)
vtimes = np.array(vtimes)
idx = {t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc): i for i, t in enumerate(times)}
sel = np.array([idx[t] for t in vtimes])
models_gpm = {"GFS": gfs[sel], "QM": qm[sel], "OLS": ols[sel], "APCNet": apc[sel], "U-Net": unet[sel]}

# ---------- 高程分层（在有效格点内三分位） ----------
e = elev[valid_el]
q1, q2 = np.quantile(e, [1/3, 2/3])
tier = {}
tier['low'] = valid_el & (elev <= q1)
tier['mid'] = valid_el & (elev > q1) & (elev <= q2)
tier['high'] = valid_el & (elev > q2)
print(f"tier thresholds: low<=%.0f mid<=%.0f high>%.0f" % (q1, q2, q2))
for k, m in tier.items():
    print(f"  {k}: {m.sum()} gridpoints, elev {elev[m].min():.0f}-{elev[m].max():.0f} m")

def metr(o, f, mask):
    o = o[:, mask].flatten(); f = f[:, mask].flatten()
    valid = np.isfinite(o) & np.isfinite(f)
    o, f = o[valid], f[valid]
    if o.size == 0:
        return None
    mse = np.mean((o - f) ** 2)
    cc = np.corrcoef(o, f)[0, 1] if np.std(o) > 0 and np.std(f) > 0 else float('nan')
    return {"n": int(o.size), "RMSE": float(np.sqrt(mse)), "MAE": float(np.mean(np.abs(o - f))),
            "CC": float(cc), "bias": float(np.mean(f - o))}

res = {"terrain_tiers": {k: {"n_grid": int(m.sum()), "elev_range": [float(elev[m].min()), float(elev[m].max())]}
                         for k, m in tier.items()}}

# ---------- GPM 分地形 ----------
res["gpm_by_terrain"] = {}
for k, m in tier.items():
    row = {"obs_rate": float(obs_gpm[:, m].mean())}
    for nm, arr in models_gpm.items():
        row[nm] = metr(obs_gpm, arr, m)
    res["gpm_by_terrain"][k] = row
    print(f"\nGPM {k}: obs_rate={row['obs_rate']:.4f}")
    for nm in models_gpm:
        r = row[nm]
        print(f"  {nm:6s} RMSE={r['RMSE']:.4f} CC={r['CC']:.3f} bias={r['bias']:+.4f}")

# ---------- CHM 分地形（测试期 2024-2025） ----------
import netCDF4 as nc
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
    from scipy.interpolate import RegularGridInterpolator
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    interp = RegularGridInterpolator((lat, lon), chm_latlon, method="linear", bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    mask = ~np.isnan(vals)
    return np.where(mask, vals, 0.0), mask

chm_all, chm_dates_all, chm_mask = [], [], None
for p in (CHM_2024, CHM_2025):
    prec, lat, lon, dates = read_chm_daily(p)
    for k in range(len(prec)):
        field, mask = regrid_chm_to_main(prec[k], lat, lon)
        chm_all.append(field)
        chm_dates_all.append(dates[k])
    chm_mask = mask if chm_mask is None else mask
chm_arr = np.array(chm_all)
chm_day_map = {d.date(): chm_arr[k] for k, d in enumerate(chm_dates_all)}

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
print("\nCHM 公共日:", len(common))
rates = {}
for nm in ("gfs", "apc", "qm", "ols", "unet"):
    dates, acc = dts[nm]
    rates[nm] = np.array([acc[dates.index(d)] / 12.0 for d in common])
chm_rate = np.array([chm_day_map[d] / 24.0 for d in common])

cm = chm_mask
res["chm_by_terrain"] = {}
for k, m in tier.items():
    m2 = cm & m
    if m2.sum() == 0:
        continue
    row = {"obs_rate": float(np.mean(chm_rate[:, m2])), "n_grid": int(m2.sum())}
    for nm in rates:
        row[nm] = metr(chm_rate, rates[nm], m2)
    res["chm_by_terrain"][k] = row
    print(f"\nCHM {k}: obs_rate={row['obs_rate']:.4f} n_grid={m2.sum()}")
    for nm in rates:
        r = row[nm]
        print(f"  {nm:6s} RMSE={r['RMSE']:.4f} CC={r['CC']:.3f} bias={r['bias']:+.4f}")

# ---------- 日循环（GPM 4 时次） ----------
res["diurnal_cycle"] = {}
for h in WINDOW_END_HOURS:
    hm = np.array([t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc) for t in vtimes]).astype('datetime64[ns]')
    hm = np.array([t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc) for t in vtimes])
    mask_h = np.array([t.hour == h for t in hm])
    if mask_h.sum() == 0:
        continue
    res["diurnal_cycle"][f"{h:02d}Z"] = {
        "n": int(mask_h.sum()),
        "obs": float(np.mean(obs_gpm[mask_h])),
        "GFS": float(np.mean(gfs[sel][mask_h])),
        "QM": float(np.mean(qm[sel][mask_h])),
        "OLS": float(np.mean(ols[sel][mask_h])),
        "APCNet": float(np.mean(apc[sel][mask_h])),
        "U-Net": float(np.mean(unet[sel][mask_h])),
    }
    print(f"\n日循环 {h:02d}Z n={mask_h.sum()}: obs={res['diurnal_cycle'][f'{h:02d}Z']['obs']:.4f} "
          f"GFS={res['diurnal_cycle'][f'{h:02d}Z']['GFS']:.4f} QM={res['diurnal_cycle'][f'{h:02d}Z']['QM']:.4f} "
          f"APC={res['diurnal_cycle'][f'{h:02d}Z']['APCNet']:.4f} UNet={res['diurnal_cycle'][f'{h:02d}Z']['U-Net']:.4f}")

# ---------- 湿面积（≥阈值格点占比） ----------
res["wet_fraction"] = {}
for th in (0.1, 10.0, 20.0):
    row = {"obs": float(np.mean((obs_gpm >= th).mean(axis=(1, 2)))),
           "GFS": float(np.mean((gfs[sel] >= th).mean(axis=(1, 2)))),
           "QM": float(np.mean((qm[sel] >= th).mean(axis=(1, 2)))),
           "OLS": float(np.mean((ols[sel] >= th).mean(axis=(1, 2)))),
           "APCNet": float(np.mean((apc[sel] >= th).mean(axis=(1, 2)))),
           "U-Net": float(np.mean((unet[sel] >= th).mean(axis=(1, 2))))}
    res["wet_fraction"][f">={th}mm"] = row
    print(f"\n湿面积 >={th}mm/3h: obs={row['obs']:.3f} GFS={row['GFS']:.3f} QM={row['QM']:.3f} APC={row['APCNet']:.3f} U={row['U-Net']:.3f}")

with open(os.path.join(WORK, "terra_phys_eval.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1, default=str)
print("\n✅ terra_phys_eval.json saved")
