# -*- coding: utf-8 -*-
"""CHM 改进率 block bootstrap 显著性（统一口径版）。

数据管线逐行复用 chm_newdata_eval.py（group_by_day 保留 12h 总量、rates=/12h、CHM=/24h、
metr=展平+isfinite），保证观测改进率 == chm_newdata_eval 权威值（APCNet -31.94% 系）。
bootstrap：月度 block / 60 天块 / 90 天块，各 2000 次。
"""
import pickle, numpy as np, os, json
from datetime import datetime, timezone
from collections import defaultdict

WORK_DIR = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
SYM42 = os.path.join(WORK_DIR, '..', 'manuscript_work_sym', 'seed42', 'predictions_apcnet.npy')
CHM_2024 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2024.nc"
CHM_2025 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2025.nc"
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]

with open(os.path.join(WORK_DIR, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
gfs = np.load(os.path.join(WORK_DIR, "gfs_test.npy"))
apc = np.load(SYM42)
qm = np.load(os.path.join(WORK_DIR, "predictions_qm.npy"))
ols = np.load(os.path.join(WORK_DIR, "predictions_ols.npy"))

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
            raise ValueError(f"CHM 维度不匹配: prec.shape={prec.shape}")
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
        accs.append(acc)
    return dates, np.array(accs)

# ---- 与 chm_newdata_eval 完全一致的数据管线 ----
chm_all, chm_mask = [], None
chm_dates_all = []
for p in (CHM_2024, CHM_2025):
    prec, lat, lon, dates = read_chm_daily(p)
    for k in range(len(prec)):
        field, mask = regrid_chm_to_main(prec[k], lat, lon)
        chm_all.append(field)
        chm_dates_all.append(dates[k])
    chm_mask = mask if chm_mask is None else mask
chm_arr = np.array(chm_all)
chm_day_map = {d.date(): chm_arr[k] for k, d in enumerate(chm_dates_all)}

dts = {}
for nm, arr in [("gfs", gfs), ("apc", apc), ("qm", qm), ("ols", ols)]:
    dates, acc = group_by_day(times, arr)
    dts[nm] = (dates, acc)
common = sorted(set(dts["gfs"][0]) & set(dts["apc"][0]) & set(chm_day_map.keys()))
print(f"公共日: {len(common)}  格点: {int(chm_mask.sum())}")

rates = {}
for nm in ("gfs", "apc", "qm", "ols"):
    dates, acc = dts[nm]
    rates[nm] = np.array([acc[dates.index(d)] / 12.0 for d in common])
chm_rate = np.array([chm_day_map[d] / 24.0 for d in common])
cm = chm_mask

def flatten_masked(a):
    return a[:, cm].flatten()

def metr(o, f):
    o = np.asarray(o).flatten(); f = np.asarray(f).flatten()
    v = np.isfinite(o) & np.isfinite(f)
    return float(np.sqrt(np.mean((o[v] - f[v]) ** 2)))

# 与 chm_newdata_eval 一致：全部 925 格点（不应用 cm 掩码）
g_full, a_full, q_full, o_full, c_full = rates["gfs"], rates["apc"], rates["qm"], rates["ols"], chm_rate

obs_g, obs_a = metr(c_full, g_full), metr(c_full, a_full)
print(f"观测: GFS RMSE={obs_g:.4f} APCNet RMSE={obs_a:.4f} 改进={100*(obs_g-obs_a)/obs_g:+.2f}%")

def day_blocks(block_len):
    n = len(common)
    n_blocks = int(np.ceil(n / block_len))
    return np.minimum(np.arange(n) // block_len, n_blocks - 1), n_blocks

def run_boot(pred_full, block_id, n_blocks, n_boot=2000, seed=2026):
    rng = np.random.default_rng(seed)
    dist = []
    for b in range(n_boot):
        sel = np.concatenate([np.where(block_id == m)[0] for m in rng.choice(n_blocks, size=n_blocks, replace=True)])
        base = max(metr(c_full[sel], g_full[sel]), 1e-9)
        imp = 100 * (base - metr(c_full[sel], pred_full[sel])) / base
        dist.append(imp)
    dist = np.array(dist)
    return dist

out = {"n_days": len(common), "n_gridpoints": int(cm.sum()), "n_bootstrap": 2000,
       "obs_rmse_gfs": obs_g, "obs_apcnet": obs_a,
       "obs_improve_pct": {"apcnet": 100 * (obs_g - obs_a) / obs_g,
                           "qm": 100 * (obs_g - metr(c_full, q_full)) / obs_g,
                           "ols": 100 * (obs_g - metr(c_full, o_full)) / obs_g}}
# 月块
month_of = np.array([(d.year, d.month) for d in common])
months = sorted(set(map(tuple, month_of)))
month_id = np.array([months.index((d.year, d.month)) for d in common])
print(f"\n=== 月块 ({len(months)} 块) ===")
for nm, P in [("apcnet", a_full), ("qm", q_full), ("ols", o_full)]:
    dist = run_boot(P, month_id, len(months))
    print(f"  {nm:<8} 观测={out['obs_improve_pct'][nm]:+.2f}%  95%CI=[{np.percentile(dist,2.5):+.2f},{np.percentile(dist,97.5):+.2f}]%  P(负)={np.mean(dist<0):.3f}")
    out[f"month_block_{nm}"] = {"ci95": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
                                "p_negative": float(np.mean(dist < 0)), "boot_mean": float(dist.mean())}
# 块长敏感性
for bl in [60, 90]:
    bid, nb = day_blocks(bl)
    print(f"\n=== {bl} 天块 ({nb} 块) ===")
    for nm, P in [("apcnet", a_full), ("qm", q_full), ("ols", o_full)]:
        dist = run_boot(P, bid, nb)
        print(f"  {nm:<8} 95%CI=[{np.percentile(dist,2.5):+.2f},{np.percentile(dist,97.5):+.2f}]%  P(负)={np.mean(dist<0):.3f}")
        out[f"block{bl}_{nm}"] = {"n_blocks": nb, "ci95": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
                                  "p_negative": float(np.mean(dist < 0)), "boot_mean": float(dist.mean())}

with open(os.path.join(WORK_DIR, "chm_bootstrap.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False, default=float)
print("\n✅ chm_bootstrap.json 已保存（统一口径，含月块/60天/90天）")
