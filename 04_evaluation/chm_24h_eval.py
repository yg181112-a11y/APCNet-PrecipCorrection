# -*- coding: utf-8 -*-
"""CHM 24h 独立验证（24h 实验，干净口径：模型 [00Z,24h] 累积 vs CHM UTC 日值）
- 只取 init=00Z 的测试样本（窗 = UTC 日，与 CHM 日边界完美对齐）
- CHM 2024/2025 日值 regrid 到主网格（25×37）后直接比较 24h 总量
"""
import os, json, pickle
from datetime import datetime, timezone
import numpy as np

E24 = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\24h_exp'
CHM_2024 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2024.nc"
CHM_2025 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2025.nc"
OUT = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\24h_exp\chm_24h_eval.json'
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
utc = timezone.utc


def read_chm_daily(path):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    lat = ds.variables["lat"][:].astype(np.float64)
    lon = ds.variables["lon"][:].astype(np.float64)
    prec = ds.variables["prec"][:]
    tv = ds.variables["time"]
    dates = nc.num2date(tv[:], units=tv.units, only_use_cftime_datetimes=False)
    ds.close()
    if prec.ndim == 3 and prec.shape[1] == len(lat) and prec.shape[2] == len(lon):
        pass
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


# ---------- 24h 测试样本（init=00Z 子集） ----------
inits = np.load(os.path.join(E24, 'init_times.npy'))
split = np.load(os.path.join(E24, 'split_mask.npy'))
g24 = np.load(os.path.join(E24, 'gfs_24h_accum.npy'))
t24 = np.load(os.path.join(E24, 'era5_24h_accum.npy'))
p24 = np.load(os.path.join(E24, 'pred_apcnet_24h.npy'))

valid = ~(np.isnan(g24).any((1, 2)) | np.isnan(t24).any((1, 2)))
tm = (split == 2) & valid
g24, t24 = g24[tm], t24[tm]
p24 = p24[:len(g24)]
init_dt = np.array([np.datetime64(s, 'h') for s in inits[tm]], dtype='datetime64[h]').astype(datetime)
sel = np.array([d.hour == 0 for d in init_dt])  # init=00Z → [00,24] UTC 窗
g24s, t24s, p24s, dates = g24[sel], t24[sel], p24[sel], [d.date() for d in init_dt[sel]]
print(f'init=00Z 测试样本: {len(dates)} ({dates[0]} → {dates[-1]})')

# ---------- CHM ----------
chm_all, chm_mask, chm_dates = [], None, []
for p in (CHM_2024, CHM_2025):
    prec, lat, lon, dts = read_chm_daily(p)
    print(f'CHM {os.path.basename(p)}: {prec.shape}')
    for k in range(len(prec)):
        field, mask = regrid_chm_to_main(prec[k], lat, lon)
        chm_all.append(field)
        chm_dates.append(dts[k].date())
    chm_mask = mask if chm_mask is None else mask
chm_arr = np.array(chm_all)
chm_day_map = {d: chm_arr[k] for k, d in enumerate(chm_dates)}
print(f'CHM 日数: {len(chm_day_map)}, 主网格覆盖格点: {int(chm_mask.sum())}')

# ---------- 公共日 ----------
common = sorted(set(dates) & set(chm_day_map.keys()))
print(f'公共日: {len(common)}')
if len(common) < 30:
    raise SystemExit('公共日不足')

def flatten(a):
    return a[:, chm_mask].flatten()

G = np.array([g24s[dates.index(d)] for d in common])
T = np.array([t24s[dates.index(d)] for d in common])
P = np.array([p24s[dates.index(d)] for d in common])
C = np.array([chm_day_map[d] for d in common])

def metr(obs, fcst, tag):
    o = obs[:, chm_mask].flatten(); f = fcst[:, chm_mask].flatten()
    v = np.isfinite(o) & np.isfinite(f)
    o, f = o[v], f[v]
    mse = np.mean((o - f) ** 2)
    cc = np.corrcoef(o, f)[0, 1] if np.std(o) > 0 and np.std(f) > 0 else 0.0
    out = {"n": int(o.size), "RMSE": float(np.sqrt(mse)), "MAE": float(np.mean(np.abs(o - f))),
           "CC": float(cc), "bias": float(np.mean(f - o)),
           "obs_mean": float(np.mean(o)), "fcst_mean": float(np.mean(f))}
    # 极端（≥20 mm/24h）
    for thr in (10, 20, 30):
        oo, ff = o[o >= thr], f[o >= thr]
        out[f'obs_n_ge{thr}'] = int(oo.size)
        out[f'fcst_n_ge{thr}'] = int(ff.size)
    # ETS（≥10 / ≥20）
    def ets(o, f, thr):
        a = ((o >= thr) & (f >= thr)).sum(); b = ((o < thr) & (f >= thr)).sum()
        c = ((o >= thr) & (f < thr)).sum(); d = ((o < thr) & (f < thr)).sum()
        den = a + b + c
        return float((a - (a + b) * (a + c) / max(den, 1)) / max(den - (a + b) * (a + c) / max(den, 1), 1e-9)) if den > 0 else 0.0
    for thr in (10, 20):
        out[f'ETS{thr}'] = ets(o, f, thr)
    out['_tag'] = tag
    return out

res = {}
for nm, arr in [('gfs', G), ('apc', P), ('era5_ref', T)]:
    res[nm] = metr(C, arr, nm)
    print(f"  {nm:9s} RMSE={res[nm]['RMSE']:.3f} MAE={res[nm]['MAE']:.3f} CC={res[nm]['CC']:.3f} "
          f"bias={res[nm]['bias']:+.3f} (obs={res[nm]['obs_mean']:.3f}, fcst={res[nm]['fcst_mean']:.3f}) "
          f"ETS10={res[nm]['ETS10']:.3f} ETS20={res[nm]['ETS20']:.3f}")
for nm in ('apc', 'era5_ref'):
    imp = 100.0 * (res['gfs']['RMSE'] - res[nm]['RMSE']) / res['gfs']['RMSE']
    res[f'rmse_improve_{nm}'] = float(imp)
    print(f"  {nm} RMSE 改进 vs GFS: {imp:+.2f}%")

# 月序列
months = sorted({(d.year, d.month) for d in common})
rows = []
for y, m in months:
    sel_m = [k for k, d in enumerate(common) if (d.year, d.month) == (y, m)]
    rows.append({"year": y, "month": m, "n_days": len(sel_m),
                 "gfs": float(np.mean(G[sel_m][:, chm_mask])),
                 "apc": float(np.mean(P[sel_m][:, chm_mask])),
                 "era5": float(np.mean(T[sel_m][:, chm_mask])),
                 "chm": float(np.mean(C[sel_m][:, chm_mask]))})
res['monthly_series'] = rows

res['protocol'] = 'CHM 24h daily verification (UTC-day window [00Z,24h], init=00Z only)'
res['n_common_days'] = len(common)
res['coverage_gridpoints'] = int(chm_mask.sum())
json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
print(f'\n✅ 已保存: {OUT}')
