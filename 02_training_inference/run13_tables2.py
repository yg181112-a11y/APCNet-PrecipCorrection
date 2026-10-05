# -*- coding: utf-8 -*-
"""补 Table 8(ERA5 分类 1/3/10/20) + Table 13 地形分层 run13。"""
import os, pickle, json
import numpy as np
from datetime import timezone

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
UNET = os.path.join(WORK, '..', 'manuscript_work_unet', 'predictions_apcnet.npy')
DEM = r'D:\liaohe\DEM\dem_merged\srtm_merged_cropped_gfsgrid.tif'
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]
utc = timezone.utc

gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
tgt = np.load(os.path.join(WORK, 'targets_test.npy'))
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
ols = np.load(os.path.join(WORK, 'predictions_ols.npy'))
unet = np.load(UNET)
apc = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
gpm3 = np.load(os.path.join(WORK, 'gpm3h_obs.npy'))
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)
sel = np.array([i for i, t in enumerate(times) if (t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)).hour in WINDOW_END_HOURS])
sel = sel[:len(gpm3)]

def cat(o, f, th):
    o, f = np.asarray(o), np.asarray(f)
    v = np.isfinite(o) & np.isfinite(f)
    o, f = o[v], f[v]
    a = o >= th; b = f >= th
    hits = np.sum(a & b); fa = np.sum(~a & b); miss = np.sum(a & ~b); cn = np.sum(~a & ~b)
    pod = hits / max(hits + miss, 1); far = fa / max(hits + fa, 1)
    tot = hits + fa + cn + miss
    ph = (hits + fa) / max(tot, 1); po = (hits + miss) / max(tot, 1)
    exp = ph * po * tot
    ets = (hits - exp) / max(hits + fa + miss - exp, 1)
    area = (hits + fa) / max(hits + miss, 1)
    return pod, far, ets, area

out = {}
print('===== Table 8 (ERA5 分类 1/3/10/20) =====')
for th in [1, 3, 10, 20]:
    pg, fg, eg, _ = cat(tgt, gfs, th)
    pa, fa_, ea, _ = cat(tgt, apc, th)
    out[f'era5_{th}'] = {'gfs': [pg, fg, eg], 'apc': [pa, fa_, ea]}
    print(f">={th:>2}: POD {pg:.3f}/{pa:.3f} FAR {fg:.3f}/{fa_:.3f} ETS {eg:.3f}/{ea:.3f}")

print('===== Table 13 (地形分层) =====')
import rasterio
with rasterio.open(DEM) as src:
    elev = src.read(1).astype(np.float32)
valid = elev > 0
e = elev[valid]
q1, q2 = np.quantile(e, [1/3, 2/3])
tiers = {'low': valid & (elev <= q1), 'mid': valid & (elev > q1) & (elev <= q2), 'high': valid & (elev > q2)}
print('tiers:', {k: int(m.sum()) for k, m in tiers.items()})

def read_chm_daily(path):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    lat = ds.variables["lat"][:].astype(np.float64); lon = ds.variables["lon"][:].astype(np.float64)
    prec = ds.variables["prec"][:]; tv = ds.variables["time"]
    dates = nc.num2date(tv[:], units=tv.units, only_use_cftime_datetimes=False)
    ds.close()
    if prec.ndim == 3 and prec.shape[1] == len(lon) and prec.shape[2] == len(lat):
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

chm_map = {}
for p in (r'D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2024.nc', r'D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2025.nc'):
    prec, lat, lon, dates = read_chm_daily(p)
    for k in range(len(prec)):
        f, m = regrid_chm_to_main(prec[k], lat, lon)
        chm_map[dates[k].date()] = f

day_map = {}
for i, t in enumerate(times):
    tt = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    day_map.setdefault(tt.date(), []).append(i)
def daily_acc(arr):
    dates, accs = [], []
    for day in sorted(day_map.keys()):
        keep = [i for i in day_map[day] if (times[i].replace(tzinfo=utc) if times[i].tzinfo is None else times[i].astimezone(utc)).hour in WINDOW_END_HOURS]
        if not keep: continue
        acc = sum(arr[i] for i in keep)
        dates.append(day); accs.append(acc)
    return dates, np.array(accs)

dts = {nm: daily_acc(a) for nm, a in [('gfs', gfs), ('apc', apc), ('qm', qm), ('ols', ols), ('unet', unet)]}
common = sorted(set.intersection(*[set(dts[nm][0]) for nm in dts]) & set(chm_map.keys()))
print('CHM common days', len(common))
r_of = lambda nm: np.array([dts[nm][1][dts[nm][0].index(d)] / 12.0 for d in common])
c = np.array([chm_map[d] / 24.0 for d in common])

def tier_rmse(o_f, m_f, mask):
    o_f = o_f[:, mask].flatten(); m_f = m_f[:, mask].flatten()
    v = np.isfinite(o_f) & np.isfinite(m_f)
    return float(np.sqrt(np.mean((o_f[v] - m_f[v])**2)))

for tn, mask in tiers.items():
    row = {'obs': float(np.nanmean(c[:, mask]))}
    for nm in ['gfs', 'qm', 'ols', 'apc', 'unet']:
        row[nm] = tier_rmse(c, r_of(nm), mask)
    out[f'chm_{tn}'] = row
    print(f"CHM {tn}: obs={row['obs']:.4f} GFS={row['gfs']:.4f} QM={row['qm']:.4f} OLS={row['ols']:.4f} APC={row['apc']:.4f} UNET={row['unet']:.4f}")

o, g = gpm3, gfs[sel]
for tn, mask in tiers.items():
    row = {'obs': float(np.nanmean(o[:, mask]))}
    for nm, arr in [('gfs', g), ('qm', qm[sel]), ('ols', ols[sel]), ('apc', apc[sel]), ('unet', unet[sel])]:
        row[nm] = tier_rmse(o, arr, mask)
    out[f'gpm_{tn}'] = row
    print(f"GPM {tn}: obs={row['obs']:.4f} GFS={row['gfs']:.4f} QM={row['qm']:.4f} OLS={row['ols']:.4f} APC={row['apc']:.4f} UNET={row['unet']:.4f}")

json.dump(out, open(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\run13_tables2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved run13_tables2.json')
