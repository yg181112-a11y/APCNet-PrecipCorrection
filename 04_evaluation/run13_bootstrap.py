# -*- coding: utf-8 -*-
"""run13_bootstrap.py — run13 S42 的 GPM/CHM/ERA5 三参考 block bootstrap（复用 gpm3h_bootstrap / chm_bootstrap 口径）。"""
import os, pickle, json
import numpy as np
from datetime import timezone

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
GPM_ROOT = r'D:\liaohe\GPM_IMERG'
APC42 = os.path.join(WORK, 'predictions_apcnet.npy')
CHM_2024 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2024.nc"
CHM_2025 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2025.nc"
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]
utc = timezone.utc

gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
tgt = np.load(os.path.join(WORK, 'targets_test.npy'))
qm = np.load(os.path.join(WORK, 'predictions_qm.npy'))
ols = np.load(os.path.join(WORK, 'predictions_ols.npy'))
apc = np.load(APC42)
gpm3 = np.load(os.path.join(WORK, 'gpm3h_obs.npy'))
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)

vt = []
for i, t in enumerate(times):
    tt = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    if tt.hour in WINDOW_END_HOURS:
        vt.append(tt)
vt = np.array(vt)
sel = np.array([i for i, t in enumerate(times) if (t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)).hour in WINDOW_END_HOURS])
sel = sel[:len(gpm3)]
n = len(sel)
print('GPM n=', n)

def block_boot(mse_g, mse_f, block_len, n_boot=2000, seed=2026):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block_len))
    bid = np.minimum(np.arange(n) // block_len, nb - 1)
    dist = np.empty(n_boot)
    base = mse_g.mean()
    for b in range(n_boot):
        blocks = rng.choice(nb, size=nb, replace=True)
        s = np.concatenate([np.where(bid == m)[0] for m in blocks])
        dist[b] = 100.0 * (base - mse_f[s].mean()) / base
    return dist, nb

def per_sample_mse(o, f):
    return np.mean((o - f) ** 2, axis=1)

O = gpm3.reshape(n, -1)
G = gfs[sel].reshape(n, -1)
A = apc[sel].reshape(n, -1)
mse_g = per_sample_mse(O, G); mse_a = per_sample_mse(O, A)
out = {'protocol': 'run13 S42 block bootstrap', 'n_samples': int(n)}
for bl in [8, 16, 32]:
    dist, nb = block_boot(mse_g, mse_a, bl)
    out[f'gpm_bl{bl}'] = {'ci95': [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
                          'p_neg': float(np.mean(dist < 0)), 'obs_imp': float(100*(mse_g.mean()-mse_a.mean())/mse_g.mean())}
    print(f"GPM bl={bl}: imp={100*(mse_g.mean()-mse_a.mean())/mse_g.mean():+.2f}% CI=[{np.percentile(dist,2.5):+.2f},{np.percentile(dist,97.5):+.2f}] P(neg)={np.mean(dist<0):.3f}")

# CHM 月块 bootstrap（复用 chm_newdata_eval 日序列）
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

chm_all, chm_dates_all, chm_mask = [], [], None
for p in (CHM_2024, CHM_2025):
    prec, lat, lon, dates = read_chm_daily(p)
    for k in range(len(prec)):
        f, m = regrid_chm_to_main(prec[k], lat, lon)
        chm_all.append(f); chm_dates_all.append(dates[k])
    chm_mask = m if chm_mask is None else m
chm_map = {d.date(): np.array(chm_all)[k] for k, d in enumerate(chm_dates_all)}

day_map = {}
for i, t in enumerate(times):
    tt = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    day_map.setdefault(tt.date(), []).append(i)

def daily_acc(arr):
    dates, accs = [], []
    for day in sorted(day_map.keys()):
        keep = [i for i in day_map[day] if (times[i].replace(tzinfo=utc) if times[i].tzinfo is None else times[i].astimezone(utc)).hour in WINDOW_END_HOURS]
        if not keep:
            continue
        acc = sum(arr[i] for i in keep)
        dates.append(day); accs.append(acc)
    return dates, np.array(accs)

dts = {nm: daily_acc(a) for nm, a in [('gfs', gfs), ('apc', apc), ('qm', qm), ('ols', ols)]}
common = sorted(set(dts['gfs'][0]) & set(dts['apc'][0]) & set(chm_map.keys()))
cm = chm_mask
def rates_of(nm):
    dates, acc = dts[nm]
    return np.array([acc[dates.index(d)] / 12.0 for d in common])
r_g = rates_of('gfs'); r_a = rates_of('apc')
c = np.array([chm_map[d] / 24.0 for d in common])
rf = lambda r: r[:, cm].flatten()
o_f = rf(c); g_f = rf(r_g); a_f = rf(r_a)
v = np.isfinite(o_f)
o_f, g_f, a_f = o_f[v], g_f[v], a_f[v]
def chm_imp(a_f):
    return 100.0 * (np.sqrt(np.mean((o_f - g_f)**2)) - np.sqrt(np.mean((o_f - a_f)**2))) / np.sqrt(np.mean((o_f - g_f)**2))
print('CHM obs imp =', round(chm_imp(a_f), 2), '%')
# 月块 bootstrap（按天索引）
day_index = {d: k for k, d in enumerate(common)}
days = np.array([day_index[d] for d in common])
months = np.array([(d.year, d.month) for d in common])
rng = np.random.default_rng(2026)
n_m = len(set(map(tuple, months)))
dist = np.empty(2000)
base = np.sqrt(np.mean((o_f - g_f)**2))
for b in range(2000):
    mkeys = list(set(map(tuple, months)))
    pick = rng.choice(len(mkeys), size=n_m, replace=True)
    s = np.concatenate([np.where((months == mkeys[k]).all(axis=1))[0] for k in pick])
    sub = np.unique(s)
    a_sub = a_f[np.concatenate([np.where((months == mkeys[k]).all(axis=1))[0] for k in pick])]
    g_sub = g_f[np.concatenate([np.where((months == mkeys[k]).all(axis=1))[0] for k in pick])]
    o_sub = o_f[np.concatenate([np.where((months == mkeys[k]).all(axis=1))[0] for k in pick])]
    dist[b] = 100.0 * (base - np.sqrt(np.mean((o_sub - a_sub)**2))) / base
out['chm_monthblock'] = {'obs_imp': float(chm_imp(a_f)), 'ci95': [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
                         'p_pos': float(np.mean(dist > 0)), 'n_days': len(common)}
print(f"CHM 月块: imp={chm_imp(a_f):+.2f}% CI=[{np.percentile(dist,2.5):+.2f},{np.percentile(dist,97.5):+.2f}] P(imp>0)={np.mean(dist>0):.3f}")

# ERA5 3h 月块 bootstrap
T = tgt.reshape(len(tgt), -1); G2 = gfs.reshape(len(gfs), -1); A2 = apc.reshape(len(apc), -1)
dates_all = np.array([(t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)).date() for t in times])
months_all = np.array([(d.year, d.month) for d in dates_all])
def era5_imp(a2):
    return 100.0 * (np.mean((T - G2)**2) - np.mean((T - a2)**2)) / np.mean((T - G2)**2)
mkeys = list(set(map(tuple, months_all)))
n_m = len(mkeys)
dist2 = np.empty(2000)
for b in range(2000):
    pick = rng.choice(len(mkeys), size=n_m, replace=True)
    s = np.concatenate([np.where((months_all == mkeys[k]).all(axis=1))[0] for k in pick])
    d2 = 100.0 * (np.mean((T[s] - G2[s])**2) - np.mean((T[s] - A2[s])**2)) / np.mean((T[s] - G2[s])**2)
    dist2[b] = d2
out['era5_monthblock'] = {'obs_imp': float(era5_imp(A2)), 'ci95': [float(np.percentile(dist2, 2.5)), float(np.percentile(dist2, 97.5))],
                          'p_pos': float(np.mean(dist2 > 0))}
print(f"ERA5 月块: imp={era5_imp(A2):+.2f}% CI=[{np.percentile(dist2,2.5):+.2f},{np.percentile(dist2,97.5):+.2f}] P(imp>0)={np.mean(dist2>0):.3f}")

json.dump(out, open(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\run13_bootstrap.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved run13_bootstrap.json')
