# -*- coding: utf-8 -*-
"""run13_boot2.py — CHM/U-Net bootstrap（925 口径，对齐 chm_newdata_eval），GPM U-Net。"""
import os, pickle, json
import numpy as np
from datetime import timezone

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
GPM_ROOT = r'D:\liaohe\GPM_IMERG'
APC42 = os.path.join(WORK, 'predictions_apcnet.npy')
UNET = os.path.join(WORK, '..', 'manuscript_work_unet', 'predictions_apcnet.npy')
CHM_2024 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2024.nc"
CHM_2025 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2025.nc"
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]
utc = timezone.utc

gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
tgt = np.load(os.path.join(WORK, 'targets_test.npy'))
apc = np.load(APC42)
unet = np.load(UNET)
gpm3 = np.load(os.path.join(WORK, 'gpm3h_obs.npy'))
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)

sel = np.array([i for i, t in enumerate(times) if (t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)).hour in WINDOW_END_HOURS])
sel = sel[:len(gpm3)]
n = len(sel)

def block_boot(mse_g, mse_f, bl, n_boot=2000, seed=2026):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / bl))
    bid = np.minimum(np.arange(n) // bl, nb - 1)
    dist = np.empty(n_boot)
    base = mse_g.mean()
    for b in range(n_boot):
        blocks = rng.choice(nb, size=nb, replace=True)
        s = np.concatenate([np.where(bid == m)[0] for m in blocks])
        dist[b] = 100.0 * (base - mse_f[s].mean()) / base
    return dist, nb

O = gpm3.reshape(n, -1); G = gfs[sel].reshape(n, -1)
def pms(o, f): return np.mean((o - f) ** 2, axis=1)
mg = pms(O, G)
out = {}
for nm, arr in [('apc', apc), ('unet', unet)]:
    ma = pms(O, arr[sel].reshape(n, -1))
    imp = 100 * (mg.mean() - ma.mean()) / mg.mean()
    for bl in [8, 16, 32]:
        dist, nb = block_boot(mg, ma, bl)
        out[f'gpm_{nm}_bl{bl}'] = {'obs_imp': float(imp), 'ci95': [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))], 'p_neg': float(np.mean(dist < 0))}
    print(f"GPM {nm}: imp={imp:+.2f}% bl16 CI=[{np.percentile(block_boot(mg,ma,16)[0],2.5):+.2f},{np.percentile(block_boot(mg,ma,16)[0],97.5):+.2f}] P(neg)={np.mean(block_boot(mg,ma,16)[0]<0):.3f}")

# CHM 925 口径（metr 风格 isfinite）
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
for p in (CHM_2024, CHM_2025):
    prec, lat, lon, dates = read_chm_daily(p)
    for k in range(len(prec)):
        f, m = regrid_chm_to_main(prec[k], lat, lon)
        chm_map[dates[k].date()] = f
        mask925 = m

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

dts = {nm: daily_acc(a) for nm, a in [('gfs', gfs), ('apc', apc), ('unet', unet)]}
common = sorted(set(dts['gfs'][0]) & set(dts['apc'][0]) & set(dts['unet'][0]) & set(chm_map.keys()))
def r_of(nm):
    dates, acc = dts[nm]
    return np.array([acc[dates.index(d)] / 12.0 for d in common])
c = np.array([chm_map[d] / 24.0 for d in common])

def metr(o, f):
    o = o.flatten(); f = f.flatten()
    v = np.isfinite(o) & np.isfinite(f)
    return float(np.sqrt(np.mean((o[v] - f[v]) ** 2)))
def imp_of(f):
    return 100.0 * (metr(c, r_of('gfs')) - metr(c, f)) / metr(c, r_of('gfs'))
print('CHM 天数', len(common), '格点(isfinite) 925')
g_f = r_of('gfs'); a_f = r_of('apc'); u_f = r_of('unet')
print(f"CHM APCNet imp={imp_of(a_f):+.2f}%  U-Net imp={imp_of(u_f):+.2f}%")
# 月块 bootstrap（925，isfinite）
months = np.array([(d.year, d.month) for d in common])
mkeys = list(set(map(tuple, months))); nm = len(mkeys)
rng = np.random.default_rng(2026)
def chm_boot(f_pred, n_boot=2000):
    dist = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(mkeys), size=nm, replace=True)
        s = np.concatenate([np.where((months == mkeys[k]).all(axis=1))[0] for k in pick])
        o_s = c[s]; g_s = g_f[s]; f_s = f_pred[s]
        base = metr(o_s, g_s); imp_b = 100.0 * (base - metr(o_s, f_s)) / base
        dist[b] = imp_b
    return dist
for nm_, f_ in [('apc', a_f), ('unet', u_f)]:
    dist = chm_boot(f_)
    out[f'chm_{nm_}'] = {'obs_imp': float(imp_of(f_)), 'ci95': [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))], 'p_pos': float(np.mean(dist > 0)), 'n_days': len(common)}
    print(f"CHM {nm_}: imp={imp_of(f_):+.2f}% 月块CI=[{np.percentile(dist,2.5):+.2f},{np.percentile(dist,97.5):+.2f}] P(imp>0)={np.mean(dist>0):.3f}")

json.dump(out, open(r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\run13_boot2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved')
