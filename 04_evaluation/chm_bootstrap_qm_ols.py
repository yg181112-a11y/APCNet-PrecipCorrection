# -*- coding: utf-8 -*-
"""M2 补实验：QM / OLS 在 CHM 验证下的月度 block bootstrap 显著性。
与 APCNet 同一数据管线（713 公共日、12h/24h 率口径、月度 block、2000 次）。
"""
import pickle, numpy as np, os, json
from datetime import timezone
from collections import defaultdict

WORK_DIR = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
CHM_2024 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2024.nc"
CHM_2025 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2025.nc"
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]

with open(os.path.join(WORK_DIR, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
gfs = np.load(os.path.join(WORK_DIR, "gfs_test.npy"))
qm = np.load(os.path.join(WORK_DIR, "predictions_qm.npy"))
ols = np.load(os.path.join(WORK_DIR, "predictions_ols.npy"))

def read_chm_daily(path):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    lat = ds.variables["lat"][:].astype(np.float64)
    lon = ds.variables["lon"][:].astype(np.float64)
    prec = ds.variables["prec"][:]
    dates = nc.num2date(ds.variables["time"][:], units=ds.variables["time"].units, only_use_cftime_datetimes=False)
    ds.close()
    if prec.ndim == 3 and prec.shape[1] == len(lon) and prec.shape[2] == len(lat):
        prec = prec.transpose(0, 2, 1)
    return prec, lat, lon, dates

from scipy.interpolate import RegularGridInterpolator
def regrid(field, lat, lon):
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    interp = RegularGridInterpolator((lat, lon), field, method="linear", bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    mask = ~np.isnan(vals)
    return np.where(mask, vals, 0.0), mask

chm_all, mask = [], None
for p in (CHM_2024, CHM_2025):
    prec, lat, lon, dates = read_chm_daily(p)
    for k in range(len(prec)):
        f, m = regrid(prec[k], lat, lon)
        chm_all.append(f); mask = m
chm_day_map = {}
k = 0
for p in (CHM_2024, CHM_2025):
    prec, lat, lon, dates = read_chm_daily(p)
    for d in dates:
        chm_day_map[d.date()] = chm_all[k]; k += 1

def group_by_day(arr):
    utc = timezone.utc
    dm = defaultdict(list)
    for i, t in enumerate(times):
        t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
        if t.hour in WINDOW_END_HOURS:
            dm[t.date()].append(i)
    dates, accs = [], []
    for day in sorted(dm.keys()):
        idx = dm[day]
        acc = np.zeros_like(arr[0])
        for i in idx:
            acc = acc + arr[i]
        dates.append(day); accs.append(acc)  # FIX B2: 总量
    return dates, np.array(accs)

dg, ag = group_by_day(gfs)
dq, aq = group_by_day(qm)
do_, ao = group_by_day(ols)
dates = sorted(set(dg) & set(dq) & set(do_) & set(chm_day_map.keys()))
print('公共日:', len(dates))
idx = {d: i for i, d in enumerate(dg)}
idq = {d: i for i, d in enumerate(dq)}
ido = {d: i for i, d in enumerate(do_)}
g = np.array([ag[idx[d]] / 12.0 for d in dates])
q = np.array([aq[idq[d]] / 12.0 for d in dates])
o = np.array([ao[ido[d]] / 12.0 for d in dates])
c = np.array([chm_day_map[d] / 24.0 for d in dates])
g_f, q_f, o_f, c_f = g[:, mask], q[:, mask], o[:, mask], c[:, mask]

def rmse(o, f):
    return float(np.sqrt(np.mean((o - f) ** 2)))

month_of = np.array([(d.year, d.month) for d in dates])
months = sorted(set(map(tuple, month_of)))
n_months = len(months)
rng = np.random.default_rng(2026)

out = {}
for nm, p_f in [('QM', q_f), ('OLS', o_f)]:
    obs_g, obs_p = rmse(c_f, g_f), rmse(c_f, p_f)
    obs_imp = 100 * (obs_g - obs_p) / obs_g
    dist = []
    for b in range(2000):
        sel_blocks = rng.choice(n_months, size=n_months, replace=True)
        sel_rows = np.concatenate([np.where((month_of[:, 0] == months[m][0]) & (month_of[:, 1] == months[m][1]))[0] for m in sel_blocks])
        imp = 100 * (rmse(c_f[sel_rows], g_f[sel_rows]) - rmse(c_f[sel_rows], p_f[sel_rows])) / rmse(c_f[sel_rows], g_f[sel_rows])
        dist.append(imp)
    dist = np.array(dist)
    out[nm] = {'obs_improve_pct': obs_imp,
               'ci95': [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
               'p_negative': float(np.mean(dist < 0))}
    print('%s: obs=%.2f%% 95%%CI=[%.2f, %.2f]%% P(neg)=%.3f' % (nm, obs_imp,
          np.percentile(dist, 2.5), np.percentile(dist, 97.5), np.mean(dist < 0)))

with open(os.path.join(WORK_DIR, 'chm_bootstrap_qm_ols.json'), 'w') as f:
    json.dump({'n_days': len(dates), 'n_month_blocks': n_months, 'n_bootstrap': 2000, **out}, f, indent=1)
print('✅ chm_bootstrap_qm_ols.json')
