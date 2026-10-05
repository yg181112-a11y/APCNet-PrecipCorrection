# -*- coding: utf-8 -*-
"""P2-1 CHM 改进率 bootstrap 块长敏感性（60 天 / 90 天块 vs 原 24 月块）。
复用 chm_bootstrap.py 的日聚合逻辑（B2 修正口径，12h 总量/12.0 → 率）。
检验"月块"假设对显著性结论的稳健性。
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
        dates.append(day); accs.append(acc)
    return dates, np.array(accs)

def build_daily(arrs):
    dates = None; out = []
    for arr in arrs:
        d, a = group_by_day(arr)
        dates = d
        out.append(a)
    return dates, out

dates_g, ag = group_by_day(gfs)
dates = sorted(set(dates_g) & set(chm_day_map.keys()))
idx = {d: i for i, d in enumerate(dates_g)}
g = np.array([ag[idx[d]] / 12.0 for d in dates])
a = np.array([group_by_day(apc)[1][group_by_day(apc)[0].index(d)] / 12.0 for d in dates])
q = np.array([group_by_day(qm)[1][group_by_day(qm)[0].index(d)] / 12.0 for d in dates])
o = np.array([group_by_day(ols)[1][group_by_day(ols)[0].index(d)] / 12.0 for d in dates])
c = np.array([chm_day_map[d] / 24.0 for d in dates])
cm = mask
g_f, a_f, q_f, o_f, c_f = g[:, cm], a[:, cm], q[:, cm], o[:, cm], c[:, cm]
print('公共日:', len(dates))

def rmse(o, f):
    """展平口径 + isfinite 过滤（与 chm_newdata_eval.metr 一致）。"""
    o = np.asarray(o).flatten(); f = np.asarray(f).flatten()
    v = np.isfinite(o) & np.isfinite(f)
    return float(np.sqrt(np.mean((o[v] - f[v]) ** 2)))

def run_block_boot(pred, block_len_days, n_boot=2000, seed=2026):
    """block_len_days: 每块天数（连续日期）。改进率分布（展平 RMSE 口径）。"""
    n = len(dates)
    n_blocks = int(np.ceil(n / block_len_days))
    block_id = np.minimum(np.arange(n) // block_len_days, n_blocks - 1)
    obs_imp = 100 * (rmse(c_f, g_f) - rmse(c_f, pred)) / rmse(c_f, g_f)
    rng = np.random.default_rng(seed)
    dist = []
    for b in range(n_boot):
        sel = np.concatenate([np.where(block_id == m)[0] for m in rng.choice(n_blocks, size=n_blocks, replace=True)])
        imp = 100 * (rmse(c_f[sel], g_f[sel]) - rmse(c_f[sel], pred[sel])) / rmse(c_f[sel], g_f[sel])
        dist.append(imp)
    dist = np.array(dist)
    return obs_imp, dist

out = {}
for bl in [60, 90]:
    out[str(bl)] = {}
    print(f"\n=== 块长 {bl} 天 ===")
    for nm, P in [("APCNet", a_f), ("QM", q_f), ("OLS", o_f)]:
        obs_imp, dist = run_block_boot(P, bl)
        out[str(bl)][nm] = {"obs_improve_pct": obs_imp, "boot_mean": float(dist.mean()),
                            "ci95": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
                            "p_negative": float(np.mean(dist < 0)), "n_blocks": int(np.ceil(len(dates) / bl))}
        print(f"  {nm:<8} 观测={obs_imp:+.2f}%  95%CI=[{np.percentile(dist,2.5):+.2f},{np.percentile(dist,97.5):+.2f}]%  P(负)={np.mean(dist<0):.3f}")

with open(os.path.join(WORK_DIR, "chm_bootstrap_blocklen.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False, default=float)
print("\n✅ chm_bootstrap_blocklen.json 已保存")
