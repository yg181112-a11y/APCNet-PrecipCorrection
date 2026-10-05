# -*- coding: utf-8 -*-
"""A 项补充：验证期(2022-2023) CHM block-bootstrap 显著性（60d/90d 块），口径同 chm_bootstrap_blocklen。
另算验证期逐样本空间 CC（ERA5 参照，3h 尺度）。
输出：chm_val_boot.json
"""
import pickle, numpy as np, os, json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import netCDF4 as nc
from scipy.interpolate import RegularGridInterpolator

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
CHM_FILES = [r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2022.nc",
             r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2023.nc"]
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]
utc = timezone.utc

with open(os.path.join(WORK, "sample_times_val.pkl"), "rb") as f:
    times = pickle.load(f)
gfs = np.load(os.path.join(WORK, "gfs_val_precip.npy")).astype(np.float32)
apc = np.load(os.path.join(WORK, "predictions_val_apcnet.npy")).astype(np.float32)
unet = np.load(os.path.join(WORK, "predictions_val_unet.npy")).astype(np.float32)

# ---- QM/OLS 重建（训练期拟合，同 chm_val_eval） ----
tr_g = np.load(os.path.join(WORK, 'train_gfs.npy')).astype(np.float32)
tr_e = np.load(os.path.join(WORK, 'train_era5.npy')).astype(np.float32)
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
print("QM/OLS val rebuilt", flush=True)

# ---- CHM 2022-2023 ----
def read_chm_daily(path):
    ds = nc.Dataset(path)
    lat = ds.variables["lat"][:].astype(np.float64)
    lon = ds.variables["lon"][:].astype(np.float64)
    prec = ds.variables["prec"][:]
    dates = nc.num2date(ds.variables["time"][:], units=ds.variables["time"].units, only_use_cftime_datetimes=False)
    ds.close()
    if prec.ndim == 3 and prec.shape[1] == len(lon) and prec.shape[2] == len(lat):
        prec = prec.transpose(0, 2, 1)
    return prec, lat, lon, dates

def regrid(field, lat, lon):
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    interp = RegularGridInterpolator((lat, lon), field, method="linear", bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    mask = ~np.isnan(vals)
    return np.where(mask, vals, 0.0), mask

chm_all, mask = [], None
for p in CHM_FILES:
    prec, lat, lon, dates = read_chm_daily(p)
    for k in range(len(prec)):
        f, m = regrid(prec[k], lat, lon)
        chm_all.append(f); mask = m
chm_day_map = {}
k = 0
for p in CHM_FILES:
    prec, lat, lon, dates = read_chm_daily(p)
    for d in dates:
        chm_day_map[d.date()] = chm_all[k]; k += 1

def group_by_day(arr):
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

dates_g, ag = group_by_day(gfs)
dates = sorted(set(dates_g) & set(chm_day_map.keys()))
idx = {d: i for i, d in enumerate(dates_g)}
g = np.array([ag[idx[d]] / 12.0 for d in dates])
a = np.array([group_by_day(apc)[1][group_by_day(apc)[0].index(d)] / 12.0 for d in dates])
u = np.array([group_by_day(unet)[1][group_by_day(unet)[0].index(d)] / 12.0 for d in dates])
q = np.array([group_by_day(qm)[1][group_by_day(qm)[0].index(d)] / 12.0 for d in dates])
o = np.array([group_by_day(ols)[1][group_by_day(ols)[0].index(d)] / 12.0 for d in dates])
c = np.array([chm_day_map[d] / 24.0 for d in dates])
cm = mask
g_f, a_f, u_f, q_f, o_f, c_f = g[:, cm], a[:, cm], u[:, cm], q[:, cm], o[:, cm], c[:, cm]
print("common days:", len(dates), flush=True)

def rmse(o, f):
    o = np.asarray(o).flatten(); f = np.asarray(f).flatten()
    v = np.isfinite(o) & np.isfinite(f)
    return float(np.sqrt(np.mean((o[v] - f[v]) ** 2)))

def run_block_boot(pred, block_len_days, n_boot=2000, seed=2026):
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
    print(f"\n=== 块长 {bl} 天 ===", flush=True)
    for nm, P in [("APCNet", a_f), ("U-Net", u_f), ("QM", q_f), ("OLS", o_f)]:
        obs_imp, dist = run_block_boot(P, bl)
        out[str(bl)][nm] = {"obs_improve_pct": float(obs_imp), "boot_mean": float(dist.mean()),
                            "ci95": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
                            "p_negative": float(np.mean(dist < 0)), "p_positive": float(np.mean(dist > 0)),
                            "n_blocks": int(np.ceil(len(dates) / bl))}
        print(f"  {nm:<8} obs={obs_imp:+.2f}%  95%CI=[{np.percentile(dist,2.5):+.2f},{np.percentile(dist,97.5):+.2f}]%  P(neg)={np.mean(dist<0):.3f}", flush=True)

# ---- 验证期逐样本空间 CC（ERA5 参照） ----
import importlib.util
M12 = r"D:\liaohe\校正优化过程\第三阶段\12优化\12.8修\12.8修_final.py"
spec = importlib.util.spec_from_file_location("m128", M12)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ep = m.ERA5DataProcessor()
era5_map = ep.load_era5_data_with_times(
    [r"D:\liaohe\ERA5-data\monthly", r"D:\liaohe\ERA5-data\processed_monthly"],
    start_date="2022-01-01", end_date="2023-12-31")
era5_t0 = np.array([era5_map[t][5] for t in times]).astype(np.float32)  # [N,25,37]
print("era5 t0 shape:", era5_t0.shape, flush=True)

def space_cc(a, b):
    cc = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        x, y = a[i], b[i]
        if np.std(x) > 0 and np.std(y) > 0:
            cc[i] = np.corrcoef(x.flatten(), y.flatten())[0, 1]
    return cc

sc = {}
sc["GFS"] = float(np.mean(space_cc(era5_t0, gfs)))
sc["APCNet"] = float(np.mean(space_cc(era5_t0, apc)))
sc["U-Net"] = float(np.mean(space_cc(era5_t0, unet)))
sc["QM"] = float(np.mean(space_cc(era5_t0, qm)))
sc["OLS"] = float(np.mean(space_cc(era5_t0, ols)))
print("per-sample space CC (ERA5 ref):", {k: round(v, 4) for k, v in sc.items()}, flush=True)
out["space_cc_era5"] = sc

with open(os.path.join(WORK, "chm_val_boot.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False, default=float)
print("\n✅ chm_val_boot.json saved")
