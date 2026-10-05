# -*- coding: utf-8 -*-
"""P2-2 GPM 插值敏感性：双线性 vs 面积平均（conservative）。
GPM 0.1° → 目标 0.25° 网格。检验强降水阈值结论（≥10/≥20mm ETS）对插值方案的稳健性。
一次聚合 GPM 3h，同时用两种插值重算各模型确定性 + 分类指标。
"""
import os, pickle, json
from datetime import datetime, timedelta, timezone
import numpy as np

WORK_DIR = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
GPM_ROOT = r"D:\liaohe\GPM_IMERG"
SYM42 = os.path.join(WORK_DIR, '..', 'manuscript_work_sym', 'seed42', 'predictions_apcnet.npy')
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]

with open(os.path.join(WORK_DIR, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
gfs = np.load(os.path.join(WORK_DIR, "gfs_test.npy"))
tgt = np.load(os.path.join(WORK_DIR, "targets_test.npy"))
apc = np.load(SYM42)
qm = np.load(os.path.join(WORK_DIR, "predictions_qm.npy"))
ols = np.load(os.path.join(WORK_DIR, "predictions_ols.npy"))
bincm = np.load(os.path.join(WORK_DIR, "bin_cm_pred.npy"))

def gpm_3h_window(dt):
    files = []
    for i in range(6):
        e = dt - timedelta(minutes=30 * (5 - i))
        files.append(os.path.join(GPM_ROOT, f"imerg_{e.year}{e.month:02d}",
                                  f"imerg_{e.year}{e.month:02d}{e.day:02d}_{e.hour:02d}{e.minute:02d}00.nc4"))
    return files

def load_gpm_field(path):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    g = ds.groups['Grid']
    prec = g.variables['precipitation'][0]
    lat = g.variables['lat'][:]; lon = g.variables['lon'][:]
    ds.close()
    return prec.T, lat, lon  # (60, 91)

def regrid_linear(field, lat, lon):
    from scipy.interpolate import RegularGridInterpolator
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    interp = RegularGridInterpolator((lat, lon), field, method="linear", bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    return np.where(np.isnan(vals), 0.0, vals)

def regrid_area(field, lat, lon):
    """面积平均：目标格点中心 ±0.125° 窗口内 GPM 格点，cos(lat) 加权平均。"""
    out = np.zeros((25, 37))
    for i, tlat in enumerate(GLOBAL_LATS):
        for j, tlon in enumerate(GLOBAL_LONS):
            m = (np.abs(lat - tlat)[:, None] <= 0.125) & (np.abs(lon - tlon)[None, :] <= 0.125)  # (60,91)
            if not m.any():
                out[i, j] = 0.0
                continue
            w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
            w = w[m]
            out[i, j] = np.sum(field[m] * w) / np.sum(w)
    return out

utc = timezone.utc
obs_lin, obs_area, vt = [], [], []
for i, t in enumerate(times):
    t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    if t.hour not in WINDOW_END_HOURS:
        continue
    fs = gpm_3h_window(t)
    if any(not os.path.exists(f) for f in fs):
        continue
    acc_l = None; acc_a = None
    for f in fs:
        field, lat, lon = load_gpm_field(f)
        rl = regrid_linear(field, lat, lon)
        acc_l = rl if acc_l is None else acc_l + rl
        ra = regrid_area(field, lat, lon)
        acc_a = ra if acc_a is None else acc_a + ra
    obs_lin.append(acc_l * 0.5); obs_area.append(acc_a * 0.5)
    vt.append(t)

obs_lin = np.array(obs_lin); obs_area = np.array(obs_area)
print(f"样本 {len(vt)}；线性 vs 面积 域均: {obs_lin.mean():.4f} vs {obs_area.mean():.4f}")

idx = {t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc): i for i, t in enumerate(times)}
sel = [idx[t] for t in vt]
G, T, A, Q, O, B = gfs[sel], tgt[sel], apc[sel], qm[sel], ols[sel], bincm[sel]

def metric(o, f, ths=(10.0, 20.0)):
    o = o.flatten(); f = f.flatten()
    mse = np.mean((o - f) ** 2)
    r = {"MSE": float(mse), "RMSE": float(np.sqrt(mse)), "CC": float(np.corrcoef(o, f)[0, 1]),
         "bias": float(np.mean(f - o))}
    for th in ths:
        hits = ((f >= th) & (o >= th)).sum(); fa = ((f >= th) & (o < th)).sum(); miss = ((f < th) & (o >= th)).sum()
        expect = 1.0 * (hits + fa) * (hits + miss) / o.size
        ets = (hits - expect) / max(hits + fa + miss - expect, 1e-9)
        r[f"ETS_{th}"] = float(ets)
    return r

def imp(o, m_g, f):
    return 100 * (m_g - metric(o, f)["MSE"]) / m_g

res = {"n_samples": len(vt)}
for tag, obs in [("linear", obs_lin), ("area", obs_area)]:
    m_g = metric(obs, G)["MSE"]
    res[tag] = {}
    for nm, P in [("GFS", G), ("ERA5目标", T), ("BinCM", B), ("QM", Q), ("OLS", O), ("APCNet", A)]:
        m = metric(obs, P)
        m["improve_pct"] = 100 * (m_g - m["MSE"]) / m_g
        res[tag][nm] = m
        print(f"[{tag}] {nm:<9} MSE={m['MSE']:.4f} RMSE={m['RMSE']:.4f} CC={m['CC']:.4f} bias={m['bias']:+.4f} "
              f"改进={m['improve_pct']:+.2f}%  ETS10={m['ETS_10.0']:.3f} ETS20={m['ETS_20.0']:.3f}")

with open(os.path.join(WORK_DIR, "gpm3h_interp_sens.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False, default=float)
print("\n✅ gpm3h_interp_sens.json 已保存")
