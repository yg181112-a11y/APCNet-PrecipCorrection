# -*- coding: utf-8 -*-
"""P1-5c GPM 3h 观测尺度验证分箱基准（理论最优是否跨尺度存活）。
聚合 GPM IMERG 3h（2475 样本，与 gpm3h_eval 同口径），保存 gpm3h_obs.npy 供复用；
对 GFS/ERA5目标/BinCM/QM/OLS/APCNet 做确定性指标。
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
    return prec.T, lat, lon

def regrid_to_main(field, lat, lon):
    from scipy.interpolate import RegularGridInterpolator
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    interp = RegularGridInterpolator((lat, lon), field, method="linear", bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    return np.where(np.isnan(vals), 0.0, vals)

utc = timezone.utc
gpm3, vt = [], []
for i, t in enumerate(times):
    t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    if t.hour not in WINDOW_END_HOURS:
        continue
    fs = gpm_3h_window(t)
    if any(not os.path.exists(f) for f in fs):
        continue
    acc = None
    for f in fs:
        field, lat, lon = load_gpm_field(f)
        r = regrid_to_main(field, lat, lon)
        acc = r if acc is None else acc + r
    gpm3.append(acc * 0.5)
    vt.append(t)

gpm3 = np.array(gpm3)
vt = np.array(vt)
np.save(os.path.join(WORK_DIR, "gpm3h_obs.npy"), gpm3.astype(np.float32))
print(f"GPM 3h 有效样本: {len(vt)}  ({vt.min()} ~ {vt.max()}) 已保存 gpm3h_obs.npy")

idx = {t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc): i for i, t in enumerate(times)}
sel = [idx[t] for t in vt]
G, T, A, Q, O, B = gfs[sel], tgt[sel], apc[sel], qm[sel], ols[sel], bincm[sel]

def metric(o, f):
    o = o.flatten(); f = f.flatten()
    mse = np.mean((o - f) ** 2)
    cc = np.corrcoef(o, f)[0, 1] if np.std(o) > 0 and np.std(f) > 0 else 0.0
    return {"MSE": float(mse), "RMSE": float(np.sqrt(mse)), "MAE": float(np.mean(np.abs(o - f))),
            "CC": float(cc), "bias": float(np.mean(f - o)),
            "obs_mean": float(o.mean()), "fcst_mean": float(f.mean())}

print("\n=== 确定性指标（真值 = GPM 3h 累积，2475 样本）===")
res = {"n_samples": len(vt)}
m_g = metric(gpm3, G)["MSE"]
for nm, P in [("GFS", G), ("ERA5目标", T), ("BinCM", B), ("QM", Q), ("OLS", O), ("APCNet", A)]:
    m = metric(gpm3, P)
    m["improve_pct"] = 100 * (m_g - m["MSE"]) / m_g
    res[nm] = m
    print(f"{nm:<9} MSE={m['MSE']:.4f} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} CC={m['CC']:.4f} "
          f"bias={m['bias']:+.4f} fcst={m['fcst_mean']:.4f} obs={m['obs_mean']:.4f} 改进={m['improve_pct']:+.2f}%")

with open(os.path.join(WORK_DIR, "gpm3h_bin_eval.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False, default=float)
print("\n✅ gpm3h_bin_eval.json 已保存")
