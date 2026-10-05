# -*- coding: utf-8 -*-
"""C2 UNet 版：GPM IMERG 3h 尺度独立观测验证（与 gpm3h_eval.py 同口径，仅输入数组换成 UNet 权威版）。"""
import os, sys, json, glob, pickle
from datetime import datetime, timedelta, timezone
import numpy as np

WORK_DIR = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
UNET = os.path.join(WORK_DIR, '..', 'manuscript_work_unet', 'predictions_apcnet.npy')
GPM_ROOT = r"D:\liaohe\GPM_IMERG"
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]

with open(os.path.join(WORK_DIR, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
gfs = np.load(os.path.join(WORK_DIR, "gfs_test.npy"))
tgt = np.load(os.path.join(WORK_DIR, "targets_test.npy"))
apc = np.load(UNET)
qm = np.load(os.path.join(WORK_DIR, "predictions_qm.npy"))
ols = np.load(os.path.join(WORK_DIR, "predictions_ols.npy"))

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
gpm3 = []
valid_times = []
for i, t in enumerate(times):
    t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    if t.hour not in WINDOW_END_HOURS:
        continue
    fs = gpm_3h_window(t)
    miss = [f for f in fs if not os.path.exists(f)]
    if miss:
        continue
    acc = None
    for f in fs:
        field, lat, lon = load_gpm_field(f)
        r = regrid_to_main(field, lat, lon)
        acc = r if acc is None else acc + r
    gpm3.append(acc * 0.5)
    valid_times.append(t)

gpm3 = np.array(gpm3)
vt = np.array(valid_times)
print(f"GPM 3h 有效样本: {len(vt)}  ({vt.min()} ~ {vt.max()})")

idx = {t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc): i for i, t in enumerate(times)}
sel = [idx[t] for t in vt]
G = gfs[sel]; T = tgt[sel]; A = apc[sel]; Q = qm[sel]; O = ols[sel]

def metric(o, f, th=None):
    o = o.flatten(); f = f.flatten()
    mse = np.mean((o - f) ** 2)
    cc = np.corrcoef(o, f)[0, 1] if np.std(o) > 0 and np.std(f) > 0 else 0.0
    r = {"n": int(o.size), "MSE": float(mse), "RMSE": float(np.sqrt(mse)),
         "MAE": float(np.mean(np.abs(o - f))), "CC": float(cc),
         "bias": float(np.mean(f - o)), "obs_mean": float(o.mean()), "fcst_mean": float(f.mean())}
    if th is not None:
        hits = ((f >= th) & (o >= th)).sum(); fa = ((f >= th) & (o < th)).sum(); miss = ((f < th) & (o >= th)).sum()
        pod = hits / max(hits + miss, 1); far = fa / max(hits + fa, 1)
        c = np.sqrt(max((hits + fa) * (hits + miss), 1) / max((hits + fa + miss) * max(o.size - (hits + fa + miss), 1), 1))
        ets = (hits - 1.0 * (hits + fa) * (hits + miss) / o.size) / max(hits + fa + miss - 1.0 * (hits + fa) * (hits + miss) / o.size, 1e-9)
        r[f"POD_{th}"] = float(pod); r[f"FAR_{th}"] = float(far); r[f"ETS_{th}"] = float(ets)
    return r

print("\n=== 确定性指标（真值=GPM 3h 累积） ===")
res = {"n_samples": len(vt), "n_gridpoints": G[0].size, "model": "UNet"}
for nm, P in [("GFS", G), ("ERA5目标", T), ("APCNet(UNet)", A), ("QM", Q), ("OLS", O)]:
    m = metric(gpm3, P)
    res[nm] = m
    print(f"  {nm:12s} MSE={m['MSE']:.4f} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} CC={m['CC']:.4f} "
          f"bias={m['bias']:+.4f} (obs={m['obs_mean']:.3f}, fcst={m['fcst_mean']:.3f})")
for nm in ("APCNet(UNet)", "QM", "OLS", "ERA5目标"):
    imp = 100 * (res["GFS"]["MSE"] - res[nm]["MSE"]) / res["GFS"]["MSE"]
    res[f"mse_improve_{nm}"] = float(imp)
    print(f"  {nm} MSE 改进 vs GFS: {imp:.2f}%")

print("\n=== 分类指标（3h 阈值） ===")
for th in (1, 3, 10, 20):
    print(f"  --- ≥{th} mm/3h ---")
    for nm, P in [("GFS", G), ("ERA5目标", T), ("APCNet(UNet)", A), ("QM", Q), ("OLS", O)]:
        m = metric(gpm3, P, th)
        res.setdefault("categorical", {})[f"{nm}_{th}"] = {k: m[k] for k in (f"POD_{th}", f"FAR_{th}", f"ETS_{th}")}
        print(f"    {nm:12s} POD={m[f'POD_{th}']:.3f} FAR={m[f'FAR_{th}']:.3f} ETS={m[f'ETS_{th}']:.3f}")

for nm, P in [("GFS", G), ("ERA5目标", T), ("APCNet(UNet)", A), ("QM", Q), ("OLS", O)]:
    cc = np.corrcoef(gpm3.mean(axis=(1, 2)), P.mean(axis=(1, 2)))[0, 1]
    res[f"time_cc_{nm}"] = float(cc)
    print(f"  域均时间相关 {nm}: {cc:.4f}")

with open(os.path.join(WORK_DIR, "gpm3h_unet_eval.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)
print("\n✅ gpm3h_unet_eval.json")
