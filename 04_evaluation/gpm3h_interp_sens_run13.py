# -*- coding: utf-8 -*-
"""GPM 插值敏感性 run13 v2：area 用 scipy.sparse 预计算权重矩阵（向量化）。"""
import os, pickle, json
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy import sparse

WORK_DIR = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
GPM_ROOT = r"D:\liaohe\GPM_IMERG"
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]
utc = timezone.utc

with open(os.path.join(WORK_DIR, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
gfs = np.load(os.path.join(WORK_DIR, "gfs_test.npy"))
tgt = np.load(os.path.join(WORK_DIR, "targets_test.npy"))
apc = np.load(os.path.join(WORK_DIR, "predictions_apcnet.npy"))
qm = np.load(os.path.join(WORK_DIR, "predictions_qm.npy"))
ols = np.load(os.path.join(WORK_DIR, "predictions_ols.npy"))
bincm = np.load(os.path.join(WORK_DIR, "bin_cm_pred.npy"))
obs_lin = np.load(os.path.join(WORK_DIR, "gpm3h_obs.npy"))

sel = np.array([i for i, t in enumerate(times)
                if ((t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)).hour in WINDOW_END_HOURS)])
sel = sel[:len(obs_lin)]
vt = [(times[i].replace(tzinfo=utc) if times[i].tzinfo is None else times[i].astimezone(utc)) for i in sel]
print(f'sel {len(sel)}')

def gpm_3h_files(dt):
    return [os.path.join(GPM_ROOT, f"imerg_{e.year}{e.month:02d}",
                         f"imerg_{e.year}{e.month:02d}{e.day:02d}_{e.hour:02d}{e.minute:02d}00.nc4")
            for i in range(6)
            for e in [dt - timedelta(minutes=30 * (5 - i))]]

def load_gpm_field(path):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    g = ds.groups['Grid']
    prec = g.variables['precipitation'][0]
    lat = g.variables['lat'][:]; lon = g.variables['lon'][:]
    ds.close()
    return prec.T, lat, lon

# 预计算 area 权重矩阵 W: (925, 5460)
rows, cols, vals = [], [], []
lat0 = None; lon0 = None
# 用第一个存在文件获取网格
for t in vt:
    fs = gpm_3h_files(t)
    ok = next((f for f in fs if os.path.exists(f)), None)
    if ok:
        field, lat0, lon0 = load_gpm_field(ok)
        break
GLAT, GLON = np.meshgrid(GLOBAL_LATS, GLOBAL_LONS, indexing='ij')
for i in range(25):
    for j in range(37):
        m = (np.abs(lat0 - GLOBAL_LATS[i])[:, None] <= 0.125) & (np.abs(lon0 - GLOBAL_LONS[j])[None, :] <= 0.125)
        idx = np.where(m.ravel())[0]
        w = np.cos(np.deg2rad(lat0))[:, None] * np.ones((1, lon0.size))
        wv = w.ravel()[idx]
        wv = wv / wv.sum()
        rows.extend([i * 37 + j] * len(idx))
        cols.extend(idx.tolist())
        vals.extend(wv.tolist())
W = sparse.csr_matrix((vals, (rows, cols)), shape=(925, 60 * 91))
print('W built', W.shape)

obs_area = []
n_ok = 0
for k, t in enumerate(vt):
    fs = gpm_3h_files(t)
    if any(not os.path.exists(f) for f in fs):
        continue
    acc = np.zeros(925)
    for f in fs:
        field, _, _ = load_gpm_field(f)
        acc += W @ field.ravel()
    obs_area.append(acc * 0.5)
    n_ok += 1
    if n_ok % 500 == 0:
        print(f'  {n_ok} done', flush=True)
obs_area = np.array(obs_area).reshape(-1, 25, 37)
print(f'obs_area {obs_area.shape}')

G, T, A, Q, O, B = gfs[sel], tgt[sel], apc[sel], qm[sel], ols[sel], bincm[sel]
obs_lin2 = obs_lin
if len(obs_area) < len(obs_lin2):
    G, T, A, Q, O, B = [x[:len(obs_area)] for x in (G, T, A, Q, O, B)]
    obs_lin2 = obs_lin2[:len(obs_area)]

def mse(o, f):
    return float(np.mean((o - f) ** 2))

res = {"n_samples": len(obs_area)}
for tag, obs in [("linear", obs_lin2), ("area", obs_area)]:
    m_g = mse(obs, G)
    res[tag] = {}
    for nm, P in [("GFS", G), ("ERA5", T), ("BinCM", B), ("QM", Q), ("OLS", O), ("APCNet", A)]:
        m = mse(obs, P)
        res[tag][nm] = {'mse': m, 'imp': 100 * (m_g - m) / m_g}
        print(f'[{tag}] {nm:<8} imp={res[tag][nm]["imp"]:+.2f}%')
with open(os.path.join(WORK_DIR, "gpm3h_interp_sens_run13.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print('saved gpm3h_interp_sens_run13.json')
