# -*- coding: utf-8 -*-
"""gpm3h_bootstrap.py — GPM 3h 独立验证 MSE 改进率的 block bootstrap 显著性检验。

数据管线逐行复用 gpm3h_eval.py（GPM 30min×6 累加为 3h、双线性插值、覆盖掩码），
保证观测改进率 == gpm3h_eval.json 权威值（APCNet -29.31% / QM +8.25% / OLS +19.63%）。
bootstrap：时间块自助（GPM 3h 连续时次存在自相关——同一对流系统跨多个时次），
主块长 16 时次（48h），敏感性 8/32 时次，各 2000 次。
审稿人 R1 曾点名"bootstrap 未处理自相关"，此脚本统一 CHM/GPM 两处统计口径。
"""
import os, json, pickle
from datetime import datetime, timedelta, timezone
import numpy as np

WORK_DIR = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
GPM_ROOT = r"D:\liaohe\GPM_IMERG"
SYM42 = os.path.join(WORK_DIR, '..', 'manuscript_work_sym', 'seed42', 'predictions_apcnet.npy')
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]
UTC = timezone.utc

with open(os.path.join(WORK_DIR, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
gfs = np.load(os.path.join(WORK_DIR, "gfs_test.npy"))
apc = np.load(SYM42)
qm = np.load(os.path.join(WORK_DIR, "predictions_qm.npy"))
ols = np.load(os.path.join(WORK_DIR, "predictions_ols.npy"))

def gpm_3h_files(dt):
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
    prec = g.variables['precipitation'][0]  # (91,60) mm/hr
    lat = g.variables['lat'][:]; lon = g.variables['lon'][:]
    ds.close()
    return prec.T, lat, lon  # (60,91)

def regrid_to_main(field, lat, lon):
    from scipy.interpolate import RegularGridInterpolator
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    interp = RegularGridInterpolator((lat, lon), field, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    return np.where(np.isnan(vals), 0.0, vals)

# ---- 逐样本 GPM 3h 累积（与 gpm3h_eval.py 完全一致） ----
gpm3, vt = [], []
for i, t in enumerate(times):
    t = t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)
    if t.hour not in WINDOW_END_HOURS:
        continue
    fs = gpm_3h_files(t)
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
n = len(vt)
print(f"GPM 3h 有效样本: {n}  ({vt[0]} ~ {vt[-1]})")

idx = {t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC): i for i, t in enumerate(times)}
sel = [idx[t] for t in vt]
G = gfs[sel].reshape(n, -1)
A = apc[sel].reshape(n, -1)
Q = qm[sel].reshape(n, -1)
O = ols[sel].reshape(n, -1)
OBS = gpm3.reshape(n, -1)

def per_sample_mse(o, f):
    return np.mean((o - f) ** 2, axis=1)

mse_g = per_sample_mse(OBS, G)
mse_a = per_sample_mse(OBS, A)
mse_q = per_sample_mse(OBS, Q)
mse_o = per_sample_mse(OBS, O)

def imp(mse_f):
    return 100.0 * (mse_g.mean() - mse_f.mean()) / mse_g.mean()

obs_imp = {"apcnet": imp(mse_a), "qm": imp(mse_q), "ols": imp(mse_o)}
print(f"观测改进率: APCNet={obs_imp['apcnet']:+.2f}% QM={obs_imp['qm']:+.2f}% OLS={obs_imp['ols']:+.2f}%")

# 与权威值核对
ref = json.load(open(os.path.join(WORK_DIR, "gpm3h_eval.json")))
assert abs(obs_imp["apcnet"] - ref["mse_improve_APCNet"]) < 0.05, "APCNet 改进率与 gpm3h_eval.json 不一致!"
assert abs(obs_imp["qm"] - ref["mse_improve_QM"]) < 0.05, "QM 改进率不一致!"
assert abs(obs_imp["ols"] - ref["mse_improve_OLS"]) < 0.05, "OLS 改进率不一致!"
print("✓ 与 gpm3h_eval.json 权威值一致")

def block_boot(mse_f, block_len, n_boot=2000, seed=2026):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block_len))
    bid = np.minimum(np.arange(n) // block_len, nb - 1)
    dist = np.empty(n_boot)
    base = mse_g.mean()
    for b in range(n_boot):
        blocks = rng.choice(nb, size=nb, replace=True)
        sel = np.concatenate([np.where(bid == m)[0] for m in blocks])
        mf = mse_f[sel].mean()
        dist[b] = 100.0 * (base - mf) / base
    return dist, nb

out = {"n_samples": n, "n_gridpoints": G.shape[1], "n_bootstrap": 2000,
       "time_range": [str(vt[0]), str(vt[-1])],
       "block_len_hours": 48, "sensitivity_block_lens_hours": [24, 48, 96],
       "obs_improve_pct": obs_imp,
       "protocol": "block bootstrap over consecutive 3-h samples (autocorrelation aware)"}

for bl, key in [(8, "block8"), (16, "block16"), (32, "block32")]:
    print(f"\n=== {bl} 时次块（{bl*3}h） ===")
    for nm, mf in [("apcnet", mse_a), ("qm", mse_q), ("ols", mse_o)]:
        dist, nb = block_boot(mf, bl)
        ci = [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))]
        p_neg = float(np.mean(dist < 0))
        out[f"{key}_{nm}"] = {"n_blocks": nb, "ci95": ci, "p_negative": p_neg, "boot_mean": float(dist.mean())}
        print(f"  {nm:<8} 观测={obs_imp[nm]:+.2f}%  95%CI=[{ci[0]:+.2f},{ci[1]:+.2f}]%  P(负)={p_neg:.3f}")

with open(os.path.join(WORK_DIR, "gpm3h_bootstrap.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False, default=float)
print("\n✅ gpm3h_bootstrap.json 已保存（块长 8/16/32 时次敏感性，主口径 16 时次=48h）")
