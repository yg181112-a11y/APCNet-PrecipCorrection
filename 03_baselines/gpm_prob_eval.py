# -*- coding: utf-8 -*-
"""P3 提升项：GPM IMERG 观测尺度概率后处理验证。
与 prob_v3_ext 同口径：QM/APCNet dressing 参数在 ERA5 参考标定（校准段 = 测试期前一半），
评估时真值换成 GPM 3h 观测（gpm3h_obs.npy，2475 样本）。
回答审稿人必问：概率后处理在观测尺度是否同样失败。
"""
import numpy as np, os, json, pickle
from datetime import timedelta, timezone
from scipy.stats import norm

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
SYM = os.path.join(WORK, "..", "manuscript_work_sym", "seed42")
GPM_ROOT = r"D:\liaohe\GPM_IMERG"
WINDOW_END_HOURS = [3, 9, 15, 21]
import sys
sys.path.insert(0, WORK)
from qm_baseline import build_qm, apply_qm

def gaussian_crps(mu, sigma, y):
    sigma = np.maximum(sigma, 1e-4)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))

def zi_crps(mu, sigma, p0, y):
    return p0 * np.abs(y) + (1 - p0) * gaussian_crps(mu, sigma, y)

# ---------- 数据与对齐 ----------
with open(os.path.join(WORK, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
obs = np.load(os.path.join(WORK, "gpm3h_obs.npy")).astype(np.float32)
utc = timezone.utc

def gpm_3h_window(dt):
    files = []
    for i in range(6):
        e = dt - timedelta(minutes=30 * (5 - i))
        files.append(os.path.join(GPM_ROOT, f"imerg_{e.year}{e.month:02d}",
                                  f"imerg_{e.year}{e.month:02d}{e.day:02d}_{e.hour:02d}{e.minute:02d}00.nc4"))
    return files

vtimes = []
for t in times:
    t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    if t.hour not in WINDOW_END_HOURS:
        continue
    if all(os.path.exists(f) for f in gpm_3h_window(t)):
        vtimes.append(t)
vtimes = np.array(vtimes)
assert len(vtimes) == len(obs), f"{len(vtimes)} vs {len(obs)}"

idx = {t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc): i for i, t in enumerate(times)}
sel = np.array([idx[t] for t in vtimes])
print("GPM 对齐样本:", len(sel))

gfs = np.load(os.path.join(WORK, "gfs_test.npy")).astype(np.float32)[sel]
apc = np.load(os.path.join(SYM, "predictions_apcnet.npy")).astype(np.float32)[sel]
tr_g = np.load(os.path.join(WORK, "train_gfs.npy")).astype(np.float32)
tr_e = np.load(os.path.join(WORK, "train_era5.npy")).astype(np.float32)

# ---------- QM（训练期标定，盲测到 GPM 样本） ----------
qm_gfs, qm_era5, p_dry_g, p_dry_e, qs = build_qm(tr_g, tr_e)
qm_tr = apply_qm(tr_g, qm_gfs, qm_era5, p_dry_g, p_dry_e, qs)
sigma_qm = np.maximum((tr_e - qm_tr).std(axis=0), 1e-4)
qm_g = apply_qm(gfs, qm_gfs, qm_era5, p_dry_g, p_dry_e, qs)

# ---------- APCNet dressing 参数（ERA5 参考校准段标定，同 prob_v3_ext） ----------
te_e = np.load(os.path.join(WORK, "targets_test.npy")).astype(np.float32)
apc_full = np.load(os.path.join(SYM, "predictions_apcnet.npy")).astype(np.float32)
split = len(te_e) // 2
apc_cal = apc_full[:split]; tgt_cal = te_e[:split]
dry_freq = np.mean(tgt_cal < 0.1, axis=0)
wm = tgt_cal >= 0.1
resid = np.where(wm, apc_cal - tgt_cal, np.nan)
sigma_apc = np.maximum(np.nanstd(resid, axis=0), 0.05)

sigma_qm_b = np.broadcast_to(sigma_qm, apc.shape)
sigma_apc_b = np.broadcast_to(sigma_apc, apc.shape)
dry_b = np.broadcast_to(dry_freq, apc.shape)
scale = 1.0
p0_adapt = np.clip(dry_freq[np.newaxis] * np.exp(-apc / scale), 0.01, 0.99)

# ---------- 评估（真值 = GPM） ----------
res = {}
res['n_eval'] = int(len(sel))
res['protocol'] = 'dressing params calibrated on ERA5 reference (cal half); observations = GPM IMERG 3h'
res['crps_det_gfs'] = float(np.mean(np.abs(gfs - obs)))
res['crps_det_qm'] = float(np.mean(np.abs(qm_g - obs)))
res['crps_det_apcnet'] = float(np.mean(np.abs(apc - obs)))
res['crps_qm_dress'] = float(np.mean(gaussian_crps(qm_g, sigma_qm_b, obs)))
res['crps_zig_const'] = float(np.mean(zi_crps(apc, sigma_apc_b, dry_b, obs)))
res['crps_zig_adaptive'] = float(np.mean(zi_crps(apc, sigma_apc_b, p0_adapt, obs)))
print('CRPS: GFS=%.4f QM=%.4f APCNet=%.4f | QM-dress=%.4f ZIG-const=%.4f ZIG-adapt=%.4f' % (
    res['crps_det_gfs'], res['crps_det_qm'], res['crps_det_apcnet'],
    res['crps_qm_dress'], res['crps_zig_const'], res['crps_zig_adaptive']))

for tname, tval in [('occurrence_0.1mm', 0.1), ('10mm', 10.0), ('20mm', 20.0)]:
    z_t = (tval - apc) / sigma_apc_b
    p_ext = (1 - p0_adapt) * (1 - norm.cdf(z_t))
    o_ext = (obs >= tval).astype(float)
    ne = int(o_ext.sum())
    clim = o_ext.mean()
    b = np.mean((p_ext - o_ext) ** 2)
    bss = 1 - b / np.mean((clim - o_ext) ** 2) if ne > 0 else None
    res[f'brier_{tname}'] = float(b)
    res[f'bss_{tname}'] = float(bss) if bss is not None else None
    res[f'n_{tname}'] = ne
    print('%s (n=%d): Brier=%.4f BSS=%+.3f' % (tname, ne, b, bss if bss is not None else -9))

# reliability（occurrence，10 档）
o = (obs >= 0.1).astype(float)
p_wet = (1 - p0_adapt) * (1 - norm.cdf((0.1 - apc) / sigma_apc_b))
nb = 10
edges = np.linspace(0, 1, nb + 1)
rel = []
for i in range(nb):
    m = (p_wet >= edges[i]) & (p_wet < edges[i + 1])
    if m.sum() > 100:
        rel.append({'bin_low': float(edges[i]), 'bin_high': float(edges[i + 1]),
                    'avg_prob': float(p_wet[m].mean()), 'avg_obs': float(o[m].mean()),
                    'count': int(m.sum())})
res['reliability'] = rel
print('=== reliability (GPM) ===')
for r in rel:
    print('  [%.2f,%.2f) prob=%.3f obs=%.3f n=%d' % (r['bin_low'], r['bin_high'], r['avg_prob'], r['avg_obs'], r['count']))

res['sharpness_p_wet_std'] = float(np.std(p_wet))
res['mean_spread_apc_sigma'] = float(np.mean(sigma_apc))
res['mean_spread_qm_sigma'] = float(np.mean(sigma_qm))
print('sharpness std=%.4f spread_apc=%.4f spread_qm=%.4f' % (
    res['sharpness_p_wet_std'], res['mean_spread_apc_sigma'], res['mean_spread_qm_sigma']))

with open(os.path.join(WORK, 'gpm_prob_eval.json'), 'w') as f:
    json.dump(res, f, indent=1, default=str)
print('✅ gpm_prob_eval.json saved')
