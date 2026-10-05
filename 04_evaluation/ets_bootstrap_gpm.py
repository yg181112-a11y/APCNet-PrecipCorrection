# -*- coding: utf-8 -*-
"""P1-7b GPM 参照极端阈值 ETS 的时次块 bootstrap 显著性（≥10/≥20 mm/3h）。
真值 = GPM IMERG 3h（gpm3h_obs.npy，2475 样本）；模型对齐 sel。
"""
import numpy as np, os, json, pickle
from datetime import timezone

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
SYM = os.path.join(WORK, "..", "manuscript_work_sym", "seed42")
with open(os.path.join(WORK, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
obs = np.load(os.path.join(WORK, "gpm3h_obs.npy")).astype(np.float32)
vt = np.array([t for t in times])  # 占位
# 重新构造 valid times：与 gpm3h_obs 相同逻辑
from datetime import timedelta
utc = timezone.utc
WINDOW_END_HOURS = [3, 9, 15, 21]
GPM_ROOT = r"D:\liaohe\GPM_IMERG"
def gpm_3h_window(dt):
    files = []
    for i in range(6):
        e = dt - timedelta(minutes=30 * (5 - i))
        files.append(os.path.join(GPM_ROOT, f"imerg_{e.year}{e.month:02d}",
                                  f"imerg_{e.year}{e.month:02d}{e.day:02d}_{e.hour:02d}{e.minute:02d}00.nc4"))
    return files
vtimes = []
for i, t in enumerate(times):
    t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    if t.hour not in WINDOW_END_HOURS:
        continue
    if all(os.path.exists(f) for f in gpm_3h_window(t)):
        vtimes.append(t)
vtimes = np.array(vtimes)
assert len(vtimes) == len(obs), f"{len(vtimes)} vs {len(obs)}"

gfs = np.load(os.path.join(WORK, "gfs_test.npy")).astype(np.float32)
tgt = np.load(os.path.join(WORK, "targets_test.npy")).astype(np.float32)
apc = np.load(os.path.join(SYM, "predictions_apcnet.npy")).astype(np.float32)
qm = np.load(os.path.join(WORK, "predictions_qm.npy")).astype(np.float32)
ols = np.load(os.path.join(WORK, "predictions_ols.npy")).astype(np.float32)
bincm = np.load(os.path.join(WORK, "bin_cm_pred.npy")).astype(np.float32)

idx = {t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc): i for i, t in enumerate(times)}
sel = np.array([idx[t] for t in vtimes])
models = {"GFS": gfs[sel], "APCNet": apc[sel], "QM": qm[sel], "OLS": ols[sel], "BinCM": bincm[sel]}
N = len(sel)
month_of = np.array([(t.year, t.month) for t in vtimes])
months = sorted(set(map(tuple, month_of)))

def ets_counts(pred, ref, th):
    p = pred >= th; o = ref >= th
    return (np.sum(p & o, axis=(1, 2)).astype(np.float64), np.sum(p & ~o, axis=(1, 2)).astype(np.float64),
            np.sum(~p & o, axis=(1, 2)).astype(np.float64))

def ets_from_counts(H, F, M, O_total):
    expect = (H + F) * (H + M) / O_total
    return (H - expect) / (H + F + M - expect + 1e-12)

rng = np.random.default_rng(2026)
n_boot = 2000
out = {}
for th in [10.0, 20.0]:
    cnt = {k: ets_counts(p, obs, th) for k, p in models.items()}
    O_TOTAL = obs.size
    obsv = {k: ets_from_counts(cnt[k][0].sum(), cnt[k][1].sum(), cnt[k][2].sum(), O_TOTAL) for k in models}
    dist_diff = {k: [] for k in ["APCNet", "QM", "OLS", "BinCM"]}
    for b in range(n_boot):
        sel_blocks = rng.choice(len(months), size=len(months), replace=True)
        ii = np.concatenate([np.where((month_of[:, 0] == months[m][0]) & (month_of[:, 1] == months[m][1]))[0]
                             for m in sel_blocks])
        ngp = len(ii) * obs.shape[1] * obs.shape[2]
        e_g = ets_from_counts(cnt["GFS"][0][ii].sum(), cnt["GFS"][1][ii].sum(), cnt["GFS"][2][ii].sum(), ngp)
        for k in ["APCNet", "QM", "OLS", "BinCM"]:
            e_m = ets_from_counts(cnt[k][0][ii].sum(), cnt[k][1][ii].sum(), cnt[k][2][ii].sum(), ngp)
            dist_diff[k].append(e_m - e_g)
    row = {"threshold_mm": th, "obs_ets": {k: float(v) for k, v in obsv.items()},
           "diff_vs_gfs": {k: {"boot_mean": float(np.mean(dist_diff[k])),
                               "ci95": [float(np.percentile(dist_diff[k], 2.5)), float(np.percentile(dist_diff[k], 97.5))],
                               "p_gt0": float(np.mean(np.array(dist_diff[k]) > 0)),
                               "p_lt0": float(np.mean(np.array(dist_diff[k]) < 0))} for k in ["APCNet", "QM", "OLS", "BinCM"]}}
    out[str(th)] = row
    print(f"\n=== GPM >= {th} mm/3h ===")
    for k, v in obsv.items():
        print(f"  {k:<8} ETS={v:.3f}")
    for k in ["APCNet", "QM", "OLS", "BinCM"]:
        d = row["diff_vs_gfs"][k]
        print(f"  {k} vs GFS: ΔETS={d['boot_mean']:+.4f} 95%CI=[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] P(>0)={d['p_gt0']:.3f}")

with open(os.path.join(WORK, "ets_bootstrap_gpm.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False, default=float)
print("\n✅ ets_bootstrap_gpm.json 已保存")
