# -*- coding: utf-8 -*-
"""测试期极端事件逐样本分析（GPM 3h，2475 样本）：
- 筛选观测 >=10 / >=20 mm/3h 样本（476 / 225）
- 每样本：观测域均/峰值 vs 各方法域均/峰值/样本RMSE
- 汇总：极端样本集上的平均RMSE改进、率、峰值追踪
- 逐事件表（Top 15 观测域均）
输出：extreme_events.json
"""
import pickle, numpy as np, os, json
from datetime import datetime, timedelta, timezone

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
SYM42 = os.path.join(WORK, '..', 'manuscript_work_sym', 'seed42', 'predictions_apcnet.npy')
GPM_ROOT = r"D:\liaohe\GPM_IMERG"
WINDOW_END_HOURS = [3, 9, 15, 21]
utc = timezone.utc

with open(os.path.join(WORK, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
gfs = np.load(os.path.join(WORK, "gfs_test.npy")).astype(np.float32)
apc = np.load(SYM42).astype(np.float32)
qm = np.load(os.path.join(WORK, "predictions_qm.npy")).astype(np.float32)
ols = np.load(os.path.join(WORK, "predictions_ols.npy")).astype(np.float32)
unet = np.load(os.path.join(WORK, "predictions_unet.npy")).astype(np.float32)
gpm3 = np.load(os.path.join(WORK, "gpm3h_obs.npy")).astype(np.float32)  # 2475
print("shapes:", gpm3.shape, gfs.shape, apc.shape, flush=True)
assert gpm3.shape[0] <= gfs.shape[0], "GPM samples must be subset of GFS samples"
N = gpm3.shape[0]

# 重建 GPM 对齐索引（文件存在性，与 gpm3h_eval.py 同口径）
def gpm_files_ok(dt):
    ok = True
    for i in range(6):
        e = dt - timedelta(minutes=30 * (5 - i))
        if not os.path.exists(os.path.join(GPM_ROOT, f"imerg_{e.year}{e.month:02d}",
                                           f"imerg_{e.year}{e.month:02d}{e.day:02d}_{e.hour:02d}{e.minute:02d}00.nc4")):
            ok = False
    return ok

sel = []
for i, t in enumerate(times):
    t = t.replace(tzinfo=utc) if t.tzinfo is None else t.astimezone(utc)
    if t.hour in WINDOW_END_HOURS and gpm_files_ok(t):
        sel.append(i)
sel = np.array(sel)
print("GPM-aligned model indices:", len(sel), "expect 2475", flush=True)
gfs = gfs[sel]; apc = apc[sel]; qm = qm[sel]; ols = ols[sel]; unet = unet[sel]
assert len(sel) == N

def per_sample_rmse(o, f):
    d = (o - f) ** 2
    return np.sqrt(d.mean(axis=(1, 2)))  # [N]

def domain_rate(f):
    return f.mean(axis=(1, 2))  # [N] mm/3h

def peak(f):
    return f.max(axis=(1, 2))  # [N]

out = {}
for th in (10, 20):
    sel = gpm3.mean(axis=(1, 2)) >= th  # 域均有 >=th ? 不，应按格点。极端样本定义：至少一个格点 >= th
    sel_gp = (gpm3.max(axis=(1, 2)) >= th)
    sel_dm = sel
    n_gp = int(sel_gp.sum()); n_dm = int(sel_dm.sum())
    print(f"\n=== >= {th} mm/3h: 格点级样本 {n_gp}, 域均级样本 {n_dm} ===", flush=True)
    # 格点级极端样本
    s = sel_gp
    o = gpm3[s]; 
    meth = {"GFS": gfs[s], "QM": qm[s], "OLS": ols[s], "APCNet": apc[s], "U-Net": unet[s]}
    rec = {"n_samples": int(s.sum()),
           "obs_domain_mean": float(o.mean(axis=(1,2)).mean()),
           "obs_peak_mean": float(o.max(axis=(1,2)).mean()),
           "methods": {}}
    for nm, P in meth.items():
        rec["methods"][nm] = {
            "sample_rmse": float(per_sample_rmse(o, P).mean()),
            "rmse_delta_pct": float(100 * (per_sample_rmse(o, gfs[s]).mean() - per_sample_rmse(o, P).mean()) / per_sample_rmse(o, gfs[s]).mean()),
            "domain_mean": float(domain_rate(P).mean()),
            "peak_mean": float(peak(P).mean()),
            "peak_bias": float(peak(P).mean() - float(o.max(axis=(1,2)).mean()))}
        print(f"  {nm:8s} sampRMSE={rec['methods'][nm]['sample_rmse']:.4f} dRMSE={rec['methods'][nm]['rmse_delta_pct']:+.1f}%  rate={rec['methods'][nm]['domain_mean']:.4f}  peak={rec['methods'][nm]['peak_mean']:.3f} (bias {rec['methods'][nm]['peak_bias']:+.3f})", flush=True)
    out[str(th)] = rec

# 逐事件表：观测域均最大的 15 个样本（不限阈值）
order = np.argsort(-gpm3.mean(axis=(1, 2)))
rows = []
for k in order[:15]:
    i = int(k)
    t = times[i]
    o = gpm3[i]; dm_o = o.mean(); pk_o = o.max()
    rows.append({"time": str(t), "obs_dm": float(dm_o), "obs_peak": float(pk_o),
                 "gfs_dm": float(gfs[i].mean()), "gfs_peak": float(gfs[i].max()),
                 "qm_dm": float(qm[i].mean()), "apc_dm": float(apc[i].mean()),
                 "apc_peak": float(apc[i].max()), "ols_dm": float(ols[i].mean()),
                 "unet_dm": float(unet[i].mean())})
out["top15_events"] = rows
for r in rows[:8]:
    print(f"\n  {r['time']} obs dm={r['obs_dm']:.3f} pk={r['obs_peak']:.2f} | GFS {r['gfs_dm']:.3f}/{r['gfs_peak']:.2f} APC {r['apc_dm']:.3f}/{r['apc_peak']:.2f} QM {r['qm_dm']:.3f}", flush=True)

with open(os.path.join(WORK, "extreme_events.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False, default=float)
print("\n✅ extreme_events.json saved")
