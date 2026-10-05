# -*- coding: utf-8 -*-
"""P1-7 ERA5 参照极端阈值 ETS 的时次块 bootstrap 显著性（≥10/≥20 mm/3h）。
模型：GFS / APCNet(S42) / QM / OLS；对 ETS 差值（model - GFS）做 2000 次月块 bootstrap。
"""
import numpy as np, os, json, pickle

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
SYM = os.path.join(WORK, "..", "manuscript_work_sym", "seed42")
with open(os.path.join(WORK, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
tgt = np.load(os.path.join(WORK, "targets_test.npy")).astype(np.float32)
gfs = np.load(os.path.join(WORK, "gfs_test.npy")).astype(np.float32)
models = {
    "GFS": gfs,
    "APCNet": np.load(os.path.join(SYM, "predictions_apcnet.npy")).astype(np.float32),
    "QM": np.load(os.path.join(WORK, "predictions_qm.npy")).astype(np.float32),
    "OLS": np.load(os.path.join(WORK, "predictions_ols.npy")).astype(np.float32),
}
N = len(times)
month_of = np.array([(t.year, t.month) for t in times])
months = sorted(set(map(tuple, month_of)))

def ets_counts(pred, ref, th):
    """逐时次 2x2 计数 (hits, fa, miss, n_obs, n_fcst_yes, n_obs_yes)"""
    p = pred >= th; o = ref >= th
    hits = np.sum(p & o, axis=(1, 2)).astype(np.float64)
    fa = np.sum(p & ~o, axis=(1, 2))
    miss = np.sum(~p & o, axis=(1, 2))
    return hits, fa, miss, o.sum(axis=(1, 2)).astype(np.float64)

def ets_from_counts(H, F, M, O_total):
    denom = H + F + M
    expect = (H + F) * (H + M) / O_total
    val = (H - expect) / (denom - expect + 1e-12)
    return val

rng = np.random.default_rng(2026)
n_boot = 2000
out = {}
for th in [10.0, 20.0]:
    cnt = {k: ets_counts(p, tgt, th) for k, p in models.items()}
    obs = {}
    O_TOTAL = tgt.size
    for k in models:
        H, F, M, O = cnt[k]
        obs[k] = ets_from_counts(H.sum(), F.sum(), M.sum(), O_TOTAL)
    dist_diff = {k: [] for k in models if k != "GFS"}
    for b in range(n_boot):
        sel_blocks = rng.choice(len(months), size=len(months), replace=True)
        idx = np.concatenate([np.where((month_of[:, 0] == months[m][0]) & (month_of[:, 1] == months[m][1]))[0]
                              for m in sel_blocks])
        n_gp = len(idx) * tgt.shape[1] * tgt.shape[2]
        e_g = ets_from_counts(cnt["GFS"][0][idx].sum(), cnt["GFS"][1][idx].sum(),
                              cnt["GFS"][2][idx].sum(), n_gp)
        for k in ["APCNet", "QM", "OLS"]:
            e_m = ets_from_counts(cnt[k][0][idx].sum(), cnt[k][1][idx].sum(),
                                  cnt[k][2][idx].sum(), n_gp)
            dist_diff[k].append(e_m - e_g)
    row = {"threshold_mm": th, "obs_ets": obs,
           "diff_vs_gfs": {k: {"boot_mean": float(np.mean(dist_diff[k])),
                               "ci95": [float(np.percentile(dist_diff[k], 2.5)), float(np.percentile(dist_diff[k], 97.5))],
                               "p_gt0": float(np.mean(np.array(dist_diff[k]) > 0)),
                               "p_lt0": float(np.mean(np.array(dist_diff[k]) < 0))} for k in ["APCNet", "QM", "OLS"]}}
    out[str(th)] = row
    print(f"\n=== >= {th} mm/3h ===")
    for k, v in obs.items():
        print(f"  {k:<8} ETS={v:.3f}")
    for k in ["APCNet", "QM", "OLS"]:
        d = row["diff_vs_gfs"][k]
        print(f"  {k} vs GFS: ΔETS={d['boot_mean']:+.4f} 95%CI=[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] P(>0)={d['p_gt0']:.3f} P(<0)={d['p_lt0']:.3f}")

with open(os.path.join(WORK, "ets_bootstrap_era5.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False, default=float)
print("\n✅ ets_bootstrap_era5.json 已保存")
