# -*- coding: utf-8 -*-
"""verify_gpm_lead.py — GPM IMERG 24h 累积独立验证（多时效 GPM 观测，R3 稿 §3j 补充）。
口径：
  - 样本 = 24h 实验测试期（split==2）的 init 00Z 时刻，窗口 [init, init+24h]
  - GPM IMERG V07B 半小时降水率，48 个 bin 累加 -> 24h 累积 mm（与 GFS f024 / ERA5 24h 同窗）
  - 双线性插值 0.1° -> 主网格 0.25°（25x37）+ 覆盖掩码
  - 方法：GFS / APCNet / U-Net / QM / BM(BinCM) / OLS / ERA5(参照)，指标 vs GPM
  - QM/BM/OLS 用训练期标定（与 train_24h.py 同实现）
输出：gpm_24h_eval.json
"""
import os, sys, glob, json, pickle, argparse
import numpy as np
from datetime import datetime, timedelta

GPM_ROOT = r"D:\liaohe\GPM_IMERG"
BASE = r"C:\Users\yg181\Desktop\论文三\13.0修复重跑"
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)   # 降序
GLOBAL_LONS = np.linspace(117.0, 126.0, 37) # 升序
BIN_H = 0.5


def gpm_file_for(dt):
    d = dt.date()
    ym = d.strftime("%Y%m")
    ymd_hms = d.strftime("%Y%m%d") + "_" + dt.strftime("%H%M%S")
    return os.path.join(GPM_ROOT, f"imerg_{ym}", f"imerg_{ymd_hms}.nc4")


def read_gpm_rate(path):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    g = ds.groups["Grid"]
    lat = g.variables["lat"][:].astype(np.float64)
    lon = g.variables["lon"][:].astype(np.float64)
    rate = g.variables["precipitation"][0, :, :].astype(np.float64)
    ds.close()
    return rate, lon, lat


def gpm_accum_lead(t0, cache, lead_h):
    """t0 = init 时刻 (UTC)。返回 [t0, t0+lead_h] 的累积场。任一 bin 缺失 -> None。"""
    n_bins = int(lead_h * 2)
    bins = []
    for k in range(n_bins):
        bt = t0 + timedelta(minutes=int(30 * k))
        key = bt.strftime("%Y%m%d_%H%M%S")
        if key in cache:
            r = cache[key]
            if r is None:
                return None, None, None
        else:
            p = gpm_file_for(bt)
            if not os.path.exists(p):
                cache[key] = None
                return None, None, None
            try:
                r, lon, lat = read_gpm_rate(p)
            except Exception:
                cache[key] = None
                return None, None, None
            cache[key] = (r, lon, lat)
        bins.append(cache[key])
    lon, lat = bins[0][1], bins[0][2]
    acc = np.zeros_like(bins[0][0])
    for r, _, _ in bins:
        acc += r * BIN_H
    return acc, lon, lat


def regrid_to_main(acc_lonlat, lon, lat):
    from scipy.interpolate import RegularGridInterpolator
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lon2d.ravel(), lat2d.ravel()], axis=-1)
    interp = RegularGridInterpolator((lon, lat), acc_lonlat, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    mask = ~np.isnan(vals)
    vals = np.where(mask, vals, 0.0)
    return vals, mask


def cont_metrics(obs, fcst):
    obs_f, fcst_f = obs.flatten(), fcst.flatten()
    valid = np.isfinite(obs_f) & np.isfinite(fcst_f)
    obs_f, fcst_f = obs_f[valid], fcst_f[valid]
    n = obs_f.size
    if n == 0:
        return None
    mse = np.mean((obs_f - fcst_f) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(obs_f - fcst_f))
    cc = np.corrcoef(obs_f, fcst_f)[0, 1] if n > 1 else 0.0
    bias = float(np.mean(fcst_f - obs_f))
    return {"n": int(n), "MSE": float(mse), "RMSE": float(rmse),
            "MAE": float(mae), "CC": float(cc), "bias": bias}


# ---- 基线（与 train_24h.py 同实现）----
def fit_qm(g, e, nb=200):
    qa = np.linspace(0.001, 0.999, nb)
    gq = np.zeros((25, 37, nb)); eq = np.zeros((25, 37, nb))
    for i in range(25):
        for j in range(37):
            gq[i, j] = np.nanquantile(g[:, i, j], qa)
            eq[i, j] = np.nanquantile(e[:, i, j], qa)
    return gq, eq


def apply_qm(g, gq, eq):
    out = np.zeros_like(g)
    for i in range(25):
        for j in range(37):
            out[:, i, j] = np.interp(g[:, i, j], gq[i, j], eq[i, j],
                                     left=eq[i, j, 0], right=eq[i, j, -1])
    return out


def fit_bm(g, e, nb=20):
    """BinCM：按 GFS 值分箱 -> 训练期 ERA5 条件均值 / GFS 箱均值 比率。"""
    gmin, gmax = np.nanmin(g), np.nanmax(g)
    edges = np.linspace(gmin, gmax, nb + 1)
    ratio = np.zeros(nb)
    for b in range(nb):
        m = (g >= edges[b]) & (g < edges[b + 1]) & np.isfinite(g) & np.isfinite(e)
        gm = g[m].mean() if m.sum() else np.nan
        em = e[m].mean() if m.sum() else np.nan
        ratio[b] = em / gm if gm and gm > 0 else 1.0
    return edges, ratio


def apply_bm(g, edges, ratio):
    out = g.copy()
    idx = np.clip(np.searchsorted(edges, g, side="right") - 1, 0, len(ratio) - 1)
    out = g * ratio[idx]
    return out


def fit_ols(g, e):
    a = np.zeros((25, 37)); b = np.zeros((25, 37))
    for i in range(25):
        for j in range(37):
            X = g[:, i, j]; Y = e[:, i, j]
            m = np.isfinite(X) & np.isfinite(Y)
            if m.sum() > 5:
                aa, bb = np.polyfit(X[m], Y[m], 1)
                a[i, j], b[i, j] = aa, bb
            else:
                a[i, j], b[i, j] = 1.0, 0.0
    return a, b


def apply_ols(g, a, b):
    return g * a[None, :, :] + b[None, :, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=24, choices=[24, 72, 120])
    ap.add_argument("--exp", default=None, help="实验目录（默认 BASE\\{lead}h_exp 或 multi_lead_exp\\{lead}h）")
    args = ap.parse_args()
    global LEAD_H
    LEAD_H = args.lead
    if args.exp:
        EXP = args.exp
    else:
        cand = os.path.join(BASE, f"{LEAD_H}h_exp")
        if not os.path.isdir(cand):
            cand = os.path.join(BASE, "multi_lead_exp", f"{LEAD_H}h")
        EXP = cand
    OUT = EXP
    print(f"lead={LEAD_H}h, EXP={EXP}", flush=True)

    init = np.load(os.path.join(EXP, "init_times.npy"), allow_pickle=True)
    mask = np.load(os.path.join(EXP, "split_mask.npy"))
    gfs = np.load(os.path.join(EXP, f"gfs_{LEAD_H}h_accum.npy"))
    era5 = np.load(os.path.join(EXP, f"era5_{LEAD_H}h_accum.npy"))
    apc = np.load(os.path.join(EXP, f"pred_apcnet_{LEAD_H}h.npy"))
    unet = np.load(os.path.join(EXP, f"unet_{LEAD_H}h.npy"))

    # 时间对象
    times = [datetime(1970, 1, 1) + timedelta(hours=int(t)) for t in init.astype("datetime64[h]").astype(np.int64)]
    times = [t.replace(tzinfo=None) for t in times]

    # 选测试期 00Z 且窗口在 GPM 覆盖内 (init+lead <= 2025-09-30)
    sel = []
    for i in range(len(times)):
        if int(mask[i]) != 2:
            continue
        if times[i].hour != 0:
            continue
        if times[i] < datetime(2024, 1, 1):
            continue
        if times[i] + timedelta(hours=LEAD_H) > datetime(2025, 9, 30, 23, 59):
            continue
        sel.append(i)
    n_sel = len(sel)
    print(f"候选样本 (test 00Z, GPM 窗内, lead={LEAD_H}h): {n_sel}", flush=True)

    # 逐样本 GPM 累积
    gpm_all = np.zeros((n_sel, 25, 37))
    mask_all = np.zeros((n_sel, 25, 37), dtype=bool)
    cache = {}
    ok_i = []
    for r, i in enumerate(sel):
        t0 = times[i]
        acc, lon, lat = gpm_accum_lead(t0, cache, LEAD_H)
        if acc is None:
            continue
        field, msk = regrid_to_main(acc, lon, lat)
        if not msk.any():
            continue
        gpm_all[r] = field
        mask_all[r] = msk
        ok_i.append(r)
        if (len(ok_i) % 100) == 0:
            print(f"  进度 {len(ok_i)}/{n_sel}", flush=True)

    ok_rows = np.array(ok_i)
    n_ok = len(ok_rows)
    print(f"GPM 有效样本: {n_ok}/{n_sel}", flush=True)
    if n_ok < 50:
        print("!! 有效样本过少"); sys.exit(1)

    # 共同覆盖掩码
    cov = mask_all[ok_rows].all(axis=0)
    n_cov = int(cov.sum())
    print(f"共同覆盖格点: {n_cov}/925", flush=True)

    def restrict(a, rows):
        return a[rows][:, cov]

    sel_global = np.array(sel)[ok_rows]
    gpm_v = restrict(gpm_all, ok_rows)
    gfs_v = restrict(gfs, sel_global)
    era5_v = restrict(era5, sel_global)
    # APCNet/U-Net 数组 = 测试期有效样本 (2916 行, 对应全局索引 valid_te_global)
    te_all = np.where(mask == 2)[0]
    te_nan = np.isnan(gfs[te_all]).any(axis=(1, 2))
    valid_te_global = te_all[~te_nan]
    pos = np.searchsorted(valid_te_global, sel_global)
    assert np.all(pos < apc.shape[0]), "APCNet 索引越界"
    apc_v = apc[pos][:, cov]
    unet_v = unet[pos][:, cov]

    # QM/BM/OLS 标定（训练期 2019-2021，剔 NaN 行）
    tr_all = np.where(mask == 0)[0]
    tr_nan = np.isnan(gfs[tr_all]).any(axis=(1, 2)) | np.isnan(era5[tr_all]).any(axis=(1, 2))
    tr = tr_all[~tr_nan]
    gfs_tr = gfs[tr]; era5_tr = era5[tr]
    print("标定 QM/BM/OLS (训练期样本 %d)..." % len(tr), flush=True)
    gq, eq = fit_qm(gfs_tr, era5_tr)
    pred_qm = apply_qm(gfs[sel_global], gq, eq)
    edges, ratio = fit_bm(gfs_tr, era5_tr)
    pred_bm = apply_bm(gfs[sel_global], edges, ratio)
    ols_a, ols_b = fit_ols(gfs_tr, era5_tr)
    pred_ols = apply_ols(gfs[sel_global], ols_a, ols_b)

    qm_v = restrict(pred_qm, np.arange(len(sel_global)))
    bm_v = restrict(pred_bm, np.arange(len(sel_global)))
    ols_v = restrict(pred_ols, np.arange(len(sel_global)))

    # 指标
    result = {
        "protocol": f"GPM IMERG V07B {LEAD_H}h accumulation validation (init 00Z, window [init, init+{LEAD_H}h])",
        "n_samples_ok": int(n_ok), "n_samples_total": int(n_sel),
        "coverage_gridpoints": int(n_cov),
        "time_range": [str(times[sel[ok_rows[0]]]), str(times[sel[ok_rows[-1]]])],
        "unit": f"mm/{LEAD_H}h",
        "aggregation": f"{int(LEAD_H*2)} half-hourly rates * 0.5h",
    }
    print("\n[连续指标] 真值 = GPM 24h")
    for name, arr in [("GFS", gfs_v), ("APCNet", apc_v), ("U-Net", unet_v),
                      ("QM", qm_v), ("BinCM", bm_v), ("OLS", ols_v), ("ERA5", era5_v)]:
        m = cont_metrics(gpm_v, arr)
        result[f"cont_{name}"] = m
        imp = 100.0 * (cont_metrics(gpm_v, gfs_v)["MSE"] - m["MSE"]) / cont_metrics(gpm_v, gfs_v)["MSE"]
        result[f"mse_improve_pct_{name}"] = float(imp)
        print(f"  {name:7s} RMSE={m['RMSE']:.3f} CC={m['CC']:.3f} bias={m['bias']:+.2f} MSE改进={imp:+.1f}%", flush=True)

    # 季节分层（暖季 JJAS vs 冷季）
    month_arr = np.array([times[sel[i]].month for i in ok_rows])
    result["seasonal"] = {}
    for season, selm in [("warm_JJAS", (month_arr >= 6) & (month_arr <= 9)),
                         ("cool_others", (month_arr < 6) | (month_arr > 9))]:
        if int(selm.sum()) < 20:
            continue
        row = {"n_samples": int(selm.sum()), "months": sorted(set(month_arr[selm].tolist()))}
        m_g = cont_metrics(gpm_v[selm], gfs_v[selm])
        row["cont_GFS"] = m_g
        for name, arr in [("APCNet", apc_v), ("U-Net", unet_v), ("QM", qm_v),
                          ("BinCM", bm_v), ("OLS", ols_v), ("ERA5", era5_v)]:
            m_x = cont_metrics(gpm_v[selm], arr[selm])
            if m_x is None:
                continue
            row[f"cont_{name}"] = m_x
            row[f"mse_improve_pct_{name}"] = 100.0 * (m_g["MSE"] - m_x["MSE"]) / m_g["MSE"]
        result["seasonal"][season] = row
        print(f"  {season} (n={row['n_samples']}): GFS RMSE={m_g['RMSE']:.3f} | "
              + " | ".join(f"{k.split('_pct_')[-1]} {v:+.1f}%" for k, v in row.items() if k.startswith('mse_improve_pct_')), flush=True)

    out_path = os.path.join(OUT, f"gpm_{LEAD_H}h_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✅ 已保存: {out_path}")


if __name__ == "__main__":
    main()
