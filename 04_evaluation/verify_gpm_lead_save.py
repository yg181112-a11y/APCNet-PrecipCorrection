# -*- coding: utf-8 -*-
"""verify_gpm_lead_save.py — GPM IMERG 多时效独立验证：保存 per-sample 数组（供 bootstrap CI 补算）
与 verify_gpm_lead.py 同口径，但把 GPM 累积场/掩码/预测逐样本落盘为 npz，
供 obs_lead_bootstrap.py 做月块 bootstrap（CHM/GPM 24-72-120h 观测验证 CI）。
三个 lead 一次运行，共享 GPM 文件缓存（全 2024-01~2025-09 半小时 bin）。
"""
import os, sys, json, argparse
from datetime import datetime, timedelta
import numpy as np

GPM_ROOT = r"D:\liaohe\GPM_IMERG"
BASE = r"C:\Users\yg181\Desktop\论文三\13.0修复重跑"
OUTD = os.path.join(BASE, "gpm_lead_arrays")
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
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


_read_fail = {"n": 0, "first": None}


def gpm_accum_lead(t0, cache, lead_h):
    """返回 (acc, lon, lat) 或 (None, None, None)。"""
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
            except Exception as e:
                _read_fail["n"] += 1
                if _read_fail["first"] is None:
                    _read_fail["first"] = f"{p}: {e!r}"
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
    # GPM 文件布局为 (lon, lat)：points=(lon, lat)，查询点须按 (lon, lat) 列序
    from scipy.interpolate import RegularGridInterpolator
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lon2d.ravel(), lat2d.ravel()], axis=-1)
    interp = RegularGridInterpolator((lon, lat), acc_lonlat, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    mask = ~np.isnan(vals)
    vals = np.where(mask, vals, 0.0)
    return vals, mask


def load_lead_arrays(fhr):
    if fhr == 24:
        d = os.path.join(BASE, "24h_exp")
    else:
        d = os.path.join(BASE, "multi_lead_exp", f"{fhr}h")
    init = np.load(os.path.join(d, "init_times.npy"), allow_pickle=True)
    mask = np.load(os.path.join(d, "split_mask.npy"))
    gfs = np.load(os.path.join(d, f"gfs_{fhr}h_accum.npy"))
    era5 = np.load(os.path.join(d, f"era5_{fhr}h_accum.npy"))
    apc = np.load(os.path.join(d, f"pred_apcnet_{fhr}h.npy"))
    unet = np.load(os.path.join(d, f"unet_{fhr}h.npy"))
    times = [datetime(1970, 1, 1) + timedelta(hours=int(t)) for t in
             init.astype("datetime64[h]").astype(np.int64)]
    sel = []
    for i in range(len(times)):
        if int(mask[i]) != 2:
            continue
        if times[i].hour != 0:
            continue
        if times[i] < datetime(2024, 1, 1):
            continue
        if times[i] + timedelta(hours=fhr) > datetime(2025, 9, 30, 23, 59):
            continue
        sel.append(i)
    sel_global = np.array(sel)
    te_all = np.where(mask == 2)[0]
    te_nan = np.isnan(gfs[te_all]).any(axis=(1, 2))
    valid_te_global = te_all[~te_nan]
    pos = np.searchsorted(valid_te_global, sel_global)
    assert np.all(pos < apc.shape[0]), "APCNet 索引越界"
    return times, sel_global, gfs, era5, apc, unet, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", default="24,72,120")
    args = ap.parse_args()
    leads = [int(x) for x in args.leads.split(",")]
    os.makedirs(OUTD, exist_ok=True)
    cache = {}

    meta = {}
    for fhr in leads:
        times, sel_global, gfs, era5, apc, unet, pos = load_lead_arrays(fhr)
        meta[fhr] = (times, sel_global, gfs, era5, apc, unet, pos)
        print(f"lead={fhr}h: 候选样本 {len(sel_global)}", flush=True)

    for fhr in leads:
        times, sel_global, gfs, era5, apc, unet, pos = meta[fhr]
        n_sel = len(sel_global)
        gpm_all = np.zeros((n_sel, 25, 37))
        mask_all = np.zeros((n_sel, 25, 37), dtype=bool)
        ok_i = []
        for r, gi in enumerate(sel_global):
            t0 = times[gi]
            acc, lon, lat = gpm_accum_lead(t0, cache, fhr)
            if acc is None:
                continue
            field, msk = regrid_to_main(acc, lon, lat)
            if not msk.any():
                continue
            gpm_all[r] = field
            mask_all[r] = msk
            ok_i.append(r)
            if (len(ok_i) % 100) == 0:
                print(f"  lead={fhr}h 进度 {len(ok_i)}/{n_sel}", flush=True)
        ok_rows = np.array(ok_i)
        n_ok = len(ok_rows)
        print(f"lead={fhr}h: GPM 有效样本 {n_ok}/{n_sel}", flush=True)
        if n_ok < 50:
            print(f"!! 有效样本过少 n={n_ok}; read_fail={_read_fail}"); sys.exit(1)
        cov = mask_all[ok_rows].all(axis=0)
        n_cov = int(cov.sum())
        print(f"lead={fhr}h: 共同覆盖格点 {n_cov}/925", flush=True)

        sel_ok = np.array(sel_global)[ok_rows]
        gpm_v = gpm_all[ok_rows][:, cov]
        gfs_v = gfs[sel_ok][:, cov]
        era5_v = era5[sel_ok][:, cov]
        apc_v = apc[pos[ok_rows]][:, cov]
        unet_v = unet[pos[ok_rows]][:, cov]
        months = np.array([(times[i].year, times[i].month) for i in sel_ok])
        dates = [times[i].strftime("%Y-%m-%d") for i in sel_ok]

        out = os.path.join(OUTD, f"gpm_{fhr}h.npz")
        np.savez(out, gpm=gpm_v, gfs=gfs_v, era5=era5_v, apc=apc_v, unet=unet_v,
                 months=months, dates=np.array(dates), cov_mask=cov)
        print(f"✅ saved {out}  n={n_ok} cov={n_cov} months={len(set(map(tuple, months)))}", flush=True)

    print("done")


if __name__ == "__main__":
    main()
