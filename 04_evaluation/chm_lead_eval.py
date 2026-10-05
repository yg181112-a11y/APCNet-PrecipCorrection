# -*- coding: utf-8 -*-
"""CHM 多时效独立验证（24/72/120h）
- init=00Z 子集：窗 [00Z, init+fhr] 对齐 CHM UTC 日累积（fhr/24 天）
- CHM 2024/2025 日值 regrid 到主网格后按窗口累加
"""
import os, json, argparse
from datetime import datetime, timedelta
import numpy as np

BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
CHM_2024 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2024.nc"
CHM_2025 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2025.nc"
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)


def read_chm_daily(path):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    lat = ds.variables["lat"][:].astype(np.float64)
    lon = ds.variables["lon"][:].astype(np.float64)
    prec = ds.variables["prec"][:]
    tv = ds.variables["time"]
    dates = nc.num2date(tv[:], units=tv.units, only_use_cftime_datetimes=False)
    ds.close()
    return prec, lat, lon, dates


def regrid_chm_to_main(chm_latlon, lat, lon):
    from scipy.interpolate import RegularGridInterpolator
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    interp = RegularGridInterpolator((lat, lon), chm_latlon, method="linear", bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    mask = ~np.isnan(vals)
    return np.where(mask, vals, 0.0), mask


def load_lead(fhr):
    if fhr == 24:
        d = os.path.join(BASE, '24h_exp')
        g = np.load(os.path.join(d, 'gfs_24h_accum.npy'))
        t = np.load(os.path.join(d, 'era5_24h_accum.npy'))
        p = np.load(os.path.join(d, 'pred_apcnet_24h.npy'))
        split = np.load(os.path.join(d, 'split_mask.npy'))
        inits = np.load(os.path.join(d, 'init_times.npy'))
        valid = ~(np.isnan(g).any((1, 2)) | np.isnan(t).any((1, 2)))
        tm = (split == 2) & valid
        g, t, p = g[tm], t[tm], p[:tm.sum()]
        init_s = inits[tm]
        pu = np.load(os.path.join(d, 'unet_24h.npy'))
        pu = pu[:tm.sum()]
    else:
        d = os.path.join(BASE, 'multi_lead_exp', f'{fhr}h')
        g = np.load(os.path.join(d, f'gfs_{fhr}h_accum.npy'))
        t = np.load(os.path.join(d, f'era5_{fhr}h_accum.npy'))
        p = np.load(os.path.join(d, f'pred_apcnet_{fhr}h.npy'))
        split = np.load(os.path.join(d, 'split_mask.npy'))
        inits = np.load(os.path.join(d, 'init_times.npy'))
        valid = ~(np.isnan(g).any((1, 2)) | np.isnan(t).any((1, 2)))
        tm = (split == 2) & valid
        g, t, p = g[tm], t[tm], p[:tm.sum()]
        init_s = inits[tm]
        pu = np.load(os.path.join(d, f'unet_{fhr}h.npy'))
        pu = pu[:tm.sum()]
    init_dt = np.array([np.datetime64(s, 'h') for s in init_s], dtype='datetime64[h]').astype(datetime)
    return g, t, p, pu, init_dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fhr', type=int, required=True)
    args = ap.parse_args()
    fhr = args.fhr
    g, t, p, pu, init_dt = load_lead(fhr)
    sel = np.array([d.hour == 0 for d in init_dt])
    gs, ts, ps, pus = g[sel], t[sel], p[sel], pu[sel]
    dates = [d.date() for d in init_dt[sel]]
    win_days = fhr // 24
    print(f'fhr={fhr}: init=00Z 测试样本 {len(dates)}, 窗口 {win_days} 个 CHM 日')

    # CHM 日 map
    chm_day_map, chm_mask = {}, None
    for pth in (CHM_2024, CHM_2025):
        prec, lat, lon, dts = read_chm_daily(pth)
        for k in range(len(prec)):
            field, mask = regrid_chm_to_main(prec[k], lat, lon)
            chm_day_map[dts[k].date()] = field
            chm_mask = mask
    # 窗口累积：需要 init 日 + 后 win_days-1 天
    G, T, P, U, C = [], [], [], [], []
    common = []
    for i, dd in enumerate(dates):
        days = [dd + timedelta(days=k) for k in range(win_days)]
        if all(x in chm_day_map for x in days):
            c = sum(chm_day_map[x] for x in days)
            G.append(gs[i]); T.append(ts[i]); P.append(ps[i]); U.append(pus[i]); C.append(c)
            common.append(dd)
    G, T, P, U, C = np.array(G), np.array(T), np.array(P), np.array(U), np.array(C)
    print(f'公共日: {len(common)}')
    if len(common) < 30:
        raise SystemExit('公共日不足')

    def metr(obs, fcst, tag):
        o = obs[:, chm_mask].flatten(); f = fcst[:, chm_mask].flatten()
        v = np.isfinite(o) & np.isfinite(f)
        o, f = o[v], f[v]
        mse = np.mean((o - f) ** 2)
        cc = np.corrcoef(o, f)[0, 1] if np.std(o) > 0 and np.std(f) > 0 else 0.0
        def ets(o, f, thr):
            a = ((o >= thr) & (f >= thr)).sum(); b = ((o < thr) & (f >= thr)).sum()
            c = ((o >= thr) & (f < thr)).sum(); d = ((o < thr) & (f < thr)).sum()
            den = a + b + c
            ref = (a + b) * (a + c) / max(den, 1)
            return float((a - ref) / max(den - ref, 1e-9)) if den > 0 else 0.0
        out = {"n": int(o.size), "RMSE": float(np.sqrt(mse)), "MAE": float(np.mean(np.abs(o - f))),
               "CC": float(cc), "bias": float(np.mean(f - o)),
               "obs_mean": float(np.mean(o)), "fcst_mean": float(np.mean(f)),
               "ETS10": ets(o, f, 10), "ETS20": ets(o, f, 20)}
        return out

    res = {}
    for nm, arr in [('gfs', G), ('apc', P), ('unet', U), ('era5_ref', T)]:
        res[nm] = metr(C, arr, nm)
        m = res[nm]
        print(f"  {nm:9s} RMSE={m['RMSE']:.3f} MAE={m['MAE']:.3f} CC={m['CC']:.3f} bias={m['bias']:+.3f} "
              f"(obs={m['obs_mean']:.3f}, fcst={m['fcst_mean']:.3f}) ETS10={m['ETS10']:.3f} ETS20={m['ETS20']:.3f}")
    for nm in ('apc', 'unet', 'era5_ref'):
        imp = 100.0 * (res['gfs']['RMSE'] - res[nm]['RMSE']) / res['gfs']['RMSE']
        res[f'rmse_improve_{nm}'] = float(imp)
        print(f"  {nm} RMSE 改进 vs GFS: {imp:+.2f}%")

    res['protocol'] = f'CHM {fhr}h verification (UTC window [00Z,+{fhr}h], init=00Z only)'
    res['n_common_days'] = len(common)
    res['coverage_gridpoints'] = int(chm_mask.sum())
    out_p = os.path.join(BASE, 'multi_lead_exp', f'chm_{fhr}h_eval.json') if fhr != 24 else os.path.join(BASE, '24h_exp', 'chm_24h_eval.json')
    json.dump(res, open(out_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    print(f'✅ 已保存: {out_p}')


if __name__ == '__main__':
    main()
