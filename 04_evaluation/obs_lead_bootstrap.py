# -*- coding: utf-8 -*-
"""obs_lead_bootstrap.py — CHM/GPM 24-72-120h 观测验证 bootstrap CI 补算
- CHM：RMSE 改进 vs GFS（月块，与 chm_lead_eval.py 同数据流；CHM nc regrid 窗口累积）
- GPM：MSE 改进 vs GFS（月块，per-sample 数组来自 verify_gpm_lead_save.py）
- 块 = (年,月)，i.i.d. 重采样 2000 次（seed 42），2.5%/97.5% 分位
输出：WAF/r3_media/obs_lead_bootstrap.json + 13.0修复重跑/obs_lead_bootstrap.json
"""
import os, json
from datetime import datetime, timedelta
import numpy as np

BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
OUTD = r'C:\Users\yg181\Desktop\论文三\WAF\r3_media'
CHM_2024 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2024.nc"
CHM_2025 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2025.nc"
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
NBOOT = 2000
RNG = np.random.default_rng(42)


# ---------- CHM ----------
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
    interp = RegularGridInterpolator((lat, lon), chm_latlon, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    mask = ~np.isnan(vals)
    return np.where(mask, vals, 0.0), mask


def load_chm_series(fhr):
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
        pu = np.load(os.path.join(d, 'unet_24h.npy'))[:tm.sum()]
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
        pu = np.load(os.path.join(d, f'unet_{fhr}h.npy'))[:tm.sum()]
    init_dt = np.array([np.datetime64(s, 'h') for s in init_s], dtype='datetime64[h]').astype(datetime)
    sel = np.array([d0.hour == 0 for d0 in init_dt])
    return g[sel], t[sel], p[sel], pu[sel], init_dt[sel]


def chm_bootstrap(fhr):
    g, t, p, pu, init_dt = load_chm_series(fhr)
    dates = [d0.date() for d0 in init_dt]
    win_days = fhr // 24

    chm_day_map, chm_mask = {}, None
    for pth in (CHM_2024, CHM_2025):
        prec, lat, lon, dts = read_chm_daily(pth)
        for k in range(len(prec)):
            field, mask = regrid_chm_to_main(prec[k], lat, lon)
            chm_day_map[dts[k].date()] = field
            chm_mask = mask

    G, T, P, U, C, months = [], [], [], [], [], []
    for i, dd in enumerate(dates):
        days = [dd + timedelta(days=k) for k in range(win_days)]
        if all(x in chm_day_map for x in days):
            c = sum(chm_day_map[x] for x in days)
            G.append(g[i]); T.append(t[i]); P.append(p[i]); U.append(pu[i]); C.append(c)
            months.append((dd.year, dd.month))
    G = np.array(G); T = np.array(T); P = np.array(P); U = np.array(U); C = np.array(C)
    months = np.array(months)
    msk = chm_mask
    # 展平为 (n_day, n_grid)
    def flat(a): return a[:, msk]
    Gf, Cf, Pf, Uf, Tf = flat(G), flat(C), flat(P), flat(U), flat(T)

    um = sorted({tuple(m) for m in months})
    idx = {m: np.where((months == m).all(axis=1))[0] for m in um}
    n = len(Gf)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def imp_rmse(g_rmse, x_rmse):
        return 100.0 * (g_rmse - x_rmse) / g_rmse

    gfs_rmse = rmse(Gf, Cf)
    res = {'n_days': int(n), 'n_months': len(um), 'gfs_rmse': gfs_rmse}
    for tag, arr in [('apc', Pf), ('unet', Uf), ('era5_ref', Tf)]:
        obs_imp = imp_rmse(gfs_rmse, rmse(arr, Cf))
        boots = []
        for _ in range(NBOOT):
            bs = [tuple(m) for m in RNG.choice(um, size=len(um), replace=True)]
            s = np.concatenate([idx[m] for m in bs])
            gg = Gf[s]; oo = Cf[s]; xx = arr[s]
            boots.append(imp_rmse(rmse(gg, oo), rmse(xx, oo)))
        boots = np.array(boots)
        res[f'{tag}_imp'] = float(obs_imp)
        res[f'{tag}_ci95'] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
        res[f'{tag}_p_pos'] = float(np.mean(boots > 0))
        print(f"  CHM {fhr}h {tag}: {obs_imp:+.2f}% CI [{res[f'{tag}_ci95'][0]:+.2f},{res[f'{tag}_ci95'][1]:+.2f}] P>0={res[f'{tag}_p_pos']:.3f}")
    return res


# ---------- GPM ----------
def gpm_bootstrap(fhr):
    z = np.load(os.path.join(BASE, 'gpm_lead_arrays', f'gpm_{fhr}h.npz'))
    obs = z['gpm']; gfs = z['gfs']; apc = z['apc']; unet = z['unet']; months = z['months']
    # 共同有效样本（剔除任一方法含 NaN 的行，如 120h 缺一天 GFS f120）
    bad = np.isnan(gfs).any(axis=1) | np.isnan(obs).any(axis=1) \
        | np.isnan(apc).any(axis=1) | np.isnan(unet).any(axis=1)
    n_drop = int(bad.sum())
    if n_drop:
        print(f"  {fhr}h: 剔除 {n_drop} 行 NaN 样本（共同有效 {bad.size - n_drop}）")
        obs, gfs, apc, unet, months = obs[~bad], gfs[~bad], apc[~bad], unet[~bad], months[~bad]
    n = len(obs)
    um = sorted({tuple(m) for m in months})
    idx = {m: np.where((months == m).all(axis=1))[0] for m in um}

    def mse(a, b):
        return float(np.mean((a - b) ** 2))

    def imp_mse(g_mse, x_mse):
        return 100.0 * (g_mse - x_mse) / g_mse

    gfs_mse = mse(gfs, obs)
    res = {'n_samples': int(n), 'n_months': len(um), 'gfs_mse': gfs_mse}
    for tag, arr in [('apc', apc), ('unet', unet)]:
        obs_imp = imp_mse(gfs_mse, mse(arr, obs))
        boots = []
        for _ in range(NBOOT):
            bs = [tuple(m) for m in RNG.choice(um, size=len(um), replace=True)]
            s = np.concatenate([idx[m] for m in bs])
            boots.append(imp_mse(mse(gfs[s], obs[s]), mse(arr[s], obs[s])))
        boots = np.array(boots)
        res[f'{tag}_imp'] = float(obs_imp)
        res[f'{tag}_ci95'] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
        res[f'{tag}_p_pos'] = float(np.mean(boots > 0))
        print(f"  GPM {fhr}h {tag}: {obs_imp:+.2f}% CI [{res[f'{tag}_ci95'][0]:+.2f},{res[f'{tag}_ci95'][1]:+.2f}] P>0={res[f'{tag}_p_pos']:.3f}")
    return res


if __name__ == '__main__':
    import sys, json as _json
    only = sys.argv[1] if len(sys.argv) > 1 else 'all'
    # 合并更新：先读已有结果，避免单分支运行覆盖另一分支
    out = {}
    for p in (os.path.join(OUTD, 'obs_lead_bootstrap.json'),
              os.path.join(BASE, 'obs_lead_bootstrap.json')):
        if os.path.exists(p):
            try:
                out.update(_json.load(open(p, encoding='utf-8')))
            except Exception:
                pass
    if only in ('all', 'chm'):
        for fhr in (24, 72, 120):
            print(f'== CHM {fhr}h ==', flush=True)
            out[f'chm_{fhr}h'] = chm_bootstrap(fhr)
    if only in ('all', 'gpm'):
        for fhr in (24, 72, 120):
            print(f'== GPM {fhr}h ==', flush=True)
            out[f'gpm_{fhr}h'] = gpm_bootstrap(fhr)
    os.makedirs(OUTD, exist_ok=True)
    for p in (os.path.join(OUTD, 'obs_lead_bootstrap.json'),
              os.path.join(BASE, 'obs_lead_bootstrap.json')):
        json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print('✅ saved', p)
