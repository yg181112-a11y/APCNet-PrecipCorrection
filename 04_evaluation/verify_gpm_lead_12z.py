# -*- coding: utf-8 -*-
"""verify_gpm_lead_12z.py — 12Z init 鲁棒性验证（24h，GPM 观测）。
与 verify_gpm_lead.py 同口径，仅将 init 00Z 改为 12Z：
窗口 [init, init+24h] = 当天12Z → 次日11:30，48 个 bin。
输出: gpm_24h_eval_12z.json
"""
import os, sys, json, numpy as np
from datetime import datetime, timedelta
import netCDF4 as nc

GPM_ROOT = r'D:\liaohe\GPM_IMERG'
BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
EXP = os.path.join(BASE, '24h_exp')
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
BIN_H = 0.5

def gpm_file_for(dt):
    d = dt.date()
    ym = d.strftime('%Y%m')
    ymd_hms = d.strftime('%Y%m%d') + '_' + dt.strftime('%H%M%S')
    return os.path.join(GPM_ROOT, f'imerg_{ym}', f'imerg_{ymd_hms}.nc4')

def read_gpm_rate(path):
    ds = nc.Dataset(path)
    g = ds.groups['Grid']
    lat = g.variables['lat'][:].astype(np.float64)
    lon = g.variables['lon'][:].astype(np.float64)
    rate = g.variables['precipitation'][0, :, :].astype(np.float64)
    ds.close()
    return rate, lon, lat

def gpm_accum_lead(t0, cache, lead_h=24):
    n_bins = int(lead_h * 2)
    bins = []
    for k in range(n_bins):
        bt = t0 + timedelta(minutes=int(30 * k))
        key = bt.strftime('%Y%m%d_%H%M%S')
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
    interp = RegularGridInterpolator((lon, lat), acc_lonlat, method='linear',
                                     bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    mask = ~np.isnan(vals)
    vals = np.where(mask, vals, 0.0)
    return vals, mask

def cont_metrics(obs, fcst):
    obs_f, fcst_f = obs.flatten(), fcst.flatten()
    valid = np.isfinite(obs_f) & np.isfinite(fcst_f)
    obs_f, fcst_f = obs_f[valid], fcst_f[valid]
    if obs_f.size == 0:
        return None
    mse = np.mean((obs_f - fcst_f) ** 2)
    cc = np.corrcoef(obs_f, fcst_f)[0, 1] if obs_f.size > 1 else 0.0
    return {'n': int(obs_f.size), 'MSE': float(mse), 'RMSE': float(np.sqrt(mse)),
            'MAE': float(np.mean(np.abs(obs_f - fcst_f))), 'CC': float(cc),
            'bias': float(np.mean(fcst_f - obs_f))}

def main():
    init = np.load(os.path.join(EXP, 'init_times.npy'), allow_pickle=True)
    mask = np.load(os.path.join(EXP, 'split_mask.npy'))
    gfs = np.load(os.path.join(EXP, 'gfs_24h_accum.npy'))
    era5 = np.load(os.path.join(EXP, 'era5_24h_accum.npy'))
    apc = np.load(os.path.join(EXP, 'pred_apcnet_24h.npy'))
    unet = np.load(os.path.join(EXP, 'unet_24h.npy'))

    times = [datetime(1970, 1, 1) + timedelta(hours=int(t)) for t in init.astype('datetime64[h]').astype(np.int64)]
    times = [t.replace(tzinfo=None) for t in times]

    # 测试期 12Z 且窗口在 GPM 覆盖内
    sel = [i for i in range(len(times)) if int(mask[i]) == 2 and times[i].hour == 12
           and times[i] >= datetime(2024, 1, 1)
           and times[i] + timedelta(hours=24) <= datetime(2025, 9, 30, 23, 59)]
    print(f'候选 12Z 样本: {len(sel)}', flush=True)

    cache = {}
    gpm_all = np.zeros((len(sel), 25, 37))
    mask_all = np.zeros((len(sel), 25, 37), dtype=bool)
    ok_i = []
    for r, i in enumerate(sel):
        t0 = times[i]
        acc, lon, lat = gpm_accum_lead(t0, cache, 24)
        if acc is None:
            continue
        field, msk = regrid_to_main(acc, lon, lat)
        if not msk.any():
            continue
        gpm_all[r] = field; mask_all[r] = msk
        ok_i.append(r)
        if len(ok_i) % 100 == 0:
            print(f'  进度 {len(ok_i)}/{len(sel)}', flush=True)

    ok_rows = np.array(ok_i)
    n_ok = len(ok_rows)
    print(f'GPM 有效 12Z 样本: {n_ok}/{len(sel)}', flush=True)
    if n_ok < 50:
        print('!! 有效样本过少'); sys.exit(1)

    cov = mask_all[ok_rows].all(axis=0)
    n_cov = int(cov.sum())
    sel_global = np.array(sel)[ok_rows]

    def restrict(a, rows):
        return a[rows][:, cov]
    gpm_v = restrict(gpm_all, ok_rows)
    gfs_v = restrict(gfs, sel_global)
    era5_v = restrict(era5, sel_global)
    te_all = np.where(mask == 2)[0]
    te_nan = np.isnan(gfs[te_all]).any(axis=(1, 2))
    valid_te_global = te_all[~te_nan]
    pos = np.searchsorted(valid_te_global, sel_global)
    apc_v = apc[pos][:, cov]
    unet_v = unet[pos][:, cov]

    # 基线（训练期标定，与 train_24h 同实现）
    tr_all = np.where(mask == 0)[0]
    tr_nan = np.isnan(gfs[tr_all]).any(axis=(1, 2)) | np.isnan(era5[tr_all]).any(axis=(1, 2))
    tr = tr_all[~tr_nan]
    gfs_tr, era5_tr = gfs[tr], era5[tr]
    qa = np.linspace(0, 1, 201)
    gq = np.zeros((25, 37, 201)); eq = np.zeros((25, 37, 201))
    for i in range(25):
        for j in range(37):
            gq[i, j] = np.quantile(gfs_tr[:, i, j], qa)
            eq[i, j] = np.quantile(era5_tr[:, i, j], qa)
    pred_qm = np.zeros_like(gfs[sel_global])
    for i in range(25):
        for j in range(37):
            pred_qm[:, i, j] = np.interp(gfs[sel_global][:, i, j], gq[i, j], eq[i, j])
    qm_v = restrict(pred_qm, np.arange(len(sel_global)))

    result = {'protocol': 'GPM 24h accumulation validation (init 12Z)',
              'n_samples_ok': int(n_ok), 'coverage_gridpoints': int(n_cov)}
    print('\n[真值 = GPM 24h, 12Z init]')
    for name, arr in [('GFS', gfs_v), ('APCNet', apc_v), ('U-Net', unet_v),
                      ('QM', qm_v), ('ERA5', era5_v)]:
        m = cont_metrics(gpm_v, arr)
        result[f'cont_{name}'] = m
        imp = 100.0 * (cont_metrics(gpm_v, gfs_v)['MSE'] - m['MSE']) / cont_metrics(gpm_v, gfs_v)['MSE']
        result[f'mse_improve_pct_{name}'] = float(imp)
        print(f'  {name:7s} RMSE={m["RMSE"]:.3f} CC={m["CC"]:.3f} bias={m["bias"]:+.2f} MSE改进={imp:+.1f}%', flush=True)

    out_path = os.path.join(EXP, 'gpm_24h_eval_12z.json')
    json.dump(result, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    print(f'✅ 已保存: {out_path}')

if __name__ == '__main__':
    main()
