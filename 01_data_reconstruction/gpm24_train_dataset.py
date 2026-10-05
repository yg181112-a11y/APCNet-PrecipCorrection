# -*- coding: utf-8 -*-
"""gpm24_train_dataset.py — 构建 00Z init 的 GPM 24h 累积训练目标（2018-2025）。
- 00Z init 窗口 [00:00, 24:00] = 当天 48 个 30min IMERG bin 之和
- 训练期 GPM: C:\\Users\\yg181\\Downloads\\GPM_IMERG_test (2018-2023)
- 测试期 GPM: D:\\liaohe\\GPM_IMERG (2024-2025)
- 输出 (00Z 子集, 与 gfs/era5/feats/split 对齐):
    gpm24_accum_00z.npy   [N00,25,37]  GPM 24h 累积 mm
    gpm24_mask_00z.npy    [25,37]      覆盖掩码 (GPM 插值有效)
    gfs24_accum_00z.npy / era5_24h_accum_00z.npy / features_00z.npy / split_00z.npy / init_00z.npy
"""
import os, sys, glob, numpy as np
from datetime import datetime, timedelta
import netCDF4 as nc

BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
EXP = os.path.join(BASE, '24h_exp')
GPM_TRAIN = r'C:\Users\yg181\Downloads\GPM_IMERG_test'
GPM_TEST = r'D:\liaohe\GPM_IMERG'
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)    # 降序
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)  # 升序
BIN_H = 0.5

def gpm_file_for(dt):
    d = dt.date()
    ym = d.strftime('%Y%m')
    ymd_hms = d.strftime('%Y%m%d') + '_' + dt.strftime('%H%M%S')
    root = GPM_TEST if d.year >= 2024 else GPM_TRAIN
    return os.path.join(root, f'imerg_{ym}', f'imerg_{ymd_hms}.nc4')

def read_gpm_rate(path):
    ds = nc.Dataset(path)
    g = ds.groups['Grid']
    lat = g.variables['lat'][:].astype(np.float64)
    lon = g.variables['lon'][:].astype(np.float64)
    rate = g.variables['precipitation'][0, :, :].astype(np.float64)
    ds.close()
    return rate, lon, lat

def day_accum(date):
    """当天 00Z-24Z 48 bin 累积。任一 bin 缺失 -> (None, None, None)。"""
    acc = None; lon = lat = None
    for k in range(48):
        bt = datetime(date.year, date.month, date.day) + timedelta(minutes=int(30 * k))
        p = gpm_file_for(bt)
        if not os.path.exists(p):
            return None, None, None
        try:
            r, lon, lat = read_gpm_rate(p)
        except Exception:
            return None, None, None
        acc = r * BIN_H if acc is None else acc + r * BIN_H
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

def main():
    init = np.load(os.path.join(EXP, 'init_times.npy'), allow_pickle=True)
    split = np.load(os.path.join(EXP, 'split_mask.npy'))
    gfs = np.load(os.path.join(EXP, 'gfs_24h_accum.npy'))
    era5 = np.load(os.path.join(EXP, 'era5_24h_accum.npy'))
    feats = np.load(os.path.join(EXP, 'features_24h.npy'))

    times = [datetime(1970, 1, 1) + timedelta(hours=int(t)) for t in init.astype('datetime64[h]').astype(np.int64)]
    times = [t.replace(tzinfo=None) for t in times]

    # 00Z init
    sel = [i for i, t in enumerate(times) if t.hour == 0]
    print(f'00Z 样本: {len(sel)} (train/val/test = '
          f'{(split[sel]==0).sum()}/{(split[sel]==1).sum()}/{(split[sel]==2).sum()})', flush=True)

    gpm_all = np.zeros((len(sel), 25, 37))
    mask_ok = []
    ok_i = []
    cache_days = {}
    for r, i in enumerate(sel):
        date = times[i].date()
        if date in cache_days:
            acc, lon, lat = cache_days[date]
        else:
            acc, lon, lat = day_accum(date)
            cache_days[date] = (acc, lon, lat)
        if acc is None:
            continue
        field, msk = regrid_to_main(acc, lon, lat)
        if not msk.any():
            continue
        gpm_all[r] = field
        mask_ok.append(msk)
        ok_i.append(r)
        if len(ok_i) % 100 == 0:
            print(f'  进度 {len(ok_i)}/{len(sel)} (缺 {r+1-len(ok_i)})', flush=True)

    ok = np.array(ok_i)
    n_ok = len(ok)
    print(f'GPM 有效 00Z 样本: {n_ok}/{len(sel)}', flush=True)
    if n_ok < 500:
        print('!! 有效样本过少'); sys.exit(1)

    cov = np.array(mask_ok).all(axis=0)
    n_cov = int(cov.sum())
    print(f'共同覆盖格点: {n_cov}/925', flush=True)

    sel_arr = np.array(sel)[ok]
    np.save(os.path.join(EXP, 'gpm24_accum_00z.npy'), gpm_all[ok])
    np.save(os.path.join(EXP, 'gpm24_mask_00z.npy'), cov)
    np.save(os.path.join(EXP, 'gfs24_accum_00z.npy'), gfs[sel_arr])
    np.save(os.path.join(EXP, 'era5_24h_accum_00z.npy'), era5[sel_arr])
    np.save(os.path.join(EXP, 'features_00z.npy'), feats[sel_arr])
    np.save(os.path.join(EXP, 'split_00z.npy'), split[sel_arr])
    np.save(os.path.join(EXP, 'init_00z.npy'), init[sel_arr])
    print('✅ 已保存 gpm24_accum_00z.npy + 掩码 + 00Z 对齐数组')
    print('时间范围:', times[sel[ok[0]]], '→', times[sel[ok[-1]]])

if __name__ == '__main__':
    main()
