# -*- coding: utf-8 -*-
"""build_gpm3h_ols.py — 3h GPM 目标 OLS 对照数据集。
样本 = init (00/06/12/18Z)，GFS 3h = f003 A_PCP（[init, init+3h]，主网格 25×37），
GPM 3h = 6×30min bin 累加（[init, init+3h]）→ 插值主网格 + coverage mask。
训练期 2018-2023（GPM_IMERG_test），测试期 2024-2025（GPM_IMERG）。
输出: 3h_exp/gfs3h_{train,test}.npy, gpm3h_{train,test}.npy, gpm3h_cov.npy, init3h_{train,test}.npy
"""
import os, numpy as np
from datetime import datetime, timedelta
import netCDF4 as nc

EXP = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\3h_exp'
os.makedirs(EXP, exist_ok=True)
GFS_ROOT = r'D:\liaohe\GFS-data\gfs.0p25.2015-2025.f003\jiangshui'
GPM_TRAIN = r'C:\Users\yg181\Downloads\GPM_IMERG_test'
GPM_TEST = r'D:\liaohe\GPM_IMERG'
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
BIN_H = 0.5

def gfs_3h(init):
    p = os.path.join(GFS_ROOT, f'gfs.0p25.{init.strftime("%Y%m%d%H")}.f003.grib2.nc')
    if not os.path.exists(p):
        return None
    ds = nc.Dataset(p)
    a = ds.variables['A_PCP_L1_Accum_1'][0].astype(np.float64)
    ds.close()
    return a

def gpm_file_for(dt):
    d = dt.date()
    root = GPM_TEST if d.year >= 2024 else GPM_TRAIN
    return os.path.join(root, f'imerg_{d.strftime("%Y%m")}',
                        f'imerg_{d.strftime("%Y%m%d")}_{dt.strftime("%H%M%S")}.nc4')

def read_gpm_rate(path):
    ds = nc.Dataset(path)
    g = ds.groups['Grid']
    lat = g.variables['lat'][:].astype(np.float64)
    lon = g.variables['lon'][:].astype(np.float64)
    rate = g.variables['precipitation'][0, :, :].astype(np.float64)
    ds.close()
    return rate, lon, lat

def load_day(date):
    arrs = []
    for k in range(48):
        bt = datetime(date.year, date.month, date.day) + timedelta(minutes=int(30 * k))
        p = gpm_file_for(bt)
        if not os.path.exists(p):
            return None
        try:
            r, lon, lat = read_gpm_rate(p)
        except Exception:
            return None
        arrs.append(r)
    return np.stack(arrs, axis=0), lon, lat

def build(inits, tag):
    from scipy.interpolate import RegularGridInterpolator
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lon2d.ravel(), lat2d.ravel()], axis=-1)
    N = len(inits)
    gfs_a = np.zeros((N, 25, 37)); gpm_a = np.zeros((N, 25, 37))
    msk_a = np.zeros((N, 25, 37), dtype=bool)
    cache = {}
    ok = 0
    for i, t0 in enumerate(inits):
        g = gfs_3h(t0)
        if g is None:
            continue
        bins = []
        bad = False
        for k in range(6):
            bt = t0 + timedelta(minutes=int(30 * k))
            key = bt.date()
            if key not in cache:
                cache[key] = load_day(key)
                if len(cache) > 40:
                    cache.pop(next(iter(cache)))
            day = cache[key]
            if day is None:
                bad = True; break
            bins.append(day[0][int((bt.hour * 60 + bt.minute) / 30)])
        if bad:
            continue
        acc = np.zeros_like(bins[0])
        for b in bins:
            acc += b * BIN_H
        lon, lat = cache[t0.date()][1], cache[t0.date()][2]
        interp = RegularGridInterpolator((lon, lat), acc, method='linear',
                                         bounds_error=False, fill_value=np.nan)
        vals = interp(pts).reshape(25, 37)
        msk = ~np.isnan(vals)
        if not msk.any():
            continue
        gfs_a[i] = g; gpm_a[i] = np.where(msk, vals, 0.0); msk_a[i] = msk
        ok += 1
        if ok % 500 == 0:
            print(f'  {tag} 进度 {ok}/{N}', flush=True)
    keep = msk_a.any(axis=(1, 2))
    print(f'{tag}: 有效 {int(keep.sum())}/{N}', flush=True)
    return gfs_a[keep], gpm_a[keep], msk_a[keep], np.array(inits)[keep]

def gen_inits(start, end):
    out = []
    t = start
    while t <= end:
        out.append(t)
        t += timedelta(hours=6)
    return out

# 训练期 2018-2023
tr_inits = gen_inits(datetime(2018, 1, 1, 0), datetime(2023, 12, 31, 18))
gfs_tr, gpm_tr, msk_tr, init_tr = build(tr_inits, 'train')
cov = msk_tr.all(axis=0)
print('共同覆盖格点:', int(cov.sum()), '/925', flush=True)
# 测试期 2024-2025（至 2025-09-30，GPM 已确认覆盖）
te_inits = gen_inits(datetime(2024, 1, 1, 0), datetime(2025, 9, 30, 18))
gfs_te, gpm_te, msk_te, init_te = build(te_inits, 'test')

np.save(os.path.join(EXP, 'gfs3h_train.npy'), gfs_tr)
np.save(os.path.join(EXP, 'gpm3h_train.npy'), gpm_tr)
np.save(os.path.join(EXP, 'init3h_train.npy'), np.array([t.strftime('%Y%m%d%H') for t in init_tr], dtype='U10'))
np.save(os.path.join(EXP, 'gfs3h_test.npy'), gfs_te)
np.save(os.path.join(EXP, 'gpm3h_test.npy'), gpm_te)
np.save(os.path.join(EXP, 'init3h_test.npy'), np.array([t.strftime('%Y%m%d%H') for t in init_te], dtype='U10'))
np.save(os.path.join(EXP, 'gpm3h_cov.npy'), cov)
print('✅ 已保存 3h OLS 数据集: train %d, test %d, cov %d' % (len(gfs_tr), len(gfs_te), int(cov.sum())))
