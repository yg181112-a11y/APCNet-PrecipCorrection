# -*- coding: utf-8 -*-
"""
24h 实验特征预提取：f024 大气场 7 通道 + f072 PWAT 补第 2 通道 + 24h 降水通道
输入: 1a 构建的 init_times.npy (24h_exp)
输出: features_24h.npy [N, 8, 25, 37] float32
通道: 0 CAPE, 1 PWAT(f072), 2 U850, 3 V850, 4 U500, 5 V500, 6 VVEL500, 7 Precip24h
"""
import numpy as np, os, glob, time, json, multiprocessing as mp
from datetime import datetime
from scipy.io import netcdf_file
import netCDF4

GFS_F024 = r'D:\liaohe\GFS-data\gfs.0p25.2015-2025.f024'
GFS_F072 = r'D:\liaohe\GFS-data\gfs.0p25.2015-2025.f072'
OUT = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\24h_exp'

def load_init_times():
    it = np.load(os.path.join(OUT, 'init_times.npy'))   # datetime64[h]
    return [datetime.utcfromtimestamp(int(t)) for t in (it.astype('datetime64[s]').astype(np.int64))]

def extract_one(args):
    idx, init = args
    fname = init.strftime('gfs.0p25.%Y%m%d%H.f024.grib2.nc')
    fpath = os.path.join(GFS_F024, fname)
    ch = np.zeros((8, 25, 37), np.float32)
    try:
        ds = netcdf_file(fpath, 'r', mmap=False)
        cape = ds.variables['CAPE_L1'].data
        ch[0] = cape[0] if cape.ndim == 3 else cape
        # 风场 level3 坐标（2021-03 后 41 层，之前 26 层，均含 850/500）
        u = ds.variables['U_GRD_L100'].data
        v = ds.variables['V_GRD_L100'].data
        lv3 = ds.variables['level3'].data
        i850 = int(np.argmin(np.abs(lv3 - 850))); i500 = int(np.argmin(np.abs(lv3 - 500)))
        ch[2] = u[0, i850]; ch[3] = v[0, i850]; ch[4] = u[0, i500]; ch[5] = v[0, i500]
        # VVEL: 2021-03 前用 level5(21层)，之后用 level3(41层)；自适应
        vv = ds.variables['V_VEL_L100'].data
        if 'level5' in ds.variables:
            lv = ds.variables['level5'].data
        else:
            lv = lv3
        i500v = int(np.argmin(np.abs(lv - 500)))
        ch[6] = vv[0, i500v] * 0.01
        ds.close()
    except Exception as e:
        print(f'  [warn] f024 fail {fname}: {e}')
        return idx, ch
    # PWAT from f072
    try:
        fname72 = init.strftime('gfs.0p25.%Y%m%d%H.f072.grib2.nc')
        ds72 = netCDF4.Dataset(os.path.join(GFS_F072, fname72))
        pwat = ds72.variables['P_WAT_L200'][0]
        ds72.close()
        ch[1] = pwat
    except Exception as e:
        print(f'  [warn] f072 fail {fname72}: {e}')
    return idx, ch

def main():
    t0 = time.time()
    inits = load_init_times()
    N = len(inits)
    print(f'init 数: {N}')
    feats = np.zeros((N, 8, 25, 37), np.float32)
    nproc = min(12, mp.cpu_count())
    print(f'多进程 {nproc} workers...')
    with mp.Pool(nproc) as pool:
        for i, (idx, ch) in enumerate(pool.imap_unordered(extract_one, enumerate(inits), chunksize=16)):
            feats[idx] = ch
            if (i + 1) % 1000 == 0:
                print(f'  {i+1}/{N}, {time.time()-t0:.0f}s')
    np.save(os.path.join(OUT, 'features_24h.npy'), feats)
    # 快检
    fm = feats.mean(axis=(0,2,3))
    print('通道均值:', np.round(fm, 4))
    print('完成, 耗时', round(time.time()-t0, 1), 's')

if __name__ == '__main__':
    mp.freeze_support()
    main()
