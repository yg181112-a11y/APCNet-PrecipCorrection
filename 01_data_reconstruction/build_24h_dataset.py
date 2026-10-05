# -*- coding: utf-8 -*-
"""
阶段 1a（修正版）：24h 累积配对数据集构建
- GFS 24h 预报 = jiangshui f024 直接累积 [init, init+24h]（f003 每天仅 4 个时次，无法累加成 24h）
- ERA5 24h 目标 = hourly tp 在 valid_time ∈ (init, init+24h] 的 24 个 1h 值累加 (×1000 m->mm)
- init 每天 00/06/12/18Z，2019-06-12 00Z ~ 2025-12-30 18Z（f024 覆盖起点）
输出：gfs_24h_accum.npy / era5_24h_accum.npy / init_times.npy / split_mask.npy / 24h_acceptance.json
"""
import netCDF4, numpy as np, os, glob, json, time
from datetime import datetime, timedelta

GFS_F024 = r'D:\liaohe\GFS-data\jiangshui\gfs.0p25.2019061212-25.2026011400.f024'
ERA5_H = r'D:\liaohe\ERA5-data\new_hourly_tp'
OUT = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\24h_exp'
os.makedirs(OUT, exist_ok=True)

t0 = time.time()

# ---------- 1. f024 文件索引: init -> path ----------
print('[1] 索引 jiangshui f024 文件...')
f024_by_init = {}
for f in glob.glob(os.path.join(GFS_F024, '*.nc')):
    base = os.path.basename(f)          # gfs.0p25.2019061212.f024.grib2.nc
    s = base.split('.')[2]
    f024_by_init[datetime.strptime(s, '%Y%m%d%H')] = f
print(f'    f024 init 数: {len(f024_by_init)}')

# ---------- 2. init 列表 ----------
print('[2] 构建 init 列表 (2019-06-12 00Z ~ 2025-12-30 18Z, 6h)...')
inits = []
t = datetime(2019, 6, 12, 0)
end = datetime(2025, 12, 30, 18)
while t <= end:
    inits.append(t)
    t += timedelta(hours=6)
N = len(inits)
print(f'    init 数: {N}')

# ---------- 3. ERA5 hourly 句柄 ----------
print('[3] 加载 ERA5 hourly...')
era5_ds = {}
era5_vt0 = {}
def get_era5(year):
    if year not in era5_ds:
        p = os.path.join(ERA5_H, f'era5_tp_hourly_{year}.nc')
        ds = netCDF4.Dataset(p)
        era5_ds[year] = ds
        era5_vt0[year] = int(ds.variables['valid_time'][0])
    return era5_ds[year]

# 方向核对
ds0 = get_era5(2019)
era5_lat = ds0.variables['latitude'][:]
gf = netCDF4.Dataset(next(iter(f024_by_init.values())))
gfs_lat = gf.variables['lat'][:]
gf.close()
era5_flip = np.sign(era5_lat[0] - era5_lat[-1]) != np.sign(gfs_lat[0] - gfs_lat[-1])
print(f'    ERA5 lat 需翻转: {era5_flip}')

# ---------- 4. 主循环 ----------
print(f'[4] 构建 {N} 个 24h 样本...')
gfs24 = np.zeros((N, 25, 37), np.float32)
era524 = np.zeros((N, 25, 37), np.float32)
miss_g, miss_e = 0, 0

for i, init in enumerate(inits):
    # GFS: f024 直接累积
    f = f024_by_init.get(init)
    if f is None:
        miss_g += 1
        gfs24[i] = np.nan
    else:
        ds = netCDF4.Dataset(f)
        gfs24[i] = ds.variables['A_PCP_L1_Accum_1'][0, :, :]
        ds.close()

    # ERA5: valid ∈ (init, init+24h]
    ref = datetime(1970, 1, 1, 0)
    acc_e = np.zeros((25, 37), np.float32)
    ok_e = True
    for k in range(1, 25):
        tv = init + timedelta(hours=k)
        ds = get_era5(tv.year)
        idx = int((tv - ref).total_seconds())
        rel = (idx - era5_vt0[tv.year]) // 3600
        try:
            v = ds.variables['tp'][rel, :, :]
        except Exception:
            ok_e = False
            break
        acc_e += v
    if not ok_e:
        miss_e += 1
        era524[i] = np.nan
        continue
    acc_e = acc_e * 1000.0
    if era5_flip:
        acc_e = acc_e[::-1, :]
    era524[i] = acc_e

    if (i + 1) % 1000 == 0:
        print(f'    {i+1}/{N}, {time.time()-t0:.0f}s')

print(f'    完成: GFS 缺失 {miss_g}, ERA5 缺失 {miss_e}')
valid = ~(np.isnan(gfs24).any(axis=(1,2)) | np.isnan(era524).any(axis=(1,2)))
print(f'    有效配对: {valid.sum()}/{N}')

# ---------- 5. 划分掩码 ----------
split = np.zeros(N, np.int8)
for i, init in enumerate(inits):
    split[i] = 0 if init.year <= 2021 else (1 if init.year <= 2023 else 2)

# ---------- 6. 保存 ----------
np.save(os.path.join(OUT, 'gfs_24h_accum.npy'), gfs24)
np.save(os.path.join(OUT, 'era5_24h_accum.npy'), era524)
np.save(os.path.join(OUT, 'init_times.npy'), np.array([np.datetime64(i, 'h') for i in inits]))
np.save(os.path.join(OUT, 'split_mask.npy'), split)

# ---------- 7. 验收 ----------
acc = {}
g, e = gfs24[valid], era524[valid]
gm, em = np.nanmean(g), np.nanmean(e)
acc['domain_mean_gfs_mm24h'] = float(gm)
acc['domain_mean_era5_mm24h'] = float(em)
acc['gfs_era5_ratio'] = float(gm / em)
acc['annual_gfs_mm'] = float(gm * 365)
acc['annual_era5_mm'] = float(em * 365)
tv_mask = split[valid] == 2
acc['test_gfs_era5_ratio'] = float(np.nanmean(g[tv_mask]) / np.nanmean(e[tv_mask]))
g_flat = g.reshape(len(g), -1); e_flat = e.reshape(len(e), -1)
acc['pixel_corr_gfs_era5'] = float(np.corrcoef(g_flat.T, e_flat.T)[0, 1])

with open(os.path.join(OUT, '24h_acceptance.json'), 'w', encoding='utf-8') as f:
    json.dump(acc, f, ensure_ascii=False, indent=2)
print('[完成] 验收:')
print(json.dumps(acc, ensure_ascii=False, indent=2))
print(f'总耗时 {time.time()-t0:.0f}s')
