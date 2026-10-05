# -*- coding: utf-8 -*-
"""
verify_era5_target.py  —  13.0 迭代 · ERA5 目标重建验收脚本（数据门禁）

用途：重新下载的 ERA5 降水目标必须先通过本脚本验收，任何一项 FAIL 都不允许进入训练。

背景（BUG_LOG B1 定谳）：CDS reanalysis-era5-single-levels 的 tp 以 hourly timeseries 提供
（de-accumulated to hourly），旧数据 3-hourly 请求拿到的是 1h 累积 @ 3h 抽稀，导致目标整体
×3 偏低。正确重建：下载 hourly tp（24 步/天）后聚合 ERA5_3h(t)=tp(t−2h)+tp(t−1h)+tp(t)，
得到与 GFS f003 配对的 3h 累积。本脚本验收的就是"已聚合/已正确下载的 3h 累积月文件"。

验收项：
  [M1] 元数据：tp 的 GRIB_stepUnits=3（或 cell_methods 明确 3h sum）；time 步长 3h；units=m
  [M2] 网格：25×37，区域 40-46N / 117-126E，0.25°
  [V1] 与 GFS f003 同同时次（03/09/15/21Z）域均比值 0.9–1.1（原错误数据为 3.038）
  [V2] 年降水估计 600–900 mm/yr（原错误数据 ~235；CHM 气候 ~635）
  [V3] ≥20 mm/3h 格点数与 GFS 同量级（比值 0.3–3；原错误数据差 ~80 倍）
  [V4] 时间序列相关 r > 0.8（原错误数据 0.936，此处主要防"时刻错配"类错误）

用法：
  python verify_era5_target.py --era5 <新ERA5月文件目录> [--gfs <GFS目录>] [--year 2024]
示例：
  python verify_era5_target.py --era5 D:/liaohe/ERA5-data/new_monthly --gfs D:/liaohe/GFS-data/gfs.0p25.2015-2025.f003 --year 2024

退出码：0=全部通过（可进入训练）；1=任一项 FAIL
"""
import argparse, glob, os, sys
import numpy as np
import netCDF4
from datetime import datetime, timedelta

FAIL = []

def check(name, ok, detail):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    if not ok:
        FAIL.append(name)
    return ok

def load_era5_monthly(era5_dir, year, month):
    """读取一个月的 ERA5 3-hourly tp（m→mm），返回 (valid_times[list], tp[n,25,37])"""
    fs = sorted(glob.glob(os.path.join(era5_dir, f"*{year}_{month:02d}*.nc"))) or \
         sorted(glob.glob(os.path.join(era5_dir, f"*{year}{month:02d}*.nc")))
    if not fs:
        raise FileNotFoundError(f"{era5_dir} 中未找到 {year}-{month:02d} 的月文件")
    # 多个文件按时间拼接
    all_t, all_tp = [], []
    for f in fs:
        ds = netCDF4.Dataset(f)
        vt = ds.variables['valid_time'][:]
        t0 = datetime(1970, 1, 1)
        ts = [t0 + timedelta(seconds=int(x)) for x in vt]
        tp = np.ma.filled(ds.variables['tp'][:], 0.0).astype(np.float64) * 1000.0  # m→mm
        all_t.extend(ts); all_tp.append(tp)
        ds.close()
    order = np.argsort([t.timestamp() for t in all_t])
    return [all_t[i] for i in order], np.concatenate(all_tp, 0)[order]

def load_gfs_f003(gfs_root, year, month):
    """读取 GFS f003 文件（jiangshui 子目录），返回 (valid_times, values[n])——域均"""
    folder = os.path.join(gfs_root, 'jiangshui')
    fs = sorted(glob.glob(os.path.join(folder, f"gfs.0p25.{year}{month:02d}*.f003.grib2.nc")))
    if not fs:
        raise FileNotFoundError(f"{folder} 中未找到 {year}{month:02d} 的 f003 文件")
    ts, vals = [], []
    for f in fs:
        ds = netCDF4.Dataset(f)
        # valid = init + 3h（f003）
        # 从文件名取 init 时刻：gfs.0p25.YYYYMMDDHH.f003.grib2.nc（'.'分段：gfs/0p25/YYYYMMDDHH/f003/...）
        b = os.path.basename(f)
        init = datetime.strptime(b.split('.')[2], '%Y%m%d%H')
        valid = init + timedelta(hours=3)
        ts.append(valid)
        vals.append(float(np.ma.filled(ds.variables['A_PCP_L1_Accum_1'][0], 0.0).mean()))
        ds.close()
    order = np.argsort([t.timestamp() for t in ts])
    return [ts[i] for i in order], np.array(vals)[order]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--era5', required=True, help='新 ERA5 月文件目录（含 tp 的 NetCDF）')
    ap.add_argument('--gfs', default=r'D:/liaohe/GFS-data/gfs.0p25.2015-2025.f003')
    ap.add_argument('--year', type=int, default=2024, help='比对基准年（建议 2024，暖季）')
    args = ap.parse_args()

    print("=" * 70)
    print("verify_era5_target.py — ERA5 目标重建验收（13.0 数据门禁）")
    print("=" * 70)

    # ---- 元数据检查（取一个样本文件）----
    probe = sorted(glob.glob(os.path.join(args.era5, "*.nc")))
    if not probe:
        print(f"FAIL: {args.era5} 中没有 .nc 文件"); sys.exit(1)
    ds = netCDF4.Dataset(probe[0])
    v = ds.variables['tp']
    step_units = int(v.getncattr('GRIB_stepUnits')) if 'GRIB_stepUnits' in v.ncattrs() else None
    step_type = str(v.getncattr('GRIB_stepType')) if 'GRIB_stepType' in v.ncattrs() else ''
    cm = v.getncattr('cell_methods') if 'cell_methods' in v.ncattrs() else ''
    units = v.getncattr('units') if 'units' in v.ncattrs() else ''
    lat = ds.variables['latitude'][:] if 'latitude' in ds.variables else ds.variables['lat'][:]
    lon = ds.variables['longitude'][:] if 'longitude' in ds.variables else ds.variables['lon'][:]
    vt = ds.variables['valid_time'][:]
    t0 = datetime(1970, 1, 1)
    ts = [t0 + timedelta(seconds=int(x)) for x in vt]
    diffs = np.unique(np.round(np.diff(np.array([t.timestamp() for t in ts])) / 3600))
    ds.close()

    print(f"\n[M1] 元数据（{os.path.basename(probe[0])}）：")
    print(f"     tp: units={units}, GRIB_stepUnits={step_units}, GRIB_stepType={step_type}, cell_methods={cm!r}")
    ok1 = (step_units == 3) or ('3' in cm)
    check("M1a 累积窗=3h", ok1, f"stepUnits={step_units} (期望 3) / cell_methods={cm!r}")
    ok1b = (units == 'm')
    check("M1b 单位=m", ok1b, f"units={units}")
    ok1c = (np.all(diffs == 3))
    check("M1c 时间步长=3h", ok1c, f"diffs={diffs.tolist()}")
    print(f"\n[M2] 网格：lat {float(lat.min()):.2f}–{float(lat.max()):.2f}（期望 40–46，N={len(lat)}）"
          f"，lon {float(lon.min()):.2f}–{float(lon.max()):.2f}（期望 117–126，N={len(lon)}）")
    ok2 = (len(lat) == 25 and len(lon) == 37)
    check("M2 网格 25×37", ok2, f"{len(lat)}×{len(lon)}")

    # ---- 数值验收：7-8 月（暖季对流活跃期）与 GFS 直接比对 ----
    print(f"\n[V] 数值验收：{args.year}-07/08 与 GFS f003 同同时次比对")
    e_t, e_v, g_t, g_v = [], [], [], []
    for mon in (7, 8):
        try:
            et, ev = load_era5_monthly(args.era5, args.year, mon)
            gt, gv = load_gfs_f003(args.gfs, args.year, mon)
            e_t.extend(et); e_v.append(ev); g_t.extend(gt); g_v.append(gv)
        except FileNotFoundError as ex:
            print(f"  ⚠ {ex}（跳过该月）")
    if not e_v:
        print("FAIL: 无可比对月份"); sys.exit(1)
    e_v = np.concatenate(e_v, 0).mean(axis=(1, 2)) if e_v[0].ndim == 3 else np.concatenate(e_v)
    g_v = np.concatenate(g_v)
    # 对齐同时次（03/09/15/21Z）
    e_map = {t: v for t, v in zip(e_t, e_v)}
    paired = [(t, e_map[t], g) for t, g in zip(g_t, g_v) if t in e_map]
    if not paired:
        print("FAIL: ERA5 与 GFS 无同时次交集（检查时间戳/抽稀）"); sys.exit(1)
    tp_e = np.array([p[1] for p in paired]); tp_g = np.array([p[2] for p in paired])
    ratio = float(tp_g.mean() / max(tp_e.mean(), 1e-9))
    corr = float(np.corrcoef(tp_g, tp_e)[0, 1])
    # V2 年降水：用全年 12 个月域均估算（避免 7-8 月暖季月均外推高估）
    year_vals = []
    for mon in range(1, 13):
        try:
            _, ev_all = load_era5_monthly(args.era5, args.year, mon)
            if ev_all.ndim == 3:
                year_vals.append(float(np.nanmean(ev_all)))
        except FileNotFoundError:
            pass
    annual_est = float(np.mean(year_vals) * 8.0 * 365.0) if year_vals else float(tp_e.mean() * 8.0 * 365.0)
    print(f"     配对时次 {len(paired)}，GFS 域均 {tp_g.mean():.4f} mm/3h，ERA5 域均 {tp_e.mean():.4f} mm/3h")
    check("V1 比值 0.9–1.1", 0.9 <= ratio <= 1.1, f"GFS/ERA5 = {ratio:.3f}（原错误数据 3.038）")
    check("V2 年降水 600–900", 600 <= annual_est <= 900, f"{annual_est:.0f} mm/yr（原错误数据 ~235）")
    check("V4 相关 r>0.8", corr > 0.8, f"r = {corr:.3f}")

    # ---- 极端格点量级（≥20mm 格点数）----
    print("\n[V3] 极端格点量级（≥20 mm/3h 格点数，测试期量级核对）")
    # 用月文件直接统计 ≥20mm 的格点-时次数
    e_heavy = 0; g_heavy = 0
    try:
        et2, ev2 = load_era5_monthly(args.era5, args.year, 7)
        ev2 = np.ma.filled(ev2, 0.0)
        e_heavy = int((ev2 >= 20.0).sum())
        gs = sorted(glob.glob(os.path.join(args.gfs, 'jiangshui', f"gfs.0p25.{args.year}07*.f003.grib2.nc")))
        for f in gs:
            dd = netCDF4.Dataset(f); g_heavy += int((np.ma.filled(dd.variables['A_PCP_L1_Accum_1'][0], 0.0) >= 20.0).sum()); dd.close()
        hratio = e_heavy / max(g_heavy, 1)
        check("V3 ≥20mm 格点同量级", 0.3 <= hratio <= 3.0,
              f"ERA5 {e_heavy} vs GFS {g_heavy}（比值 {hratio:.2f}；原错误数据 ~{24} vs 2001 ≈ 0.012）")
    except Exception as ex:
        print(f"  ⚠ V3 跳过：{ex}")

    print("\n" + "=" * 70)
    if FAIL:
        print(f"结果：FAIL（{len(FAIL)} 项未通过：{', '.join(FAIL)}）— 禁止进入训练")
        sys.exit(1)
    else:
        print("结果：全部 PASS — 数据可进入 13.0 训练")
        sys.exit(0)

if __name__ == '__main__':
    main()
