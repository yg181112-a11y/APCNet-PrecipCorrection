# -*- coding: utf-8 -*-
"""
aggregate_era5_tp.py — ERA5 目标重建：hourly tp → 3h 累积 → 合并为新 monthly
流程：
  1) 读 new_hourly_tp/era5_tp_hourly_YYYYMM.nc（24 步/天，1h 累积）
  2) 连续时间轴滑动 3 步求和：tp3h(t) = tp(t−2h) + tp(t−1h) + tp(t)，t ∈ {00,03,...,21}Z
  3) 与旧 monthly（分析变量瞬时场正确）合并，替换 tp 通道
  4) 输出 D:\\liaohe\\ERA5-data\\monthly_new\\era5_merged_YYYYMM.nc（248 步/月）
用法：
  python aggregate_era5_tp.py            # 全量 2015-2025
  python aggregate_era5_tp.py --year 2024  # 单年
"""
import argparse, glob, os, sys
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

HOURLY_DIR = r"D:\liaohe\ERA5-data\new_hourly_tp"
OLD_MONTHLY = r"D:\liaohe\ERA5-data\monthly"
NEW_MONTHLY = r"D:\liaohe\ERA5-data\monthly_new"

def aggregate_all(years):
    os.makedirs(NEW_MONTHLY, exist_ok=True)
    # 1) 逐月加载 hourly（netCDF4，完全可控）；glob 全部年份文件（含 2014 补文件，
    #    使 2015-01-01 00/03Z 的 3h 窗可跨年聚合），排除测试样本 202407
    files = sorted(glob.glob(os.path.join(HOURLY_DIR, "era5_tp_hourly_*.nc")))
    files = [f for f in files if "202407" not in f]
    print(f"加载 {len(files)} 个 hourly 文件...")
    import netCDF4
    from datetime import datetime as _dt
    v0 = _dt(1970, 1, 1)
    all_vt, all_tp = [], []
    for f in files:
        ds = netCDF4.Dataset(f)
        vt = ds.variables["valid_time"][:]
        tp = np.ma.filled(ds.variables["tp"][:], 0.0).astype(np.float64) * 1000.0
        all_vt.append(vt); all_tp.append(tp)
        ds.close()
    vt = np.concatenate(all_vt).astype(np.int64)
    tp = np.concatenate(all_tp, 0)
    order = np.argsort(vt)
    vt, tp = vt[order], tp[order]
    # 去重（防御）
    _, uid = np.unique(vt, return_index=True)
    if len(uid) < len(vt):
        vt, tp = vt[uid], tp[uid]
    n = len(vt)
    print(f"  hourly 序列: {n} 步, {v0+timedelta(seconds=int(vt[0]))} -> {v0+timedelta(seconds=int(vt[-1]))}")

    # 2) 滑动 3 步求和 → 每 3h 时刻的 3h 累积
    tp3h = np.zeros_like(tp)
    tp3h[2:] = tp[2:] + tp[1:-1] + tp[:-2]   # tp3h[i] = tp[i]+tp[i-1]+tp[i-2]（i 时刻结束的 3h 窗）
    tp3h[:2] = np.nan                         # 前 2 步无法聚合
    # 缺口防御：valid_time 步长 != 3600s 处及其后 2 步置 NaN（避免跨缺口错误聚合）
    dt_sec = np.diff(vt.astype(np.int64))
    bad = np.where(dt_sec != 3600)[0] + 1     # 缺口后的第一个索引
    for b in bad:
        tp3h[max(0, b-2):min(n, b+1)] = np.nan
    if len(bad):
        print(f"  ⚠ 检测到 {len(bad)} 处时间缺口，已对应置 NaN")
    # 目标时刻：valid_time 为 00/03/06/.../21Z（%3==0 的整点）
    mask3h = (vt % 3600).astype(np.int64) == 0  # valid_time 单位 seconds；再按小时校验
    hours = (vt // 3600) % 24
    mask3h = hours % 3 == 0
    target_idx = np.where(mask3h)[0]
    target_idx = target_idx[target_idx >= 2]
    print(f"  3h 目标时刻: {len(target_idx)} 个")

    # 3) 按年月分组输出 tp3h，并合并旧 monthly
    groups = {}
    for i in target_idx:
        t = v0 + timedelta(seconds=int(vt[i]))
        key = (t.year, t.month)
        groups.setdefault(key, []).append(i)
    n_ok = 0
    for (y, m), idxs in sorted(groups.items()):
        idxs = np.array(sorted(idxs))
        tp3h_month = tp3h[idxs]
        old_path = os.path.join(OLD_MONTHLY, f"era5_merged_{y}_{m:02d}.nc")
        out_path = os.path.join(NEW_MONTHLY, f"era5_merged_{y}_{m:02d}.nc")
        if not os.path.exists(old_path):
            print(f"  ⚠ {y}-{m:02d}: 旧 monthly 缺失，跳过 merge")
            continue
        old = xr.open_dataset(old_path)
        # 防御：旧文件 valid_time 存在重复时间戳时去重（保留每条唯一时刻首条）
        # 例：2025-02~08 旧文件每时刻重复 2 次（448=224×2），去重后与 new 对齐
        if "valid_time" in old.coords:
            vt_arr = np.asarray(old.valid_time.values, dtype="datetime64[ns]")
            u_vals, first_idx = np.unique(vt_arr, return_index=True)  # first_idx 已按唯一值（时间）顺序
            if len(u_vals) < len(vt_arr):
                old = old.isel(valid_time=first_idx)  # 注意：first_idx 本身有序，勿再 sort
                print(f"  ⚠ {y}-{m:02d}: 旧文件去除 {len(vt_arr)-len(u_vals)} 个重复时间戳")
        new_times = [v0 + timedelta(seconds=int(vt[i])) for i in idxs]
        # 检查与旧文件 valid_time 对齐（datetime64[s] 比较，避免 int 溢出）
        align_ok = True
        if "valid_time" in old.coords:
            old_t_dt = np.asarray(old.valid_time.values, dtype="datetime64[s]")
            new_t_dt = np.array(new_times, dtype="datetime64[s]")
            if len(old_t_dt) != len(new_t_dt) or not np.all(old_t_dt == new_t_dt):
                align_ok = False
        if not align_ok:
            print(f"  ⚠ {y}-{m:02d}: valid_time 不对齐（old={len(old.valid_time)} new={len(new_times)}），"
                  f"跳过该月（全量 hourly 就绪后重跑可对齐）")
            old.close()
            continue
        # 网格方向校验：新 tp 与旧文件 lat/lon 顺序一致（46→40, 117→126）
        new_lat = np.linspace(46.0, 40.0, tp3h_month.shape[1])
        new_lon = np.linspace(117.0, 126.0, tp3h_month.shape[2])
        if "latitude" in old.coords:
            if not np.allclose(np.asarray(old.latitude.values, dtype=float), new_lat) or \
               not np.allclose(np.asarray(old.longitude.values, dtype=float), new_lon):
                print(f"  ⚠ {y}-{m:02d}: 网格坐标与旧文件不一致，尝试按旧坐标对齐")
                # 按旧坐标重排新 tp（处理 lat 方向翻转等）
                old_lat = np.asarray(old.latitude.values, dtype=float)
                old_lon = np.asarray(old.longitude.values, dtype=float)
                if np.allclose(old_lat, new_lat[::-1]):
                    tp3h_month = tp3h_month[:, ::-1, :]
                if np.allclose(old_lon, new_lon[::-1]):
                    tp3h_month = tp3h_month[:, :, ::-1]
        # 替换 tp 通道（保持旧文件其他变量与属性）
        out = old.copy()
        # 数值存 m（与旧文件 units=m 一致）：tp3h_month 目前是 mm 值，转回 m
        out["tp"].values = tp3h_month / 1000.0
        # 更新 tp 元数据：现在已是 3h 累积（旧文件 stepUnits=1 会误导 M1 验收）
        tp_attrs = dict(out["tp"].attrs)
        tp_attrs["GRIB_stepUnits"] = 3
        tp_attrs["GRIB_stepType"] = "accum"
        tp_attrs["cell_methods"] = "time: sum over hours (3-hourly aggregated from hourly tp)"
        tp_attrs["units"] = "m"
        tp_attrs["long_name"] = "Total precipitation (3h accumulation, BUG_LOG B1 rebuilt)"
        out["tp"].attrs = tp_attrs
        out.attrs["history"] = "13.0: tp replaced with hourly-aggregated 3h accumulation (BUG_LOG B1 fix)"
        out.to_netcdf(out_path)
        old.close()
        out.close()
        mean3h = float(np.nanmean(tp3h_month))
        annual = mean3h * 8 * 365
        print(f"  ✅ {y}-{m:02d}: {len(idxs)} 步, 域均 {mean3h:.4f} mm/3h (年估 {annual:.0f}) -> {os.path.basename(out_path)}")
        n_ok += 1
    print(f"完成：{n_ok} 个月已重建并合并。下一步跑 verify_era5_target.py 验收。")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    years = [args.year] if args.year else list(range(args.start, args.end + 1))
    aggregate_all(years)

if __name__ == "__main__":
    main()
