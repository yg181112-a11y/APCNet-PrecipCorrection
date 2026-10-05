# -*- coding: utf-8 -*-
"""
download_era5_tp_hourly.py — 下载 ERA5 逐小时 total_precipitation（辽河流域 40-46N/117-126E）
使用 cdsapi 官方库（wheel 解压至 D:\\liaohe\\ERA5-data\\_cdsapi_lib，sys.path 注入，无需 pip 安装）
输出：D:\\liaohe\\ERA5-data\\new_hourly_tp\\era5_tp_hourly_YYYYMM.nc（24 步/天，1h 累积）

用法：
  python download_era5_tp_hourly.py --year 2024 --month 7      # 单月
  python download_era5_tp_hourly.py --start 2015 --end 2025    # 全量（断点续传）
"""
import argparse, calendar, os, sys, time

CDSAPI_LIB = r"D:\liaohe\ERA5-data\_cdsapi_lib"
sys.path.insert(0, CDSAPI_LIB)
import cdsapi  # noqa: E402

OUT_DIR = r"D:\liaohe\ERA5-data\new_hourly_tp"
LOG_FILE = os.path.join(OUT_DIR, "download_tp.log")
MIN_SIZE = 500_000

def log(msg):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

def build_request(year, month):
    days = [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    return {
        "product_type": "reanalysis",
        "variable": ["total_precipitation"],
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [46, 117, 40, 126],
        "grid": [0.25, 0.25],
        "data_format": "netcdf",
    }

def download_month(year, month, retries=3):
    out_path = os.path.join(OUT_DIR, f"era5_tp_hourly_{year}{month:02d}.nc")
    if os.path.exists(out_path) and os.path.getsize(out_path) > MIN_SIZE:
        log(f"[{year}-{month:02d}] 已存在，跳过")
        return True
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            c = cdsapi.Client(quiet=True, progress=False, delete=False)
            c.retrieve("reanalysis-era5-single-levels", build_request(year, month), out_path)
            sz = os.path.getsize(out_path)
            if sz > MIN_SIZE:
                log(f"[{year}-{month:02d}] ✅ {sz/1e6:.2f} MB ({time.time()-t0:.0f}s)")
                return True
            log(f"[{year}-{month:02d}] ⚠ 文件过小 {sz}，重试")
        except Exception as e:
            log(f"[{year}-{month:02d}] attempt {attempt} 失败: {str(e)[:200]}")
            time.sleep(20 * attempt)
    log(f"[{year}-{month:02d}] ❌ 重试耗尽")
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    ap.add_argument("--month", type=int)
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    if args.year and args.month:
        months = [(args.year, args.month)]
    else:
        months = [(y, m) for y in range(args.start, args.end + 1) for m in range(1, 13)]
    log(f"开始下载 {len(months)} 个月（hourly tp, 2015-2025）")
    ok = 0
    for y, m in months:
        if download_month(y, m):
            ok += 1
    log(f"完成：{ok}/{len(months)} 成功")
    sys.exit(0 if ok == len(months) else 1)

if __name__ == "__main__":
    main()
