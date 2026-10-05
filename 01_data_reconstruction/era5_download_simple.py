# -*- coding: utf-8 -*-
"""
era5_download_simple.py — ERA5 逐小时总降水（total_precipitation）独立下载脚本
=====================================================================
可手动运行：下载东北区域（40-46N, 117-126E）0.25° 网格、指定年份的逐小时降水。

用法（任选其一）：
  python era5_download_simple.py --year 2020              # 下载 2020 全年
  python era5_download_simple.py --start 2015 --end 2017  # 下载 2015-2017
  python era5_download_simple.py --year 2020 --month 07   # 只下载 2020 年 7 月

输出：D:\\liaohe\\ERA5-data\\new_hourly_tp\\era5_tp_hourly_{YYYY}.nc
     （整年文件约 8-15 MB；每月一个文件的体积约 0.6-1.3 MB）

依赖：
  1. cdsapi（Python 库）
     - 已有本地解压库：D:\\liaohe\\ERA5-data\\_cdsapi_lib（脚本自动注入）
     - 或自行安装：pip install cdsapi
  2. 凭据文件 C:\\Users\\yg181\\.cdsapirc（已配置，无需改动）

注意：
  - CDS 请求是"提交-排队-处理-下载"模式，高峰期可能排队几十分钟。
  - 若提交后长时间卡在 accepted（>30 分钟无进展），请删除该任务后重跑本脚本。
  - 本脚本是"跑完即退出"的一次性版本；持续自动重试请用 era5_download_yearly.py。
"""
import argparse, json, os, sys, time
from datetime import datetime, timezone

# 若系统未安装 cdsapi，注入本地解压库
try:
    import cdsapi
except ImportError:
    sys.path.insert(0, r"D:\liaohe\ERA5-data\_cdsapi_lib")
    import cdsapi  # noqa: E402

OUT_DIR = r"D:\liaohe\ERA5-data\new_hourly_tp"
MIN_SIZE = 5_000_000  # 整年 hourly tp 约 8-15MB


def build_request(year, months=None):
    return {
        "product_type": "reanalysis",
        "variable": ["total_precipitation"],
        "year": str(year),
        "month": [f"{m:02d}" for m in (months or range(1, 13))],
        "day": [f"{d:02d}" for d in range(1, 32)],  # CDS 自动忽略无效日期
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [46, 117, 40, 126],   # N W S E（东北区域）
        "grid": [0.25, 0.25],
        "data_format": "netcdf",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, help="下载单一年份")
    ap.add_argument("--start", type=int, help="起始年份（与 --end 连用）")
    ap.add_argument("--end", type=int, default=None, help="结束年份（与 --start 连用）")
    ap.add_argument("--month", type=int, default=None, help="只下载某个月份（1-12）")
    args = ap.parse_args()

    if args.year:
        years = [args.year]
    elif args.start:
        years = list(range(args.start, args.end or args.start + 1))
    else:
        ap.error("请提供 --year 或 --start/--end")

    os.makedirs(OUT_DIR, exist_ok=True)
    client = cdsapi.Client(quiet=True, progress=False, delete=False, wait_until_complete=False)

    for y in years:
        out_path = os.path.join(OUT_DIR, f"era5_tp_hourly_{y}.nc")
        if os.path.exists(out_path) and os.path.getsize(out_path) > MIN_SIZE:
            print(f"[{y}] 已存在，跳过")
            continue

        print(f"[{y}] 提交请求……")
        req = build_request(y, [args.month] if args.month else None)
        r = client.retrieve("reanalysis-era5-single-levels", req)
        print(f"[{y}] 请求 ID: {r.request_id}，等待处理（排队可能 10-60 分钟）……")

        # 轮询（最多 3 小时；卡死超 30 分钟则删除重提）
        last_update = time.time()
        while True:
            try:
                status = r.status
                j = r.json
                updated = j.get("updated") or j.get("created")
                if updated:
                    t_up = datetime.strptime(updated[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
                    last_update = t_up
            except Exception as e:
                print(f"[{y}] 轮询异常: {str(e)[:80]}，30 秒后重试")
                time.sleep(30); continue

            print(f"[{y}] 状态: {status}")

            if status == "successful":
                break
            if status in ("failed", "rejected"):
                print(f"[{y}] {status}，删除并退出（请稍后重跑）")
                try: r.delete()
                except Exception: pass
                sys.exit(1)
            if status == "accepted" and time.time() - last_update > 1800:
                print(f"[{y}] 卡死超 30 分钟，删除重提（仅一次）")
                r.delete()
                r = client.retrieve("reanalysis-era5-single-levels", req)
                print(f"[{y}] 新请求 ID: {r.request_id}")
                last_update = time.time()
                time.sleep(60); continue
            time.sleep(30)

        print(f"[{y}] 开始下载……")
        r.download(out_path)
        sz = os.path.getsize(out_path)
        if sz > MIN_SIZE:
            print(f"[{y}] ✅ 完成 {sz/1e6:.2f} MB -> {out_path}")
        else:
            print(f"[{y}] ⚠ 文件异常小 ({sz} B)，请删除重跑")


if __name__ == "__main__":
    main()
