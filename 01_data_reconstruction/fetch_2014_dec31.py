# -*- coding: utf-8 -*-
"""一次性补下载 2014-12-31 hourly tp（供 2015-01-01 00/03Z 跨年聚合）"""
import sys, os, time
sys.path.insert(0, r"D:\liaohe\ERA5-data\_cdsapi_lib")
import cdsapi

OUT = r"D:\liaohe\ERA5-data\new_hourly_tp\era5_tp_hourly_2014.nc"

c = cdsapi.Client(quiet=True, progress=False, delete=False, wait_until_complete=False)
print("提交 2014-12-31...", flush=True)
r = c.retrieve("reanalysis-era5-single-levels", {
    "product_type": "reanalysis",
    "variable": ["total_precipitation"],
    "year": "2014",
    "month": "12",
    "day": "31",
    "time": [f"{h:02d}:00" for h in range(24)],
    "area": [46, 117, 40, 126],
    "grid": [0.25, 0.25],
    "data_format": "netcdf",
})
rid = r.request_id
print(f"request_id={rid}", flush=True)
# 轮询直到完成（小请求，超时保护 60 分钟）
t0 = time.time()
while time.time() - t0 < 3600:
    st = c.client.get_remote(rid).status
    print(f"  status={st}", flush=True)
    if st == "successful":
        r.download(OUT)
        print(f"完成: {OUT} {os.path.getsize(OUT)/1e6:.2f} MB", flush=True)
        break
    elif st in ("failed", "rejected"):
        print(f"任务 {st}，退出", flush=True)
        break
    time.sleep(60)
else:
    print("超时，未完成", flush=True)
