# -*- coding: utf-8 -*-
"""
era5_download_guarded.py — 下载器守护包装器（根治"启动即崩溃"问题）

问题：era5_download_yearly.py 在轮询/下载阶段偶发未捕获异常导致进程退出，
      状态文件中 accepted 任务已从服务器消失时，重启后轮询失效任务再次崩溃，
      形成"看门狗反复拉起-反复崩溃"循环（已发生 2 次，各致 3 小时停机）。

方案：本脚本常驻运行，将下载器作为子进程启动：
  1. 子进程退出（任何原因）→ 自动清理状态文件中失效的 accepted/running
     任务（改为 resubmit、清除 request_id）→ 5 秒后重启子进程
  2. 永不退出（除非手动终止）

用法：python era5_download_guarded.py
注意：启用本脚本后请停用计划任务 ERA5DownloadWatchdog，避免双重拉起。
"""
import json, os, subprocess, sys, time, traceback

PY = sys.executable
SCRIPT = r"C:\Users\yg181\Desktop\论文三\13.0修复重跑\era5_download_yearly.py"
STATE_FILE = r"D:\liaohe\ERA5-data\new_hourly_tp\download_year_state.json"
LOG_FILE = r"D:\liaohe\ERA5-data\new_hourly_tp\guarded.log"


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)


def cleanup_stale():
    """把 accepted/running 状态重置为 resubmit（任务可能已从服务器消失）。"""
    try:
        if not os.path.exists(STATE_FILE):
            return
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            st = json.load(f)
        changed = False
        for k, v in st.items():
            if isinstance(v, dict) and v.get("status") in ("accepted", "running"):
                v["status"] = "resubmit"
                v.pop("request_id", None)
                changed = True
        if changed:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False, indent=1)
            log("已重置失效任务 accepted/running -> resubmit")
    except Exception:
        log("cleanup_stale 异常: " + traceback.format_exc()[-300:])


def main():
    log("守护启动，目标下载器: " + SCRIPT)
    while True:
        try:
            cleanup_stale()
            log("启动下载器...")
            p = subprocess.run([PY, SCRIPT])
            log(f"下载器退出，returncode={p.returncode}")
        except Exception:
            log("守护循环异常: " + traceback.format_exc()[-300:])
        time.sleep(5)


if __name__ == "__main__":
    main()
