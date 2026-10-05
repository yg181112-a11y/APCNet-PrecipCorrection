# -*- coding: utf-8 -*-
"""
era5_download_yearly.py — ERA5 hourly tp 按年串行下载器（2015-2025，11 个请求）

策略（CDS 高峰拥堵 + 排队上限~2 的应对）：
  - 每年一个请求（year + 全部 month），132 个月 → 11 个请求
  - 严格串行：同一时刻服务器只挂 1 个任务
  - rejected/failed → 冷却 15 分钟重试（≤4 次）
  - 断点续传：整年文件存在（>5MB）跳过；状态存 JSON
  - 心跳日志

输出：D:\\liaohe\\ERA5-data\\new_hourly_tp\\era5_tp_hourly_{YYYY}.nc（全年 hourly，1h 累积）
用法：python era5_download_yearly.py [--start 2015] [--end 2025]
"""
import argparse, calendar, json, os, sys, time
from datetime import datetime, timezone

sys.path.insert(0, r"D:\liaohe\ERA5-data\_cdsapi_lib")
import cdsapi  # noqa: E402

OUT_DIR = r"D:\liaohe\ERA5-data\new_hourly_tp"
STATE_FILE = os.path.join(OUT_DIR, "download_year_state.json")
LOG_FILE = os.path.join(OUT_DIR, "download_tp.log")
MIN_SIZE = 4_000_000     # 整年 hourly tp 4.6-5.8MB（2017 完整文件实测 4.66MB）     # 整年 hourly tp 约 8-15MB
COOLDOWN_REJECT = 900
MAX_TRIES = 4
HEARTBEAT = 600

def log(msg):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

def build_request(year):
    return {
        "product_type": "reanalysis",
        "variable": ["total_precipitation"],
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],  # CDS 自动忽略无效日期（如 2-30）
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [46, 117, 40, 126],
        "grid": [0.25, 0.25],
        "data_format": "netcdf",
    }

def out_path(y):
    return os.path.join(OUT_DIR, f"era5_tp_hourly_{y}.nc")

def is_done(y):
    p = out_path(y)
    return os.path.exists(p) and os.path.getsize(p) > MIN_SIZE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    years = list(range(args.start, args.end + 1))

    state = {}
    if os.path.exists(STATE_FILE):
        try:
            old = json.load(open(STATE_FILE, encoding="utf-8"))
            state = {k: v for k, v in old.items() if int(k) in years and not is_done(int(k))}
        except Exception as e:
            log(f"状态读取失败（重建）: {e}")

    client = cdsapi.Client(quiet=True, progress=False, delete=False, wait_until_complete=False)
    last_beat = time.time()
    log(f"按年下载器启动：{len(years)} 年（窗口=1，冷却={COOLDOWN_REJECT}s）")

    while True:
        current = None
        for y in years:
            if is_done(y):
                state.pop(str(y), None); continue
            st = state.get(str(y))
            if st and st.get("status") in ("accepted", "running"):
                current = y; break
        if current is None:
            for y in years:
                if is_done(y): continue
                st = state.get(str(y))
                tries = (st or {}).get("tries", 0)
                revive = (st or {}).get("revive_at", 0)
                if st is None or (st.get("status") in ("submit_error", "resubmit") and tries < MAX_TRIES) \
                        or (st.get("status") == "give_up" and time.time() >= revive):
                    current = y; break
        if current is None:
            done = sum(1 for y in years if is_done(y))
            gave = sum(1 for k, v in state.items() if v.get("status") == "give_up")
            if done == len(years):
                log("🎉 全部年份完成")
            else:
                log(f"❌ 无可执行任务：完成 {done}/{len(years)}，放弃 {gave}")
            break

        key = str(current)
        st = state.get(key) or {}
        tries = st.get("tries", 0)
        rid = st.get("request_id")

        if not rid:
            if st.get("status") == "resubmit" and time.time() - st.get("submitted", 0) < COOLDOWN_REJECT:
                time.sleep(30); continue
            try:
                r = client.retrieve("reanalysis-era5-single-levels", build_request(current))
                rid = r.request_id
                state[key] = {"request_id": rid, "status": "accepted",
                              "submitted": time.time(), "tries": tries + 1}
                log(f"[{key}] 已提交 {rid} (tries={tries+1})")
            except Exception as e:
                log(f"[{key}] 提交失败: {str(e)[:140]}")
                state[key] = {"status": "submit_error", "submitted": time.time(), "tries": tries + 1}
                json.dump(state, open(STATE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                time.sleep(60); continue

        try:
            r = client.client.get_remote(rid)
            status = r.status
        except Exception as e:
            log(f"[{key}] 轮询异常: {str(e)[:120]}（30s 后重试）")
            time.sleep(30); continue

        # 悬挂保护：accepted 2 小时无 updated 变化 → 删除重提
        # （实测 CDS 任务排队处理耗时 6 秒~86 分钟不等，30 分钟删除会浪费队列位置）
        if status == "accepted":
            try:
                j = r.json
                created = j.get("created")
                updated = j.get("updated") or j.get("created")
                if created:
                    t_created = datetime.strptime(created[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
                    t_updated = datetime.strptime(updated[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
                    stale = time.time() - max(t_created, t_updated)
                    if stale > 7200:
                        log(f"[{key}] 任务 2 小时无推进（created/updated={str(created)[:19]}），删除重提")
                        r.delete()
                        if tries + 1 >= MAX_TRIES:
                            state[key] = {"status": "give_up", "revive_at": time.time() + 7200, "tries": tries + 1}
                        else:
                            state[key] = {"status": "resubmit", "submitted": time.time(), "tries": tries + 1}
                        json.dump(state, open(STATE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                        time.sleep(60)
                        continue
            except Exception:
                pass

        if status == "successful":
            try:
                r.download(out_path(current))
                sz = os.path.getsize(out_path(current))
                if sz > MIN_SIZE:
                    log(f"[{key}] ✅ 下载完成 {sz/1e6:.2f} MB")
                    state.pop(key, None)
                else:
                    log(f"[{key}] ⚠ 文件过小 {sz}，重试")
                    state[key] = {"status": "resubmit", "submitted": time.time(), "tries": tries + 1}
            except Exception as e:
                log(f"[{key}] 下载异常: {str(e)[:140]}")
                state[key] = {"status": "resubmit", "submitted": time.time(), "tries": tries + 1}
        elif status in ("failed", "rejected"):
            log(f"[{key}] {status}，冷却 {COOLDOWN_REJECT}s 后重试（tries={tries+1}）")
            try:
                r.delete()
            except Exception:
                pass
            # CDS 故障期不永久放弃：重试达上限后进入 give_up，2 小时后自动复活
            if tries + 1 >= MAX_TRIES:
                state[key] = {"status": "give_up", "revive_at": time.time() + 7200, "tries": tries + 1}
            else:
                state[key] = {"status": "resubmit", "submitted": time.time(), "tries": tries + 1}
        else:
            state[key] = {"request_id": rid, "status": status, "submitted": time.time(), "tries": tries}
            if time.time() - last_beat > HEARTBEAT:
                done = sum(1 for y in years if is_done(y))
                log(f"心跳：{done}/{len(years)} 年完成，当前 [{key}] {status}")
                last_beat = time.time()

        json.dump(state, open(STATE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        time.sleep(30)

if __name__ == "__main__":
    main()
