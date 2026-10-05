# -*- coding: utf-8 -*-
"""
era5_download_serial.py — ERA5 hourly tp 严格串行下载器（2015-2025，132 个月）

设计（吸取 rejected 教训）：
  - 同一时刻服务器上只挂 1 个任务（串行）：提交 → 轮询到 successful → 下载 → 下一个
  - rejected/failed → 冷却 15 分钟再重试（≤4 次）
  - 提交前查服务器，已有同月 accepted 任务则沿用（防重复）
  - 断点续传：已下载文件（>500KB）跳过；状态存 JSON
  - 心跳日志每 10 分钟一条

用法：python era5_download_serial.py [--start 2015] [--end 2025]
"""
import argparse, calendar, json, os, sys, time

sys.path.insert(0, r"D:\liaohe\ERA5-data\_cdsapi_lib")
import cdsapi  # noqa: E402

OUT_DIR = r"D:\liaohe\ERA5-data\new_hourly_tp"
STATE_FILE = os.path.join(OUT_DIR, "download_state.json")
LOG_FILE = os.path.join(OUT_DIR, "download_tp.log")
MIN_SIZE = 500_000
COOLDOWN_REJECT = 900     # rejected 后冷却 15 分钟
MAX_TRIES = 4
HEARTBEAT = 600            # 心跳日志间隔（秒）

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

def out_path(key):
    return os.path.join(OUT_DIR, f"era5_tp_hourly_{key.replace('-', '')}.nc")

def is_done(key):
    p = out_path(key)
    return os.path.exists(p) and os.path.getsize(p) > MIN_SIZE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    months = [f"{y}-{m:02d}" for y in range(args.start, args.end + 1) for m in range(1, 13)]

    state = {}
    if os.path.exists(STATE_FILE):
        try:
            old = json.load(open(STATE_FILE, encoding="utf-8"))
            state = {k: v for k, v in old.items() if k in months and not is_done(k)}
        except Exception as e:
            log(f"状态读取失败（重建）: {e}")

    client = cdsapi.Client(quiet=True, progress=False, delete=False, wait_until_complete=False)
    last_beat = time.time()
    log(f"串行下载器启动：{len(months)} 个月（窗口=1，冷却={COOLDOWN_REJECT}s）")

    while True:
        # 找下一个未完成且未在进行的月份
        current = None
        for k in months:
            if is_done(k):
                state.pop(k, None); continue
            st = state.get(k)
            if st and st.get("status") in ("accepted", "running"):
                current = k; break
        if current is None:
            for k in months:
                if is_done(k): continue
                st = state.get(k)
                tries = (st or {}).get("tries", 0)
                if st is None:
                    current = k; break
                if st.get("status") in ("submit_error", "resubmit") and tries < MAX_TRIES:
                    current = k; break
        if current is None:
            done = sum(1 for k in months if is_done(k))
            gave = sum(1 for k in state if state[k].get("status") == "give_up")
            if done == len(months):
                log("🎉 全部月份完成")
            else:
                log(f"❌ 无可执行任务：完成 {done}/{len(months)}，放弃 {gave}（剩余均达重试上限）")
            break

        st = state.get(current) or {}
        tries = st.get("tries", 0)
        rid = st.get("request_id")

        if not rid:
            # 提交新任务
            if time.time() - st.get("submitted", 0) < COOLDOWN_REJECT and st.get("status") == "resubmit":
                time.sleep(30); continue
            # 防重复：查服务器同月任务
            try:
                jobs = client.client.get_jobs()
                for it in jobs._json_dict.get("jobs", []):
                    if it.get("status") not in ("accepted", "running"): continue
                    ids = it.get("metadata", {}).get("request", {}).get("ids", {})
                    if f"{ids.get('year')}-{ids.get('month')}" == current:
                        rid = it.get("jobID"); break
            except Exception:
                pass
            if not rid:
                try:
                    r = client.retrieve("reanalysis-era5-single-levels", build_request(int(current[:4]), int(current[5:7])))
                    rid = r.request_id
                    state[current] = {"request_id": rid, "status": "accepted",
                                      "submitted": time.time(), "tries": tries + 1}
                    log(f"[{current}] 已提交 {rid} (tries={tries+1})")
                except Exception as e:
                    log(f"[{current}] 提交失败: {str(e)[:140]}")
                    state[current] = {"status": "submit_error", "submitted": time.time(), "tries": tries + 1}
                    json.dump(state, open(STATE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                    time.sleep(60); continue
            else:
                state[current] = {"request_id": rid, "status": "accepted", "submitted": time.time(), "tries": tries}
                log(f"[{current}] 沿用服务器任务 {rid}")

        # 轮询当前任务
        try:
            r = client.client.get_remote(rid)
            status = r.status
        except Exception as e:
            log(f"[{current}] 轮询异常: {str(e)[:120]}（30s 后重试）")
            time.sleep(30); continue

        if status == "successful":
            try:
                r.download(out_path(current))
                sz = os.path.getsize(out_path(current))
                if sz > MIN_SIZE:
                    log(f"[{current}] ✅ 下载完成 {sz/1e6:.2f} MB")
                    state.pop(current, None)
                else:
                    log(f"[{current}] ⚠ 文件过小 {sz}，重试")
                    state[current] = {"status": "resubmit", "submitted": time.time(), "tries": tries + 1}
            except Exception as e:
                log(f"[{current}] 下载异常: {str(e)[:140]}")
                state[current] = {"status": "resubmit", "submitted": time.time(), "tries": tries + 1}
        elif status in ("failed", "rejected"):
            log(f"[{current}] {status}，冷却 {COOLDOWN_REJECT}s 后重试（tries={tries+1}/{MAX_TRIES}）")
            try:
                r.delete()
            except Exception:
                pass
            if tries + 1 >= MAX_TRIES:
                state[current] = {"status": "give_up", "tries": tries + 1}
            else:
                state[current] = {"status": "resubmit", "submitted": time.time(), "tries": tries + 1}
        else:
            state[current] = {"status": status, "submitted": time.time(), "tries": tries}
            if time.time() - last_beat > HEARTBEAT:
                done = sum(1 for k in months if is_done(k))
                log(f"心跳：{done}/{len(months)} 完成，当前 [{current}] {status}")
                last_beat = time.time()

        json.dump(state, open(STATE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        time.sleep(30)

if __name__ == "__main__":
    main()
