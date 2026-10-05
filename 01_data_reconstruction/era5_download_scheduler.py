# -*- coding: utf-8 -*-
"""
era5_download_scheduler.py — ERA5 hourly tp 并发窗口调度器（2015-2025，132 个月）

设计：
  - 并发窗口 WINDOW=3：同时挂起的服务器任务 ≤3，完成一个再补一个（避免 CDS 限流/排队风暴）
  - 异步提交（wait_until_complete=False）→ 状态存 JSON → 周期性轮询 status → successful 后下载
  - 断点续传：状态文件 + 已下载文件（>500KB）双保险，可随时中断重启
  - 超时保护：accepted 超 120 分钟 → 放弃重提（≤3 次）；failed/rejected → 重提（≤3 次）

用法：python era5_download_scheduler.py [--start 2015] [--end 2025] [--window 3]
状态/日志：D:\\liaohe\\ERA5-data\\new_hourly_tp\\{download_state.json, download_tp.log}
"""
import argparse, calendar, json, os, sys, time

CDSAPI_LIB = r"D:\liaohe\ERA5-data\_cdsapi_lib"
sys.path.insert(0, CDSAPI_LIB)
import cdsapi  # noqa: E402

OUT_DIR = r"D:\liaohe\ERA5-data\new_hourly_tp"
STATE_FILE = os.path.join(OUT_DIR, "download_state.json")
LOG_FILE = os.path.join(OUT_DIR, "download_tp.log")
MIN_SIZE = 500_000
STALE_MIN = 120        # 排队超过该分钟数视为卡死，重提
MAX_TRIES = 3

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

def month_key(y, m):
    return f"{y}-{m:02d}"

def out_path(key):
    return os.path.join(OUT_DIR, f"era5_tp_hourly_{key.replace('-', '')}.nc")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--window", type=int, default=3)
    args = ap.parse_args()

    months = [month_key(y, m) for y in range(args.start, args.end + 1) for m in range(1, 13)]
    # 加载状态（旧状态文件里的月份若不在本次范围则忽略）
    state = {}
    if os.path.exists(STATE_FILE):
        try:
            old = json.load(open(STATE_FILE, encoding="utf-8"))
            state = {k: v for k, v in old.items() if k in months}
        except Exception as e:
            log(f"状态文件读取失败（重建）: {e}")

    client = cdsapi.Client(quiet=True, progress=False, delete=False, wait_until_complete=False)
    log(f"调度器启动：{len(months)} 个月，窗口 {args.window}")

    def is_done(key):
        p = out_path(key)
        return os.path.exists(p) and os.path.getsize(p) > MIN_SIZE

    for k in list(state):
        if is_done(k):
            state.pop(k, None)

    def server_active_map():
        """返回 { 'YYYY-MM': request_id }：服务器上 accepted/running 的 ERA5 tp 任务"""
        try:
            jobs = client.client.get_jobs()
            m = {}
            for it in jobs._json_dict.get("jobs", []):
                if it.get("status") not in ("accepted", "running"):
                    continue
                md = it.get("metadata", {})
                ids = md.get("request", {}).get("ids", {})
                y, mo = ids.get("year"), ids.get("month")
                if y and mo:
                    m[f"{y}-{mo}"] = it.get("jobID")
            return m
        except Exception as e:
            log(f"查询服务器任务失败: {str(e)[:120]}")
            return {}

    last_reject = {}   # key -> 上次 rejected 时间

    def submit(key):
        y, m = int(key[:4]), int(key[5:7])
        try:
            remote = client.retrieve("reanalysis-era5-single-levels", build_request(y, m))  # wait=False → Remote
            rid = remote.request_id
            state[key] = {"request_id": rid, "status": "accepted",
                          "submitted": time.time(), "tries": state.get(key, {}).get("tries", 0) + 1}
            log(f"[{key}] 已提交 request_id={rid} (tries={state[key]['tries']})")
            return True
        except Exception as e:
            log(f"[{key}] 提交失败: {str(e)[:160]}")
            state[key] = {"status": "submit_error", "submitted": time.time(),
                          "tries": state.get(key, {}).get("tries", 0) + 1}
            return False

    def download(key):
        try:
            rid = state[key]["request_id"]
            remote = client.client.get_remote(rid)
            remote.download(out_path(key))
            sz = os.path.getsize(out_path(key))
            if sz > MIN_SIZE:
                log(f"[{key}] ✅ 下载完成 {sz/1e6:.2f} MB")
                state.pop(key, None)
                return True
            else:
                log(f"[{key}] ⚠ 文件过小 {sz}，重提")
                state[key] = {"status": "resubmit", "submitted": time.time(),
                              "tries": state.get(key, {}).get("tries", 0) + 1}
                return False
        except Exception as e:
            log(f"[{key}] 下载异常: {str(e)[:160]}")
            state[key] = {"status": "resubmit", "submitted": time.time(),
                          "tries": state.get(key, {}).get("tries", 0) + 1}
            return False

    round_no = 0
    while True:
        round_no += 1
        changed = False
        # 1) 补提交：活跃任务数 < 窗口 且 有未完成任务
        active = [k for k, v in state.items() if v.get("status") in ("accepted", "running")]
        # 清掉已完成的（防御）
        for k in [k for k in state if is_done(k)]:
            state.pop(k, None); changed = True
        # 待提交：未下载完成，且 不在状态 或 处于可重试的错误态（submit_error / resubmit / 未给 up 且 tries 未超）
        def retryable(k):
            st = state.get(k)
            if st is None:
                return True
            if st.get("status") in ("submit_error", "resubmit"):
                return st.get("tries", 0) < MAX_TRIES
            return False
        pending = [k for k in months if not is_done(k) and retryable(k)]
        # 服务器去重：服务器已有同月活跃任务时沿用其 request_id，不重复提交
        srv = server_active_map()
        for k in list(pending):
            if k in srv and k not in active:
                state[k] = {"request_id": srv[k], "status": "accepted",
                            "submitted": time.time(), "tries": state.get(k, {}).get("tries", 0) + 1}
                log(f"[{k}] 服务器已有活跃任务 {srv[k]}，沿用")
                active.append(k); pending.remove(k); changed = True
        for k in pending:
            if len(active) >= args.window:
                break
            if submit(k):
                active.append(k); changed = True
            time.sleep(1)
        # 2) 轮询所有活跃任务
        for k in list(active):
            st = state.get(k, {})
            tries = st.get("tries", 0)
            # 超时保护
            if st.get("status") == "accepted" and time.time() - st.get("submitted", 0) > STALE_MIN * 60:
                log(f"[{k}] 排队超 {STALE_MIN} 分钟（可能卡死），重提")
                state.pop(k, None)
                if tries < MAX_TRIES:
                    if submit(k):
                        changed = True
                else:
                    log(f"[{k}] ❌ 重试耗尽，跳过")
                    state[k] = {"status": "give_up", "tries": tries}
                continue
            try:
                rid = st["request_id"]
                remote = client.client.get_remote(rid)
                status = remote.status
                if status != st.get("status"):
                    state[k]["status"] = status; changed = True
                if status == "successful":
                    log(f"[{k}] 服务器就绪，下载中...")
                    if download(k):
                        changed = True
                elif status in ("failed", "rejected"):
                    # 冷却 5 分钟后重提（避免 rejected→重提→rejected 风暴）
                    now = time.time()
                    if now - last_reject.get(k, 0) < 300:
                        continue
                    last_reject[k] = now
                    log(f"[{k}] 服务器返回 {status}，删除旧任务并重提")
                    try:
                        remote.delete()
                    except Exception:
                        pass
                    state.pop(k, None)
                    if tries < MAX_TRIES:
                        submit(k); changed = True
                    else:
                        state[k] = {"status": "give_up", "tries": tries}
            except Exception as e:
                log(f"[{k}] 轮询异常: {str(e)[:120]}")
        # 3) 保存状态
        json.dump(state, open(STATE_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        # 4) 完成检查
        done = sum(1 for k in months if is_done(k))
        gave = sum(1 for k in state if state[k].get("status") == "give_up")
        if done == len(months):
            log(f"🎉 全部 {len(months)} 个月完成")
            break
        if gave == len(state) and not any(v.get("status") in ("accepted", "running") for v in state.values()):
            log(f"❌ {gave} 个任务全部放弃，停止")
            break
        if round_no % 10 == 1:
            pending_n = len([k for k in months if not is_done(k) and retryable(k)])
            log(f"进度：{done}/{len(months)} 完成，活跃 {len(active)}，待提交 {pending_n}")
        time.sleep(30)

if __name__ == "__main__":
    main()
