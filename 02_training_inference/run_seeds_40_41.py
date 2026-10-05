# -*- coding: utf-8 -*-
"""3-seeds 补跑 runner：SEED=40/41 顺序跑 13.0_main.py（FIX-12 残差正则版），
OUT_WORK 各自归档产物，日志落盘 seed{}/run_seed{}.log。
SEED=42 已有当前 manuscript_work 产物，不重跑。
"""
import os
import subprocess
import sys
import time

PY = r'C:\ProgramData\anaconda3\envs\dl_5050\python.exe'
MAIN = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\13.0_main.py'
CWD = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
BASE = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'

seeds = [40, 41]
for s in seeds:
    out = os.path.join(BASE, f'seed{s}')
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    env['SEED'] = str(s)
    env['OUT_WORK'] = out
    log_path = os.path.join(out, f'run_seed{s}.log')
    print(f'[{time.strftime("%H:%M:%S")}] 启动 SEED={s} ...', flush=True)
    t0 = time.time()
    with open(log_path, 'wb') as f:
        p = subprocess.run([PY, MAIN], env=env, cwd=CWD,
                           stdout=f, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    print(f'[{time.strftime("%H:%M:%S")}] SEED={s} exit={p.returncode} 耗时 {dt:.1f} min -> {log_path}', flush=True)

print('ALL SEEDS DONE', flush=True)
