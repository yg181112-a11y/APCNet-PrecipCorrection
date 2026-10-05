# -*- coding: utf-8 -*-
import os, subprocess, sys, time
PY = r'C:\ProgramData\anaconda3\envs\dl_5050\python.exe'
SCR = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\controlled_exp_grid.py'
LOG = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\controlled_grid_run.log'
jobs = [('2.0', '24'), ('1.16', '24'), ('0.5', '24')]
with open(LOG, 'a', encoding='utf-8') as f:
    for ns, ep in jobs:
        env = dict(os.environ, NOISE_STD=ns, EPOCHS=ep)
        t0 = time.time()
        f.write(f'\n===== sigma={ns} ep={ep} start {time.strftime("%H:%M:%S")} =====\n')
        f.flush()
        r = subprocess.run([PY, SCR], env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        f.write(r.stdout)
        f.write('--- stderr tail ---\n')
        f.write(r.stderr[-3000:])
        f.write(f'===== sigma={ns} ep={ep} end {time.strftime("%H:%M:%S")} rc={r.returncode} ({time.time()-t0:.0f}s) =====\n')
        f.flush()
print('ALL DONE')
