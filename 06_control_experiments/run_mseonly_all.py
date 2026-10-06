# -*- coding: utf-8 -*-
"""重启后一键运行 M1 纯 MSE 受控实验：4 档(identity/0.5/1.16/2.0) × 3 种子(42/40/41) = 12 个训练。
日志追加到 controlled_exp/run_mseonly_all.log；每个训练结果落 res_<ns>_s<seed>_mseonly.json。"""
import subprocess, os

PY = r'C:\ProgramData\anaconda3\envs\dl_5050\python.exe'
SCRIPT = r'D:\liaohe\论文三\03_重建成稿代_2026_R3全链主实验\controlled_experiment_mseonly.py'
OUT = r'D:\liaohe\校正优化过程\第三阶段\12优化\controlled_exp'
LOG = os.path.join(OUT, 'run_mseonly_all.log')

combos = [(ns, s) for ns in ('identity',) for s in (42, 40, 41)] + \
         [(ns, s) for ns in ('0.5', '1.16', '2.0') for s in (42, 40, 41)]

os.makedirs(OUT, exist_ok=True)
with open(LOG, 'a', encoding='utf-8') as lg:
    for ns, seed in combos:
        env = os.environ.copy()
        env['NOISE_STD'] = ns
        env['SEED'] = str(seed)
        print(f'===== MSE-only {ns} seed {seed} =====', flush=True)
        r = subprocess.run([PY, SCRIPT], env=env, capture_output=True, text=True, timeout=7200)
        lg.write(f'===== MSE-only {ns} seed {seed} rc={r.returncode} =====\n')
        lg.write(r.stdout)
        lg.write('--- stderr ---\n' + r.stderr[-2000:] + '\n\n')
        lg.flush()
        print(r.stdout[-1200:], flush=True)
        if r.returncode != 0:
            print('!!! rc=', r.returncode, r.stderr[-800:], flush=True)
print('ALL DONE. 12 runs finished. Log:', LOG)
