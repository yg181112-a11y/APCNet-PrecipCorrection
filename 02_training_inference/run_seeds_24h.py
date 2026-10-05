# -*- coding: utf-8 -*-
"""24h APCNet 种子稳定性 v2：SEED=40/41。
- 中文路径经 base64 传环境变量，避免 GBK 破坏
- 数据从主 24h_exp 读取（DATA_DIR），模型/预测/结果写 seed 子目录（OUT）
"""
import os, subprocess, time, base64

PY = r'C:\ProgramData\anaconda3\envs\dl_5050\python.exe'
SRC = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\train_24h.py'
DST = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\train_24h_seed.py'
CWD = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\24h_exp'
DATA = BASE  # 数据主目录

def b64(s):
    return base64.b64encode(s.encode('utf-8')).decode('ascii')

src = open(SRC, encoding='utf-8').read()
# 头部替换：OUT 由 OUT_WORK_B64 解码；DATA 由 DATA_DIR_B64 解码；seed 设置
old_head = "OUT = r'C:\\Users\\yg181\\Desktop\\论文三\\13.0修复重跑\\24h_exp'"
new_head = (
    "import base64 as _b64\n"
    "OUT = _b64.b64decode(os.environ['OUT_WORK_B64']).decode('utf-8')\n"
    "DATA = _b64.b64decode(os.environ['DATA_DIR_B64']).decode('utf-8')\n"
    "SEED = int(os.environ.get('SEED', '42'))\n"
    "import random\n"
    "random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)\n"
    "if torch.cuda.is_available():\n"
    "    torch.cuda.manual_seed_all(SEED)\n"
    "print(f'SEED={SEED} OUT={OUT}')"
)
assert old_head in src, 'head not found'
src = src.replace(old_head, new_head, 1)
# 数据读取路径 OUT -> DATA（仅 np.load 行）
import re
src = src.replace("np.load(os.path.join(OUT, 'features_24h.npy'))", "np.load(os.path.join(DATA, 'features_24h.npy'))")
src = src.replace("np.load(os.path.join(OUT, 'gfs_24h_accum.npy'))", "np.load(os.path.join(DATA, 'gfs_24h_accum.npy'))")
src = src.replace("np.load(os.path.join(OUT, 'era5_24h_accum.npy'))", "np.load(os.path.join(DATA, 'era5_24h_accum.npy'))")
src = src.replace("np.load(os.path.join(OUT, 'split_mask.npy'))", "np.load(os.path.join(DATA, 'split_mask.npy'))")
open(DST, 'w', encoding='utf-8').write(src)
print('生成 train_24h_seed.py OK (v2)')

seeds = [40, 41]
for s in seeds:
    out = os.path.join(BASE, f'seed{s}')
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    env['SEED'] = str(s)
    env['OUT_WORK_B64'] = b64(out)
    env['DATA_DIR_B64'] = b64(DATA)
    log_path = os.path.join(out, f'run_seed{s}.log')
    print(f'[{time.strftime("%H:%M:%S")}] 启动 SEED={s} ...', flush=True)
    t0 = time.time()
    with open(log_path, 'wb') as f:
        p = subprocess.run([PY, DST], env=env, cwd=CWD, stdout=f, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    print(f'[{time.strftime("%H:%M:%S")}] SEED={s} exit={p.returncode} 耗时 {dt:.1f} min -> {log_path}', flush=True)

print('ALL SEEDS DONE', flush=True)
