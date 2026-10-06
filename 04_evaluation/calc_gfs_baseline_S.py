# -*- coding: utf-8 -*-
"""Table S2 补 GFS 基线行（v4 m4）：
在验证窗（2022-2023, ERA5 参考）上计算 raw GFS 的 ETS15/POD15/FAR15、ETS20/POD20/FAR20 与
合成得分 S（S = 3.0*ETS20 + 1.8*POD20 - 0.45*FAR20 + 1.4*ETS15 + 0.9*POD15 - 0.20*FAR15
      - 0.003*L_val - 2.0*max(0, 0.20 - POD20)；GFS 无 L_val → 该项记 '—'，hinge 项按 POD20 实测）。
口径：与主实验 monitor_extreme_event_performance 的阈值定义一致（gfs_base 与 era5 对比，mm/3h）。
用法：等 train_mseonly_select.py 全部结束后运行（复用同一 val_loader，避免与训练争 GPU）。
"""
import sys, os, json
import numpy as np
import torch

MAIN_DIR = r'D:\liaohe\论文三\03_重建成稿代_2026_R3全链主实验'
sys.path.insert(0, MAIN_DIR)
os.environ['PYTHONPATH'] = MAIN_DIR + ';' + os.path.dirname(os.path.abspath(__file__)) + ';' + os.environ.get('PYTHONPATH', '')
import main13 as M

GFS_BASE = r"D:\liaohe\GFS-data\gfs.0p25.2015-2025.f003"
ERA5_BASE = r"D:\liaohe\ERA5-data"

def threshold_metrics(pred, obs, th):
    """与 monitor 一致的 ETS/POD/FAR（rain mask: pred>=th 与 obs>=th）"""
    p = (pred >= th).astype(np.float32)
    o = (obs >= th).astype(np.float32)
    a = float(((p == 1) & (o == 1)).sum())   # hits
    b = float(((p == 1) & (o == 0)).sum())   # false alarms
    c = float(((p == 0) & (o == 1)).sum())   # misses
    pod = a / max(a + c, 1e-9)
    far = b / max(a + b, 1e-9)
    ar = float((p == 1).sum())               # forecast area
    ob = float((o == 1).sum())               # observed area
    n = float(p.size)
    hss_base = (a + b) * (a + c) / max(n, 1e-9)
    ets = (a - hss_base) / max(a + b + c - hss_base, 1e-9)
    return ets, pod, far

def main():
    print('[data] 构建数据集（与训练共享口径）...', flush=True)
    d = M.create_datasets_with_dem(GFS_BASE, ERA5_BASE, dem_tensor=None)
    cfg = M.ultra_fast_training_config()
    def fast_loader(ds, shuffle, drop_last):
        return M.DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle, num_workers=0,
                            pin_memory=True, drop_last=drop_last, collate_fn=M.custom_collate_fn)
    val_loader = fast_loader(d['correction_val'], False, True)
    sf = float(d.get('scaling_factor', 1.0))
    device = M.get_device()
    print(f'val={len(d["correction_val"])} scaling={sf} device={device}', flush=True)

    g15, g20 = [], []
    o15, o20 = [], []
    with torch.no_grad():
        for v_in, v_tar in val_loader:
            v_in = v_in.to(device, non_blocking=True)
            v_tar = v_tar.to(device, non_blocking=True)
            gfs_base = v_in[:, -1, -1:, :, :].cpu().numpy()
            era5_abs = (gfs_base + v_tar.cpu().numpy() / sf)
            # 阈值直接比较原单位（mm/3h）
            g15.append(gfs_base.ravel()); o15.append(era5_abs.ravel())
            g20.append(gfs_base.ravel()); o20.append(era5_abs.ravel())
    g15 = np.concatenate(g15); o15 = np.concatenate(o15)
    g20 = np.concatenate(g20); o20 = np.concatenate(o20)

    e15, p15, f15 = threshold_metrics(g15, o15, 15.0)
    e20, p20, f20 = threshold_metrics(g20, o20, 20.0)
    hinge = max(0.0, 0.20 - p20) * 2.0
    S = (3.0*e20 + 1.8*p20 - 0.45*f20 + 1.4*e15 + 0.9*p15 - 0.20*f15 - hinge)
    print('GFS baseline (validation window 2022-2023, ERA5 ref):')
    print(f'  ETS20={e20:.4f} POD20={p20:.4f} FAR20={f20:.4f}')
    print(f'  ETS15={e15:.4f} POD15={p15:.4f} FAR15={f15:.4f}')
    print(f'  hinge_term={hinge:.4f}  S={S:.4f}   (L_val: n/a)')
    out = {'gfs_ets20': e20, 'gfs_pod20': p20, 'gfs_far20': f20,
           'gfs_ets15': e15, 'gfs_pod15': p15, 'gfs_far15': f15,
           'hinge_term': hinge, 'S': S, 'val_n': len(d['correction_val'])}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gfs_baseline_S.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('saved gfs_baseline_S.json')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
