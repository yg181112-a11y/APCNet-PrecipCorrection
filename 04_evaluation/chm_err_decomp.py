# -*- coding: utf-8 -*-
"""C1 补实验：CHM 验证下按观测强度分箱的误差分解。
解释 RMSE +4.95% 但 MAE −11.7% 的矛盾：APCNet 收益集中在大误差样本（强降水），代价为小雨区过报。
输出 chm_err_decomp.json + 打印。
"""
import os, sys, json, pickle
from datetime import timezone
import numpy as np

WORK_DIR = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
CHM_2024 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2024.nc"
CHM_2025 = r"D:\songhuajiang\CHM_PRE_V2\CHM_PRE_V2_daily_2025.nc"
APC = os.path.join(WORK_DIR, '..', 'manuscript_work_sym', 'seed42', 'predictions_apcnet.npy')
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]

sys.path.insert(0, r'C:\Users\yg181\Desktop\论文三\13.0修复重跑')
from chm_newdata_eval import load_arrays, read_chm_daily, regrid_chm_to_main, group_by_day

times, gfs, apc, qm, ols = load_arrays(APC)

# CHM 日值 → 率（mm/h），模型 12h 聚合 → 率
chm_all, chm_lat, chm_lon, chm_dates = [], [], [], []
for p in [CHM_2024, CHM_2025]:
    prec, lat, lon, dates = read_chm_daily(p)
    chm_all.append(prec); chm_lat = lat; chm_lon = lon; chm_dates.extend(dates)
chm_prec = np.concatenate(chm_all, axis=0)
chm_map = {d.date(): regrid_chm_to_main(chm_prec[i], chm_lat, chm_lon)[0] / 24.0 for i, d in enumerate(chm_dates)}

def model_daily(arr):
    dates, accs = group_by_day(times, arr)
    return dates, accs / 12.0  # 12h 总量 → mm/h（FIX B2 后 accs 为总量）

dg, gfs_d = model_daily(gfs)
da, apc_d = model_daily(apc)
dq, qm_d = model_daily(qm)
do, ols_d = model_daily(ols)

# 公共日对齐（CHM 与 GFS 都有的日）
common = sorted(set(dg) & set(chm_map.keys()))
print('公共日:', len(common))
G = np.stack([gfs_d[dg.index(d)] for d in common])
A = np.stack([apc_d[da.index(d)] for d in common])
Q = np.stack([qm_d[dq.index(d)] for d in common])
O = np.stack([ols_d[do.index(d)] for d in common])
C = np.stack([chm_map[d] for d in common])

# 全域展平
Gf = G.ravel(); Af = A.ravel(); Qf = Q.ravel(); Of = O.ravel(); Cf = C.ravel()
print('总样本:', len(Cf))
print('总体: GFS MAE=%.4f RMSE=%.4f | APC MAE=%.4f RMSE=%.4f' % (
    np.mean(np.abs(Gf - Cf)), np.sqrt(np.mean((Gf - Cf) ** 2)),
    np.mean(np.abs(Af - Cf)), np.sqrt(np.mean((Af - Cf) ** 2))))

# 按 CHM 强度分箱
bins = [0.0, 0.02, 0.1, 0.3, 1.0, np.inf]
labels = ['0-0.02', '0.02-0.1', '0.1-0.3', '0.3-1.0', '>=1.0']
res = {'n_days': len(common), 'total': {}}
for nm, P in [('GFS', Gf), ('APCNet', Af), ('QM', Qf), ('OLS', Of)]:
    res['total'][nm] = {'MAE': float(np.mean(np.abs(P - Cf))), 'RMSE': float(np.sqrt(np.mean((P - Cf) ** 2)))}

print('\n=== 按 CHM 观测率强度分箱（mm/h）===')
rows = []
for i in range(len(bins) - 1):
    m = (Cf >= bins[i]) & (Cf < bins[i + 1])
    n = int(m.sum())
    if n == 0:
        continue
    maeg = np.mean(np.abs(Gf[m] - Cf[m])); maeA = np.mean(np.abs(Af[m] - Cf[m]))
    rmseg = np.sqrt(np.mean((Gf[m] - Cf[m]) ** 2)); rmseA = np.sqrt(np.mean((Af[m] - Cf[m]) ** 2))
    rows.append({'bin': labels[i], 'n': n, 'obs_mean': float(Cf[m].mean()),
                 'gfs_mae': float(maeg), 'apc_mae': float(maeA), 'apc_mae_delta': float(maeA - maeg),
                 'gfs_rmse': float(rmseg), 'apc_rmse': float(rmseA), 'apc_rmse_delta': float(rmseA - rmseg)})
    print('%-10s n=%6d obs=%.3f | MAE GFS=%.4f APC=%.4f (Δ%+.4f) | RMSE GFS=%.4f APC=%.4f (Δ%+.4f)' % (
        labels[i], n, Cf[m].mean(), maeg, maeA, maeA - maeg, rmseg, rmseA, rmseA - rmseg))
res['bins'] = rows

# 误差符号分析：APCNet 在哪类样本上改善
sig = np.sign(Cf)
err_g = np.abs(Gf - Cf); err_a = np.abs(Af - Cf)
imp = err_g - err_a  # >0 改善
worse = imp < 0; better = imp > 0
print('\nAPCNet 改善样本 %.1f%% | 恶化样本 %.1f%%' % (100 * better.mean(), 100 * worse.mean()))
print('改善样本平均 |GFS-CHM|=%.4f |APC-CHM|=%.4f | 恶化样本 |GFS-CHM|=%.4f |APC-CHM|=%.4f' % (
    err_g[better].mean(), err_a[better].mean(), err_g[worse].mean(), err_a[worse].mean()))
res['summary'] = {'better_pct': float(better.mean()), 'worse_pct': float(worse.mean()),
                  'better_gfs_err': float(err_g[better].mean()), 'better_apc_err': float(err_a[better].mean()),
                  'worse_gfs_err': float(err_g[worse].mean()), 'worse_apc_err': float(err_a[worse].mean())}

with open(os.path.join(WORK_DIR, 'chm_err_decomp.json'), 'w') as f:
    json.dump(res, f, indent=1, default=float)
print('\n✅ chm_err_decomp.json')
