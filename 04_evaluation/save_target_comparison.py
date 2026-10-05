# -*- coding: utf-8 -*-
"""汇总训练目标对照实验结果 -> target_comparison.json"""
import json, os
E = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\24h_exp'
out = {
    'experiment': 'GPM观测目标 vs ERA5再分析目标 对照训练（24h 累积, U-Net 同构同掩码同loss）',
    'loss': '纯 MSE（无 storm_weight；GPM 目标下 storm_weight 已被证明有害: weighted +23.1% -> -8.5%）',
    'coverage': '828/925 格点（GPM 共同覆盖）',
    'split': 'train 2019-2021 / val 2022-2023 / test 2024-2025（按 init 时刻）',
    'results_00z': {
        'n_test': 639,
        'GFS': {'RMSE': 5.298, 'CC': 0.697, 'bias': -0.11, 'mse_improve': 0.0},
        'U-Net(GPM目标, weighted)': {'RMSE': 5.518, 'CC': 0.741, 'bias': 1.27, 'mse_improve': -8.5},
        'U-Net(GPM目标, no-weight)': {'RMSE': 4.647, 'CC': 0.767, 'bias': 0.44, 'mse_improve': 23.1},
        'U-Net(ERA5目标, no-weight)': {'RMSE': 4.674, 'CC': 0.754, 'bias': -0.22, 'mse_improve': 22.2},
        'U-Net(ERA5目标, weighted, 主实验)': {'RMSE': 4.786, 'CC': 0.746, 'bias': 0.13, 'mse_improve': 18.4},
        'APCNet(ERA5目标, 主实验)': {'RMSE': 4.914, 'CC': 0.728, 'bias': 0.17, 'mse_improve': 14.0},
    },
    'results_allinit': {
        'n_test': 2549,
        'GFS': {'RMSE': 5.281, 'CC': 0.706, 'bias': -0.17, 'mse_improve': 0.0},
        'U-Net(GPM目标)': {'RMSE': 4.730, 'CC': 0.757, 'bias': 0.38, 'mse_improve': 19.8},
        'U-Net(ERA5目标)': {'RMSE': 4.675, 'CC': 0.762, 'bias': -0.20, 'mse_improve': 21.7},
    },
    'conclusion': ('24h 累积尺度上, 观测目标(GPM)与再分析目标(ERA5)在 GPM 独立验证上几乎等价'
                   '(全体init: +19.8% vs +21.7%, 差1.9pp; 00Z: +23.1% vs +22.2%)。'
                   '训练参照并非 24h DL 订正技巧的主要决定因素; 多时效反转由尺度可预测性/信噪比驱动。'
                   '该对照排除了"模型仅过拟合ERA5"的质疑, 证明 24h+ DL 技巧为真实可预测性收益。'
                   '附发现: storm_weight 强度加权对高噪声观测目标(GPM)有害(00Z GPM: -8.5% -> +23.1% 当去除权重)。'),
}
with open(os.path.join(E, 'target_comparison.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('已保存 target_comparison.json')
