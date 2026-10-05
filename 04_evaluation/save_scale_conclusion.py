# -*- coding: utf-8 -*-
"""整合目标对照 + QQ + 季节 -> 3h/24h 目标对照总汇 JSON"""
import json, os
BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
E24 = os.path.join(BASE, '24h_exp')
E3 = os.path.join(BASE, '3h_exp')

gpm3h_ols = json.load(open(os.path.join(E3, 'gpm3h_ols_eval.json'), encoding='utf-8'))
tc = json.load(open(os.path.join(E24, 'target_comparison.json'), encoding='utf-8'))

out = {
    'experiment': '训练目标对照（GPM 观测 vs ERA5 再分析）× 尺度（3h/24h）— 因果归因完整版',
    'summary': (
        '3h: 线性基线在两种目标下均为正收益(OLS-ERA5目标 +19.6%, OLS-GPM目标 +15.4%, 差4pp), '
        'DL 在 3h 大败(APCNet -29.3%, U-Net -47%) -> 3h DL 失败源于模型复杂度×低信噪比, 非训练参照。'
        '24h: DL 大幅改进(ERA5目标 +18.4~21.7%, GPM目标 +19.8~23.1%), 目标影响<2pp; '
        'QM 近无效(+1~3%)。受控实验(目标=GFS+σ噪声, σ=1.16 -> -27.2%)佐证 3h 失败机制。'
        '结论: 尺度反转(DL 技巧 3h 恶化 -> 24h+ 大幅提升)是稳健的、对训练参照不敏感的事实; '
        '驱动力为尺度可预测性/信噪比, 而非 ERA5 平滑。观测目标不改变结论, 排除"仅过拟合ERA5"质疑。'
    ),
    'scale_3h': {
        'n_train': gpm3h_ols['n_train'], 'n_test': gpm3h_ols['n_test'],
        'cov': gpm3h_ols['cov'],
        'GFS': {'RMSE': gpm3h_ols['GFS']['RMSE'], 'CC': gpm3h_ols['GFS']['CC'], 'bias': gpm3h_ols['GFS']['bias']},
        'OLS_ERA5_target': {'mse_improve_pct': 19.63, 'note': '主实验 split 训练(2015-2021), gpm3h_eval.json'},
        'OLS_GPM_target': {'RMSE': gpm3h_ols['OLS_GPM_target']['RMSE'], 'CC': gpm3h_ols['OLS_GPM_target']['CC'],
                           'bias': gpm3h_ols['OLS_GPM_target']['bias'], 'mse_improve_pct': gpm3h_ols['OLS_GPM_target']['mse_improve_pct'],
                           'note': '训练期 2018-2023(GPM 覆盖), slope_mean=%.3f' % gpm3h_ols['slope_mean']},
        'QM': {'mse_improve_pct': 8.25},
        'APCNet_ERA5_target': {'mse_improve_pct': -29.31},
        'U-Net_ERA5_target': {'mse_improve_pct': -47.3, 'note': 'gpm3h_unet_eval'},
        'ERA5_reference': {'mse_improve_pct': 4.60},
    },
    'scale_24h': tc,
    'qq_season': {
        'note': 'fig_qq_24h.pdf/png 已出（00Z 测试期, GPM 真值, 200 分位点）; 季节分解(00Z): '
                'warm JJAS n=244: GPM目标U-Net +23.1%(RMSE 7.07), ERA5目标U-Net +18.7%, APCNet +14.8%, QM +1.6%; '
                'cool n=395: GPM目标 +22.6%, ERA5目标 +16.1%, APCNet +7.4%, QM -1.0%',
        'figs': ['fig_qq_24h.pdf', 'fig_qq_24h.png'],
    },
    'evidence_chain': [
        '受控实验(受控): 目标=GFS+高斯噪声, σ=0 -> CC 0.9984(架构无瓶颈), σ=1.16(实测残差水平) -> -27.2%',
        '3h 线性双目标: OLS(ERA5) +19.6% vs OLS(GPM) +15.4% -> 目标影响 ~4pp',
        '24h 线性: QM +1.3~3.0% 近无效; DL 大幅改进 -> 高可预测性尺度上 DL 的优势',
        '24h DL 双目标: 全体init GPM +19.8% vs ERA5 +21.7% -> 目标影响 <2pp',
        '24h 季节: GPM 目标在暖/冷季均为最优(+23.1/+22.6%)',
        '12Z 鲁棒性: 24h APCNet +12.0%, U-Net +16.6% (与 00Z +14.0/+18.4 一致) -> init 时刻不敏感',
    ],
}
with open(os.path.join(BASE, 'target_scale_conclusion.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('✅ 已保存 target_scale_conclusion.json')
