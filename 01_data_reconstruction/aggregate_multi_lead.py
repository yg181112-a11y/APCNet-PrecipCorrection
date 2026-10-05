# -*- coding: utf-8 -*-
"""汇总多时效权威数字 → r3_media/multi_lead_summary.json（R3 稿写作直接引用）"""
import json, os, numpy as np

BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
OUTD = r'C:\Users\yg181\Desktop\论文三\WAF\r3_media'
os.makedirs(OUTD, exist_ok=True)

S = {}

# 1) ERA5 参照：MSE 改进（% vs GFS）
# 3h（run13 权威）
S['era5_3h'] = {
    'gfs_mse': 1.3462, 'apcnet_imp': 14.48, 'unet_imp': -55.1,
    'qm_imp': -11.1, 'ols_imp': -22.2, 'bincm_imp': -27.3,
}
# 多时效（来自 24h_results.json / 72h/120h *_results.json 的 mse）
def load(fhr):
    d = os.path.join(BASE, 'multi_lead_exp' if fhr != 24 else '24h_exp')
    if fhr == 24:
        r = json.load(open(os.path.join(BASE, '24h_exp', '24h_results.json'), encoding='utf-8'))
        u = json.load(open(os.path.join(BASE, '24h_exp', 'unet_24h_results.json'), encoding='utf-8'))
    else:
        d2 = os.path.join(d, f'{fhr}h')
        r = json.load(open(os.path.join(d2, f'{fhr}h_results.json'), encoding='utf-8'))
        u = json.load(open(os.path.join(d2, f'unet_{fhr}h_results.json'), encoding='utf-8'))
    g = r['gfs_raw']['mse']
    def imp(m):
        return (m / g - 1) * 100
    return dict(
        gfs_mse=g, gfs_rmse=float(np.sqrt(g)), gfs_cc=r['gfs_raw']['cc'],
        apcnet_mse=r['apcnet']['mse'], apcnet_imp=imp(r['apcnet']['mse']),
        apcnet_cc=r['apcnet']['cc'], apcnet_ets10=r['apcnet']['ets_10'],
        unet_mse=u['unet_%dh' % fhr]['mse'], unet_imp=imp(u['unet_%dh' % fhr]['mse']),
        unet_cc=u['unet_%dh' % fhr]['cc'],
        qm_mse=r['qm']['mse'], qm_imp=imp(r['qm']['mse']),
        bm_mse=r['bm']['mse'], bm_imp=imp(r['bm']['mse']),
        ols_mse=r['ols']['mse'], ols_imp=imp(r['ols']['mse']),
    )
for fhr in (24, 72, 120):
    S[f'era5_{fhr}h'] = load(fhr)

# 2) bootstrap CI（月块）
bs = json.load(open(os.path.join(BASE, 'multi_lead_exp', 'bootstrap_summary.json'), encoding='utf-8'))
S['bootstrap'] = bs

# 3) CHM 独立验证（含 U-Net）
for fhr in (24, 72, 120):
    p = os.path.join(BASE, 'multi_lead_exp' if fhr != 24 else '24h_exp', f'chm_{fhr}h_eval.json')
    r = json.load(open(p, encoding='utf-8'))
    S[f'chm_{fhr}h'] = {
        'ndays': r['n_common_days'], 'gfs_rmse': r['gfs']['RMSE'],
        'apcnet_imp': r['rmse_improve_apc'], 'unet_imp': r['rmse_improve_unet'],
        'era5_ref_imp': r['rmse_improve_era5_ref'],
        'apcnet_cc': r['apc']['CC'], 'unet_cc': r['unet']['CC'], 'era5_cc': r['era5_ref']['CC'],
        'apcnet_bias': r['apc']['bias'], 'unet_bias': r['unet']['bias'],
    }

with open(os.path.join(OUTD, 'multi_lead_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(S, f, ensure_ascii=False, indent=2)
print('written:', os.path.join(OUTD, 'multi_lead_summary.json'))
# 打印核对
for k, v in S.items():
    if k.startswith('era5_') and 'apcnet_cc' in v:
        print(k, 'APCNet %+.1f | U-Net %+.1f | QM %+.1f | CC(apc/unet)=%.3f/%.3f' % (
            v['apcnet_imp'], v['unet_imp'], v['qm_imp'], v['apcnet_cc'], v['unet_cc']))
    elif k.startswith('chm_'):
        print(k, 'APC %+.2f%% U %+.2f%% E5 %+.2f%% | CC a/u/e=%.3f/%.3f/%.3f' % (
            v['apcnet_imp'], v['unet_imp'], v['era5_ref_imp'], v['apcnet_cc'], v['unet_cc'], v['era5_cc']))
print('bootstrap keys:', list(bs.keys()))
