# -*- coding: utf-8 -*-
"""M1 分支A 最终结果：三种子 impr% + block-bootstrap 95% CI（口径与主实验一致），
并与主实验 S 选点对照，输出 m1_branchA_result.json。"""
import numpy as np, json

D = r'D:\liaohe\_waf_review\m1_diagnostic\mse_select'
data = np.load(D + r'\era5_test_preds.npz')
preds = {42: data['pred42'], 40: data['pred40'], 41: data['pred41']}
target = data['target']; gfs = data['gfs']
n_samples = target.shape[0]

def impr_pct(p, o, g):
    mse_m = float(np.mean((p - o) ** 2)); mse_g = float(np.mean((g - o) ** 2))
    return (mse_g - mse_m) / mse_g * 100.0

def block_boot_ci(p, o, g, n_boot=1000, block=28, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    n_blocks = int(np.ceil(n_samples / block))
    for _ in range(n_boot):
        idx = []
        for _b in range(n_blocks):
            s = int(rng.integers(0, max(1, n_samples - block)))
            idx.append(np.arange(s, min(s + block, n_samples)))
        sel = np.concatenate(idx)[:n_samples]
        vals.append(impr_pct(p[sel], o[sel], g[sel]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

rows = []
for s in (42, 40, 41):
    imp = impr_pct(preds[s], target, gfs)
    lo, hi = block_boot_ci(preds[s], target, gfs)
    rows.append({'seed': s, 'impr_pct': round(imp, 1), 'ci95': [round(lo, 1), round(hi, 1)]})
    print(f"seed{s}: {imp:+.1f}%  CI95 [{lo:+.1f}, {hi:+.1f}]")

means = np.mean([r['impr_pct'] for r in rows])
# 主实验 S 选点对照（ERA5 测试期 3h，三种子）
main = {'42': -14.5, '40': -46.7, '41': -82.8}
main_mean = float(np.mean(list(main.values())))
print(f"\nMSE-selection three-seed mean = {means:+.1f}%  (vs composite-S mean {main_mean:+.1f}%)")
print(f"mean diff = {means - main_mean:+.1f} pct-points")

out = {'mse_select': {'rows': rows, 'three_seed_mean_impr_pct': round(means, 1),
                      'samples': int(n_samples), 'bootstrap': 'block28_n1000_rng42'},
       'composite_S_main': {'42': -14.5, '40': -46.7, '41': -82.8,
                            'three_seed_mean_impr_pct': round(main_mean, 1)},
       'mean_diff_pct_pts': round(means - main_mean, 1)}
with open(D + r'\m1_branchA_result.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("saved m1_branchA_result.json")
