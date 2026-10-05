# -*- coding: utf-8 -*-
"""
P1a: 强降水加权重训（审稿 R1·2.1 敏感性实验）
- 训练真值仍为 ERA5，仅放大损失中 >=10/>=20mm 像素权重（默认 x2，可 --mult 调整）
- 训练协议与主实验完全一致（12.8修_final.py 的 staged_training_strategy_with_monitoring）
- 输出：p1a 预测 npy + 全套评估（ERA5 真值下）+ 模型权重 + sample_times
用法: python p1a_heavy_train.py [--mult 2.0] [--workdir D:\\...\\12.8修\\p1a_run]
"""
import importlib.util
import os, sys, json, gc, argparse
import numpy as np
import torch

MAIN_PY = r'D:\liaohe\校正优化过程\第三阶段\12优化\12.8修\12.8修_final.py'
OUT_BASE = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work\p1a'


def load_main():
    spec = importlib.util.spec_from_file_location('main128', MAIN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mult', type=float, default=2.0)
    ap.add_argument('--workdir', default=r'D:\liaohe\校正优化过程\第三阶段\12优化\12.8修\p1a_run')
    args = ap.parse_args()
    mult = float(args.mult)
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    mod = load_main()
    torch.manual_seed(42)
    np.random.seed(42)

    # ---- P1a 损失：放大强降水段权重 ----
    class HeavyPrecipMultiTaskLoss(mod.MultiTaskLoss):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.heavy_mult = mult
            self.storm_mse_w1 = self.storm_mse_w1 * mult
            self.storm_mse_w2 = self.storm_mse_w2 * mult

        def _precip_weight(self, target_abs):
            w = super()._precip_weight(target_abs)
            w = torch.where(target_abs >= 10.0, w * self.heavy_mult, w)
            return w

    mod.MultiTaskLoss = HeavyPrecipMultiTaskLoss

    device = mod.get_device()
    print(f'[P1a] device={device} mult={mult} workdir={workdir}')

    # ---- 与主实验一致：门控配置 ----
    mod.GATE_CFG['adaptive'] = True
    mod.GATE_CFG['hard_gate'] = True
    mod.GATE_CFG['threshold_base'] = 0.22
    mod.GATE_CFG['threshold_min'] = 0.10
    mod.GATE_CFG['threshold_max'] = 0.40
    mod.GATE_CFG['gate_power'] = 0.90
    mod.GATE_CFG['storm_gate_p'] = 0.25

    # ---- 数据集 ----
    print('[P1a] 创建数据集（严格配对）...', flush=True)
    datasets = mod.create_datasets_with_dem(
        r'D:\liaohe\GFS-data\gfs.0p25.2015-2025.f003',
        r'D:\liaohe\ERA5-data', dem_tensor=None)
    correction_train = datasets['correction_train']
    correction_val = datasets['correction_val']
    correction_test = datasets['correction_test']
    scaling_factor = float(datasets.get('scaling_factor', 1.0))
    print(f'[P1a] train={len(correction_train)} val={len(correction_val)} '
          f'test={len(correction_test)} sf={scaling_factor}', flush=True)

    # ---- 过采样（与主实验一致） ----
    intensity_stats = mod.ExtremeEventDataLoader.analyze_dataset_intensity(
        correction_train, thresholds=[0.1, 1.0, 5.0, 10.0, 20.0])
    sampled_total = max(1, sum(intensity_stats.values()))
    heavy_ratio = (intensity_stats.get('10.0-20.0mm', 0) + intensity_stats.get('>=20.0mm', 0)) / sampled_total
    print(f'[P1a] 暴雨样本比例 heavy_ratio={heavy_ratio*100:.2f}%', flush=True)
    ratios = [0.4, 0.8, 1.2, 5.0, 20.0, 45.0, 70.0]
    storm_patch_train = mod.StormPatchWrapper(correction_train, patch=20, storm_th=10.0, storm_prob=1.0)
    train_loader = mod.ExtremeEventDataLoader.create_adaptive_oversampled_loader(
        dataset=storm_patch_train, batch_size=64, num_workers=0, oversample_ratios=ratios)
    cfg = mod.ultra_fast_training_config()

    def _fast(ds, shuffle=False, drop_last=True):
        return mod.DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle, num_workers=0,
                              pin_memory=True, drop_last=drop_last, collate_fn=mod.custom_collate_fn)

    val_loader = _fast(correction_val, shuffle=False, drop_last=True)
    test_loader = _fast(correction_test, shuffle=False, drop_last=False)
    print(f'[P1a] loader batches: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}', flush=True)

    # ---- 模型 ----
    model = mod.AdvancedPrecipCorrectionNet(
        input_channels=8, hidden_channels=24, sequence_length=6,
        prediction_horizon=mod.PREDICTION_HORIZON, spatial_dims=(25, 37), dropout_rate=0.1).to(device)
    print('[P1a] 模型初始化完成', flush=True)

    # ---- 两阶段训练 ----
    print('[P1a] 阶段1训练 (10ep)...', flush=True)
    model, res1 = mod.staged_training_strategy_with_monitoring(
        model=model, train_loader=train_loader, val_loader=val_loader, device=device,
        scaling_factor=scaling_factor, epochs=10, second_stage=False)
    bp20 = float(np.max(res1['storm_pod_20'])) if res1['storm_pod_20'] else 0.0
    be20 = float(np.max(res1['storm_ets_20'])) if res1['storm_ets_20'] else 0.0
    print(f'[P1a] stage1 best POD20={bp20:.4f} ETS20={be20:.4f}', flush=True)
    if bp20 >= 0.10 or be20 >= 0.02:
        print('[P1a] 触发阶段2微调 (8ep)...', flush=True)
        if os.path.exists('best_correction_model.pth'):
            ckpt = torch.load('best_correction_model.pth', map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        model, res2 = mod.staged_training_strategy_with_monitoring(
            model=model, train_loader=train_loader, val_loader=val_loader, device=device,
            scaling_factor=scaling_factor, epochs=8, second_stage=True)
    else:
        print('[P1a] 未触发阶段2，使用阶段1最佳模型', flush=True)

    # ---- 加载 best + 门控校准 ----
    ckpt = torch.load('best_correction_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    torch.save(ckpt, os.path.join(workdir, f'best_p1a_mult{mult}.pth'))
    print('[P1a] 门控校准...', flush=True)
    old_th = float(mod.GATE_CFG.get('threshold_base', 0.22))
    old_pow = float(mod.GATE_CFG.get('gate_power', 0.90))
    calib = mod.calibrate_gate_threshold_on_val(
        model=model, val_loader=val_loader, device=device, scaling_factor=scaling_factor,
        search=np.linspace(0.08, 0.24, 9), power_search=(0.80, 0.90, 1.00),
        storm_gate_p_search=[0.10, 0.15, 0.20, 0.25, 0.30],
        min_pod20=0.25, min_pod15=0.25, max_far20=0.999, min_ets20=0.005)
    if calib.get('is_valid', False) or (calib.get('ets20', -1) >= 0.005 and calib.get('pod20', 0) >= 0.20):
        mod.GATE_CFG['threshold_base'] = float(calib['th'])
        mod.GATE_CFG['gate_power'] = float(calib['pow'])
        mod.GATE_CFG['storm_gate_p'] = float(calib['gate_p'])
    else:
        mod.GATE_CFG['threshold_base'] = old_th
        mod.GATE_CFG['gate_power'] = old_pow
    print(f'[P1a] 门控: th={mod.GATE_CFG["threshold_base"]:.3f} '
          f'pow={mod.GATE_CFG["gate_power"]:.2f} storm_gate_p={mod.GATE_CFG["storm_gate_p"]:.3f}', flush=True)

    # ---- 测试集预测 ----
    print('[P1a] 测试集推理...', flush=True)
    model.eval()
    all_pred, all_tgt, all_gfs = [], [], []
    with torch.no_grad():
        for inputs, tgt in test_loader:
            inputs = inputs.to(device)
            tgt = tgt.to(device)
            p, o, g, _, _, _, _ = mod.get_model_eval_tensors(
                model=model, inputs=inputs, targets_scaled=tgt,
                scaling_factor=scaling_factor, max_precip=200.0)
            all_pred.append(p[:, -1].cpu().numpy())
            all_tgt.append(o[:, -1].cpu().numpy())
            all_gfs.append(g[:, -1].cpu().numpy())
    preds = np.concatenate(all_pred, 0).astype(np.float32)
    targets = np.concatenate(all_tgt, 0).astype(np.float32)
    gfs = np.concatenate(all_gfs, 0).astype(np.float32)

    # ---- 全套评估（ERA5 真值） ----
    full = mod.evaluate_model_predictions(preds, targets, gfs, model_name=f'APCNet_P1A_mult{mult}')
    os.makedirs(OUT_BASE, exist_ok=True)
    np.save(os.path.join(OUT_BASE, f'predictions_p1a_mult{mult}.npy'), preds)
    np.save(os.path.join(OUT_BASE, f'targets_test_p1a.npy'), targets)
    np.save(os.path.join(OUT_BASE, f'gfs_test_p1a.npy'), gfs)
    try:
        import pickle
        st = correction_test.sample_times[:len(preds)] if hasattr(correction_test, 'sample_times') else []
        with open(os.path.join(OUT_BASE, f'sample_times_test_p1a.pkl'), 'wb') as f:
            pickle.dump(st, f)
        print(f'[P1a] sample_times saved: {len(st)}', flush=True)
    except Exception as e:
        print(f'[P1a] sample_times 保存失败: {e}', flush=True)
    with open(os.path.join(OUT_BASE, f'p1a_mult{mult}_eval.json'), 'w', encoding='utf-8') as f:
        json.dump({'mult': mult, 'full_eval': full, 'gate': dict(mod.GATE_CFG)}, f, indent=2, ensure_ascii=False)
    print(f'[P1a] 完成。结果保存在 {OUT_BASE}', flush=True)
    c = full['continuous']
    print(f'[P1a] MSE改进={c["mse_improve_pct"]:.2f}% RMSE={c["RMSE_model"]:.4f} CC={c["CC_model"]:.4f}', flush=True)
    print(f'[P1a] POD20={full["categorical"]["20mm"]["POD"]:.4f} '
          f'ETS20={full["categorical"]["20mm"]["ETS"]:.4f} '
          f'FSS10={full["fss"]["fss_model_10mm"]:.4f}', flush=True)
    print('[P1a] DONE', flush=True)


if __name__ == '__main__':
    main()
