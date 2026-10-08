# -*- coding: utf-8 -*-
"""M1 路径A 评估：纯MSE损失+验证MSE选点 检查点 在 ERA5 测试期(2024-2025)的 3h 技巧。
口径与 eval_mse_select.py 完全一致（test_loader drop_last=False、门控校准后推理、improvement vs raw GFS）。
唯一差异：checkpoint 目录 mseonly_select（纯MSE重训产物）。
"""
import sys, os, json
import numpy as np
import torch

MAIN_DIR = r'D:\liaohe\论文三\03_重建成稿代_2026_R3全链主实验'
OUT_ROOT = r'D:\liaohe\_waf_review\m1_diagnostic\mseonly_select'
sys.path.insert(0, MAIN_DIR)
os.environ['PYTHONPATH'] = MAIN_DIR + ';' + os.path.dirname(os.path.abspath(__file__)) + ';' + os.environ.get('PYTHONPATH', '')
import main13 as M

SEEDS = [42, 40, 41]
GFS_BASE = r"D:\liaohe\GFS-data\gfs.0p25.2015-2025.f003"
ERA5_BASE = r"D:\liaohe\ERA5-data"

def collect(model, test_loader, device, scaling_factor):
    model.eval()
    all_preds, all_targets, all_gfs = [], [], []
    with torch.no_grad():
        for inputs, targets_batch in test_loader:
            inputs = inputs.to(device); targets_batch = targets_batch.to(device)
            pred_abs, true_abs, gfs_expand, _, _, _, _ = M.get_model_eval_tensors(
                model=model, inputs=inputs, targets_scaled=targets_batch,
                scaling_factor=scaling_factor, max_precip=200.0)
            all_preds.append(pred_abs[:, -1].cpu().numpy())
            all_targets.append(true_abs[:, -1].cpu().numpy())
            all_gfs.append(gfs_expand[:, -1].cpu().numpy())
    return (np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_gfs))

def main():
    device = M.get_device()
    datasets_dict = M.create_datasets_with_dem(GFS_BASE, ERA5_BASE, dem_tensor=None)
    correction_test = datasets_dict['correction_test']
    scaling_factor = float(datasets_dict.get('scaling_factor', 1.0))
    cfg = M.ultra_fast_training_config()
    test_loader = M.DataLoader(correction_test, batch_size=cfg['batch_size'], shuffle=False,
                               num_workers=0, pin_memory=True, drop_last=False,
                               collate_fn=M.custom_collate_fn)

    rows = []
    all_data = {}
    for seed in SEEDS:
        out_dir = os.path.join(OUT_ROOT, f'seed{seed}')
        ckpt_path = os.path.join(out_dir, 'best_correction_model.pth')
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = M.AdvancedPrecipCorrectionNet(
            input_channels=8, hidden_channels=24, sequence_length=6,
            prediction_horizon=M.PREDICTION_HORIZON, spatial_dims=(25, 37), dropout_rate=0.1).to(device)
        model.load_state_dict(ck['model_state_dict'])
        hist = json.load(open(os.path.join(out_dir, 'history.json'), encoding='utf-8'))
        gate = hist.get('gate', {})
        for k, v in gate.items():
            if v is not None:
                M.GATE_CFG[k] = v
        print(f"[seed {seed}] gate={gate} selected_epoch={ck.get('epoch')}", flush=True)

        pred, target, gfs = collect(model, test_loader, device, scaling_factor)
        met = M.evaluate_model_predictions(pred, target, gfs, model_name=f'APCNet-MSEonly-S{seed}')
        rows.append({'seed': seed, 'epoch': ck.get('epoch'), 'val_loss': ck.get('val_loss'),
                     'mse': float(met['continuous']['MSE_model']),
                     'rmse': float(met['continuous']['RMSE_model']),
                     'mse_improve_pct': float(met['continuous']['mse_improve_pct'])})
        all_data[f'seed{seed}'] = {'pred': pred, 'target': target, 'gfs': gfs,
                                   'epoch': ck.get('epoch')}
        print(f"[seed {seed}] MSE={met['continuous']['MSE_model']:.4f} "
              f"RMSE={met['continuous']['RMSE_model']:.4f} "
              f"improve_vs_GFS={met['continuous']['mse_improve_pct']:.1f}%", flush=True)

    print("\n=== MSE-only checkpoints: ERA5 test-period 3-h skill ===")
    print(f"{'seed':>4} {'epoch':>5} {'val_loss':>10} {'MSE':>8} {'impr%':>8}")
    for r in rows:
        print(f"{r['seed']:>4} {r['epoch']:>5} {r['val_loss']:>10.4f} {r['mse']:>8.4f} {r['mse_improve_pct']:>8.1f}")
    mean_imp = np.mean([r['mse_improve_pct'] for r in rows])
    print(f"three-seed mean improvement = {mean_imp:.1f}%")
    with open(os.path.join(OUT_ROOT, 'era5_test_skill_mseonly.json'), 'w') as f:
        json.dump({'rows': rows, 'three_seed_mean_improve': mean_imp}, f, indent=1)
    np.savez(os.path.join(OUT_ROOT, 'era5_test_preds_mseonly.npz'),
             pred42=all_data['seed42']['pred'], pred40=all_data['seed40']['pred'],
             pred41=all_data['seed41']['pred'],
             target=all_data['seed42']['target'], gfs=all_data['seed42']['gfs'])

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
