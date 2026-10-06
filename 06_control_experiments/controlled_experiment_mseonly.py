# -*- coding: utf-8 -*-
"""
受控实验变体 B：纯 MSE 损失（MSE-only loss）
================================================================
审稿人 M1 最低接受条件之一：把合成目标实验的损失换成纯 MSE
（去掉 FSS/ETS/rain/storm/CC/storm-MSE/peak/residual-reg 全部加权项），
检验 3h 负技巧是否仍出现 —— 从而把归因从"损失设计"剥离。

训练过程其余环节与原受控实验完全一致（保持唯一变量）：
  - 数据划分/噪声构造：create_datasets_with_dem + make_noise_target_ds（同一 seed 规则）
  - 过采样：StormPatchWrapper(patch=20, storm_th=10) + ExtremeEventDataLoader adaptive oversample
  - 优化：AdamW(lr=8e-5, wd=8e-6) + ReduceLROnPlateau(patience=3) + EMA(0.999) + AMP
  - 选点：EMA 权重按验证集残差 MSE 最小保存（与原受控实验一致，纯 MSE 选点）
  - 早停：val 无改善 4 个 epoch 且 >=6 则停

损失（唯一改动）：
  L = mean((pred_res - targets)^2)     # 残差空间 MSE，与 val_mse 同口径

输出（与原受控实验同目录，_mseonly 后缀）：
  - res_<ns>_s<seed>_mseonly.json     最终测试评估（improvement/r/…，口径与原版一致）
  - hist_<ns>_s<seed>_mseonly.json    每 epoch train loss / val MSE / test MSE（收敛曲线）
  - 每档 stdout 打印 epX train/val/test
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

WORK = r'D:\liaohe\论文三\03_重建成稿代_2026_R3全链主实验'
os.chdir(WORK)
sys.path.insert(0, WORK)

import importlib.util
_spec = importlib.util.spec_from_file_location('m13', os.path.join(WORK, '13.0_main.py'))
m13 = importlib.util.module_from_spec(_spec)
sys.modules['m13'] = m13
_spec.loader.exec_module(m13)

from m13 import (AdvancedPrecipCorrectionNet, ModelEMA,
                 get_device, StormPatchWrapper, ExtremeEventDataLoader,
                 custom_collate_fn, ultra_fast_training_config,
                 create_datasets_with_dem)

OUT = r'D:\liaohe\校正优化过程\第三阶段\12优化\controlled_exp'


def make_noise_target_ds(noise_std, seed):
    """与原受控实验完全相同的噪声目标构造（同一 seed*100000+i 规则）。"""
    gfs_base_path = r"D:\liaohe\GFS-data\gfs.0p25.2015-2025.f003"
    era5_base_path = r"D:\liaohe\ERA5-data"
    d = create_datasets_with_dem(gfs_base_path, era5_base_path, dem_tensor=None)
    for key in ('correction_train', 'correction_val', 'correction_test'):
        ds = d[key]
        for i, (x, y, am) in enumerate(ds.sequences):
            gfs_base = x[-1, -1, :, :]
            if noise_std == 'identity':
                new_y = torch.zeros_like(y)
                new_am = float(gfs_base.max().item())
            else:
                g = torch.Generator().manual_seed(seed * 100000 + i)
                new_y = (torch.randn(gfs_base.shape, generator=g) * noise_std).unsqueeze(0)
                new_am = float((gfs_base + 3.0 * noise_std).max().item())
            ds.sequences[i] = (x, new_y, new_am)
    return d


def run(noise_std, seed, epochs=12):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    d = make_noise_target_ds(noise_std, seed)
    tr, va, te = d['correction_train'], d['correction_val'], d['correction_test']
    print(f'[受控MSE-only {noise_std}] train={len(tr)} val={len(va)} test={len(te)}', flush=True)

    storm_patch_train = StormPatchWrapper(tr, patch=20, storm_th=10.0, storm_prob=1.0)
    ratios = [0.4, 0.8, 1.2, 5.0, 20.0, 45.0, 70.0]
    train_loader = ExtremeEventDataLoader.create_adaptive_oversampled_loader(
        dataset=storm_patch_train, batch_size=64, num_workers=0, oversample_ratios=ratios)
    cfg = ultra_fast_training_config()

    def _loader(ds, shuffle, drop_last):
        return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle,
                          num_workers=0, pin_memory=True, drop_last=drop_last,
                          collate_fn=custom_collate_fn)
    val_loader = _loader(va, shuffle=False, drop_last=True)
    test_loader = _loader(te, shuffle=False, drop_last=False)

    model = AdvancedPrecipCorrectionNet(
        input_channels=8, hidden_channels=24, sequence_length=6,
        prediction_horizon=1, spatial_dims=(25, 37), dropout_rate=0.1).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=8e-5, weight_decay=8e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    ema = ModelEMA(model, decay=0.999)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def _eval_mse(loader, ema_model):
        tot, n = 0.0, 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                res, _, _, _ = ema_model(inputs, return_residual=True, return_storm_logits=True)
                tot += float(torch.mean((res - targets) ** 2).item() * len(targets))
                n += len(targets)
        return tot / max(n, 1)

    best_val = float('inf')
    best_state = None
    no_improve = 0
    history = []
    for epoch in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                residual, _, _, _ = model(inputs, return_residual=True, return_storm_logits=True)
                loss = F.mse_loss(residual, targets)          # 纯 MSE（残差空间）
            if not torch.isfinite(loss):
                continue
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            ema.update(model)
            tot += float(loss.item())
            nb += 1

        ema_model = ema.ema
        ema_model.eval()
        vloss = _eval_mse(val_loader, ema_model)
        tloss = _eval_mse(test_loader, ema_model)
        scheduler.step(vloss)
        hist = {'epoch': epoch + 1, 'train_loss': round(tot / max(nb, 1), 6),
                'val_mse': round(vloss, 6), 'test_mse': round(tloss, 6)}
        history.append(hist)
        print(f'  ep{epoch+1} train={hist["train_loss"]:.4f} val_mse={vloss:.4f} test_mse={tloss:.4f}', flush=True)
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in ema_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 4 and epoch >= 6:
                break

    # ===== 测试评估（与原始受控实验同一口径） =====
    model.load_state_dict(best_state)
    model.eval()
    res_parts = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            res, _, _, _ = model(inputs, return_residual=True, return_storm_logits=True)
            res_parts.append(res[:, -1].cpu().numpy())
    pred_res = np.concatenate(res_parts, axis=0).astype(np.float32)

    noise_parts, gfs_parts = [], []
    for i, (x, y, am) in enumerate(te.sequences):
        gfs_base = x[-1, -1].numpy()
        if noise_std == 'identity':
            nz = np.zeros_like(gfs_base)
        else:
            g = torch.Generator().manual_seed(seed * 100000 + i)
            nz = torch.randn(gfs_base.shape, generator=g).numpy() * noise_std
        noise_parts.append(nz)
        gfs_parts.append(gfs_base)
    noise = np.stack(noise_parts)
    gfs_base_arr = np.stack(gfs_parts)
    pred_abs = gfs_base_arr + pred_res

    result = {'noise_std': str(noise_std), 'seed': seed,
              'best_val': float(best_val), 'epochs_run': epoch + 1,
              'loss': 'mse_only'}
    if noise_std == 'identity':
        result['res_mean'] = float(pred_res.mean())
        result['res_abs_mean'] = float(np.abs(pred_res).mean())
        result['cc_pred_gfs'] = float(np.corrcoef(pred_abs.flatten(), gfs_base_arr.flatten())[0, 1])
        result['mae_pred_gfs'] = float(np.mean(np.abs(pred_abs - gfs_base_arr)))
        print(f'[恒等/MSE-only] mean|res|={result["res_abs_mean"]:.4f} CC={result["cc_pred_gfs"]:.4f} '
              f'MAE={result["mae_pred_gfs"]:.4f}', flush=True)
    else:
        sig2 = float(np.mean(noise ** 2))
        mse_res = float(np.mean((pred_res - noise) ** 2))
        result['improvement'] = 1.0 - mse_res / sig2
        result['r_pred_noise'] = float(np.corrcoef(pred_res.flatten(), noise.flatten())[0, 1])
        result['res_mean'] = float(pred_res.mean())
        result['mse_res'] = mse_res
        result['sigma2'] = sig2
        print(f'[σ={noise_std}/MSE-only] MSE改进={result["improvement"]*100:.2f}% '
              f'r(pred,noise)={result["r_pred_noise"]:.4f} res_mean={result["res_mean"]:.4f}', flush=True)

    os.makedirs(OUT, exist_ok=True)
    tag = f'{noise_std}_s{seed}_mseonly'
    with open(os.path.join(OUT, f'res_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, f'hist_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump({'noise_std': str(noise_std), 'seed': seed, 'loss': 'mse_only',
                   'epochs': history}, f, indent=2, ensure_ascii=False)
    print(f'✅ saved res_{tag}.json + hist_{tag}.json', flush=True)
    return result


if __name__ == '__main__':
    seed = int(os.environ.get('SEED', '42'))
    ns = os.environ.get('NOISE_STD', 'identity')
    run('identity' if ns == 'identity' else float(ns), seed)
