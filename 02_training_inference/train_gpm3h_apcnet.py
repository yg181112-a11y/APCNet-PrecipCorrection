# -*- coding: utf-8 -*-
"""train_gpm3h_apcnet.py — 用 GPM IMERG 观测当标签训练 APCNet（3h，与主实验同构）
协议对齐点：AdvancedPrecipCorrectionNet 架构 / sym 损失（仅强度加权，无 miss-fa 不对称）/
Adam(lr=1e-3) / batch 64 / EMA(0.999) / 门控评估（GATE_CFG 同主实验）。
训练期 GPM 2018-2021 / 验证 2022-2023 / 测试 2024-2025（gpm_train_dataset 口径）。
用法: python train_gpm3h_apcnet.py [--epochs 30] [--seed 42] [--smoke 2]
"""
import sys, os, time, json, argparse, pickle
sys.path.insert(0, r'C:\Users\yg181\Desktop\论文三\13.0修复重跑')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import importlib.util

BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
OUT = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\gpm_trained_exp'
os.makedirs(OUT, exist_ok=True)


def load_main():
    spec = importlib.util.spec_from_file_location('m13', os.path.join(BASE, '13.0_main.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


M = load_main()
import gpm_train_dataset as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', type=int, default=0, help='>0 只训练 smoke 个 epoch')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', DEVICE, 'seed:', args.seed)

    # ---------- 1) 数据集（GPM 标签版）----------
    gfs_base_path = r"D:\liaohe\GFS-data\gfs.0p25.2015-2025.f003"
    print('构建 GPM 训练数据集 ...')
    ds = G.create_gpm_strict_datasets(gfs_base_path)
    correction_train = ds['correction_train']
    correction_val = ds['correction_val']
    correction_test = ds['correction_test']
    print('train %d | val %d | test %d' % (len(correction_train), len(correction_val), len(correction_test)))

    train_loader = DataLoader(correction_train, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(correction_val, batch_size=64, shuffle=False, num_workers=0)

    # ---------- 2) sym 损失（对齐 manuscript_work sym 口径：仅强度加权）----------
    def sym_loss_fn(pred_res, target_abs, gfs_base):
        pred_abs = gfs_base + pred_res
        diff = pred_abs - target_abs
        w = torch.ones_like(target_abs)
        w = torch.where(target_abs < 0.1, w * 1.0, w)
        w = torch.where(target_abs >= 0.1,  w * 1.5, w)
        w = torch.where(target_abs >= 3.0,  w * 2.0, w)
        w = torch.where(target_abs >= 10.0, w * 10.0, w)
        w = torch.where(target_abs >= 20.0, w * 60.0, w)
        w = torch.where(target_abs >= 50.0, w * 160.0, w)
        w = w / w.detach().mean().clamp(min=1e-8)
        return torch.mean((diff ** 2) * w) * 0.5

    # ---------- 3) 训练 ----------
    model = M.AdvancedPrecipCorrectionNet(input_channels=8, hidden_channels=24,
                                          prediction_horizon=1, sequence_length=6,
                                          spatial_dims=(25, 37), dropout_rate=0.1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    ema = M.ModelEMA(model, decay=0.999)
    EPOCHS = args.smoke if args.smoke > 0 else args.epochs
    best_v, best_ep, patience, no_impr = 1e18, -1, 8, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); tot = 0.0; nb = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            gfs_base = xb[:, -1, -1, :, :].unsqueeze(1)
            target_abs = (gfs_base + yb).clamp(min=0.0)
            residual, rain_prob, _ = model(xb)
            loss = sym_loss_fn(residual[:, 0].unsqueeze(1), target_abs, gfs_base)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model)
            tot += loss.item(); nb += 1
        # val (EMA)
        model_eval = ema.ema if ema.ema is not None else model
        model_eval.eval(); vt = 0.0; nv = 0; pod20 = 0.0; n20 = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                gfs_base = xb[:, -1, -1, :, :].unsqueeze(1)
                target_abs = (gfs_base + yb).clamp(min=0.0)
                residual, _, _ = model_eval(xb)
                pred_abs = (gfs_base + residual[:, 0].unsqueeze(1)).clamp(min=0.0)
                vt += torch.mean((pred_abs - target_abs) ** 2).item(); nv += 1
                t20 = (target_abs[:, 0] >= 20.0)
                if t20.any():
                    pod20 += ((pred_abs[:, 0][t20] >= 20.0).float().mean()).item(); n20 += 1
        v = vt / nv
        if n20: pod20 = pod20 / n20
        if v < best_v:
            best_v, best_ep, no_impr = v, ep, 0
            torch.save({'model_state_dict': model_eval.state_dict(), 'epoch': ep}, os.path.join(OUT, 'apcnet_gpm3h_s%d.pth' % args.seed))
        else:
            no_impr += 1
        print('ep %2d loss %.4f val_mse %.4f POD20 %.3f (%.0fs)' % (ep+1, tot/max(nb,1), v, pod20, time.time()-t0))
        if no_impr >= patience and ep >= 10:
            print('early stop @ ep %d (best ep %d)' % (ep+1, best_ep+1)); break

    print('best val mse %.4f @ ep %d' % (best_v, best_ep+1))

    # ---------- 4) 测试预测（门控评估，同主实验 GATE_CFG）----------
    for k, v in [('adaptive', True), ('hard_gate', True), ('threshold_base', 0.22),
                 ('threshold_min', 0.10), ('threshold_max', 0.40), ('gate_power', 0.90),
                 ('storm_gate_p', 0.25)]:
        M.GATE_CFG[k] = v
    model_eval = M.AdvancedPrecipCorrectionNet(input_channels=8, hidden_channels=24,
                                               prediction_horizon=1, sequence_length=6,
                                               spatial_dims=(25, 37), dropout_rate=0.1).to(DEVICE)
    ck = torch.load(os.path.join(OUT, 'apcnet_gpm3h_s%d.pth' % args.seed), map_location=DEVICE)
    model_eval.load_state_dict(ck['model_state_dict']); model_eval.eval()
    test_loader = DataLoader(correction_test, batch_size=64, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            pred_full = model_eval(xb, return_residual=False)  # gated prediction [B,2,H,W]
            pred_abs = pred_full[:, -1]                        # 主实验统一取 pred_abs[:, -1]
            preds.append(pred_abs.cpu().numpy())
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = np.clip(preds, 0, 500)
    times = correction_test.sample_times
    np.save(os.path.join(OUT, 'pred_apcnet_gpm3h_s%d.npy' % args.seed), preds)
    with open(os.path.join(OUT, 'times_test_gpm.pkl'), 'wb') as f:
        pickle.dump(times, f)
    print('predictions saved:', preds.shape, 'n_times', len(times))
    json.dump({'seed': args.seed, 'epochs_run': EPOCHS, 'best_val_mse': best_v, 'n_train': len(correction_train),
               'n_val': len(correction_val), 'n_test': len(correction_test), 'pred_shape': list(preds.shape)},
              open(os.path.join(OUT, 'train_gpm3h_s%d.json' % args.seed), 'w'), indent=2, ensure_ascii=False)
    print('done')


if __name__ == '__main__':
    main()
