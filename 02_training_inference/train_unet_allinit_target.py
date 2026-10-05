# -*- coding: utf-8 -*-
"""train_unet_allinit_target.py — 全体 init 训练目标对照（权威版）。
同架构 U-Net、同掩码（GPM cov 828 格点）、同 loss（纯 MSE）。
--target gpm: 目标 = GPM 24h 累积；--target era5: 目标 = ERA5 24h 累积。
输出: pred_unet_{target}_24h_allinit_nowt.npy + eval_{target}_allinit.json（GPM 00Z 验证口径）
"""
import argparse, os, json, time, importlib.util
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

EXP = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\24h_exp'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 与主实验 U-Net 完全同构（13.0_main.StandardUNet）
_spec = importlib.util.spec_from_file_location('m13', r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\13.0_main.py')
_m13 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m13)
StandardUNet = _m13.StandardUNet

class DS(Dataset):
    def __init__(self, x, t, m):
        self.x = torch.from_numpy(x).float()
        self.t = torch.from_numpy(t).float()
        self.m = torch.from_numpy(m).float()
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i].unsqueeze(0), self.t[i], self.m[i]   # [1,8,25,37], [25,37], [25,37]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=['gpm', 'era5'], required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    TGT = args.target
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    feats = np.load(os.path.join(EXP, 'features_24h.npy'))    # [N,8,25,37] 全体
    gfs = np.load(os.path.join(EXP, 'gfs_24h_accum.npy'))
    era5 = np.load(os.path.join(EXP, 'era5_24h_accum.npy'))
    split = np.load(os.path.join(EXP, 'split_mask.npy'))
    gpm_all = np.load(os.path.join(EXP, 'gpm24_accum_allinit.npy'))     # 全体 N 行 NaN 标记
    gpm_mask = np.load(os.path.join(EXP, 'gpm24_mask_allinit.npy'))     # [N,25,37] bool
    cov = np.load(os.path.join(EXP, 'gpm24_cov_allinit.npy'))           # [25,37] 共同覆盖

    N = len(feats)
    valid = ~(np.isnan(gfs).any((1, 2)) | np.isnan(era5).any((1, 2)) | np.isnan(gpm_all).any((1, 2)))
    feats = feats[valid]; gfs = gfs[valid]; era5 = era5[valid]
    gpm_all = gpm_all[valid]; gpm_mask = gpm_mask[valid]; split = split[valid]
    feats[:, 7] = gfs
    tr = split == 0; val = split == 1; te = split == 2
    print('样本 %d: train %d, val %d, test %d' % (len(feats), tr.sum(), val.sum(), te.sum()))

    # 标准化
    mu = feats[tr][:, :7].mean(axis=(0, 2, 3), keepdims=True)
    sd = feats[tr][:, :7].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    feats[:, :7] = (feats[:, :7] - mu) / sd

    if TGT == 'gpm':
        tgt = gpm_all
        label = 'GPM'
    else:
        tgt = era5
        label = 'ERA5'
    msk = gpm_mask  # 全体掩码

    def make_ds(idx):
        return DS(feats[idx], tgt[idx] - gfs[idx], msk[idx])

    tr_ds = make_ds(tr); val_ds = make_ds(val); te_ds = make_ds(te)
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = StandardUNet(input_channels=8, hidden_channels=32).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

    best_v, best_ep = 1e9, 0
    t0 = time.time()
    for ep in range(60):
        model.train(); tot = 0.0; nb = 0
        for xb, tb, mb in tr_loader:
            xb, tb, mb = xb.to(DEVICE), tb.to(DEVICE), mb.to(DEVICE)
            res, _, _ = model(xb)
            r = res[:, 0]
            loss = (((r - tb) ** 2) * mb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        model.eval(); vt = 0.0; nv = 0
        with torch.no_grad():
            for xb, tb, mb in val_loader:
                xb, tb, mb = xb.to(DEVICE), tb.to(DEVICE), mb.to(DEVICE)
                res, _, _ = model(xb)
                r = res[:, 0]
                vt += (((r - tb) ** 2) * mb).mean().item(); nv += 1
        v = vt / nv
        if v < best_v:
            best_v = v; best_ep = ep
            torch.save(model.state_dict(), os.path.join(EXP, f'unet_{TGT}_24h_allinit_nowt.pth'))
        if ep % 10 == 0 or ep == 59:
            print(f'  ep {ep}: train {tot/nb:.4f} val {v:.4f} ({time.time()-t0:.0f}s)', flush=True)
    print(f'best val {best_v:.4f} @ ep {best_ep}', flush=True)

    # 推理
    model.load_state_dict(torch.load(os.path.join(EXP, f'unet_{TGT}_24h_allinit_nowt.pth'), map_location=DEVICE))
    model.eval()
    preds = np.zeros((len(te_ds), 25, 37), np.float32)
    with torch.no_grad():
        for i in range(0, len(te_ds), 64):
            xb = torch.from_numpy(feats[te][i:i + 64]).float().unsqueeze(1).to(DEVICE)
            res, _, _ = model(xb)
            r = res[:, 0].cpu().numpy()
            preds[i:i + 64] = np.clip(gfs[te][i:i + 64] + r, 0, 500)
    np.save(os.path.join(EXP, f'pred_unet_{TGT}_24h_allinit_nowt.npy'), preds)

    # 评估（GPM 真值, cov 格点）
    g_te = gfs[te][:, cov]; p_te = preds[:, cov]
    t_te = np.load(os.path.join(EXP, 'gpm24_accum_allinit.npy'))[valid][te][:, cov]

    def cont(o, f):
        o, f = o.flatten(), f.flatten()
        v = np.isfinite(o) & np.isfinite(f)
        o, f = o[v], f[v]
        mse = np.mean((o - f) ** 2)
        return {'RMSE': float(np.sqrt(mse)), 'CC': float(np.corrcoef(o, f)[0, 1]),
                'bias': float(np.mean(f - o)), 'MSE': float(mse)}

    mg = cont(t_te, g_te); mp = cont(t_te, p_te)
    imp = 100.0 * (mg['MSE'] - mp['MSE']) / mg['MSE']
    res = {'target': label, 'GFS': mg, 'U-Net': mp, 'mse_improve_pct': imp,
           'n_test': int(te.sum()), 'cov_grids': int(cov.sum())}
    json.dump(res, open(os.path.join(EXP, f'eval_{TGT}_allinit.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print('完成')

if __name__ == '__main__':
    main()
