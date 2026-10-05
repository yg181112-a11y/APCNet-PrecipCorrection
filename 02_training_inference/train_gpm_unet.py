# -*- coding: utf-8 -*-
"""train_gpm_unet.py — 24h 00Z 严格对照训练：训练目标对照实验（第一层核心）。
--target gpm : target = GPM24h - GFS   (观测目标)
--target era5: target = ERA5 24h - GFS  (再分析目标, 00Z-only 重训)
两者同架构(StandardUNet hidden=32)、同超参、同掩码(cov 828 格点)、同 epoch。
输出: unet_{target}_24h_00z.pth / pred_unet_{target}_24h_00z.npy / eval_{target}_00z.json
"""
import importlib.util, numpy as np, os, json, time, argparse
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
EXP = os.path.join(BASE, '24h_exp')
spec = importlib.util.spec_from_file_location('m13', os.path.join(BASE, '13.0_main.py'))
m13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m13)
UNet = m13.StandardUNet
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE)

ap = argparse.ArgumentParser()
ap.add_argument('--target', choices=['gpm', 'era5'], required=True)
ap.add_argument('--loss', choices=['weighted', 'mse'], default='weighted',
                help='weighted = storm_weight(默认, 与 ERA5 目标一致); mse = 纯 MSE 诊断')
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()
TGT = args.target
LOSS = args.loss
OUT_PTH = os.path.join(EXP, f'unet_{TGT}_24h_00z{"" if LOSS=="weighted" else "_nowt"}.pth')
OUT_PRED = os.path.join(EXP, f'pred_unet_{TGT}_24h_00z{"" if LOSS=="weighted" else "_nowt"}.npy')
OUT_JSON = os.path.join(EXP, f'eval_{TGT}_00z{"" if LOSS=="weighted" else "_nowt"}.json')

# ---- 数据（00Z 子集）----
feats = np.load(os.path.join(EXP, 'features_00z.npy'))       # [N00,8,25,37]
gfs = np.load(os.path.join(EXP, 'gfs24_accum_00z.npy'))
era5 = np.load(os.path.join(EXP, 'era5_24h_accum_00z.npy'))
split = np.load(os.path.join(EXP, 'split_00z.npy'))
cov = np.load(os.path.join(EXP, 'gpm24_mask_00z.npy'))       # [25,37] bool
if TGT == 'gpm':
    obs = np.load(os.path.join(EXP, 'gpm24_accum_00z.npy'))
else:
    obs = era5.copy()

# 有效样本（gfs/目标有限值）
valid = ~(np.isnan(gfs).any((1,2)) | np.isnan(obs).any((1,2)))
feats = feats[valid]; gfs = gfs[valid]; era5 = era5[valid]; obs = obs[valid]; split = split[valid]
feats[:, 7] = gfs
N = len(feats)
print(f'[{TGT}] 样本 {N}: train {(split==0).sum()}, val {(split==1).sum()}, test {(split==2).sum()}')

# 标准化（通道0-6 z-score 用训练集；通道7物理值）
tr = split == 0
mu = feats[tr][:, :7].mean(axis=(0,2,3), keepdims=True)
sd = feats[tr][:, :7].std(axis=(0,2,3), keepdims=True) + 1e-6
feats[:, :7] = (feats[:, :7] - mu) / sd

class DS(Dataset):
    def __init__(self, x, t, m):
        self.x = torch.from_numpy(x).float()
        self.t = torch.from_numpy(t).float()
        self.m = torch.from_numpy(m.astype(np.float32)).float()  # [25,37]
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return self.x[i].unsqueeze(0), self.t[i], self.m

train_ds = DS(feats[tr], obs[tr] - gfs[tr], cov)
val_mask = split == 1; val_ds = DS(feats[val_mask], obs[val_mask] - gfs[val_mask], cov)
te_mask = split == 2
gfs_te, obs_te, feats_te = gfs[te_mask], obs[te_mask], feats[te_mask]
tr_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

def storm_weight(t):
    w = torch.ones_like(t)
    w[t > 5.0] = 3.0
    w[t > 20.0] = 8.0
    return w

print(f'[训练 U-Net-{TGT} 00Z]')
if os.path.exists(OUT_PTH):
    print('  已存在权重，跳过训练')
else:
    model = UNet(input_channels=8, hidden_channels=32).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    best_v, best_ep = 1e9, 0
    t0 = time.time()
    for ep in range(60):
        model.train(); tot = 0.0; nb = 0
        for xb, tb, mb in tr_loader:
            xb, tb, mb = xb.to(DEVICE), tb.to(DEVICE), mb.to(DEVICE)
            res, _, _ = model(xb)
            mse = (res[:, 0] - tb) ** 2 * mb[None]   # 掩码格点
            if LOSS == 'weighted':
                loss = (mse * storm_weight(tb)).mean()
            else:
                loss = mse.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        model.eval(); vt = 0.0; nv = 0
        with torch.no_grad():
            for xb, tb, mb in val_loader:
                xb, tb, mb = xb.to(DEVICE), tb.to(DEVICE), mb.to(DEVICE)
                res, _, _ = model(xb)
                if LOSS == 'weighted':
                    vt += ((res[:, 0] - tb) ** 2 * mb[None]).mean().item(); nv += 1
                else:
                    vt += ((res[:, 0] - tb) ** 2 * mb[None]).mean().item(); nv += 1
        v = vt / nv
        if v < best_v:
            best_v = v; best_ep = ep
            torch.save(model.state_dict(), OUT_PTH)
        if ep % 10 == 0 or ep == 59:
            print(f'  ep {ep}: train {tot/nb:.4f} val {v:.4f} ({time.time()-t0:.0f}s)')
    print(f'best val {best_v:.4f} @ ep {best_ep}')

print('[推理]')
model = UNet(input_channels=8, hidden_channels=32).to(DEVICE)
model.load_state_dict(torch.load(OUT_PTH, map_location=DEVICE))
model.eval()
preds = np.zeros((len(gfs_te), 25, 37), np.float32)
with torch.no_grad():
    for i in range(0, len(te_mask), 64):
        xb = torch.from_numpy(feats_te[i:i+64]).float().unsqueeze(1).to(DEVICE)
        res, _, _ = model(xb)
        r = res[:, 0].cpu().numpy()
        preds[i:i+len(r)] = np.clip(gfs_te[i:i+len(r)] + r, 0, 500)
np.save(OUT_PRED, preds)

# ---- 评估（GPM 覆盖格点；真值 = GPM 或 ERA5）----
print('[评估]')
if TGT == 'gpm':
    truth = obs_te
    label = 'GPM'
else:
    truth = era5[te_mask]
    label = 'ERA5'
g_v = gfs_te[:, cov]
t_v = truth[:, cov]
p_v = preds[:, cov]

def cont(obs_f, fcst_f):
    obs_f, fcst_f = obs_f.flatten(), fcst_f.flatten()
    valid_ = np.isfinite(obs_f) & np.isfinite(fcst_f)
    obs_f, fcst_f = obs_f[valid_], fcst_f[valid_]
    if obs_f.size == 0:
        return None
    mse = np.mean((obs_f - fcst_f) ** 2)
    rmse = np.sqrt(mse)
    cc = np.corrcoef(obs_f, fcst_f)[0, 1]
    bias = float(np.mean(fcst_f - obs_f))
    return {'MSE': float(mse), 'RMSE': float(rmse), 'CC': float(cc), 'bias': bias}

res = {}
m_g = cont(t_v, g_v)
res['GFS'] = m_g
m_u = cont(t_v, p_v)
res['U-Net'] = m_u
res['U-Net_mse_improve'] = 100.0 * (m_g['MSE'] - m_u['MSE']) / m_g['MSE']
json.dump(res, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps(res, ensure_ascii=False, indent=2))
print('完成')
