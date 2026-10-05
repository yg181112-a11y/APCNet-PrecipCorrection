# -*- coding: utf-8 -*-
"""
阶段 1b：24h 订正实验（APCNet-24h 训练 + QM/BM/OLS 基线 + 统一评估，ERA5-24h 参照）
输入: 24h_exp/features_24h.npy, gfs_24h_accum.npy, era5_24h_accum.npy, split_mask.npy
输出: 24h_exp/24h_results.json, pred_apcnet_24h.npy
划分: train 2019-2021 / val 2022-2023 / test 2024-2025
"""
import importlib.util, numpy as np, os, json, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import base64 as _b64
OUT = _b64.b64decode(os.environ['OUT_WORK_B64']).decode('utf-8')
DATA = _b64.b64decode(os.environ['DATA_DIR_B64']).decode('utf-8')
SEED = int(os.environ.get('SEED', '42'))
import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f'SEED={SEED} OUT={OUT}')
spec = importlib.util.spec_from_file_location('m13', r'C:\Users\yg181\Desktop\论文三\13.0修复重跑\13.0_main.py')
m13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m13)
APCNet = m13.AdvancedPrecipCorrectionNet
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE)

# ---------------- 1. 数据 ----------------
feats = np.load(os.path.join(DATA, 'features_24h.npy'))     # [N,8,25,37]
gfs   = np.load(os.path.join(DATA, 'gfs_24h_accum.npy'))    # [N,25,37]
era5  = np.load(os.path.join(DATA, 'era5_24h_accum.npy'))   # [N,25,37]
split = np.load(os.path.join(DATA, 'split_mask.npy'))
valid = ~(np.isnan(gfs).any((1,2)) | np.isnan(era5).any((1,2)))
feats = feats[valid]; gfs = gfs[valid]; era5 = era5[valid]; split = split[valid]
feats[:, 7] = gfs   # 降水通道 = GFS 24h 物理值
N = len(feats)
print(f'样本 {N}: train {(split==0).sum()}, val {(split==1).sum()}, test {(split==2).sum()}')

# 标准化（通道0-6 z-score，通道7物理值；用训练集统计）
tr = split == 0
mu = feats[tr][:, :7].mean(axis=(0,2,3), keepdims=True)
sd = feats[tr][:, :7].std(axis=(0,2,3), keepdims=True) + 1e-6
feats[:, :7] = (feats[:, :7] - mu) / sd
np.save(os.path.join(OUT, 'std24_params.npy'), np.array([mu.flatten(), sd.flatten()]))

class DS(Dataset):
    def __init__(self, x, t):
        self.x = torch.from_numpy(x).float()
        self.t = torch.from_numpy(t).float()
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return self.x[i].unsqueeze(0), self.t[i]  # [1,C,H,W], target

train_ds = DS(feats[tr], era5[tr] - gfs[tr])       # 残差目标
val_mask = split == 1; val_ds = DS(feats[val_mask], era5[val_mask] - gfs[val_mask])
te_mask = split == 2;  te_ds = DS(feats[te_mask], era5[te_mask] - gfs[te_mask])
gfs_tr, era5_tr = gfs[tr], era5[tr]
gfs_te, era5_te = gfs[te_mask], era5[te_mask]
feats_te = feats[te_mask]

tr_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

# ---------------- 2. 训练 APCNet-24h ----------------
print('\n[训练 APCNet-24h]')
model = APCNet(input_channels=8, hidden_channels=24, sequence_length=1, spatial_dims=(25,37), dropout_rate=0.1).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

def storm_weight(t):
    w = torch.ones_like(t)
    w[t > 5.0] = 3.0
    w[t > 20.0] = 8.0
    return w

best_v, best_ep = 1e9, 0
t0 = time.time()
for ep in range(60):
    model.train(); tot = 0.0; nb = 0
    for xb, tb in tr_loader:
        xb, tb = xb.to(DEVICE), tb.to(DEVICE)
        res, _, _ = model(xb)
        mse = (res - tb.unsqueeze(1)) ** 2
        loss = (mse * storm_weight(tb.unsqueeze(1))).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    sched.step()
    model.eval(); vt = 0.0; nv = 0
    with torch.no_grad():
        for xb, tb in val_loader:
            xb, tb = xb.to(DEVICE), tb.to(DEVICE)
            res, _, _ = model(xb)
            vt += ((res - tb.unsqueeze(1)) ** 2).mean().item(); nv += 1
    v = vt / nv
    if v < best_v:
        best_v = v; best_ep = ep
        torch.save(model.state_dict(), os.path.join(OUT, 'apcnet_24h.pth'))
    if ep % 10 == 0 or ep == 59:
        print(f'  ep {ep}: train {tot/nb:.4f} val {v:.4f} ({time.time()-t0:.0f}s)')
print(f'best val {best_v:.4f} @ ep {best_ep}')

# ---------------- 3. 推理 ----------------
print('\n[推理]')
model.load_state_dict(torch.load(os.path.join(OUT, 'apcnet_24h.pth'), map_location=DEVICE))
model.eval()
preds = np.zeros((len(te_ds), 25, 37), np.float32)
with torch.no_grad():
    for i in range(0, len(te_ds), 64):
        xb = torch.from_numpy(feats_te[i:i+64]).float().unsqueeze(1).to(DEVICE)
        res, _, _ = model(xb)
        r = res.cpu().numpy()[:, 0]
        preds[i:i+64] = np.clip(gfs_te[i:i+64] + r, 0, 500)
np.save(os.path.join(OUT, 'pred_apcnet_24h.npy'), preds)

# ---------------- 4. 基线 ----------------
print('\n[基线]')
# QM：逐格点分位数映射
def fit_qm(g, e, nb=200):
    qa = np.linspace(0, 1, nb + 1)
    H, W = g.shape[1], g.shape[2]
    gq = np.zeros((H, W, nb + 1)); eq = np.zeros((H, W, nb + 1))
    for i in range(H):
        for j in range(W):
            gq[i, j] = np.quantile(g[:, i, j], qa)
            eq[i, j] = np.quantile(e[:, i, j], qa)
    return gq, eq
def apply_qm(g, gq, eq):
    H, W = g.shape[1], g.shape[2]
    out = np.zeros_like(g)
    for i in range(H):
        for j in range(W):
            out[:, i, j] = np.interp(g[:, i, j], gq[i, j], eq[i, j])
    return out
gq, eq = fit_qm(gfs_tr, era5_tr)
pred_qm = apply_qm(gfs_te, gq, eq)

# BM：分箱条件偏差
bins = np.array([0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500])
def fit_bm(g, e):
    H, W = g.shape[1], g.shape[2]
    ratio = np.zeros((H, W, len(bins) - 1))
    for i in range(H):
        for j in range(W):
            for k in range(len(bins) - 1):
                m = (g[:, i, j] >= bins[k]) & (g[:, i, j] < bins[k + 1])
                ratio[i, j, k] = np.mean(e[m, i, j]) / np.mean(g[m, i, j]) if m.sum() > 5 else 1.0
    return ratio
def apply_bm(g, ratio):
    out = np.zeros_like(g)
    for i in range(g.shape[1]):
        for j in range(g.shape[2]):
            x = g[:, i, j]
            kk = np.clip(np.searchsorted(bins, x) - 1, 0, len(bins) - 2)
            out[:, i, j] = x * ratio[i, j, kk]
    return out
bm_r = fit_bm(gfs_tr, era5_tr)
pred_bm = apply_bm(gfs_te, bm_r)

# OLS：逐格点线性
def fit_ols(g, e):
    H, W = g.shape[1], g.shape[2]
    a = np.zeros((H, W)); b = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            X = g[:, i, j]; Y = e[:, i, j]
            a[i, j], b[i, j] = np.polyfit(X, Y, 1)
    return a, b
def apply_ols(g, a, b):
    return g * a + b
ols_a, ols_b = fit_ols(gfs_tr, era5_tr)
pred_ols = apply_ols(gfs_te, ols_a, ols_b)

# ---------------- 5. 评估 ----------------
print('\n[评估]')
def stats(pred, obs, gfs_raw, tag):
    mse = np.mean((pred - obs) ** 2)
    mae = np.mean(np.abs(pred - obs))
    cc = np.corrcoef(pred.ravel(), obs.ravel())[0, 1]
    rmse_g = np.sqrt(np.mean((gfs_raw - obs) ** 2))
    r = {'mse': float(mse), 'mae': float(mae), 'cc': float(cc), 'gfs_rmse': float(rmse_g)}
    for th in [0.1, 1, 5, 10, 20, 30]:
        p, o = pred >= th, obs >= th
        g0 = gfs_raw >= th
        hit = ((p) & (o)).sum(); fa = ((p) & (~o)).sum(); mi = ((~p) & (o)).sum()
        pod = hit / (hit + mi + 1e-9); far = fa / (hit + fa + 1e-9)
        exp = (p.sum() * o.sum()) / (p.size + 1e-9)
        ets = (hit - exp) / (hit + fa + mi - exp + 1e-9)
        r[f'pod_{th}'] = float(pod); r[f'far_{th}'] = float(far); r[f'ets_{th}'] = float(ets)
    # FSS 5x5 邻域
    for th in [1, 5, 10, 20]:
        p = (pred >= th).astype(np.float32); o = (obs >= th).astype(np.float32)
        from scipy.ndimage import uniform_filter
        ps = uniform_filter(p, size=5); os_ = uniform_filter(o, size=5)
        mse_f = np.mean((ps - os_) ** 2)
        denom = np.mean(ps ** 2) + np.mean(os_ ** 2)
        r[f'fss_{th}'] = float(1 - mse_f / denom) if denom > 0 else 1.0
    return r

results = {}
results['gfs_raw'] = stats(gfs_te, era5_te, gfs_te, 'gfs')
results['apcnet'] = stats(preds, era5_te, gfs_te, 'apcnet')
results['qm'] = stats(pred_qm, era5_te, gfs_te, 'qm')
results['bm'] = stats(pred_bm, era5_te, gfs_te, 'bm')
results['ols'] = stats(pred_ols, era5_te, gfs_te, 'ols')

with open(os.path.join(OUT, '24h_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(json.dumps(results, ensure_ascii=False, indent=2))
print('完成')
