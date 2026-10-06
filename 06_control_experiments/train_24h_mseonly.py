# -*- coding: utf-8 -*-
"""
M5 复核（v5 审稿意见 + 风险对冲）：24h 纯 MSE 单 seed 复核。
协议与 train_24h_seed.py 完全一致，仅损失改为纯 MSE（去掉 storm_weight 强度加权），
选点仍为验证 MSE（与 24h 主实验一致），seed 42。
输出: 24h_exp/pred_apcnet_24h_mseonly.npy, apcnet_24h_mseonly.pth, mseonly_24h_results.json
"""
import base64 as _b64
import importlib.util, numpy as np, os, json, time
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

OUT = _b64.b64decode(os.environ['OUT_WORK_B64']).decode('utf-8')
DATA = _b64.b64decode(os.environ['DATA_DIR_B64']).decode('utf-8')
SEED = int(os.environ.get('SEED', '42'))
import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f'SEED={SEED} OUT={OUT}')

spec = importlib.util.spec_from_file_location('m13', r'D:\liaohe\论文三\03_重建成稿代_2026_R3全链主实验\13.0_main.py')
m13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m13)
APCNet = m13.AdvancedPrecipCorrectionNet
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE)

# ---------------- 1. 数据（与 24h 主实验一致） ----------------
feats = np.load(os.path.join(DATA, 'features_24h.npy'))
gfs   = np.load(os.path.join(DATA, 'gfs_24h_accum.npy'))
era5  = np.load(os.path.join(DATA, 'era5_24h_accum.npy'))
split = np.load(os.path.join(DATA, 'split_mask.npy'))
valid = ~(np.isnan(gfs).any((1,2)) | np.isnan(era5).any((1,2)))
feats = feats[valid]; gfs = gfs[valid]; era5 = era5[valid]; split = split[valid]
feats[:, 7] = gfs
N = len(feats)
print(f'样本 {N}: train {(split==0).sum()}, val {(split==1).sum()}, test {(split==2).sum()}')

tr = split == 0
mu = feats[tr][:, :7].mean(axis=(0,2,3), keepdims=True)
sd = feats[tr][:, :7].std(axis=(0,2,3), keepdims=True) + 1e-6
feats[:, :7] = (feats[:, :7] - mu) / sd

class DS(Dataset):
    def __init__(self, x, t):
        self.x = torch.from_numpy(x).float()
        self.t = torch.from_numpy(t).float()
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return self.x[i].unsqueeze(0), self.t[i]

train_ds = DS(feats[tr], era5[tr] - gfs[tr])
val_mask = split == 1; val_ds = DS(feats[val_mask], era5[val_mask] - gfs[val_mask])
te_mask = split == 2;  te_ds = DS(feats[te_mask], era5[te_mask] - gfs[te_mask])
gfs_tr, era5_tr = gfs[tr], era5[tr]
gfs_te, era5_te = gfs[te_mask], era5[te_mask]
feats_te = feats[te_mask]

tr_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

# ---------------- 2. 训练：纯 MSE 损失（无 storm_weight）+ MSE 选点 ----------------
print('\n[训练 APCNet-24h 纯 MSE, seed %d]' % SEED)
model = APCNet(input_channels=8, hidden_channels=24, sequence_length=1, spatial_dims=(25,37), dropout_rate=0.1).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

best_v, best_ep = 1e9, 0
t0 = time.time()
for ep in range(60):
    model.train(); tot = 0.0; nb = 0
    for xb, tb in tr_loader:
        xb, tb = xb.to(DEVICE), tb.to(DEVICE)
        res, _, _ = model(xb)
        mse = (res - tb.unsqueeze(1)) ** 2
        loss = mse.mean()                      # 纯 MSE（关键改动）
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
        torch.save(model.state_dict(), os.path.join(OUT, 'apcnet_24h_mseonly.pth'))
    if ep % 10 == 0 or ep == 59:
        print(f'  ep {ep}: train {tot/nb:.4f} val {v:.4f} ({time.time()-t0:.0f}s)')
print(f'best val {best_v:.4f} @ ep {best_ep}')

# ---------------- 3. 推理 ----------------
print('\n[推理]')
model.load_state_dict(torch.load(os.path.join(OUT, 'apcnet_24h_mseonly.pth'), map_location=DEVICE))
model.eval()
preds = np.zeros((len(te_ds), 25, 37), np.float32)
with torch.no_grad():
    for i in range(0, len(te_ds), 64):
        xb = torch.from_numpy(feats_te[i:i+64]).float().unsqueeze(1).to(DEVICE)
        res, _, _ = model(xb)
        r = res.cpu().numpy()[:, 0]
        preds[i:i+64] = np.clip(gfs_te[i:i+64] + r, 0, 500)
np.save(os.path.join(OUT, 'pred_apcnet_24h_mseonly.npy'), preds)

# ---------------- 4. 基线（与主实验同协议） ----------------
print('\n[基线]')
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

# ---------------- 5. 评估（ERA5 参考） ----------------
print('\n[评估 ERA5 参考]')
def stats(pred, obs, gfs_raw, tag):
    mse = np.mean((pred - obs) ** 2)
    mae = np.mean(np.abs(pred - obs))
    cc = np.corrcoef(pred.ravel(), obs.ravel())[0, 1]
    rmse_g = np.sqrt(np.mean((gfs_raw - obs) ** 2))
    r = {'mse': float(mse), 'mae': float(mae), 'cc': float(cc), 'gfs_rmse': float(rmse_g),
         'mse_imp': float(1 - mse / np.mean((gfs_raw - obs) ** 2))}
    return r

results = {}
results['gfs_raw'] = stats(gfs_te, era5_te, gfs_te, 'gfs')
results['apcnet_mseonly'] = stats(preds, era5_te, gfs_te, 'apcnet_mseonly')
results['qm'] = stats(pred_qm, era5_te, gfs_te, 'qm')
results['bm'] = stats(pred_bm, era5_te, gfs_te, 'bm')
results['ols'] = stats(pred_ols, era5_te, gfs_te, 'ols')

with open(os.path.join(OUT, 'mseonly_24h_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(json.dumps(results, ensure_ascii=False, indent=2))
print('完成')
