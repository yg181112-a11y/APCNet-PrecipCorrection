# -*- coding: utf-8 -*-
"""U-Net 多时效对照训练（24/72/120h）——尺度效应与架构无关性检验
- 模型：StandardUNet（与 3h 主实验 U-Net 完全一致，hidden=32）
- 数据/评估：复用 train_24h.py 管线（残差目标、storm_weight、统一评估）
"""
import importlib.util, numpy as np, os, json, time, argparse
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
spec = importlib.util.spec_from_file_location('m13', os.path.join(BASE, '13.0_main.py'))
m13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m13)
UNet = m13.StandardUNet
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE)

ap = argparse.ArgumentParser()
ap.add_argument('--fhr', type=int, required=True)
args = ap.parse_args()
fhr = args.fhr

if fhr == 24:
    D = os.path.join(BASE, '24h_exp')
    gf, tf, ff = 'gfs_24h_accum.npy', 'era5_24h_accum.npy', 'features_24h.npy'
    out_tag = 'unet_24h'
else:
    D = os.path.join(BASE, 'multi_lead_exp', f'{fhr}h')
    gf, tf, ff = f'gfs_{fhr}h_accum.npy', f'era5_{fhr}h_accum.npy', f'features_{fhr}h.npy'
    out_tag = f'unet_{fhr}h'
OUT = os.path.join(D, out_tag + '.npy')
OUT_PTH = os.path.join(D, out_tag + '.pth')

feats = np.load(os.path.join(D, ff))
gfs = np.load(os.path.join(D, gf))
era5 = np.load(os.path.join(D, tf))
split = np.load(os.path.join(D, 'split_mask.npy'))
valid = ~(np.isnan(gfs).any((1,2)) | np.isnan(era5).any((1,2)))
feats = feats[valid]; gfs = gfs[valid]; era5 = era5[valid]; split = split[valid]
feats[:, 7] = gfs
N = len(feats)
print(f'[{fhr}h] 样本 {N}: train {(split==0).sum()}, val {(split==1).sum()}, test {(split==2).sum()}')

tr = split == 0
mu = feats[tr][:, :7].mean(axis=(0,2,3), keepdims=True)
sd = feats[tr][:, :7].std(axis=(0,2,3), keepdims=True) + 1e-6
feats[:, :7] = (feats[:, :7] - mu) / sd

class DS(Dataset):
    def __init__(self, x, t):
        self.x = torch.from_numpy(x).float(); self.t = torch.from_numpy(t).float()
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i].unsqueeze(0), self.t[i]

train_ds = DS(feats[tr], era5[tr] - gfs[tr])
val_mask = split == 1; val_ds = DS(feats[val_mask], era5[val_mask] - gfs[val_mask])
te_mask = split == 2
gfs_te, era5_te, feats_te = gfs[te_mask], era5[te_mask], feats[te_mask]
tr_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

print(f'[训练 U-Net-{fhr}h]')
if os.path.exists(OUT_PTH):
    print('  已存在权重，跳过训练，直接推理')
else:
    model = UNet(input_channels=8, hidden_channels=32).to(DEVICE)
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
            mse = (res[:, 0] - tb) ** 2
            loss = (mse * storm_weight(tb)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        model.eval(); vt = 0.0; nv = 0
        with torch.no_grad():
            for xb, tb in val_loader:
                xb, tb = xb.to(DEVICE), tb.to(DEVICE)
                res, _, _ = model(xb)
                vt += ((res[:, 0] - tb) ** 2).mean().item(); nv += 1
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
np.save(OUT, preds)

# 评估（同 train_24h.py stats）
def stats(pred, obs, gfs_raw, tag):
    mse = np.mean((pred - obs) ** 2)
    mae = np.mean(np.abs(pred - obs))
    cc = np.corrcoef(pred.ravel(), obs.ravel())[0, 1]
    rmse_g = np.sqrt(np.mean((gfs_raw - obs) ** 2))
    r = {'mse': float(mse), 'mae': float(mae), 'cc': float(cc), 'gfs_rmse': float(rmse_g)}
    for th in [0.1, 1, 5, 10, 20, 30]:
        p, o, g0 = pred >= th, obs >= th, gfs_raw >= th
        hit = (p & o).sum(); fa = (p & ~o).sum(); mi = (~p & o).sum()
        pod = hit / (hit + mi + 1e-9); far = fa / (hit + fa + 1e-9)
        exp = (p.sum() * o.sum()) / (p.size + 1e-9)
        r[f'pod_{th}'] = float(pod); r[f'far_{th}'] = float(far); r[f'ets_{th}'] = float((hit - exp) / (hit + fa + mi - exp + 1e-9))
    from scipy.ndimage import uniform_filter
    for th in [1, 5, 10, 20]:
        p = (pred >= th).astype(np.float32); o = (obs >= th).astype(np.float32)
        ps = uniform_filter(p, size=5); os_ = uniform_filter(o, size=5)
        mse_f = np.mean((ps - os_) ** 2)
        denom = np.mean(ps ** 2) + np.mean(os_ ** 2)
        r[f'fss_{th}'] = float(1 - mse_f / denom) if denom > 0 else 1.0
    return r

results = {'gfs_raw': stats(gfs_te, era5_te, gfs_te, 'gfs'), out_tag: stats(preds, era5_te, gfs_te, out_tag)}
json.dump(results, open(os.path.join(D, f'{out_tag}_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps(results, ensure_ascii=False, indent=2))
print('完成', OUT)
