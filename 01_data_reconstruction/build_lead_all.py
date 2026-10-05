# -*- coding: utf-8 -*-
"""
多时效实验泛化版：build dataset + features + train + eval 全流程
用法: python build_lead_all.py --fhr 72   (或 120)
数据: jiangshui f0{fhr} 降水 + GFS f0{fhr} 大气场 + ERA5 hourly 聚合目标
输出: multi_lead_exp/{fhr}h/{fhr}h_results.json 等
"""
import argparse, importlib.util, numpy as np, os, glob, json, time, sys
import torch, torch.nn as nn
from datetime import datetime, timedelta
import netCDF4, multiprocessing as mp
from scipy.io import netcdf_file

BASE = r'C:\Users\yg181\Desktop\论文三\13.0修复重跑'
GFS_ROOT = r'D:\liaohe\GFS-data'
JS_ROOT = os.path.join(GFS_ROOT, 'jiangshui')
ERA5_H = r'D:\liaohe\ERA5-data\new_hourly_tp'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fhr', type=int, required=True)
    return p.parse_args()

# ---------------- 数据构建 ----------------
def build_dataset(fhr, js_dir, out_dir):
    print(f'[{fhr}h] 构建数据集...')
    t0 = time.time()
    f024_by_init = {}
    for f in glob.glob(os.path.join(js_dir, '*.nc')):
        s = os.path.basename(f).split('.')[2]
        f024_by_init[datetime.strptime(s, '%Y%m%d%H')] = f
    print(f'  降水文件 init 数: {len(f024_by_init)}')

    inits = []
    t = datetime(2019, 6, 12, 0)
    end = datetime(2025, 12, 31, 23) - timedelta(hours=fhr)
    while t <= end:
        inits.append(t)
        t += timedelta(hours=6)
    N = len(inits)
    print(f'  init 数: {N}')

    era5_ds, era5_vt0 = {}, {}
    def get_era5(year):
        if year not in era5_ds:
            ds = netCDF4.Dataset(os.path.join(ERA5_H, f'era5_tp_hourly_{year}.nc'))
            era5_ds[year] = ds
            era5_vt0[year] = int(ds.variables['valid_time'][0])
        return era5_ds[year]

    gfs_a = np.zeros((N, 25, 37), np.float32)
    era5_a = np.zeros((N, 25, 37), np.float32)
    miss_g = miss_e = 0
    ref = datetime(1970, 1, 1, 0)
    for i, init in enumerate(inits):
        f = f024_by_init.get(init)
        if f is None:
            miss_g += 1; gfs_a[i] = np.nan
        else:
            ds = netCDF4.Dataset(f); gfs_a[i] = ds.variables['A_PCP_L1_Accum_1'][0]; ds.close()
        acc_e = np.zeros((25, 37), np.float32); ok = True
        for k in range(1, fhr + 1):
            tv = init + timedelta(hours=k)
            ds = get_era5(tv.year)
            rel = (int((tv - ref).total_seconds()) - era5_vt0[tv.year]) // 3600
            try:
                acc_e += ds.variables['tp'][rel, :, :]
            except Exception:
                ok = False; break
        if not ok:
            miss_e += 1; era5_a[i] = np.nan
        else:
            era5_a[i] = acc_e * 1000.0
        if (i + 1) % 2000 == 0:
            print(f'    {i+1}/{N}')
    print(f'  完成: GFS 缺 {miss_g}, ERA5 缺 {miss_e}')
    valid = ~(np.isnan(gfs_a).any((1,2)) | np.isnan(era5_a).any((1,2)))
    print(f'  有效: {valid.sum()}/{N}')
    split = np.zeros(N, np.int8)
    for i, init in enumerate(inits):
        split[i] = 0 if init.year <= 2021 else (1 if init.year <= 2023 else 2)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'gfs_{fhr}h_accum.npy'), gfs_a)
    np.save(os.path.join(out_dir, f'era5_{fhr}h_accum.npy'), era5_a)
    np.save(os.path.join(out_dir, 'init_times.npy'), np.array([np.datetime64(i, 'h') for i in inits]))
    np.save(os.path.join(out_dir, 'split_mask.npy'), split)
    g, e = gfs_a[valid], era5_a[valid]
    acc = {'domain_mean_gfs_mm': float(np.nanmean(g)), 'domain_mean_era5_mm': float(np.nanmean(e)),
           'gfs_era5_ratio': float(np.nanmean(g) / np.nanmean(e)),
           'annual_gfs_mm': float(np.nanmean(g) * 365 / (fhr / 24.0)),
           'annual_era5_mm': float(np.nanmean(e) * 365 / (fhr / 24.0)),
           'pixel_corr': float(np.corrcoef(g.reshape(len(g), -1).T, e.reshape(len(e), -1).T)[0, 1])}
    with open(os.path.join(out_dir, f'{fhr}h_acceptance.json'), 'w') as f:
        json.dump(acc, f, ensure_ascii=False, indent=2)
    print('  验收:', json.dumps(acc, ensure_ascii=False))
    print(f'  耗时 {time.time()-t0:.0f}s')
    return inits

# ---------------- 特征提取 ----------------
def extract_one(args):
    fhr, init, pwat_from72 = args
    fname = init.strftime(f'gfs.0p25.%Y%m%d%H.f{fhr:03d}.grib2.nc')
    ch = np.zeros((8, 25, 37), np.float32)
    try:
        if fhr == 72:
            ds = netCDF4.Dataset(os.path.join(GFS_ROOT, f'gfs.0p25.2015-2025.f{fhr:03d}', fname))
            def rd(name, sl=None):
                v = ds.variables[name]
                return v[:] if sl is None else v[sl]
        else:
            ds = netcdf_file(os.path.join(GFS_ROOT, f'gfs.0p25.2015-2025.f{fhr:03d}', fname), 'r', mmap=False)
            def rd(name, sl=None):
                v = ds.variables[name].data
                return v if sl is None else v[sl]
        cape = rd('CAPE_L1', (0,)) if fhr == 72 else rd('CAPE_L1')
        ch[0] = cape[0] if cape.ndim == 3 else cape
        u = rd('U_GRD_L100', (0,)) if fhr == 72 else rd('U_GRD_L100')
        v = rd('V_GRD_L100', (0,)) if fhr == 72 else rd('V_GRD_L100')
        lv3 = ds.variables['level3'][:]
        i850 = int(np.argmin(np.abs(lv3 - 850))); i500 = int(np.argmin(np.abs(lv3 - 500)))
        ch[2] = u[0, i850]; ch[3] = v[0, i850]; ch[4] = u[0, i500]; ch[5] = v[0, i500]
        vv = rd('V_VEL_L100', (0,)) if fhr == 72 else rd('V_VEL_L100')
        lv = ds.variables['level5'][:] if 'level5' in ds.variables else lv3
        i500v = int(np.argmin(np.abs(lv - 500)))
        ch[6] = vv[0, i500v] * 0.01
        if not pwat_from72:
            pwat = ds.variables['P_WAT_L200'][0] if fhr == 72 else ds.variables['P_WAT_L200'].data[0]
            ch[1] = pwat
        if fhr == 72:
            ds.close()
        else:
            ds.close()
    except Exception as e:
        print(f'  [warn] f{fhr} fail {fname}: {e}')
        return None
    # PWAT: f120 无此变量时从 f072 读
    if pwat_from72 or ch[1].sum() == 0:
        try:
            f72 = init.strftime('gfs.0p25.%Y%m%d%H.f072.grib2.nc')
            ds72 = netCDF4.Dataset(os.path.join(GFS_ROOT, 'gfs.0p25.2015-2025.f072', f72))
            ch[1] = ds72.variables['P_WAT_L200'][0]
            ds72.close()
        except Exception as e:
            print(f'  [warn] f072 PWAT fail: {e}')
    return ch

def build_features(fhr, out_dir, inits):
    print(f'[{fhr}h] 特征提取...')
    t0 = time.time()
    pw_from72 = (fhr == 120)
    args = [(fhr, init, pw_from72) for init in inits]
    feats = np.zeros((len(inits), 8, 25, 37), np.float32)
    with mp.Pool(min(12, mp.cpu_count())) as pool:
        for i, ch in enumerate(pool.imap_unordered(extract_one, args, chunksize=16)):
            if ch is not None:
                feats[i] = ch
    np.save(os.path.join(out_dir, f'features_{fhr}h.npy'), feats)
    print('  通道均值:', np.round(feats.mean(axis=(0, 2, 3)), 3))
    print(f'  耗时 {time.time()-t0:.0f}s')

# ---------------- 训练 + 基线 + 评估 ----------------
def train_eval(fhr, out_dir):
    print(f'[{fhr}h] 训练与评估...')
    spec = importlib.util.spec_from_file_location('m13', os.path.join(BASE, '13.0_main.py'))
    m13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m13)
    APCNet = m13.AdvancedPrecipCorrectionNet
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    feats = np.load(os.path.join(out_dir, f'features_{fhr}h.npy'))
    gfs = np.load(os.path.join(out_dir, f'gfs_{fhr}h_accum.npy'))
    era5 = np.load(os.path.join(out_dir, f'era5_{fhr}h_accum.npy'))
    split = np.load(os.path.join(out_dir, 'split_mask.npy'))
    valid = ~(np.isnan(gfs).any((1,2)) | np.isnan(era5).any((1,2)))
    feats = feats[valid]; gfs = gfs[valid]; era5 = era5[valid]; split = split[valid]
    feats[:, 7] = gfs
    tr = split == 0
    mu = feats[tr][:, :7].mean(axis=(0,2,3), keepdims=True)
    sd = feats[tr][:, :7].std(axis=(0,2,3), keepdims=True) + 1e-6
    feats[:, :7] = (feats[:, :7] - mu) / sd
    from torch.utils.data import Dataset, DataLoader
    class DS(Dataset):
        def __init__(self, x, t): self.x = torch.from_numpy(x).float(); self.t = torch.from_numpy(t).float()
        def __len__(self): return len(self.x)
        def __getitem__(self, i): return self.x[i].unsqueeze(0), self.t[i]
    tr_ds = DS(feats[tr], era5[tr] - gfs[tr])
    vm = split == 1; va_ds = DS(feats[vm], era5[vm] - gfs[vm])
    te_m = split == 2
    gfs_te, era5_te, feats_te = gfs[te_m], era5[te_m], feats[te_m]
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)

    model = APCNet(input_channels=8, hidden_channels=24, sequence_length=1, spatial_dims=(25,37), dropout_rate=0.1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    def sw(t): return torch.where(t > 20.0, torch.full_like(t, 8.0), torch.where(t > 5.0, torch.full_like(t, 3.0), torch.ones_like(t)))
    best_v, best_ep = 1e9, 0
    for ep in range(60):
        model.train(); tot = 0.0; nb = 0
        for xb, tb in tr_loader:
            xb, tb = xb.to(DEVICE), tb.to(DEVICE)
            res, _, _ = model(xb)
            loss = ((res - tb.unsqueeze(1)) ** 2 * sw(tb.unsqueeze(1))).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        sched.step()
        model.eval(); vt = 0.0; nv = 0
        with torch.no_grad():
            for xb, tb in va_loader:
                xb, tb = xb.to(DEVICE), tb.to(DEVICE)
                res, _, _ = model(xb)
                vt += ((res - tb.unsqueeze(1)) ** 2).mean().item(); nv += 1
        v = vt / nv
        if v < best_v:
            best_v = v; best_ep = ep
            torch.save(model.state_dict(), os.path.join(out_dir, f'apcnet_{fhr}h.pth'))
    print(f'  best val {best_v:.4f} @ {best_ep}')
    model.load_state_dict(torch.load(os.path.join(out_dir, f'apcnet_{fhr}h.pth'), map_location=DEVICE))
    model.eval()
    n_te = int(te_m.sum())
    preds = np.zeros((n_te, 25, 37), np.float32)
    with torch.no_grad():
        for i in range(0, n_te, 64):
            xb = torch.from_numpy(feats_te[i:i+64]).float().unsqueeze(1).to(DEVICE)
            res, _, _ = model(xb)
            preds[i:i+64] = np.clip(gfs_te[i:i+64] + res.cpu().numpy()[:, 0], 0, 600)
    np.save(os.path.join(out_dir, f'pred_apcnet_{fhr}h.npy'), preds)

    # 基线
    def fit_qm(g, e, nb=200):
        qa = np.linspace(0, 1, nb + 1); H, W = g.shape[1], g.shape[2]
        gq = np.zeros((H, W, nb + 1)); eq = np.zeros((H, W, nb + 1))
        for i in range(H):
            for j in range(W):
                gq[i, j] = np.quantile(g[:, i, j], qa); eq[i, j] = np.quantile(e[:, i, j], qa)
        return gq, eq
    def apply_qm(g, gq, eq):
        H, W = g.shape[1], g.shape[2]; out = np.zeros_like(g)
        for i in range(H):
            for j in range(W):
                out[:, i, j] = np.interp(g[:, i, j], gq[i, j], eq[i, j])
        return out
    gq, eq = fit_qm(gfs[tr], era5[tr])
    pred_qm = apply_qm(gfs_te, gq, eq)
    bins = np.array([0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 600])
    ratio = np.zeros((25, 37, len(bins) - 1))
    for i in range(25):
        for j in range(37):
            for k in range(len(bins) - 1):
                m = (gfs[tr][:, i, j] >= bins[k]) & (gfs[tr][:, i, j] < bins[k+1])
                ratio[i, j, k] = np.mean(era5[tr][m, i, j]) / np.mean(gfs[tr][m, i, j]) if m.sum() > 5 else 1.0
    pred_bm = np.zeros_like(gfs_te)
    for i in range(25):
        for j in range(37):
            kk = np.clip(np.searchsorted(bins, gfs_te[:, i, j]) - 1, 0, len(bins) - 2)
            pred_bm[:, i, j] = gfs_te[:, i, j] * ratio[i, j, kk]
    a = np.zeros((25, 37)); b = np.zeros((25, 37))
    for i in range(25):
        for j in range(37):
            a[i, j], b[i, j] = np.polyfit(gfs[tr][:, i, j], era5[tr][:, i, j], 1)
    pred_ols = gfs_te * a + b

    from scipy.ndimage import uniform_filter
    def stats(pred, obs, g0, tag):
        r = {'mse': float(np.mean((pred - obs) ** 2)), 'mae': float(np.mean(np.abs(pred - obs))),
             'cc': float(np.corrcoef(pred.ravel(), obs.ravel())[0, 1])}
        for th in [0.1, 1, 5, 10, 20, 30]:
            p, o = pred >= th, obs >= th
            hit = ((p) & (o)).sum(); fa = ((p) & (~o)).sum(); mi = ((~p) & (o)).sum()
            pod = hit / (hit + mi + 1e-9); far = fa / (hit + fa + 1e-9)
            exp = (p.sum() * o.sum()) / (p.size + 1e-9)
            r[f'pod_{th}'] = float(pod); r[f'far_{th}'] = float(far); r[f'ets_{th}'] = float((hit - exp) / (hit + fa + mi - exp + 1e-9))
        for th in [1, 5, 10, 20]:
            ps = uniform_filter((pred >= th).astype(np.float32), 5); os_ = uniform_filter((obs >= th).astype(np.float32), 5)
            ms = np.mean((ps - os_) ** 2); dn = np.mean(ps ** 2) + np.mean(os_ ** 2)
            r[f'fss_{th}'] = float(1 - ms / dn) if dn > 0 else 1.0
        return r
    results = {k: stats(p, era5_te, gfs_te, k) for k, p in [('gfs_raw', gfs_te), ('apcnet', preds), ('qm', pred_qm), ('bm', pred_bm), ('ols', pred_ols)]}
    with open(os.path.join(out_dir, f'{fhr}h_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))

def main():
    args = parse_args()
    fhr = args.fhr
    out_dir = os.path.join(BASE, 'multi_lead_exp', f'{fhr}h')
    js_dir = glob.glob(os.path.join(JS_ROOT, f'*f{fhr:03d}'))[0]
    print('jiangshui dir:', js_dir)
    inits = build_dataset(fhr, js_dir, out_dir)
    build_features(fhr, out_dir, inits)
    train_eval(fhr, out_dir)

if __name__ == '__main__':
    mp.freeze_support()
    main()
