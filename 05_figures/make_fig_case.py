# -*- coding: utf-8 -*-
"""
Fig. 10 (candidate): Typical NECV heavy-rain case — spatial verification of
GPM IMERG obs vs GFS/QM/OLS/APCNet/U-Net at the 3-h scale (test period 2024-2025).
Panel layout: 2x3 spatial fields of the peak window + a domain-mean 3-h time
series around the event. Publication style: Okabe-Ito, Arial, despine, 300 dpi + PDF.
"""
import os, pickle
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm

WORK = r"D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work"
GPM_ROOT = r"D:\liaohe\GPM_IMERG"
APC42 = os.path.join(WORK, 'predictions_apcnet.npy')           # run13 S42
OUT = r"C:\Users\yg181\Desktop\论文三\WAF\r3_media"
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)
WINDOW_END_HOURS = [3, 9, 15, 21]
UTC = timezone.utc

OKABE = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9', '#F0E442']

# ---- load model fields ----
with open(os.path.join(WORK, "sample_times_test.pkl"), "rb") as f:
    times = pickle.load(f)
gfs = np.load(os.path.join(WORK, "gfs_test.npy"))
apc = np.load(APC42)
qm = np.load(os.path.join(WORK, "predictions_qm.npy"))
ols = np.load(os.path.join(WORK, "predictions_ols.npy"))
unet = np.load(os.path.join(WORK, '..', 'manuscript_work_unet', 'predictions_apcnet.npy'))

# ---- GPM 3-h accumulation (same pipeline as gpm3h_eval.py) ----
def gpm_3h_files(dt):
    return [os.path.join(GPM_ROOT, f"imerg_{e.year}{e.month:02d}",
                         f"imerg_{e.year}{e.month:02d}{e.day:02d}_{e.hour:02d}{e.minute:02d}00.nc4")
            for i in range(6)
            for e in [dt - timedelta(minutes=30 * (5 - i))]]

def load_gpm_field(path):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    g = ds.groups['Grid']
    prec = g.variables['precipitation'][0]
    lat = g.variables['lat'][:]; lon = g.variables['lon'][:]
    ds.close()
    return prec.T, lat, lon

def regrid_to_main(field, lat, lon):
    from scipy.interpolate import RegularGridInterpolator
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    interp = RegularGridInterpolator((lat, lon), field, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    vals = interp(pts).reshape(25, 37)
    return np.where(np.isnan(vals), 0.0, vals)

gpm3, vt = [], []
for i, t in enumerate(times):
    t = t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)
    if t.hour not in WINDOW_END_HOURS:
        continue
    fs = gpm_3h_files(t)
    if any(not os.path.exists(f) for f in fs):
        continue
    acc = None
    for f in fs:
        field, lat, lon = load_gpm_field(f)
        r = regrid_to_main(field, lat, lon)
        acc = r if acc is None else acc + r
    gpm3.append(acc * 0.5)
    vt.append(t)
gpm3 = np.array(gpm3)
n = len(vt)
print(f"GPM samples: {n}  ({vt[0]} ~ {vt[-1]})", flush=True)

idx = {t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC): i for i, t in enumerate(times)}
sel = [idx[t] for t in vt]
G = gfs[sel].reshape(n, 25, 37)
A = apc[sel].reshape(n, 25, 37)
Q = qm[sel].reshape(n, 25, 37)
O = ols[sel].reshape(n, 25, 37)
U = unet[sel].reshape(n, 25, 37)
OBS = gpm3.reshape(n, 25, 37)

# ---- select the event: strongest summer (JJA) episode with >=3 consecutive 3-h windows ----
dm = OBS.mean(axis=(1, 2))
mon = np.array([t.month for t in vt])
summer = (mon >= 6) & (mon <= 8)
cand = np.where(summer & (dm >= 0.8))[0]
# find consecutive runs
runs, cur = [], [cand[0]] if len(cand) else []
for k in range(1, len(cand)):
    if cand[k] == cand[k-1] + 1:
        cur.append(cand[k])
    else:
        if len(cur) >= 3:
            runs.append(cur)
        cur = [cand[k]]
if len(cur) >= 3:
    runs.append(cur)
best = max(runs, key=lambda r: float(dm[r].max()))
pk = int(best[np.argmax(dm[best])])
print("event runs found:", len(runs), "| picked peak index", pk, vt[pk],
      "domain-mean obs %.2f mm/3h" % dm[pk], flush=True)

# ---- figure ----
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
                     'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'axes.linewidth': 0.7})
vmax = 15.0
cmap = matplotlib.colormaps['viridis']

fig = plt.figure(figsize=(7.0, 6.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.13,
                      left=0.06, right=0.97, top=0.74, bottom=0.22)
panels = [("GPM IMERG (obs)", OBS, "(a)"), ("GFS (raw)", G, "(b)"), ("QM", Q, "(c)"),
          ("OLS", O, "(d)"), ("APCNet", A, "(e)"), ("U-Net", U, "(f)")]
for k, (label, F, lab) in enumerate(panels):
    ax = fig.add_subplot(gs[k // 3, k % 3])
    im = ax.imshow(F[pk], origin='upper', cmap=cmap, vmin=0, vmax=vmax, interpolation='bilinear',
                   extent=[GLOBAL_LONS[0], GLOBAL_LONS[-1], GLOBAL_LATS[-1], GLOBAL_LATS[0]],
                   aspect='auto')
    ax.set_title('%s  (%.2f)' % (label, float(F[pk].mean())))
    y_off = -0.16 if k // 3 == 0 else -0.30   # 第一行无 x 轴标签，编号贴图下方；第二行在 x 标签之下
    ax.text(0.5, y_off, lab, transform=ax.transAxes, fontsize=11, fontweight='bold',
            va='top', ha='center')
    ax.set_xticks([118, 121, 124])
    ax.set_yticks([41, 43, 45])
    if k % 3 == 0:
        ax.set_ylabel('Lat (N)')
    else:
        ax.set_yticklabels([])
    if k // 3 == 1:
        ax.set_xlabel('Lon (E)')
    else:
        ax.set_xticklabels([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)

cb_ax = fig.add_axes([0.30, 0.115, 0.40, 0.018])
cb = fig.colorbar(im, cax=cb_ax, orientation='horizontal')
cb.set_label('mm / 3 h')
cb.ax.tick_params(labelsize=6.5)

# ---- time series above panels ----
ax2 = fig.add_axes([0.06, 0.815, 0.91, 0.115])
r0, r1 = max(0, pk - 5), min(n, pk + 6)
tt = np.arange(r0, r1)
for lab, F, c, ls in [('obs', OBS, OKABE[0], '-'), ('GFS', G, OKABE[1], '--'),
                      ('QM', Q, OKABE[2], '-.'), ('OLS', O, OKABE[3], ':'),
                      ('APCNet', A, OKABE[4], '-'), ('U-Net', U, OKABE[5], ':')]:
    ax2.plot(tt, F[tt].mean(axis=(1, 2)), color=c, ls=ls, lw=1.3, marker='o', ms=2.8,
             label=lab)
ax2.axvline(pk, color='0.4', lw=0.8, ls=':')
ax2.set_xticks(tt[::2])
ax2.set_xticklabels([vt[j].strftime('%m-%d %HZ') for j in tt[::2]], fontsize=6.5)
ax2.set_ylabel('Domain-mean\n(mm/3h)', fontsize=7)
ax2.set_ylim(0, 8)
ax2.legend(ncol=6, frameon=True, fontsize=6.5, loc='upper left', facecolor='white', framealpha=0.9, edgecolor='0.7')
ax2.text(0.01, 1.08, '3-h domain-mean precipitation around the peak window', transform=ax2.transAxes, fontsize=7.5, va='bottom', ha='left')
for s in ax2.spines.values():
    s.set_linewidth(0.6)

fig.suptitle('Typical Northeast China cold-vortex heavy-rain event: %s  (peak window, domain 25 x 37)'
             % vt[pk].strftime('%Y-%m-%d %HZ'), fontsize=9.5, y=0.995)

import os as _os
_os.makedirs(OUT, exist_ok=True)
png = _os.path.join(OUT, 'fig10_case_run13.png')
pdf = _os.path.join(OUT, 'fig10_case_run13.pdf')
fig.savefig(png, dpi=300, facecolor='white')
fig.savefig(pdf, facecolor='white')
print('saved', png)
print('saved', pdf)
