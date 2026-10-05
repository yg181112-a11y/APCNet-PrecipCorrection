# -*- coding: utf-8 -*-
"""Storm case synoptic analysis v2: correct GRIB variable names."""
import numpy as np
import netCDF4 as nc
import os, glob, json
from datetime import datetime, timedelta

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
GFS_BASE = r'D:\liaohe\GFS-data\gfs.0p25.2015-2025.f003'

storm_cases = [
    {'idx': 2221, 'time': datetime(2025, 7, 29, 15, 0), 'max_precip': 38.6},
    {'idx': 773, 'time': datetime(2024, 7, 28, 3, 0), 'max_precip': 25.3},
    {'idx': 2338, 'time': datetime(2025, 8, 27, 21, 0), 'max_precip': 23.4},
    {'idx': 827, 'time': datetime(2024, 8, 10, 15, 0), 'max_precip': 22.7},
    {'idx': 873, 'time': datetime(2024, 8, 22, 3, 0), 'max_precip': 21.4},
]

IDX850 = 5
IDX500 = 12

def find_gfs_file(effective_time):
    init_time = effective_time - timedelta(hours=3)
    date_str = init_time.strftime('%Y%m%d')
    hour_str = init_time.strftime('%H')
    for folder in glob.glob(os.path.join(GFS_BASE, '*')):
        if not os.path.isdir(folder):
            continue
        for pat in [f'*{date_str}{hour_str}*f003*', f'*{date_str}*{hour_str}*f003*']:
            matches = glob.glob(os.path.join(folder, pat))
            if matches:
                return matches[0]
    return None

def extract_fields(filepath):
    ds = nc.Dataset(filepath, 'r')
    fields = {}
    try:
        fields['u850'] = np.array(ds.variables['U_GRD_L100'][0, IDX850], dtype=np.float32)
        fields['v850'] = np.array(ds.variables['V_GRD_L100'][0, IDX850], dtype=np.float32)
        fields['u500'] = np.array(ds.variables['U_GRD_L100'][0, IDX500], dtype=np.float32)
        fields['v500'] = np.array(ds.variables['V_GRD_L100'][0, IDX500], dtype=np.float32)
    except Exception as e:
        print(f"  Wind extract error: {e}")
    try:
        fields['hgt500'] = np.array(ds.variables['HGT_L100'][0, IDX500], dtype=np.float32)
    except:
        try:
            fields['hgt500'] = np.array(ds.variables['HGT_L6'][0], dtype=np.float32)
        except:
            print("  HGT500 not available")
    try:
        fields['cape'] = np.array(ds.variables['CAPE_L1'][0], dtype=np.float32)
    except:
        print("  CAPE not available")
    try:
        fields['pwat'] = np.array(ds.variables['P_WAT_L200'][0], dtype=np.float32)
    except:
        print("  PWAT not available")
    try:
        fields['w850'] = np.array(ds.variables['V_VEL_L100'][0, IDX850], dtype=np.float32)
    except:
        print("  W850 not available")
    try:
        fields['t850'] = np.array(ds.variables['TMP_L100'][0, IDX850], dtype=np.float32)
        fields['t500'] = np.array(ds.variables['TMP_L100'][0, IDX500], dtype=np.float32)
    except:
        pass
    ds.close()
    return fields

def diagnose(fields, case):
    d = {}
    r1, r2, c1, c2 = 8, 17, 12, 25
    
    for k in ['cape', 'pwat', 'hgt500', 'w850', 't850', 't500']:
        if k in fields:
            c = fields[k][r1:r2, c1:c2]
            d[f'{k}_mean'] = float(np.nanmean(c))
            d[f'{k}_max'] = float(np.nanmax(c))
            d[f'{k}_min'] = float(np.nanmin(c))
    
    if 'u850' in fields and 'v850' in fields:
        ws850 = np.sqrt(fields['u850']**2 + fields['v850']**2)
        d['ws850_mean'] = float(np.nanmean(ws850[r1:r2, c1:c2]))
        d['ws850_max'] = float(np.nanmax(ws850[r1:r2, c1:c2]))
    if 'u500' in fields and 'v500' in fields:
        ws500 = np.sqrt(fields['u500']**2 + fields['v500']**2)
        d['ws500_mean'] = float(np.nanmean(ws500[r1:r2, c1:c2]))
        d['ws500_max'] = float(np.nanmax(ws500[r1:r2, c1:c2]))
    
    if 'hgt500' in fields:
        hgt = fields['hgt500']
        d['hgt500_range'] = float(np.nanmax(hgt[r1:r2, c1:c2]) - np.nanmin(hgt[r1:r2, c1:c2]))
    
    if 'u500' in fields and 'v500' in fields:
        u500, v500 = fields['u500'], fields['v500']
        dvdx = np.gradient(v500, axis=1)
        dudy = np.gradient(u500, axis=0)
        vort = dvdx - dudy
        d['vort500_max'] = float(np.nanmax(vort[r1:r2, c1:c2]))
        d['vort500_mean'] = float(np.nanmean(vort[r1:r2, c1:c2]))
    
    # Moisture flux at 850 (simplified: q * V, use PWAT as proxy)
    d['moisture_flux'] = float(d.get('pwat_mean', 0) * d.get('ws850_mean', 0))
    
    # Classification
    drivers = []
    
    if d.get('vort500_max', 0) > 0.008 and d.get('hgt500_mean', 9999) < 5880:
        drivers.append('东北冷涡(500hPa正涡度+低高度场)')
    elif d.get('vort500_max', 0) > 0.005:
        drivers.append('低槽/切变(500hPa正涡度)')
    
    if d.get('ws850_max', 0) > 12:
        drivers.append(f'低空急流(850hPa最大{d["ws850_max"]:.1f}m/s)')
    elif d.get('ws850_max', 0) > 8:
        drivers.append(f'低空急流偏弱(850hPa最大{d["ws850_max"]:.1f}m/s)')
    
    if d.get('cape_max', 0) > 1500:
        drivers.append(f'强对流不稳定(CAPE最大{d["cape_max"]:.0f}J/kg)')
    elif d.get('cape_max', 0) > 500:
        drivers.append(f'中等对流不稳定(CAPE最大{d["cape_max"]:.0f}J/kg)')
    
    if d.get('pwat_max', 0) > 50:
        drivers.append(f'高水汽(PWAT最大{d["pwat_max"]:.1f}mm)')
    elif d.get('pwat_max', 0) > 35:
        drivers.append(f'较高水汽(PWAT最大{d["pwat_max"]:.1f}mm)')
    
    if d.get('ws500_max', 0) > 25:
        drivers.append(f'强高空急流(500hPa最大{d["ws500_max"]:.1f}m/s)')
    
    if d.get('w850_min', 0) < -0.3:
        drivers.append(f'强上升运动(850hPa omega最小{d["w850_min"]:.2f}Pa/s)')
    
    d['drivers'] = drivers
    d['max_precip'] = case['max_precip']
    d['time'] = case['time'].strftime('%Y-%m-%d %H:%M')
    return d

results = []
for case in storm_cases:
    print(f"\n{'='*60}")
    print(f"Case: {case['time']}, max={case['max_precip']}mm")
    filepath = find_gfs_file(case['time'])
    if filepath:
        fields = extract_fields(filepath)
        if fields:
            diag = diagnose(fields, case)
            print(f"  Drivers: {diag['drivers']}")
            print(f"  CAPE: mean={diag.get('cape_mean',0):.0f}, max={diag.get('cape_max',0):.0f}")
            print(f"  PWAT: mean={diag.get('pwat_mean',0):.1f}, max={diag.get('pwat_max',0):.1f}")
            print(f"  WS850: mean={diag.get('ws850_mean',0):.1f}, max={diag.get('ws850_max',0):.1f}")
            print(f"  WS500: mean={diag.get('ws500_mean',0):.1f}, max={diag.get('ws500_max',0):.1f}")
            print(f"  HGT500: mean={diag.get('hgt500_mean',0):.0f}, range={diag.get('hgt500_range',0):.0f}")
            print(f"  Vort500: mean={diag.get('vort500_mean',0):.5f}, max={diag.get('vort500_max',0):.5f}")
            print(f"  W850: mean={diag.get('w850_mean',0):.3f}, min={diag.get('w850_min',0):.3f}")
            results.append(diag)
        else:
            results.append({'time': case['time'].strftime('%Y-%m-%d %H:%M'), 'max_precip': case['max_precip'], 'drivers': ['提取失败']})
    else:
        results.append({'time': case['time'].strftime('%Y-%m-%d %H:%M'), 'max_precip': case['max_precip'], 'drivers': ['文件未找到']})

with open(os.path.join(WORK, 'storm_synoptic_analysis.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved.")
