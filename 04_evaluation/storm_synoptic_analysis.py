# -*- coding: utf-8 -*-
"""
Storm case synoptic analysis: extract GFS meteorological fields at 5 storm times
and diagnose driving mechanisms (NECV, low-level jet, CAPE, moisture flux, etc.)
"""
import numpy as np
import netCDF4 as nc
import os, glob, json
from datetime import datetime, timedelta

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'
GFS_BASE = r'D:\liaohe\GFS-data\gfs.0p25.2015-2025.f003'

# 5 storm cases (effective time)
storm_cases = [
    {'idx': 2221, 'time': datetime(2025, 7, 29, 15, 0), 'max_precip': 38.6},
    {'idx': 773, 'time': datetime(2024, 7, 28, 3, 0), 'max_precip': 25.3},
    {'idx': 2338, 'time': datetime(2025, 8, 27, 21, 0), 'max_precip': 23.4},
    {'idx': 827, 'time': datetime(2024, 8, 10, 15, 0), 'max_precip': 22.7},
    {'idx': 873, 'time': datetime(2024, 8, 22, 3, 0), 'max_precip': 21.4},
]

def find_gfs_file(effective_time):
    """Find GFS f003 file for given effective time.
    GFS f003 init time = effective_time - 3h.
    File pattern: gfs.0p25.YYYYMMDDHH.f003.nc (or .grb or similar)
    """
    init_time = effective_time - timedelta(hours=3)
    # Search in year folders
    year = init_time.year
    year_folders = glob.glob(os.path.join(GFS_BASE, f'*{year}*'))
    if not year_folders:
        # Try all folders
        year_folders = glob.glob(os.path.join(GFS_BASE, '*'))
    
    date_str = init_time.strftime('%Y%m%d')
    hour_str = init_time.strftime('%H')
    
    for folder in year_folders:
        if not os.path.isdir(folder):
            continue
        # Search for files matching date and hour
        patterns = [
            f'*{date_str}{hour_str}*f003*',
            f'*{date_str}*{hour_str}*f003*',
            f'gfs*t{date_str}{hour_str}z*f003*',
        ]
        for pat in patterns:
            matches = glob.glob(os.path.join(folder, pat))
            if matches:
                return matches[0]
    
    # List files in folder to understand naming
    if year_folders:
        files = os.listdir(year_folders[0])
        print(f"  Sample files in {year_folders[0]}: {files[:5]}")
    return None

def extract_fields(filepath):
    """Extract key meteorological fields from GFS file."""
    try:
        ds = nc.Dataset(filepath, 'r')
    except Exception as e:
        print(f"  Error opening {filepath}: {e}")
        return None
    
    fields = {}
    # List variables
    var_names = list(ds.variables.keys())
    
    # Try to find variables by common names
    var_map = {
        'cape': ['CAPE', 'cape', 'Convective_available_potential_energy_surface'],
        'pwat': ['PWAT', 'pwat', 'Precipitable_water_entire_atmosphere'],
        'u850': ['U850', 'u850', 'U-component_of_wind_isobaric_850mb', 'u-component_of_wind_isobaric'],
        'v850': ['V850', 'v850', 'V-component_of_wind_isobaric_850mb'],
        'u500': ['U500', 'u500', 'U-component_of_wind_isobaric_500mb'],
        'v500': ['V500', 'v500', 'V-component_of_wind_isobaric_500mb'],
        'w': ['V-Velocity', 'vvel', 'Vertical_velocity_pressure_isobaric', 'omega'],
        'precip': ['APCP', 'precip', 'Total_precipitation_surface', 'GFS-Precip'],
    }
    
    for key, candidates in var_map.items():
        for cand in candidates:
            if cand in var_names:
                try:
                    data = ds.variables[cand][:]
                    if hasattr(data, 'filled'):
                        data = data.filled(np.nan)
                    fields[key] = np.array(data, dtype=np.float32)
                    break
                except Exception as e:
                    print(f"  Error reading {cand}: {e}")
    
    ds.close()
    return fields

def diagnose_case(fields, case):
    """Diagnose synoptic driving mechanisms from fields."""
    diagnosis = {}
    
    # Domain stats (Liaohe basin ~ 121-126E, 40-44N -> grid indices)
    # GFS 0.25 deg, domain likely 118-128E, 38-46N (25x37 grid)
    # Center of domain
    for key in fields:
        arr = fields[key]
        if arr.ndim >= 2:
            # Take center region (rows 8-17, cols 12-25)
            center = arr[..., 8:17, 12:25] if arr.ndim == 3 else arr[8:17, 12:25]
            diagnosis[f'{key}_mean'] = float(np.nanmean(center))
            diagnosis[f'{key}_max'] = float(np.nanmax(center))
            diagnosis[f'{key}_min'] = float(np.nanmin(center))
    
    # Wind speed at 850 and 500
    if 'u850' in fields and 'v850' in fields:
        ws850 = np.sqrt(fields['u850']**2 + fields['v850']**2)
        center850 = ws850[..., 8:17, 12:25] if ws850.ndim == 3 else ws850[8:17, 12:25]
        diagnosis['ws850_mean'] = float(np.nanmean(center850))
        diagnosis['ws850_max'] = float(np.nanmax(center850))
        # Low-level jet detection (>12 m/s at 850)
        diagnosis['llj_pixels'] = int(np.sum(center850 > 12))
    
    if 'u500' in fields and 'v500' in fields:
        ws500 = np.sqrt(fields['u500']**2 + fields['v500']**2)
        center500 = ws500[..., 8:17, 12:25] if ws500.ndim == 3 else ws500[8:17, 12:25]
        diagnosis['ws500_mean'] = float(np.nanmean(center500))
        diagnosis['ws500_max'] = float(np.nanmax(center500))
    
    # CAPE > 1000 J/kg indicates convective instability
    if 'cape' in fields:
        cape = fields['cape']
        center_cape = cape[..., 8:17, 12:25] if cape.ndim == 3 else cape[8:17, 12:25]
        diagnosis['cape_gt1000'] = int(np.sum(center_cape > 1000))
        diagnosis['cape_gt500'] = int(np.sum(center_cape > 500))
    
    # PWAT > 30 kg/m2 indicates high moisture
    if 'pwat' in fields:
        pwat = fields['pwat']
        center_pwat = pwat[..., 8:17, 12:25] if pwat.ndim == 3 else pwat[8:17, 12:25]
        diagnosis['pwat_gt30'] = int(np.sum(center_pwat > 30))
        diagnosis['pwat_gt40'] = int(np.sum(center_pwat > 40))
    
    # Classify driving mechanism
    drivers = []
    if diagnosis.get('cape_gt1000', 0) > 10:
        drivers.append('强对流不稳定(CAPE>1000)')
    elif diagnosis.get('cape_gt500', 0) > 10:
        drivers.append('中等对流不稳定(CAPE>500)')
    
    if diagnosis.get('llj_pixels', 0) > 5:
        drivers.append(f'低空急流(850hPa最大{diagnosis.get("ws850_max",0):.1f}m/s)')
    
    if diagnosis.get('pwat_gt30', 0) > 10:
        drivers.append(f'高水汽(PWAT最大{diagnosis.get("pwat_max",0):.1f}mm)')
    
    if diagnosis.get('ws500_max', 0) > 25:
        drivers.append(f'强高空急流(500hPa最大{diagnosis.get("ws500_max",0):.1f}m/s)')
    
    # NECV signature: cold vortex at 500hPa = cyclonic circulation + cold core
    # Simplified: high vorticity = strong wind shear
    if 'u500' in fields and 'v500' in fields:
        u500 = fields['u500'][0] if fields['u500'].ndim == 3 else fields['u500']
        v500 = fields['v500'][0] if fields['v500'].ndim == 3 else fields['v500']
        # Relative vorticity (simplified)
        if u500.shape[0] > 3 and u500.shape[1] > 3:
            dvdx = np.gradient(v500, axis=1)
            dudy = np.gradient(u500, axis=0)
            vort = dvdx - dudy
            center_vort = vort[8:17, 12:25]
            diagnosis['vort_max'] = float(np.nanmax(center_vort))
            diagnosis['vort_mean'] = float(np.nanmean(center_vort))
            if np.nanmax(center_vort) > 0.005:
                drivers.append('正涡度区(可能冷涡/低槽)')
    
    diagnosis['drivers'] = drivers
    diagnosis['max_precip'] = case['max_precip']
    diagnosis['time'] = case['time'].strftime('%Y-%m-%d %H:%M')
    return diagnosis

# Process each case
results = []
for case in storm_cases:
    print(f"\n{'='*60}")
    print(f"Case: {case['time']}, max precip={case['max_precip']}mm")
    print(f"{'='*60}")
    
    filepath = find_gfs_file(case['time'])
    if filepath:
        print(f"  Found: {os.path.basename(filepath)}")
        fields = extract_fields(filepath)
        if fields:
            print(f"  Extracted fields: {list(fields.keys())}")
            diag = diagnose_case(fields, case)
            print(f"  Drivers: {diag['drivers']}")
            print(f"  CAPE mean={diag.get('cape_mean',0):.1f}, max={diag.get('cape_max',0):.1f}")
            print(f"  PWAT mean={diag.get('pwat_mean',0):.1f}, max={diag.get('pwat_max',0):.1f}")
            print(f"  WS850 mean={diag.get('ws850_mean',0):.1f}, max={diag.get('ws850_max',0):.1f}")
            print(f"  WS500 mean={diag.get('ws500_mean',0):.1f}, max={diag.get('ws500_max',0):.1f}")
            results.append(diag)
        else:
            print(f"  Failed to extract fields")
            results.append({'time': case['time'].strftime('%Y-%m-%d %H:%M'), 'max_precip': case['max_precip'], 'drivers': ['数据提取失败']})
    else:
        print(f"  File not found!")
        results.append({'time': case['time'].strftime('%Y-%m-%d %H:%M'), 'max_precip': case['max_precip'], 'drivers': ['文件未找到']})

# Save
with open(os.path.join(WORK, 'storm_synoptic_analysis.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to storm_synoptic_analysis.json")
