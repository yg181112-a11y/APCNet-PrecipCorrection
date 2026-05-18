import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import xarray as xr
import os
import copy
from contextlib import contextmanager
import glob
import zipfile
import tarfile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shutil
import psutil
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
import random
import math
from tqdm import tqdm
import re
from datetime import datetime, timedelta
import pandas as pd
from matplotlib.gridspec import GridSpec  # Keep, because used
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, zoom
import rasterio
from rasterio.transform import from_origin
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as RasterioResampling
import torch.backends.cudnn as cudnn
import matplotlib
import argparse
import os

parser = argparse.ArgumentParser(description="Train APCNet for Precipitation Correction")

# 定义数据输入与输出路径，默认使用相对路径
parser.add_argument('--gfs_dir', type=str, default='./data/sample/GFS', help='Directory containing GFS forecast data')
parser.add_argument('--era5_dir', type=str, default='./data/sample/ERA5', help='Directory containing ERA5 reanalysis data')
parser.add_argument('--save_dir', type=str, default='./output', help='Directory to save evaluation results and models')

# 解析命令行参数
args = parser.parse_args()

# 将解析后的参数赋值给全局变量，供后续代码调用
GFS_DIR = args.gfs_dir
ERA5_DIR = args.era5_dir
SAVE_DIR = args.save_dir

# 自动创建输出文件夹（如果不存在）
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"📁 Created output directory at: {SAVE_DIR}")
# Set immediately after importing other libraries

matplotlib.rcParams['font.family'] = 'sans-serif'
# 1. Put Arial first - Elsevier's most recommended font
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 2. Core requirement: embed TrueType fonts in PDF and EPS
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 3. DPI: 300 for screen, 600 for exports
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 600 
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.1

# 4. Increase global base font size because large figsize shrinks fonts when scaled to paper size
plt.rcParams['font.size'] = 14
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = [12, 8]
plt.style.use('seaborn-v0_8-whitegrid')

# Get CPU cores - optimize resource usage
TOTAL_CORES = psutil.cpu_count(logical=False)
TOTAL_THREADS = psutil.cpu_count(logical=True)

# ==================== Global precipitation level configuration (CMA standard) ====================
# For 3h accumulated precipitation: 0.1(rain/no-rain), 3.0(moderate), 10.0(heavy), 20.0(storm)
PRECIP_THRESHOLDS = [0.1, 3.0, 10.0, 20.0] 
PRECIP_LEVELS = ['Light', 'Moderate', 'Heavy', 'Storm']

# Only print once in main process
if mp.current_process().name == 'MainProcess':
    print(f" Unified precipitation level configuration loaded:")
    print(f"  Thresholds: {PRECIP_THRESHOLDS} mm/3h")
    print(f"  Levels: {PRECIP_LEVELS}")

if mp.current_process().name == 'MainProcess':
    print(f" Full data training mode: detected {TOTAL_CORES} physical cores, {TOTAL_THREADS} logical processors")

# ==================== Global gate configuration (Gen5: balance accuracy and capture ability) ====================
GATE_CFG = {
    "adaptive": True,
    "threshold_base": 0.22,  # moderate base threshold
    "threshold_min": 0.08,
    "threshold_max": 0.45,
    "gate_power": 0.85,      # slightly reduced power to increase sensitivity
    "min_rain_value": 0.10,
    "max_precip": 250.0,
    "storm_gate_p": 0.40     # physical bypass trigger point raised from 0.3 to 0.4
}

# ==================== Task definition: only f003 (+3h) correction ====================
PREDICTION_HORIZON = 1   # Only f003 -> must be 1
GFS_FORECAST_HOURS = 3   # f003 -> +3h
TIME_STEP_HOURS = 3      # ERA5 3h step
GLOBAL_LATS = np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = np.linspace(117.0, 126.0, 37)

def open_netcdf_file(filepath, **kwargs):
    """
    Automatically try multiple engines to open NetCDF file until success.
    Engine order: netcdf4 -> scipy -> h5netcdf (if installed)
    """
    engines = ['netcdf4', 'scipy', 'h5netcdf']
    last_error = None
    for eng in engines:
        try:
            return xr.open_dataset(filepath, engine=eng, **kwargs)
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Cannot open file {filepath} with any engine: {last_error}")

def safe_div(numer, denom, fill=np.nan, eps=1e-12):
    """Safe division: return NaN if denominator is too small (avoid pathological values like 4e8)"""
    denom = float(denom)
    if abs(denom) < eps:
        return fill
    return float(numer) / denom

def deduplicate_keep_order(seq):
    """Deduplicate but keep order"""
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def get_geo_extent_from_globals():
    """
    Unified extent for imshow to display longitude/latitude on figures.
    Depends on GLOBAL_LATS / GLOBAL_LONS (already defined in code)
    """
    lon_min = float(np.min(GLOBAL_LONS))
    lon_max = float(np.max(GLOBAL_LONS))
    lat_min = float(np.min(GLOBAL_LATS))
    lat_max = float(np.max(GLOBAL_LATS))
    return [lon_min, lon_max, lat_min, lat_max]

def setup_geo_axes(ax, with_grid=True):
    """Add lon/lat ticks to spatial plots (a basic requirement for high-quality journals)"""
    extent = get_geo_extent_from_globals()
    lon_min, lon_max, lat_min, lat_max = extent
    
    # Enlarge axis label fonts
    ax.set_xlabel("Longitude (°E)", fontweight="bold", fontsize=16)
    ax.set_ylabel("Latitude (°N)", fontweight="bold", fontsize=16)

    # Not too dense ticks
    ax.set_xticks(np.linspace(lon_min, lon_max, 5))
    ax.set_yticks(np.linspace(lat_min, lat_max, 5))
    
    # Enlarge tick label font size
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    if with_grid:
        ax.grid(True, linestyle="--", alpha=0.25)

def save_fig_multi(fig, path_no_ext, dpi=300):
    """
    Save both PNG + PDF (vector formats recommended for curves/bars in journals)
    """
    fig.savefig(f"{path_no_ext}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.pdf", dpi=dpi, bbox_inches="tight")

def ultra_fast_training_config():
    return {
        'batch_size': 64,
        'num_workers': 0,
        'pin_memory': True,
        'grad_accumulation': 2,
        'use_amp': True,
        'mixed_precision_dtype': torch.float16,
        'learning_rate': 1e-4,              # back to 1e-4 to increase convergence drive
        'weight_decay': 5e-4,              # increase weight decay to prevent overfitting extremes
        'scheduler': 'cosine',
        'model_channels': 16,
    }

class StrictStandardizer:
    """Wrapper to ensure all data uses the same fixed parameters"""
    def __init__(self, base_standardizer):
        self.base = base_standardizer
  
    def transform(self, data):
        # Add safety check
        if not hasattr(self.base, 'fitted') or not self.base.fitted:
            print(" Standardizer not fitted, returning original data")
            return data
        if not hasattr(self.base, 'transform'):
            print(" Standardizer has no transform method, returning original data")
            return data
        return self.base.transform(data)

def get_device():
    """Choose best available device and optimize CPU settings"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if mp.current_process().name == 'MainProcess':
            print(" Using CUDA GPU for acceleration")
      
        # Enable all GPU optimizations
        torch.backends.cudnn.benchmark = True  # automatically find optimal convolution algorithm
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 precision (RTX 5050 supports)
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')  # highest precision
      
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f" GPU detected: {gpu_name}")
        print(f" GPU memory: {gpu_memory:.1f}GB")
        print(f" Optimizations enabled: cudnn.benchmark=True, TF32=True")
      
    else:
        device = torch.device("cpu")
        torch.set_num_threads(TOTAL_THREADS)
        os.environ['OMP_NUM_THREADS'] = str(TOTAL_THREADS)
        os.environ['MKL_NUM_THREADS'] = str(TOTAL_THREADS)
        os.environ['OPENBLAS_NUM_THREADS'] = str(TOTAL_THREADS)
        os.environ['NUMEXPR_NUM_THREADS'] = str(TOTAL_THREADS)
        if mp.current_process().name == 'MainProcess':
            print(f" Using CPU high-performance parallel mode ({TOTAL_THREADS} threads)")
  
    return device

def extract_date_from_filename(filename):
    """Extract date from filename"""
    patterns = [
        r'(\d{4})(\d{2})(\d{2})',  # YYYYMMDD
        r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
        r'(\d{4})_(\d{2})_(\d{2})',  # YYYY_MM_DD
    ]
  
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            year, month, day = match.groups()
            try:
                return datetime(int(year), int(month), int(day))
            except ValueError:
                continue
    return None
def parse_gfs_init_time_and_fhour(name: str):
    """
    Parse GFS initialization time YYYYMMDDHH and forecast hour fXXX from filename.
    Returns: (init_dt: datetime | None, fhour: int | None)
    """
    # Init time: continuous 10 digits YYYYMMDDHH
    m_init = re.search(r'(\d{10})', name)
    init_dt = None
    if m_init:
        s = m_init.group(1)
        try:
            init_dt = datetime.strptime(s, "%Y%m%d%H")
        except Exception:
            init_dt = None

    # Forecast hour: f003 / f006 ...
    m_f = re.search(r'f(\d{3})', name)
    fhour = int(m_f.group(1)) if m_f else None
    return init_dt, fhour


def gfs_valid_time_from_filename(name: str):
    """
    Compute valid_time = init_time + fhour.
    For f003, valid_time = init + 3h.
    """
    init_dt, fhour = parse_gfs_init_time_and_fhour(name)
    if init_dt is None or fhour is None:
        return None
    return init_dt + timedelta(hours=int(fhour))


def to_py_datetime(t):
    """Convert numpy.datetime64 / pandas Timestamp to python datetime (naive, UTC semantics)"""
    try:
        return pd.to_datetime(t).to_pydatetime()
    except Exception:
        return None
def discover_data_folders(base_path, end_date=None):
    """Automatically discover data folders and apply date filtering"""
    if mp.current_process().name == 'MainProcess':
        print(f" Searching for data folders in base path: {base_path}")
        if end_date:
            print(f" Filtering by end date: {end_date}")
  
    if not os.path.exists(base_path):
        if mp.current_process().name == 'MainProcess':
            print(f" Base path does not exist: {base_path}")
        return []
  
    # Special handling for specific GFS folder
    special_folder = os.path.join(base_path, "gfs.0p25.2015011500-25.2025011418.f003.grib2.nc")
    if os.path.exists(special_folder):
        if mp.current_process().name == 'MainProcess':
            print(f" Found special GFS data folder: gfs.0p25.2015011500-25.2025011418.f003.grib2.nc")
        return [special_folder]
  
    patterns = [
        os.path.join(base_path, "gfs.0p25.*"),
        os.path.join(base_path, "gfs_0p25*"),
        os.path.join(base_path, "GFS*"),
        os.path.join(base_path, "gfs*"),
    ]
  
    all_folders = []
    for pattern in patterns:
        all_folders.extend(glob.glob(pattern))
  
    all_folders = list(set(all_folders))
    valid_folders = []
  
    for folder in all_folders:
        folder_name = os.path.basename(folder)
      
        folder_date = extract_date_from_filename(folder_name)
      
        if end_date and folder_date:
            if folder_date > end_date:
                if mp.current_process().name == 'MainProcess':
                    print(f" Skipping {folder_name} (date {folder_date.strftime('%Y%m%d')} > {end_date.strftime('%Y%m%d')})")
                continue
      
        # Support multiple archive formats
        zip_files = glob.glob(os.path.join(folder, "*.zip"))
        tar_files = glob.glob(os.path.join(folder, "*.tar"))
        nc_files = glob.glob(os.path.join(folder, "*.nc"))
      
        if zip_files or tar_files or nc_files:
            valid_folders.append(folder)
            file_info = []
            if zip_files:
                file_info.append(f"{len(zip_files)} ZIP files")
            if tar_files:
                file_info.append(f"{len(tar_files)} TAR files")
            if nc_files:
                file_info.append(f"{len(nc_files)} NetCDF files")
          
            date_info = f"Date: {folder_date.strftime('%Y%m%d')}" if folder_date else "Date: unknown"
            if mp.current_process().name == 'MainProcess':
                print(f" Found data folder: {folder_name} - {', '.join(file_info)} - {date_info}")
  
    valid_folders.sort()
    if mp.current_process().name == 'MainProcess':
        print(f" Total found {len(valid_folders)} valid data folders")
    return valid_folders

def advanced_precip_cleaning(data):
    """
    Peak-preserving precipitation cleaning:
    - Keep extreme peaks (avoid gaussian_filter reducing POD@20)
    - Only slightly smooth light rain / noise
    """
    if len(data) < 6:
        return data

    precip = data[5].astype(np.float32)

    # Physical constraints
    precip = np.clip(precip, 0, 100)

    # Remove extreme outliers (only for >0 pixels)
    if np.any(precip > 0):
        thr = np.percentile(precip[precip > 0], 99.9)
        precip[precip > thr] = thr

    # Peak protection: no smoothing for heavy rain; light smoothing for weak rain
    if precip.max() < 10.0:
        precip = gaussian_filter(precip, sigma=0.6)

    data[5] = precip
    return data
class ModelEMA:
    """Exponential Moving Average for model parameters (validation-friendly)."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v_ema in self.ema.state_dict().items():
            v = msd[k].detach()
            if v.dtype.is_floating_point:
                v_ema.copy_(v_ema * self.decay + (1.0 - self.decay) * v)
            else:
                v_ema.copy_(v)

    def state_dict(self):
        return self.ema.state_dict()
class EnhancedDataProcessor:
    def __init__(self, base_path, precip_path=None, max_workers=None, end_date=None):
        self.base_path = base_path
        self.precip_path = precip_path
        self.max_workers = max_workers if max_workers is not None else (psutil.cpu_count(logical=False) // 2)
        self.end_date = end_date
        self.memory_limit_gb = 7.0
      
        # Statistics variables
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped_date': 0,
            'precip_stats': {
                'max_values': [],
                'mean_values': [],
                'non_zero_counts': [],
                'non_zero_ratios': [],
                'file_names': []
            },
            'total_points': 0,
            'strong_precip_events': []  # specifically record heavy precipitation events
        }
    def check_memory_usage(self):
        """Check memory usage"""
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1024**3
        if memory_gb > self.memory_limit_gb:
            if mp.current_process().name == 'MainProcess':
                print(f" High memory usage: {memory_gb:.2f}GB, performing garbage collection...")
            gc.collect()
            return False
        return True
  
    def _get_precip_archive_path(self, archive_path):
        """Given the regular data archive file path, find corresponding precipitation archive file path"""
        if not self.precip_path:
            return None
      
        # Get relative path from base_path
        rel_path = os.path.relpath(archive_path, self.base_path)
      
        # Build precipitation data path
        precip_archive_path = os.path.join(self.precip_path, rel_path)
      
        # Check if file exists
        if os.path.exists(precip_archive_path):
            return precip_archive_path
      
        # If direct path does not exist, try to find similar file
        archive_name = os.path.basename(archive_path)
        archive_dir = os.path.dirname(archive_path)
        rel_dir = os.path.relpath(archive_dir, self.base_path)
      
        # Search for similar files under precipitation path
        search_pattern = os.path.join(self.precip_path, rel_dir, "*" + os.path.splitext(archive_name)[1])
        matching_files = glob.glob(search_pattern)
      
        if matching_files:
            # Try to find the file with closest date
            archive_date = extract_date_from_filename(archive_name)
            if archive_date:
                best_match = None
                min_date_diff = float('inf')
              
                for match_file in matching_files:
                    match_date = extract_date_from_filename(os.path.basename(match_file))
                    if match_date:
                        date_diff = abs((archive_date - match_date).days)
                        if date_diff < min_date_diff:
                            min_date_diff = date_diff
                            best_match = match_file
              
                if best_match and min_date_diff <= 1:  # allow 1 day difference
                    return best_match
      
        if mp.current_process().name == 'MainProcess':
            print(f" Corresponding precipitation file not found: {archive_name}")
        return None
  
    def process_archive_batch_parallel(self, archive_batch, extract_dir):
        """
        Old interface called during initial Dataset loading (no timestamp)
        For compatibility, call the version with timestamps and return only data part
        """
        # Call new method to get list of (time, data) tuples
        results_with_times = self.process_archive_batch_parallel_with_times(archive_batch, extract_dir)
        
        # Extract only data part to comply with old interface expectation
        return [data for t, data in results_with_times if data is not None]
    def process_archive_batch_parallel_with_times(self, archive_batch, extract_dir):
        if mp.current_process().name == 'MainProcess':
            print(f" Starting multi-process extraction of {len(archive_batch)} GFS files...")

        # Get number of physical cores, avoid overload; use half of TOTAL_CORES or 2/3
        num_workers = max(1, TOTAL_CORES // 2) 
        
        results = []
        # Use process pool to execute decompression, reading and cleaning logic in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit single-file processing to process pool
            # Note: ensure the called function can be pickled or is accessible outside class
            futures = [executor.submit(self._process_single_archive_silent_with_time, path, extract_dir) 
                    for path in archive_batch]
            
            # Monitor task completion with tqdm
            for future in tqdm(futures, desc="Extracting GFS data", total=len(archive_batch), leave=False):
                try:
                    res = future.result()
                    if res is not None:
                        results.append(res)
                except Exception as e:
                    continue
                    
        return results
    def _process_single_archive_silent_with_time(self, archive_path, extract_dir):
        """
        New: returns (valid_time, stacked_data)
        """
        # Parse valid time
        valid_time = gfs_valid_time_from_filename(os.path.basename(archive_path))
        if valid_time is None:
            return None

        stacked_data = self._process_single_archive_silent(archive_path, extract_dir)
        if stacked_data is None:
            return None

        return valid_time, stacked_data
    def _process_batch_with_progress(self, batch, extract_dir):
        """Process batch with progress bar - remove real-time printing of heavy rain"""
        results = []
      
        # Create progress bar
        from tqdm import tqdm
        pbar = tqdm(batch, desc="Extracting GFS data", 
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                   leave=False)  # do not keep progress bar
      
        for archive_path in pbar:
            self.stats['total_processed'] += 1
          
            # Update progress bar description
            file_name = os.path.basename(archive_path)[:25]
            if len(file_name) < 25:
                file_name = file_name.ljust(25)
            pbar.set_description(f"Processing: {file_name}")
          
            try:
                result = self._process_single_archive_silent(archive_path, extract_dir)
                if result is not None:
                    results.append(result)
                    self.stats['successful'] += 1
                  
                    # Collect precipitation statistics
                    if len(result) > 5:
                        precip_data = result[5]
                        max_precip = np.max(precip_data)
                        mean_precip = np.mean(precip_data)
                        non_zero = np.sum(precip_data > 0.001)
                        total_points = precip_data.size
                        ratio = non_zero / total_points * 100 if total_points > 0 else 0
                      
                        self.stats['total_points'] += total_points
                        self.stats['precip_stats']['max_values'].append(max_precip)
                        self.stats['precip_stats']['mean_values'].append(mean_precip)
                        self.stats['precip_stats']['non_zero_counts'].append(non_zero)
                        self.stats['precip_stats']['non_zero_ratios'].append(ratio)
                        self.stats['precip_stats']['file_names'].append(os.path.basename(archive_path))
                      
                        # Remove real-time printing, only record heavy precipitation events
                        if max_precip > 5.0:
                            self.stats['strong_precip_events'].append({
                                'file': os.path.basename(archive_path),
                                'max_precip': max_precip,
                                'coverage': ratio
                            })
                else:
                    self.stats['failed'] += 1
                  
            except Exception as e:
                self.stats['failed'] += 1
                continue
          
            # Update progress bar suffix
            pbar.set_postfix({
                'Success': f"{self.stats['successful']}",
                'Precip points': f"{sum(self.stats['precip_stats']['non_zero_counts'])}"
            })
      
        pbar.close()
        return results
  
    def _process_single_archive_silent(self, archive_path, extract_dir):
        """Silently process a single GFS archive file"""
        file_date = extract_date_from_filename(os.path.basename(archive_path))
      
        # Date filter check
        if self.end_date and file_date:
            if file_date > self.end_date:
                self.stats['skipped_date'] += 1
                return None
      
        # Get corresponding precipitation data file path
        precip_archive_path = self._get_precip_archive_path(archive_path)
      
        # Process regular data
        regular_data = self._extract_regular_variables_silent(archive_path, extract_dir)
        if regular_data is None:
            return None
      
        # Process precipitation data
        precip_data = None
        if precip_archive_path:
            precip_data = self._extract_precipitation_variables_silent(precip_archive_path, extract_dir)
      
        # If precipitation data missing, create zero-filled
        if precip_data is None:
            precip_data = np.zeros((25, 37))
      
        # Merge data
        all_data = list(regular_data) + [precip_data]
      
        # Check data dimensions
        for i, data in enumerate(all_data):
            if data.shape != (25, 37):
                all_data[i] = np.zeros((25, 37))
      
        stacked_data = np.stack(all_data, axis=0)
        return stacked_data
  
    def _process_single_archive_safe(self, archive_path, extract_dir):
        """Safe wrapper for processing a single file"""
        try:
            return self._process_single_archive_silent(archive_path, extract_dir)
        except MemoryError:
            if mp.current_process().name == 'MainProcess':
                print(f" Out of memory processing file: {os.path.basename(archive_path)}")
            gc.collect()
            return None
        except Exception as e:
            if mp.current_process().name == 'MainProcess':
                print(f" Failed processing {os.path.basename(archive_path)}: {str(e)[:100]}")
            return None
  
    def _extract_regular_variables(self, archive_path, extract_dir):
        """Extract 5 regular variables from regular data file"""

        # 1) Read .nc directly (including .grib2.nc)
        if archive_path.endswith(('.nc', '.nc4', '.netcdf')):
            try:
                with open_netcdf_file(archive_path, cache=False) as ds:
                    sample_data = self._extract_gfs_regular_variables_silent(ds, os.path.basename(archive_path))
                    if sample_data is None or len(sample_data) == 0:
                        return None
                    return sample_data
            except Exception:
                return None

        # 2) Original zip/tar extraction flow (unchanged)
        temp_dir = os.path.join(extract_dir, f"temp_regular_{hash(archive_path) & 0xFFFFFFFF}")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            elif archive_path.endswith('.tar'):
                with tarfile.open(archive_path, 'r') as tar_ref:
                    tar_ref.extractall(temp_dir)
            else:
                if mp.current_process().name == 'MainProcess':
                    print(f" Unsupported archive format: {archive_path}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None

            nc_files = self._find_nc_files(temp_dir)
            if not nc_files:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None

            nc_file = nc_files[0]
            with open_netcdf_file(nc_file, cache=False) as ds:
                sample_data = self._extract_gfs_regular_variables_silent(ds, os.path.basename(archive_path))
                return sample_data if sample_data else None
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
  
    def _extract_precipitation_variables(self, archive_path, extract_dir):
        if archive_path.endswith(('.nc', '.nc4', '.netcdf')):
            try:
                with open_netcdf_file(archive_path, cache=False) as ds:
                    return self._extract_gfs_precipitation_variable_silent(ds, os.path.basename(archive_path))
            except Exception:
                return None
  
    def _find_nc_files(self, directory):
        """Recursively find all NetCDF files in directory"""
        nc_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.nc', '.nc4', '.netcdf')):
                    nc_files.append(os.path.join(root, file))
        return nc_files
  
    def _extract_gfs_regular_variables_silent(self, ds, filename):
        """Extract GFS regular variables (silent version)"""
        try:
            variables_config = {
                'CAPE_L1': {'target_name': 'cape', 'unit_conversion': 1.0},
                'P_WAT_L200': {'target_name': 'pwat', 'unit_conversion': 1.0},
                'U_GRD_L100': {'target_name': 'u_wind', 'unit_conversion': 1.0},
                'V_GRD_L100': {'target_name': 'v_wind', 'unit_conversion': 1.0},
                'V_VEL_L100': {'target_name': 'vvel', 'unit_conversion': 0.01}
            }
          
            extracted_vars = {}
          
            for gfs_var, config in variables_config.items():
                if gfs_var in ds.variables:
                    var_data = ds[gfs_var].values
                  
                    # Handle different dimension cases
                    if len(var_data.shape) == 3:  # [time, lat, lon]
                        var_data = var_data[0] if var_data.shape[0] > 0 else np.zeros((25, 37))
                    elif len(var_data.shape) == 4:  # [time, level, lat, lon]
                        level_idx = var_data.shape[1] // 2
                        var_data = var_data[0, level_idx] if var_data.shape[0] > 0 and var_data.shape[1] > 0 else np.zeros((25, 37))
                    else:
                        var_data = np.zeros((25, 37))
                  
                    # Unit conversion
                    var_data = var_data * config['unit_conversion']
                    var_data = np.nan_to_num(var_data, nan=0.0, posinf=0.0, neginf=0.0)
                  
                    extracted_vars[config['target_name']] = var_data
                else:
                    extracted_vars[config['target_name']] = np.zeros((25, 37))
          
            # Return 5 variables in order
            all_vars = [
                extracted_vars['cape'], 
                extracted_vars['pwat'],
                extracted_vars['u_wind'], 
                extracted_vars['v_wind'], 
                extracted_vars['vvel']
            ]
          
            return all_vars
          
        except Exception as e:
            return []
  
    def _extract_gfs_precipitation_variable_silent(self, ds, filename):
        """Extract GFS precipitation variable (silent version)"""
        try:
            # Look for precipitation variable
            precipitation_variables = [
                'A_PCP_L1_Accum_1', 'APCP_P8_L1_GLL0', 'TP_P0_L1_GLL0',
                'APCP', 'TP', 'PRATE'
            ]
          
            found_precip_var = None
            for precip_var in precipitation_variables:
                if precip_var in ds.variables:
                    found_precip_var = precip_var
                    break
          
            if not found_precip_var:
                # If standard name not found, try variables containing keywords
                all_vars = list(ds.variables.keys())
                precip_like_vars = [var for var in all_vars if any(keyword in var.lower() for keyword in ['precip', 'apcp', 'tp', 'pcp', 'accum', 'rain'])]
                if precip_like_vars:
                    found_precip_var = precip_like_vars[0]
                else:
                    return np.zeros((25, 37))
          
            precip_data = ds[found_precip_var].values
          
            # Handle precipitation data dimensions
            if len(precip_data.shape) == 3:  # [time, lat, lon]
                precip_data = precip_data[0] if precip_data.shape[0] > 0 else np.zeros((25, 37))
            elif len(precip_data.shape) == 4:  # [time, level, lat, lon]
                precip_data = precip_data[0, 0] if precip_data.shape[0] > 0 and precip_data.shape[1] > 0 else np.zeros((25, 37))
            elif len(precip_data.shape) == 2:  # [lat, lon]
                pass
            else:
                precip_data = np.zeros((25, 37))
          
            precip_data = np.nan_to_num(precip_data, nan=0.0, posinf=0.0, neginf=0.0)
          
            # Unit conversion
            if hasattr(ds[found_precip_var], 'units'):
                units = ds[found_precip_var].units.lower()
                if 'kg m-2' in units or 'kg/m2' in units:
                    # Already accumulative, correct units (kg/m² = mm)
                    pass
                elif 'm' in units and 's' not in units:
                    # Convert meters to millimeters
                    precip_data = precip_data * 1000
                elif 'cm' in units:
                    # Convert cm to mm
                    precip_data = precip_data * 10
          
            # Apply threshold limit
            precip_data = np.clip(precip_data, 0, 100)
          
            return precip_data
          
        except Exception as e:
            return np.zeros((25, 37))
  
    def _extract_gfs_regular_variables(self, ds, filename):
        """Extract GFS regular variables (keep for compatibility)"""
        return self._extract_gfs_regular_variables_silent(ds, filename)
  
    def _extract_gfs_precipitation_variable(self, ds, filename):
        """Extract GFS precipitation variable (keep for compatibility)"""
        return self._extract_gfs_precipitation_variable_silent(ds, filename)
  
    def _extract_regular_variables_silent(self, archive_path, extract_dir):
        """Silent extraction of regular variables"""
        return self._extract_regular_variables(archive_path, extract_dir)
  
    def _extract_precipitation_variables_silent(self, archive_path, extract_dir):
        """Silent extraction of precipitation variables"""
        return self._extract_precipitation_variables(archive_path, extract_dir)
  
    def _print_extraction_summary(self):
        """Print extraction summary - optimized version"""
        if mp.current_process().name == 'MainProcess':
            print("\n" + "="*60)
            print(" GFS Data Extraction Report")
            print("="*60)
            print(f" File processing statistics:")
            print(f"  Successfully extracted: {self.stats['successful']} files")
            print(f"  Failed: {self.stats['failed']} files")
            if self.stats['skipped_date'] > 0:
                print(f"  Skipped by date: {self.stats['skipped_date']} files")
          
            if self.stats['precip_stats']['max_values']:
                total_non_zero = sum(self.stats['precip_stats']['non_zero_counts'])
                total_points = self.stats['total_points'] if self.stats['total_points'] > 0 else 1
                precip_ratio = total_non_zero / total_points * 100
              
                print(f"\n Precipitation statistics summary:")
                print(f"  Total precipitation grid points: {total_non_zero:,}/{total_points:,} ({precip_ratio:.2f}%)")
                print(f"  Maximum precipitation: {max(self.stats['precip_stats']['max_values']):.2f} mm")
                print(f"  Mean precipitation: {np.mean(self.stats['precip_stats']['mean_values']):.4f} mm")
                print(f"  Heavy rain events (>5mm): {len(self.stats['strong_precip_events'])}")
              
                # Show top 5 heavy rain events (in summary)
                if len(self.stats['precip_stats']['max_values']) >= 5:
                    sorted_indices = np.argsort(self.stats['precip_stats']['max_values'])[-5:][::-1]
                    print(f"\n Top 5 heavy rain events:")
                    for i, idx in enumerate(sorted_indices):
                        filename_short = self.stats['precip_stats']['file_names'][idx][:20]
                        if len(filename_short) < 20:
                            filename_short = filename_short.ljust(20)
                        print(f"  {i+1}. {filename_short}...: "
                              f"{self.stats['precip_stats']['max_values'][idx]:6.2f}mm "
                              f"({self.stats['precip_stats']['non_zero_ratios'][idx]:5.1f}% area)")
              
                # Optionally show heavy rain event statistics
                if len(self.stats['strong_precip_events']) > 0:
                    strong_precip_count = len(self.stats['strong_precip_events'])
                    strong_precip_ratio = strong_precip_count / len(self.stats['precip_stats']['max_values']) * 100
                    print(f"\n Heavy rain analysis (>5mm):")
                    print(f"  Number of events: {strong_precip_count} ({strong_precip_ratio:.1f}% files)")
                    print(f"  Maximum heavy rain: {max([e['max_precip'] for e in self.stats['strong_precip_events']]):.1f}mm")
                    print(f"  Average coverage: {np.mean([e['coverage'] for e in self.stats['strong_precip_events']]):.1f}%")
          
            print("="*60)
class ERA5DataProcessor:
    """ERA5 data processor - enhanced: automatically align time format and physical units"""
  
    def __init__(self):
        # Broader variable name matching to handle ERA5T and different download versions
        self.variable_mapping = {
            'cape': ['cape', 'CONVECTIVE_AVAILABLE_POTENTIAL_ENERGY'],
            'pwat': ['tcwv', 'TOTAL_COLUMN_WATER_VAPOUR'],
            'u_wind': ['u100', '100m_u_component_of_wind'],
            'v_wind': ['v100', '100m_v_component_of_wind'],
            'vvel': ['w', 'omega', 'vertical_velocity'],
            'precipitation': ['tp', 'total_precipitation', 'precip', 'cp', 'total_precipitation_6hr']
        }
  
    def load_era5_data(self, data_dirs, start_date=None, end_date=None):
        """Load ERA5 data and filter by time range"""
        all_samples = []
        if mp.current_process().name == 'MainProcess':
            print(" Starting ERA5 data loading...")
          
        for data_dir in data_dirs:
            if not os.path.exists(data_dir): continue
            try:
                samples = self._load_era5_directory(data_dir, start_date, end_date)
                if samples:
                    all_samples.extend(samples)
            except Exception as e:
                print(f" Failed to load ERA5 directory {data_dir}: {e}")
      
        if mp.current_process().name == 'MainProcess':
            print(f" ERA5 loading complete, total samples: {len(all_samples)}")
        return all_samples
    def load_era5_data_with_times(self, data_dirs, start_date=None, end_date=None):
        """
        New: return dict {valid_time(datetime): sample(np.ndarray [6,H,W])}
        """
        all_pairs = []
        if mp.current_process().name == 'MainProcess':
            print(" Starting ERA5 loading (with timestamps)...")

        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                continue
            try:
                pairs = self._load_era5_directory_with_times(data_dir, start_date, end_date)
                if pairs:
                    all_pairs.extend(pairs)
            except Exception as e:
                print(f" Failed to load ERA5 directory {data_dir}: {e}")

        # Merge into dict (later pair overwrites earlier if same time)
        era5_map = {}
        for t, sample in all_pairs:
            if t is not None and sample is not None:
                era5_map[t] = sample

        if mp.current_process().name == 'MainProcess':
            print(f" ERA5 loading complete (with timestamps), number of valid times: {len(era5_map)}")
        return era5_map


    def _load_era5_directory_with_times(self, data_dir, start_date=None, end_date=None):
        pairs = []
        nc_files = glob.glob(os.path.join(data_dir, "*.nc"))
        if not nc_files:
            return pairs

        surface_files = [f for f in nc_files if 'pressure' not in os.path.basename(f).lower()]
        pressure_files = [f for f in nc_files if 'pressure' in os.path.basename(f).lower()]

        surface_pairs = self._load_surface_data_with_times(surface_files, start_date, end_date)
        pressure_pairs = self._load_pressure_data_with_times(pressure_files, start_date, end_date)

        # merge w into channel 4 by time
        w_map = {t: w for t, w in pressure_pairs if t is not None}
        merged = []
        for t, s in surface_pairs:
            if t is None or s is None:
                continue
            if t in w_map:
                s = s.copy()
                s[4] = w_map[t]
            merged.append((t, s))

        return merged


    def _load_surface_data_with_times(self, nc_files, start_date=None, end_date=None):
        all_pairs = []
        target_start = np.datetime64(start_date) if start_date else None
        target_end = np.datetime64(end_date) if end_date else None

        for nc_file in nc_files:
            try:
                # Automatically try multiple engines
                ds = None
                for eng in ['netcdf4', 'h5netcdf', 'scipy']:
                    try:
                        ds = xr.open_dataset(nc_file, engine=eng)
                        break
                    except:
                        continue
                
                if ds is None:
                    print(f" Cannot open file with any engine: {os.path.basename(nc_file)}")
                    continue

                with ds:
                    if 'expver' in ds.dims:
                        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

                    time_dim_name = 'valid_time' if 'valid_time' in ds.dims else 'time'
                    if time_dim_name not in ds.dims:
                        continue

                    times = ds[time_dim_name].values
                    mask = np.ones(len(times), dtype=bool)
                    if target_start:
                        mask &= (times >= target_start)
                    if target_end:
                        mask &= (times <= target_end)
                    if not np.any(mask):
                        continue

                    idxs = np.where(mask)[0]
                    ds_slice = ds.isel({time_dim_name: idxs})

                    # Unit correction (m->mm)
                    if 'tp' in ds_slice.variables:
                        tp_max = ds_slice['tp'].max().values
                        if tp_max < 0.5:
                            ds_slice['tp'] = ds_slice['tp'] * 1000.0

                    identified_vars = self._identify_variables_in_file(list(ds_slice.variables.keys()), nc_file)

                    for local_i, t_idx in enumerate(range(len(ds_slice[time_dim_name]))):
                        t_val = to_py_datetime(ds_slice[time_dim_name].values[t_idx])

                        var_dict = {}
                        for internal_name, era5_name in identified_vars.items():
                            data = ds_slice[era5_name].values
                            if data.ndim == 3:
                                val = data[t_idx]
                            elif data.ndim == 4:
                                val = data[t_idx, 0]
                            elif data.ndim == 2:
                                val = data
                            else:
                                val = np.zeros((25, 37))
                            var_dict[internal_name] = np.nan_to_num(val, nan=0.0)

                        sample = self._create_era5_sample(var_dict)
                        if sample is not None:
                            all_pairs.append((t_val, sample))
            except Exception:
                continue

        return all_pairs


    def _load_pressure_data_with_times(self, nc_files, start_date=None, end_date=None):
        all_pairs = []
        target_start = np.datetime64(start_date) if start_date else None
        target_end = np.datetime64(end_date) if end_date else None

        for nc_file in nc_files:
            try:
                # Automatically try multiple engines
                ds = None
                for eng in ['netcdf4', 'h5netcdf', 'scipy']:
                    try:
                        ds = xr.open_dataset(nc_file, engine=eng)
                        break
                    except:
                        continue
                
                if ds is None:
                    print(f" Cannot open file with any engine: {os.path.basename(nc_file)}")
                    continue

                with ds:
                    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
                    if time_dim not in ds.dims:
                        continue

                    times = ds[time_dim].values
                    mask = np.ones(len(times), dtype=bool)
                    if target_start:
                        mask &= (times >= target_start)
                    if target_end:
                        mask &= (times <= target_end)
                    if not np.any(mask):
                        continue

                    idxs = np.where(mask)[0]
                    ds_slice = ds.isel({time_dim: idxs})

                    w_var = next((v for v in ['w', 'omega', 'vertical_velocity'] if v in ds_slice.variables), None)
                    if not w_var:
                        continue

                    data = ds_slice[w_var].values
                    for t_idx in range(len(ds_slice[time_dim])):
                        t_val = to_py_datetime(ds_slice[time_dim].values[t_idx])
                        if data.ndim == 4:
                            val = data[t_idx, data.shape[1] // 2]
                        else:
                            val = data[t_idx]
                        all_pairs.append((t_val, np.nan_to_num(val, nan=0.0)))
            except Exception:
                continue

        return all_pairs
    def _load_era5_directory(self, data_dir, start_date=None, end_date=None):
        samples = []
        nc_files = glob.glob(os.path.join(data_dir, "*.nc"))
        if not nc_files: return samples
      
        # Separate surface and pressure level files
        surface_files = [f for f in nc_files if 'pressure' not in os.path.basename(f).lower()]
        pressure_files = [f for f in nc_files if 'pressure' in os.path.basename(f).lower()]
      
        surface_samples = self._load_surface_data(surface_files, start_date, end_date)
        pressure_samples = self._load_pressure_data(pressure_files, start_date, end_date)
      
        return self._merge_surface_pressure_data(surface_samples, pressure_samples)

    def _load_surface_data(self, nc_files, start_date=None, end_date=None):
        """Load surface data - fixed ds undefined + robust time filtering"""
        all_samples = []
        target_start = np.datetime64(start_date) if start_date else None
        target_end = np.datetime64(end_date) if end_date else None

        for nc_file in nc_files:
            try:
                # Automatically try multiple engines
                ds = None
                for eng in ['netcdf4', 'h5netcdf', 'scipy']:
                    try:
                        ds = xr.open_dataset(nc_file, engine=eng)
                        break
                    except:
                        continue
                
                if ds is None:
                    print(f" Cannot open file with any engine: {os.path.basename(nc_file)}")
                    continue

                with ds:
                    # 1) expver merge
                    if 'expver' in ds.dims:
                        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

                    time_dim_name = 'valid_time' if 'valid_time' in ds.dims else 'time'
                    if time_dim_name not in ds.dims:
                        continue

                    times = ds[time_dim_name].values
                    mask = np.ones(len(times), dtype=bool)
                    if target_start is not None:
                        mask &= (times >= target_start)
                    if target_end is not None:
                        mask &= (times <= target_end)
                    if not np.any(mask):
                        continue

                    ds_slice = ds.isel({time_dim_name: np.where(mask)[0]})

                    # 2) tp unit correction (m->mm)
                    if 'tp' in ds_slice.variables:
                        tp_max = ds_slice['tp'].max().values
                        if tp_max < 0.5:
                            ds_slice['tp'] = ds_slice['tp'] * 1000.0

                    identified_vars = self._identify_variables_in_file(list(ds_slice.variables.keys()), nc_file)

                    for t_idx in range(len(ds_slice[time_dim_name])):
                        var_dict = {}
                        for internal_name, era5_name in identified_vars.items():
                            data = ds_slice[era5_name].values
                            if data.ndim == 3:
                                val = data[t_idx]
                            elif data.ndim == 4:
                                val = data[t_idx, 0]
                            elif data.ndim == 2:
                                val = data
                            else:
                                val = np.zeros((25, 37))
                            var_dict[internal_name] = np.nan_to_num(val, nan=0.0)

                        sample = self._create_era5_sample(var_dict)
                        if sample is not None:
                            all_samples.append(sample)

            except Exception as e:
                if mp.current_process().name == 'MainProcess':
                    print(f" Skipping problematic ERA5 file {os.path.basename(nc_file)}: {e}")
                continue

        return all_samples

    def _load_pressure_data(self, nc_files, start_date=None, end_date=None):
        """Load pressure level data - enhanced robustness"""
        all_samples = []
        target_start = np.datetime64(start_date) if start_date else None
        target_end = np.datetime64(end_date) if end_date else None

        for nc_file in nc_files:
            try:
                # Automatically try multiple engines
                ds = None
                for eng in ['netcdf4', 'h5netcdf', 'scipy']:
                    try:
                        ds = xr.open_dataset(nc_file, engine=eng)
                        break
                    except:
                        continue
                
                if ds is None:
                    print(f" Cannot open file with any engine: {os.path.basename(nc_file)}")
                    continue

                with ds:
                    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'

                    # Use same boolean mask logic as surface data
                    times = ds[time_dim].values
                    mask = np.ones(len(times), dtype=bool)
                    if target_start:
                        mask &= (times >= target_start)
                    if target_end:
                        mask &= (times <= target_end)
                    if not np.any(mask):
                        continue

                    ds_slice = ds.isel({time_dim: np.where(mask)[0]})

                    w_var = next((v for v in ['w', 'omega', 'vertical_velocity'] if v in ds_slice.variables), None)
                    if not w_var:
                        continue

                    data = ds_slice[w_var].values
                    for t_idx in range(len(ds_slice[time_dim])):
                        # Take middle level (typically 500hPa or 700hPa)
                        val = data[t_idx, data.shape[1]//2] if data.ndim == 4 else data[t_idx]
                        all_samples.append(np.nan_to_num(val, nan=0.0))
            except Exception:
                continue
        return all_samples

    def _merge_surface_pressure_data(self, surface_samples, pressure_samples):
        if not pressure_samples: return surface_samples
        merged = []
        for i in range(min(len(surface_samples), len(pressure_samples))):
            s = surface_samples[i]
            # Inject vertical velocity into channel 4 (index 4)
            s[4] = pressure_samples[i]
            merged.append(s)
        return merged

    def _identify_variables_in_file(self, file_vars, filename):
        identified = {}
        for internal, possible_names in self.variable_mapping.items():
            for name in possible_names:
                if name in file_vars:
                    identified[internal] = name; break
        return identified

    def _create_era5_sample(self, var_dict):
        # Ensure shape (25, 37)
        shape = (25, 37)
        channels = [
            var_dict.get('cape', np.zeros(shape)),
            var_dict.get('pwat', np.zeros(shape)),
            var_dict.get('u_wind', np.zeros(shape)),
            var_dict.get('v_wind', np.zeros(shape)),
            var_dict.get('vvel', np.zeros(shape)),
            var_dict.get('precipitation', np.zeros(shape))
        ]
        # Force shape check and stack
        final_channels = []
        for c in channels:
            if c.shape != shape:
                res = np.zeros(shape)
                h, w = min(shape[0], c.shape[0]), min(shape[1], c.shape[1])
                res[:h, :w] = c[:h, :w]
                final_channels.append(res)
            else:
                final_channels.append(c)
        return np.stack(final_channels, axis=0)


def custom_collate_fn(batch):
    """Custom collate function to handle dimension mismatches and empty batches (unified 9-channel placeholder, compatible with DEM)"""
    batch = [item for item in batch if item is not None and item[0] is not None and item[1] is not None]

    # Unified placeholder shape: seq_len=6, channels=9
    def _placeholder():
        placeholder_input = torch.zeros((1, 6, 9, 25, 37))  # [B,T,C,H,W]
        placeholder_target = torch.zeros((1, PREDICTION_HORIZON, 25, 37))
        return placeholder_input, placeholder_target

    if len(batch) == 0:
        return _placeholder()

    input_shapes = [item[0].shape for item in batch]
    target_shapes = [item[1].shape for item in batch]

    from collections import Counter
    input_shape_counter = Counter(input_shapes)
    target_shape_counter = Counter(target_shapes)
    if not input_shape_counter or not target_shape_counter:
        return _placeholder()

    most_common_input_shape = input_shape_counter.most_common(1)[0][0]
    most_common_target_shape = target_shape_counter.most_common(1)[0][0]

    processed_inputs = []
    processed_targets = []

    for inputs, targets in batch:
        try:
            # Adjust input dimensions
            if inputs.shape != most_common_input_shape:
                if inputs.dim() == 4:
                    inputs_resized = []
                    for t in range(inputs.shape[0]):
                        time_step = inputs[t].unsqueeze(0)
                        resized = F.interpolate(
                            time_step,
                            size=most_common_input_shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                        inputs_resized.append(resized.squeeze(0))
                    inputs = torch.stack(inputs_resized, dim=0)

            # Adjust target dimensions
            if targets.shape != most_common_target_shape:
                if targets.dim() == 3:
                    targets_resized = []
                    for t in range(targets.shape[0]):
                        time_step = targets[t].unsqueeze(0).unsqueeze(0)
                        resized = F.interpolate(
                            time_step,
                            size=most_common_target_shape[1:],
                            mode='bilinear',
                            align_corners=False
                        )
                        targets_resized.append(resized.squeeze(0).squeeze(0))
                    targets = torch.stack(targets_resized, dim=0)

            processed_inputs.append(inputs)
            processed_targets.append(targets)
        except Exception:
            continue

    if len(processed_inputs) == 0:
        return _placeholder()

    # Stack
    try:
        return torch.stack(processed_inputs), torch.stack(processed_targets)
    except Exception:
        min_batch_size = min(len(processed_inputs), len(processed_targets))
        return torch.stack(processed_inputs[:min_batch_size]), torch.stack(processed_targets[:min_batch_size])

class BalancedPrecipitationDataset(Dataset):
    """
    Research-grade balanced precipitation dataset
    Weight key names unified: use English keys (no_precip, trace, light, moderate, heavy, very_heavy, extreme).
    """
    def __init__(self, data_type='gfs', data_paths=None, sequence_length=6, 
                prediction_horizon=3, temp_extract_dir="temp_extract",
                max_samples=None, standardizer=None, start_date=None, end_date=None,
                target_scaling_factor=1.0, augment=False, self_supervised=False,
                precip_threshold=0.1, precip_weight=3.0, memory_limit_gb=7.0,
                precip_data_path=None, intensity_weights=None,
                cache_dir=None, mode='self_supervised',
                dem_features=None, enable_cleaning=True, **kwargs):
      
        self.data_type = data_type
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.temp_extract_dir = temp_extract_dir
        self.max_samples = max_samples
        self.standardizer = standardizer
        self.start_date = start_date
        self.end_date = end_date
        self.target_scaling_factor = target_scaling_factor
        self.augment = augment
        self.self_supervised = self_supervised
        self.precip_threshold = precip_threshold
        self.precip_weight = precip_weight
        self.memory_limit_gb = memory_limit_gb
        self.precip_data_path = precip_data_path
        self.cache_dir = cache_dir
        self.mode = mode
        self.enable_cleaning = enable_cleaning
      
        # Core fix: ensure weight dictionary is absolutely complete to avoid KeyError
        default_weights = {
            'no_precip': 5.0,     # increased from 1.0 to 5.0 to strongly suppress false alarms
            'Light': 1.0, 
            'Moderate': 5.0, 
            'Heavy': 15.0, 
            'Storm': 50.0
        }
      
        if intensity_weights is None:
            self.intensity_weights = default_weights
        else:
            self.intensity_weights = intensity_weights
            # Auto-fill missing keys
            for k, v in default_weights.items():
                if k not in self.intensity_weights:
                    self.intensity_weights[k] = v

        # DEM feature handling
        received_dem = dem_features if dem_features is not None else kwargs.get('dem_features')
        self.dem_features = received_dem.cpu() if received_dem is not None else None

        if not os.path.exists(temp_extract_dir):
            os.makedirs(temp_extract_dir, exist_ok=True)
      
        # 1. Load raw data
        self.data = self._load_data_optimized(data_paths)

        # 2. Create sequences
        self.sequences = self._create_sequences_optimized()
      
        # 3. Compute sample weights (for extreme class imbalance)
        if len(self.sequences) > 0:
            self.sample_weights = self._calculate_sample_weights()
        else:
            self.sample_weights = []

        print(f" {data_type.upper()} dataset ready: {len(self.sequences)} valid sequences")

    def _calculate_sample_weights(self):
        """Compute sample weights for multi-level precipitation intensities"""
        weights = []
        intensity_counts = {k: 0 for k in self.intensity_weights.keys()}
      
        print(f" Computing sequence weight distribution (total {len(self.sequences)} sequences)...")
      
        for i, (input_seq, target_seq) in enumerate(self.sequences):
            # Get max precipitation intensity in target sequence (de-normalized back to mm)
            max_p = torch.max(target_seq).item() / self.target_scaling_factor
          
            # Assign weight key based on precipitation level
            if max_p < 0.1:
                w_key = 'no_precip'
            elif max_p < 3.0:
                w_key = 'Light'
            elif max_p < 10.0:
                w_key = 'Moderate'
            elif max_p < 20.0:
                w_key = 'Heavy'
            else:
                w_key = 'Storm'
          
            # Safely extract weight
            weight = self.intensity_weights.get(w_key, 1.0)
            weights.append(weight)
            intensity_counts[w_key] += 1
          
            if (i+1) % 5000 == 0:
                print(f"   Scanned {i+1} sequences...")
      
        print(f" Sample level distribution: {intensity_counts}")
        return np.array(weights)

    def _check_memory(self):
        process = psutil.Process(os.getpid())
        if process.memory_info().rss / 1024**3 > self.memory_limit_gb:
            gc.collect()
            return False
        return True

    def _load_data_optimized(self, data_paths):
        if self.data_type == 'gfs':
            return self._load_gfs_data_optimized(data_paths)
        else:
            return self._load_era5_data_optimized(data_paths)

    def _load_gfs_data_optimized(self, data_folders):
        all_data = []
        for folder in data_folders:
            archive_files = (
                glob.glob(os.path.join(folder, "*.zip")) +
                glob.glob(os.path.join(folder, "*.tar")) +
                glob.glob(os.path.join(folder, "*.nc"))
            )
            archive_files.sort()
            processor = EnhancedDataProcessor(base_path=folder, precip_path=self.precip_data_path, end_date=self.end_date)
            batch_results = processor.process_archive_batch_parallel(archive_files, self.temp_extract_dir)
            for res in batch_results:
                if res is not None:
                    if self.enable_cleaning: res = advanced_precip_cleaning(res)
                    all_data.append(res)
                if len(all_data) % 500 == 0: self._check_memory()
            if self.max_samples and len(all_data) >= self.max_samples: break
        return all_data

    def _load_era5_data_optimized(self, data_dirs):
        processor = ERA5DataProcessor()
        return processor.load_era5_data(data_dirs, self.start_date, self.end_date)

    def _create_sequences_optimized(self):
        sequences = []
        if len(self.data) < (self.sequence_length + self.prediction_horizon):
            return sequences
      
        total_possible = len(self.data) - self.sequence_length - self.prediction_horizon + 1
        for i in range(total_possible):
            try:
                # 1. Extract raw input sequence (6, 6, 25, 37)
                input_seq_raw = [self.data[i + j] for j in range(self.sequence_length)]
                # 2. Extract raw precipitation channel [T, 25, 37] - keep separate for normalization
                # Channel 5 is precipitation
                raw_precip_channel = np.stack([self.data[i + j][5] for j in range(self.sequence_length)])
              
                # 3. Convert to tensor
                in_tensor = torch.FloatTensor(np.stack(input_seq_raw))
              
                # 4. Apply standardization
                if self.standardizer:
                    in_tensor = self.standardizer.transform(in_tensor)
                    # Core modification: force channel 5 (precipitation) back to physical values (mm)
                    # So the model sees real mm values like 0, 1.2, 20.5
                    in_tensor[:, 5, :, :] = torch.FloatTensor(raw_precip_channel)

                # 5. Process target values (ERA5)
                target_list = []
                for j in range(self.prediction_horizon):
                    target_list.append(self.data[i + self.sequence_length + j][5])
                out_tensor = torch.FloatTensor(np.stack(target_list))
              
                sequences.append((in_tensor, out_tensor))
            except Exception as e: 
                continue
        return sequences

    def get_sampler(self):
        if not self.sample_weights.any(): return None
        return WeightedRandomSampler(torch.DoubleTensor(self.sample_weights), len(self.sample_weights))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if idx >= len(self.sequences): idx = idx % len(self.sequences)
        input_seq, output_seq = self.sequences[idx]
      
        # Force concatenation on CPU to avoid DataLoader cross-device errors
        if self.dem_features is not None:
            dem_cpu = self.dem_features.cpu() 
            seq_len = input_seq.shape[0]
            # [T, 3, H, W]
            dem_expanded = dem_cpu.unsqueeze(0).expand(seq_len, -1, -1, -1)
            # Only concatenate if channels are still 6 (avoid repeated concat)
            if input_seq.shape[1] == 6:
                input_seq = torch.cat([input_seq, dem_expanded], dim=1)
      
        return input_seq, output_seq
class PairedGFSEra5ResidualDatasetStrict(Dataset):

    def __init__(self,
                 gfs_folders,
                 era5_paths,
                 standardizer=None,
                 sequence_length=6,
                 prediction_horizon=1,
                 temp_extract_dir="temp_extract",
                 precip_data_path=None,
                 start_date=None,
                 end_date=None,
                 dem_features=None,
                 enable_cleaning=True,
                 require_strict_step=True,
                 sequence_step_hours=None,
                 augment=False):
        """
        Parameters:
            augment: bool, whether to apply data augmentation when fetching samples
        """
        assert prediction_horizon == 1, "For f003 only, prediction_horizon must be 1"

        self.standardizer = standardizer
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.temp_extract_dir = temp_extract_dir
        self.precip_data_path = precip_data_path
        self.start_date = start_date
        self.end_date = end_date
        self.enable_cleaning = enable_cleaning
        self.require_strict_step = require_strict_step
        self.sequence_step_hours = sequence_step_hours
        self.augment = augment
        # Parse start/end to datetime (extend end to end of day)
        self.start_dt = None
        self.end_dt = None
        try:
            if start_date:
                self.start_dt = pd.to_datetime(start_date).to_pydatetime()
            if end_date:
                e = pd.to_datetime(end_date).to_pydatetime()
                self.end_dt = e + timedelta(hours=23, minutes=59, seconds=59)
        except Exception:
            self.start_dt, self.end_dt = None, None

        self.dem_features = dem_features.cpu() if dem_features is not None else None
        os.makedirs(self.temp_extract_dir, exist_ok=True)

        # 1) Build GFS: dict(valid_time -> sample[6,H,W]) (filter by time window to prevent leakage)
        self.gfs_map = self._build_gfs_time_map(gfs_folders)

        # 2) Build ERA5: dict(valid_time -> sample[6,H,W])
        era5_proc = ERA5DataProcessor()
        self.era5_map = era5_proc.load_era5_data_with_times(
            era5_paths, start_date=self.start_date, end_date=self.end_date
        )

        # 3) Find common times and sort
        common_times = sorted(set(self.gfs_map.keys()).intersection(set(self.era5_map.keys())))
        if len(common_times) == 0:
            raise RuntimeError("GFS and ERA5 have no common valid_time, check time coordinates/timezone/file coverage")
        if len(common_times) < self.sequence_length:
            raise RuntimeError(f"Insufficient common valid times: common={len(common_times)}, need>={self.sequence_length}")

        # 4) Infer sequence step (hours)
        step_hours = self.sequence_step_hours or self._infer_step_hours(common_times)
        self.inferred_step_hours = step_hours
        dt = timedelta(hours=int(step_hours))

        # 5) Build samples: use last frame time t0 as label time
        self.sample_times = []
        self.sequences = []  # (x[T,C,H,W], y[1,H,W], abs_max)

        common_set = set(common_times)

        for t0 in common_times:
            seq_times = [t0 - dt * (self.sequence_length - 1 - k) for k in range(self.sequence_length)]
            if any(t not in common_set for t in seq_times):
                continue

            if self.require_strict_step:
                ok = True
                for k in range(1, len(seq_times)):
                    if (seq_times[k] - seq_times[k - 1]) != dt:
                        ok = False
                        break
                if not ok:
                    continue

            # Input sequence (GFS)
            x_list = []
            raw_precip_list = []
            for t in seq_times:
                arr = self.gfs_map[t]
                x_list.append(arr)
                raw_precip_list.append(arr[5])

            x = torch.FloatTensor(np.stack(x_list, axis=0))  # [T,6,H,W]

            # Standardization (precipitation kept as physical values)
            if self.standardizer is not None:
                raw_precip = torch.FloatTensor(np.stack(raw_precip_list, axis=0))  # [T,H,W]
                x = self.standardizer.transform(x)
                x[:, 5] = raw_precip

            # Label residual = ERA5_abs(t0) - GFS_base(t0)
            era5_abs = torch.FloatTensor(self.era5_map[t0][5])  # [H,W]
            gfs_base = x[-1, 5]                                 # [H,W]
            y_res = (era5_abs - gfs_base).unsqueeze(0)          # [1,H,W]
            abs_max = float(torch.max(era5_abs).item())

            self.sequences.append((x, y_res, abs_max))
            self.sample_times.append(t0)

        print(f" Paired Strict Dataset ready: {len(self.sequences)} samples "
              f"(GFS={len(self.gfs_map)}, ERA5={len(self.era5_map)}, step={self.inferred_step_hours}h)")

        if len(self.sequences) == 0:
            diffs = []
            for i in range(1, min(len(common_times), 20)):
                diffs.append((common_times[i] - common_times[i-1]).total_seconds() / 3600.0)
            raise RuntimeError(
                "Paired dataset got 0 samples. Most likely reason: your GFS valid_times are not 3-hourly continuous "
                "(often 6-hourly cycles: 03/09/15/21...).\n"
                f"Suggested fix: set sequence_step_hours=6, or keep auto-infer (current inferred={self.inferred_step_hours}).\n"
                f"First diffs(hours) among common_times: {diffs}"
            )

    def _infer_step_hours(self, times):
        diffs = []
        for i in range(1, len(times)):
            dh = int(round((times[i] - times[i-1]).total_seconds() / 3600.0))
            if dh > 0:
                diffs.append(dh)
        if not diffs:
            return TIME_STEP_HOURS

        from collections import Counter
        c = Counter(diffs)
        step = c.most_common(1)[0][0]
        if step not in (1, 3, 6, 12, 24):
            step = TIME_STEP_HOURS
        return step

    def _in_time_window(self, t: datetime):
        if t is None:
            return False
        if self.start_dt is not None and t < self.start_dt:
            return False
        if self.end_dt is not None and t > self.end_dt:
            return False
        return True

    def _build_gfs_time_map(self, gfs_folders):
        gfs_map = {}
        for folder in gfs_folders:
            archive_files = (
                glob.glob(os.path.join(folder, "*.zip")) +
                glob.glob(os.path.join(folder, "*.tar")) +
                glob.glob(os.path.join(folder, "*.nc"))
            )
            archive_files.sort()
            proc = EnhancedDataProcessor(base_path=folder, precip_path=self.precip_data_path, end_date=None)
            pairs = proc.process_archive_batch_parallel_with_times(archive_files, self.temp_extract_dir)

            for t, arr in pairs:
                if arr is None or t is None:
                    continue
                if not self._in_time_window(t):
                    continue
                if self.enable_cleaning:
                    arr = advanced_precip_cleaning(arr)
                gfs_map[t] = arr
        return gfs_map

    def get_target_max_precip(self, idx):
        return self.sequences[idx][2]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y_res, abs_max = self.sequences[idx]

        # Return only the base 6 channels: [CAPE, PWAT, U-Wind, V-Wind, Vvel, GFS-Precip]
        # No DEM or artificially constructed interaction features to avoid noise
        is_storm = abs_max >= 20.0

        # ===== Basic augmentation (random flip + rotation for storm samples) =====
        if self.augment:
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-1])
                y_res = torch.flip(y_res, dims=[-1])

            if random.random() > 0.5:
                x = torch.flip(x, dims=[-2])
                y_res = torch.flip(y_res, dims=[-2])

            if is_storm:
                k = random.randint(0, 3)
                if k != 0:
                    x = torch.rot90(x, k, dims=[-2, -1])
                    y_res = torch.rot90(y_res, k, dims=[-2, -1])

        # ===== Advanced augmentation (only for storm samples during training) =====
        if is_storm and self.augment:
            # Local noise (only for precipitation channel)
            noise = torch.randn_like(x[:, 5:6]) * 0.5
            mask = (x[:, 5:6] > 0).float()
            x[:, 5:6] = x[:, 5:6] + noise * mask * 0.3
            x = torch.clamp(x, min=0.0)

        # MixUp augmentation (only for storm samples, low probability)
        if is_storm and self.augment and random.random() > 0.7:
            other_idx = random.randint(0, len(self.sequences) - 1)
            other_x, other_y_res, other_abs_max = self.sequences[other_idx]
            if other_abs_max >= 20.0:
                lam = random.betavariate(0.5, 0.5) 
                x = lam * x + (1 - lam) * other_x
                y_res = lam * y_res + (1 - lam) * other_y_res

        return x, y_res
class ExtremeEventDataLoader:
    """
    Heavy rain event resampling data loader
    Core idea: greatly increase exposure frequency of heavy rain samples in training set to solve sample imbalance
    """
    @staticmethod
    def create_adaptive_oversampled_loader(dataset, intensity_bins=None, oversample_ratios=None,
                                        batch_size=64, num_workers=0):
        """
        Create a truly effective adaptive oversampling data loader (robust version)
        Improvements:
        1. Retain real duplicate sampling
        2. Reduce oversampling ratios for extreme samples to avoid skewing overall distribution
        3. Only mild downsampling for no-rain samples to keep background distribution
        """
        print(f" Launching enhanced adaptive heavy rain oversampling strategy...")

        if intensity_bins is None:
            intensity_bins = [0, 0.1, 0.5, 3.0, 10.0, 20.0, 50.0, float('inf')]
            intensity_labels = ['No rain', 'Trace', 'Light', 'Moderate', 'Heavy', 'Storm', 'Torrential']
        else:
            intensity_labels = [f'Bin_{i}' for i in range(len(intensity_bins) - 1)]

        # More conservative than previous version to reduce degradation
        if oversample_ratios is None:
            oversample_ratios = [0.6, 1.2, 2.5, 4.0, 8.0, 16.0, 24.0]

        print(f"   Intensity bins: {intensity_labels}")
        print(f"   Oversampling ratios: {oversample_ratios}")

        indices_by_intensity = {label: [] for label in intensity_labels}
        intensity_stats = {label: {'count': 0, 'max_precip': 0.0} for label in intensity_labels}

        print(" Scanning dataset intensity distribution...")
        for idx in range(len(dataset)):
            try:
                if hasattr(dataset, 'get_target_max_precip'):
                    max_precip = float(dataset.get_target_max_precip(idx))
                else:
                    _, target = dataset[idx]
                    if isinstance(target, torch.Tensor):
                        max_precip = target.max().item()
                    else:
                        max_precip = float(np.max(target))
                assigned = False
                for i in range(len(intensity_bins) - 1):
                    if intensity_bins[i] <= max_precip < intensity_bins[i + 1]:
                        label = intensity_labels[i]
                        indices_by_intensity[label].append(idx)
                        intensity_stats[label]['count'] += 1
                        intensity_stats[label]['max_precip'] = max(
                            intensity_stats[label]['max_precip'], max_precip
                        )
                        assigned = True
                        break

                if not assigned:
                    label = intensity_labels[-1]
                    indices_by_intensity[label].append(idx)
                    intensity_stats[label]['count'] += 1
                    intensity_stats[label]['max_precip'] = max(
                        intensity_stats[label]['max_precip'], max_precip
                    )
            except Exception:
                continue

        total_samples = sum(stats['count'] for stats in intensity_stats.values())
        print(f"\n Dataset intensity distribution statistics:")
        for i, label in enumerate(intensity_labels):
            count = intensity_stats[label]['count']
            ratio = count / total_samples * 100 if total_samples > 0 else 0
            max_p = intensity_stats[label]['max_precip']
            oversample = oversample_ratios[i]
            print(f"   {label:<8}: {count:>6} samples ({ratio:6.2f}%), max precip={max_p:6.2f}mm, oversample ratio={oversample:.1f}x")

        oversampled_indices = []

        for i, label in enumerate(intensity_labels):
            indices = indices_by_intensity[label]
            if len(indices) == 0:
                continue

            ratio = oversample_ratios[i]

            if ratio < 1.0:
                keep_count = max(1, int(len(indices) * ratio))
                selected = random.sample(indices, min(keep_count, len(indices)))
                oversampled_indices.extend(selected)
            else:
                int_ratio = int(ratio)
                frac_ratio = ratio - int_ratio

                for idx in indices:
                    oversampled_indices.extend([idx] * int_ratio)
                    if random.random() < frac_ratio:
                        oversampled_indices.append(idx)

        if len(oversampled_indices) == 0:
            print(" Oversampled indices empty, falling back to original dataset loader")
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=custom_collate_fn,
                drop_last=True
            )

        print(f"\n Oversampling statistics:")
        print(f"   Original sample count: {total_samples}")
        print(f"   After oversampling: {len(oversampled_indices)}")
        print(f"   Actual oversampling factor: {len(oversampled_indices) / max(total_samples, 1):.2f}x")

        oversampled_dataset = Subset(dataset, oversampled_indices)
        _ = ExtremeEventDataLoader.analyze_dataset_intensity(oversampled_dataset, thresholds=[0.1, 1.0, 5.0, 10.0, 20.0])
        adaptive_loader = DataLoader(
            oversampled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            drop_last=True
        )

        print(" Adaptive heavy rain oversampling data loader built (robust real duplicate sampling version)")
        return adaptive_loader
    @staticmethod
    def analyze_dataset_intensity(dataset, thresholds=[0.1, 1.0, 5.0, 10.0, 20.0]):
        """
        Analyze distribution of different precipitation intensities in dataset
        (must use ERA5 absolute precipitation, not residual)
        """
        print("\n Dataset precipitation intensity distribution analysis:")

        intensity_counts = {f"<{thresholds[0]}mm": 0}
        for i in range(len(thresholds)):
            if i == len(thresholds) - 1:
                label = f">={thresholds[i]}mm"
            else:
                label = f"{thresholds[i]}-{thresholds[i+1]}mm"
            intensity_counts[label] = 0

        sample_size = min(1000, len(dataset))
        if sample_size <= 0:
            return intensity_counts

        indices = np.random.choice(len(dataset), sample_size, replace=False)

        for idx in indices:
            try:
                if hasattr(dataset, 'get_target_max_precip'):
                    max_precip = float(dataset.get_target_max_precip(idx))
                else:
                    _, target = dataset[idx]
                    max_precip = float(target.max().item()) if isinstance(target, torch.Tensor) else float(np.max(target))

                if max_precip < thresholds[0]:
                    intensity_counts[f"<{thresholds[0]}mm"] += 1
                else:
                    for i in range(len(thresholds)):
                        if i == len(thresholds) - 1:
                            if max_precip >= thresholds[i]:
                                intensity_counts[f">={thresholds[i]}mm"] += 1
                                break
                        elif thresholds[i] <= max_precip < thresholds[i+1]:
                            intensity_counts[f"{thresholds[i]}-{thresholds[i+1]}mm"] += 1
                            break
            except Exception:
                continue

        for intensity, count in intensity_counts.items():
            percentage = count / sample_size * 100
            print(f"   {intensity:<12}: {count:>4} samples ({percentage:5.1f}%)")

        return intensity_counts
class SelfSupervisedGFSDataset(BalancedPrecipitationDataset):
    """GFS self-supervised dataset - designed for self-supervised pretraining"""
  
    def __init__(self, **kwargs):
        # Force self-supervised mode
        kwargs['mode'] = 'self_supervised'
        kwargs['data_type'] = 'gfs'
      
        # Fix: set multi-level weights for GFS self-supervised training (before parent init)
        if 'intensity_weights' not in kwargs:
            kwargs['intensity_weights'] = {
                'no_precip': 1.0,
                'trace': 2.0,       # 0.1-0.5mm
                'light': 3.0,       # 0.5-1mm
                'moderate': 4.0,    # 1-2mm
                'heavy': 6.0,       # 2-5mm
                'very_heavy': 8.0,  # 5-15mm
                'extreme': 10.0     # >15mm (raised threshold!)
            }
      
        # Ensure necessary parameters
        if 'temp_extract_dir' not in kwargs:
            kwargs['temp_extract_dir'] = "temp_extract"
      
        if 'memory_limit_gb' not in kwargs:
            kwargs['memory_limit_gb'] = 7.0
      
        # Key fix: set augment parameter
        if 'augment' not in kwargs:
            kwargs['augment'] = False
      
        # Ensure intensity_weights parameter is passed correctly
        print(f" SelfSupervisedGFSDataset initializing:")
        print(f"   intensity_weights passed: {'intensity_weights' in kwargs}")
      
        # Call parent init
        super().__init__(**kwargs)
      
        # Verify attributes are set correctly
        if hasattr(self, 'intensity_weights'):
            print(f" intensity_weights attribute set: {list(self.intensity_weights.keys())}")
        else:
            print(f" intensity_weights attribute not set, manually setting defaults")
            self.intensity_weights = kwargs.get('intensity_weights', {
                'no_precip': 1.0,
                'trace': 2.0,       # 0.1-0.5mm
                'light': 3.0,       # 0.5-1mm
                'moderate': 4.0,    # 1-2mm
                'heavy': 6.0,       # 2-5mm
                'very_heavy': 8.0,  # 5-15mm
                'extreme': 10.0     # >15mm
            })
class StormPatchWrapper(Dataset):
    """
    During training, preferentially crop patches from heavy rain samples to mitigate "background dilution".
    Compatible with PairedGFSEra5ResidualDatasetStrict: target is residual, but can restore ERA5_abs = gfs_last + residual
    """
    def __init__(self, base_ds, patch=20, storm_th=20.0, storm_prob=1.0):
        self.ds = base_ds
        self.patch = int(patch)
        self.storm_th = float(storm_th)
        self.storm_prob = float(storm_prob)

    def __len__(self):
        return len(self.ds)

    def get_target_max_precip(self, idx):
        # Proxy to base (ensure oversampling statistics still based on "absolute precipitation max")
        if hasattr(self.ds, "get_target_max_precip"):
            return float(self.ds.get_target_max_precip(idx))
        # fallback: compute itself
        x, y_res = self.ds[idx]
        gfs_last = x[-1, 5]
        era5_abs = gfs_last + y_res[0]
        return float(torch.max(era5_abs).item())

    def __getitem__(self, idx):
        x, y_res = self.ds[idx]  # x:[T,C,H,W], y_res:[1,H,W]
        H, W = x.shape[-2], x.shape[-1]

        gfs_last = x[-1, 5]            # [H,W]
        era5_abs = gfs_last + y_res[0] # [H,W]

        # Choose center: for storm samples, pick peak point; otherwise random
        if (era5_abs.max() >= self.storm_th) and (random.random() < self.storm_prob):
            iy, ix = torch.unravel_index(torch.argmax(era5_abs), era5_abs.shape)
            cy, cx = int(iy), int(ix)
        else:
            cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)

        p = self.patch
        y0 = max(0, cy - p // 2)
        y1 = min(H, y0 + p)
        y0 = y1 - p   # ensure y1 - y0 = p
        x0 = max(0, cx - p // 2)
        x1 = min(W, x0 + p)
        x0 = x1 - p

        x_patch = x[..., y0:y1, x0:x1]
        y_patch = y_res[..., y0:y1, x0:x1]

        # ===== New: add noise for storm samples (only during training) =====
        # Gen-3 fix: safely get augment attribute (prevent Subset penetration error)
        is_augment = getattr(self.ds, 'augment', False)
        if not is_augment and hasattr(self.ds, 'dataset'):
            is_augment = getattr(self.ds.dataset, 'augment', False)
        if is_augment and era5_abs.max() >= 20.0:  # storm sample and augmentation enabled
            # Add small Gaussian noise to GFS precipitation channel (only where original precip >0)
            noise = torch.randn_like(x_patch[:, 5:6]) * 0.3
            mask = (x_patch[:, 5:6] > 0).float()
            x_patch[:, 5:6] = torch.clamp(x_patch[:, 5:6] + noise * mask * 0.3, min=0.0)

        return x_patch, y_patch
class ResearchVisualizer:
    @staticmethod
    def get_professional_labels():
        """
        CMA style discrete colormap (mm/3h)
        Unify cmap/norm for all spatial plots to avoid distortion across figures.
        """
        levels = [0, 0.1, 1, 3, 10, 20, 50, 100]  # mm/3h
        colors = ['#FFFFFF', '#A6F28F', '#3DBA3D', '#61B8FF', '#0000FF', '#FA00FA', '#800040']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        return cmap, norm
    @staticmethod
    def plot_density_scatter(preds, obs, gfs, save_path='density_scatter_comparison.png'):
        """Generate journal-standard 2D density scatter plot (Joint PDF)"""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LogNorm

        # Flatten data
        p_f = preds.flatten()
        o_f = obs.flatten()
        g_f = gfs.flatten()

        # Filter out very light precipitation to highlight main rain areas
        mask = (o_f > 0.1) | (p_f > 0.1) | (g_f > 0.1)
        o_f, p_f, g_f = o_f[mask], p_f[mask], g_f[mask]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        max_val = min(100.0, max(np.max(o_f), np.max(p_f), np.max(g_f)))

        # Panel A: GFS vs Obs
        hb1 = axes[0].hexbin(o_f, g_f, gridsize=80, cmap='Spectral_r', norm=LogNorm(), mincnt=1, extent=[0, max_val, 0, max_val])
        axes[0].plot([0, max_val], [0, max_val], 'k--', lw=2, alpha=0.7)
        axes[0].set_xlabel('Observed Precipitation (mm/3h)', fontweight='bold')
        axes[0].set_ylabel('GFS Forecast (mm/3h)', fontweight='bold')
        axes[0].set_title('(a) GFS Baseline vs Observation', fontweight='bold')
        axes[0].grid(True, linestyle=':', alpha=0.6)

        # Panel B: Model vs Obs
        hb2 = axes[1].hexbin(o_f, p_f, gridsize=80, cmap='Spectral_r', norm=LogNorm(), mincnt=1, extent=[0, max_val, 0, max_val])
        axes[1].plot([0, max_val], [0, max_val], 'k--', lw=2, alpha=0.7)
        axes[1].set_xlabel('Observed Precipitation (mm/3h)', fontweight='bold')
        axes[1].set_ylabel('Model Corrected (mm/3h)', fontweight='bold')
        axes[1].set_title('(b) Model vs Observation', fontweight='bold')
        axes[1].grid(True, linestyle=':', alpha=0.6)

        cb = fig.colorbar(hb2, ax=axes.ravel().tolist(), pad=0.02)
        cb.set_label('log10(Data Points Count)', fontweight='bold')

        base_path = save_path.replace('.png', '').replace('.pdf', '')
        save_fig_multi(fig, base_path, dpi=300)
        plt.close(fig)
    @staticmethod
    def plot_taylor_diagram(preds, obs, gfs, save_path='taylor_diagram.png'):
        """
        Patch 1: Taylor Diagram
        Displays standard deviation, correlation coefficient, and RMSE.
        """
        from scipy import stats
        import matplotlib.pyplot as plt

        # Normalize data (observation as reference 1.0)
        def get_stats(p, o):
            std_p = np.std(p) / (np.std(o) + 1e-8)
            cc = np.corrcoef(p, o)[0, 1]
            return std_p, cc

        std_m, cc_m = get_stats(preds.flatten(), obs.flatten())
        std_g, cc_g = get_stats(gfs.flatten(), obs.flatten())

        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = fig.add_subplot(111, projection='polar')

        # Draw CC arc lines
        theta = np.arccos(np.linspace(0, 1, 100))
        ax.set_theta_direction(-1)
        ax.set_theta_offset(0)

        # Draw scatter points
        ax.scatter(np.arccos(cc_m), std_m, color='red', s=100, label='Model', marker='o', edgecolors='k')
        ax.scatter(np.arccos(cc_g), std_g, color='blue', s=100, label='GFS', marker='s', edgecolors='k')
        ax.scatter(np.arccos(1.0), 1.0, color='green', s=120, label='Observed (Ref)', marker='*')

        ax.set_thetamin(0); ax.set_thetamax(90)
        ax.set_xlabel("Normalized Std (Ref=1.0)", fontweight='bold')
        plt.title("Taylor Diagram: Global Prediction Skill", fontweight='bold', pad=20)
        plt.legend()

        base_path = save_path.replace('.png', '').replace('.pdf', '')
        save_fig_multi(fig, base_path, dpi=300)
        plt.close(fig)
    @staticmethod
    def plot_training_history(history, save_path='training_convergence.png'):
        """
        Patch 2: Training convergence curve (Loss Curve)
        """
        plt.figure(figsize=(10, 5), dpi=300)
        plt.plot(history['stage_c_losses'], label='Train Loss', color='blue', lw=2)
        plt.plot(history['stage_c_val_losses'], label='Val Loss', color='red', ls='--', lw=2)
        plt.xlabel('Epochs'); plt.ylabel('Multi-Task Loss')
        plt.title('Model Convergence Analysis', fontweight='bold')
        plt.grid(True, alpha=0.3); plt.legend()

        fig = plt.gcf()
        base_path = save_path.replace('.png', '').replace('.pdf', '')
        save_fig_multi(fig, base_path, dpi=300)
        plt.close(fig)
    @staticmethod
    def plot_spatial_bias_map(preds, obs, gfs, save_path='spatial_bias_gain.png'):
        """
        Patch 3: Spatial bias comparison heatmap
        Shows where model significantly reduces GFS inherent bias.
        """
        extent = get_geo_extent_from_globals()
        gfs_bias = np.mean(np.abs(gfs - obs), axis=0)
        mod_bias = np.mean(np.abs(preds - obs), axis=0)
        gain = gfs_bias - mod_bias  # positive means model better

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        im = ax.imshow(gain, cmap='RdYlGn', extent=extent, origin='upper')
        setup_geo_axes(ax)
        plt.colorbar(im, label='MAE Reduction (mm/3h)')
        plt.title("Spatial Bias Improvement (GFS_MAE - Model_MAE)\nPositive indicates Model is better", fontweight='bold')

        base_path = save_path.replace('.png', '').replace('.pdf', '')
        save_fig_multi(fig, base_path, dpi=300)
        plt.close(fig)
    @staticmethod
    def plot_high_res_spatial_pro(preds, obs, gfs, idx=0):
        cmap, norm = ResearchVisualizer.get_professional_labels()
        fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=300)
      
        titles = [
            '(a) Observed Precipitation (ERA5)',
            '(b) GFS Baseline Forecast',
            '(c) Model Corrected Forecast',
            '(d) Correction Error (Model - GFS)'
        ]
      
        def prepare_slice(data):
            if torch.is_tensor(data): data = data.cpu().numpy()
            while data.ndim > 2: data = data[0]
            return data

        p = prepare_slice(preds[idx])
        o = prepare_slice(obs[idx])
        g = prepare_slice(gfs[idx])
      
        data = [o, g, p, p - o]
        for i, ax in enumerate(axes):
            if i < 3:
                im = ax.imshow(data[i], cmap=cmap, norm=norm, origin='lower')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                vmax_err = max(abs(data[i].min()), abs(data[i].max()), 1.0)
                im = ax.imshow(data[i], cmap='RdBu_r', vmin=-vmax_err, vmax=vmax_err, origin='lower')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(titles[i], fontweight='bold')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    @staticmethod
    def plot_seasonal_comparison_bar(monthly_stats, save_path='monthly_rmse_improvement.png'):
        """
        Optimized research-grade monthly RMSE improvement rate distribution plot
        monthly_stats: dict, contains gfs_se and mod_se lists per month
        """
        import matplotlib.pyplot as plt
        import numpy as np

        months = list(range(1, 13))
        improvements = []

        # 1. Compute relative RMSE improvement per month
        for m in months:
            data = monthly_stats.get(m, {'gfs_se': [], 'mod_se': []})
            if len(data['gfs_se']) > 0:
                rmse_gfs = np.sqrt(np.mean(data['gfs_se']))
                rmse_mod = np.sqrt(np.mean(data['mod_se']))
                imp = (rmse_gfs - rmse_mod) / (rmse_gfs + 1e-8) * 100
                improvements.append(imp)
            else:
                improvements.append(0)

        # 2. Plot settings
        plt.figure(figsize=(13, 6), dpi=300)
        ax = plt.gca()

        colors = ['#66b3ff' if m not in [6, 7, 8] else '#ff6666' for m in months]
        bars = plt.bar(months, improvements, color=colors, edgecolor='white', linewidth=0.8, alpha=0.85)

        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='black')

        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(months, month_labels, fontsize=11)
        plt.ylabel('RMSE Improvement Rate (%)', fontsize=12, fontweight='bold')
        plt.title('Liaohe Basin: Monthly Distribution of Model Correction Efficiency',
                fontsize=14, fontweight='bold', pad=20)
        plt.axvspan(5.5, 8.5, color='gray', alpha=0.05, label='Main Flood Season')

        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='#ff6666', lw=4),
                        Line2D([0], [0], color='#66b3ff', lw=4)]
        ax.legend(custom_lines, ['Flood Season (Critical)', 'Non-Flood Season'],
                loc='upper right', frameon=True)

        plt.tight_layout()

        fig = plt.gcf()
        base_path = save_path.replace('.png', '').replace('.pdf', '')
        save_fig_multi(fig, base_path, dpi=300)
        plt.close(fig)
    @staticmethod
    def plot_reliability_diagram(probs, targets_bin,
                                 bins=10,
                                 save_path="reliability_diagram"):
        """
        Reliability diagram for rain occurrence probability.
        probs: [N,H,W] or [N,1,H,W], values in [0,1]
        targets_bin: same shape, {0,1}
        """
        if probs is None or targets_bin is None:
            print(" reliability: probs/targets_bin missing, skip.")
            return None
        p = probs.reshape(-1).astype(np.float64)
        y = targets_bin.reshape(-1).astype(np.float64)
        # binning
        edges = np.linspace(0, 1, bins + 1)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        obs_freq = np.full(bins, np.nan)
        pred_mean = np.full(bins, np.nan)
        counts = np.zeros(bins, dtype=int)
        for i in range(bins):
            m = (p >= edges[i]) & (p < edges[i+1])
            counts[i] = int(np.sum(m))
            if counts[i] > 0:
                obs_freq[i] = float(np.mean(y[m]))
                pred_mean[i] = float(np.mean(p[m]))
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Perfect reliability")
        ax.plot(pred_mean, obs_freq, 'o-', lw=2.5, color="#ff7f0e", label="Model")
        # count as bars (optional)
        ax2 = ax.twinx()
        ax2.bar(bin_centers, counts, width=0.08, alpha=0.20, color="gray", label="Counts")
        ax2.set_ylabel("sample count", fontweight="bold", color="gray")
        ax2.tick_params(axis='y', labelcolor="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Forecast probability", fontweight="bold")
        ax.set_ylabel("Observed frequency", fontweight="bold")
        ax.set_title("Reliability diagram (rain occurrence)", fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(loc="upper left", frameon=True)
        save_fig_multi(fig, save_path, dpi=300)
        plt.close(fig)
        print(f" reliability diagram saved: {save_path}.png/.pdf")
        return save_path
    @staticmethod
    def plot_storm_contour_comparison(era5, gfs, model,
                                      case_time=None,
                                      threshold=20.0,
                                      save_dir="detailed_storm_cases"):
        """
        Three-way storm object-oriented spatial comparison:
        - Background uses ERA5 (or DEM) base map
        - Overlay contours of ERA5/GFS/Model at given threshold
        - Annotate respective centroids to visually see position shifts
        """
        import os
        from scipy import ndimage
        os.makedirs(save_dir, exist_ok=True)
        extent = get_geo_extent_from_globals()
        def centroid(mask):
            ys, xs = np.where(mask)
            if len(ys) == 0:
                return None
            return float(np.mean(xs)), float(np.mean(ys))  # (x,y) in grid coords
        def contour_and_cent(ax, field, color, label):
            mask = field >= threshold
            # contours
            ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=2.0,
                       extent=extent, origin="upper")
            c = centroid(mask)
            if c is not None:
                # convert to lon/lat approximately: linear mapping using extent (sufficient for paper display)
                lon_min, lon_max, lat_min, lat_max = extent
                x, y = c
                H, W = field.shape
                lon = lon_min + (lon_max - lon_min) * (x / max(W-1, 1))
                lat = lat_max - (lat_max - lat_min) * (y / max(H-1, 1))
                ax.plot(lon, lat, marker='o', markersize=6, color=color, markeredgecolor='white', markeredgewidth=1.0)
                ax.text(lon, lat, f" {label}", color=color, fontweight="bold", fontsize=9)
        # title time
        if case_time is not None and hasattr(case_time, "strftime"):
            tstr = case_time.strftime("%Y-%m-%d %H:%M UTC")
            fname = case_time.strftime("%Y%m%d_%H%M")
        else:
            tstr = "case"
            fname = "case"
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.5), dpi=300)
        # Background: ERA5 precipitation (can also use DEM)
        cmap, norm = ResearchVisualizer.get_professional_labels()
        im = ax.imshow(era5, cmap=cmap, norm=norm, extent=extent, origin="upper")
        setup_geo_axes(ax, with_grid=False)
        contour_and_cent(ax, era5, color="#2ca02c", label="ERA5")
        contour_and_cent(ax, gfs,  color="#1f77b4", label="GFS")
        contour_and_cent(ax, model,color="#ff7f0e", label="Model")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("mm/3h", fontweight="bold")
        ax.set_title(f"Storm contour comparison (≥{threshold} mm/3h)\n{tstr}", fontweight="bold")
        out_no_ext = os.path.join(save_dir, f"storm_contours_th{int(threshold)}_{fname}")
        save_fig_multi(fig, out_no_ext, dpi=300)
        plt.close(fig)
        print(f" contour comparison saved: {out_no_ext}.png/.pdf")
        return out_no_ext
    @staticmethod
    def create_event_composite_maps(test_metrics_summary,
                                    thresholds=(10.0, 20.0),
                                    select_by='max',  # 'max' or 'area'
                                    min_area=5,
                                    save_dir='storm_composites'):
        """
        Event composite spatial maps (key journal figure):
        - Compute composite mean precipitation for set of samples meeting condition
        - Compute exceedance frequency (frequency exceeding threshold)
        - Compute bias maps (GFS-ERA5, Model-ERA5)
      
        Parameters:
        - thresholds: e.g., (10,20); suggest main plot >=10, supplementary >=20
        - select_by:
            'max': select samples where max(ERA5) >= th (stable, easy)
            'area': select samples where connected area >= min_area (more object-oriented)
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        preds = test_metrics_summary.get('predictions')   # [N,H,W]
        tars  = test_metrics_summary.get('targets')       # [N,H,W]
        gfs   = test_metrics_summary.get('gfs_baseline')  # [N,H,W]
        sample_times = test_metrics_summary.get('sample_times', None)
        if preds is None or tars is None or gfs is None:
            print(" composite maps: predictions/targets/gfs_baseline missing, skip.")
            return {}
        cmap, norm = ResearchVisualizer.get_professional_labels()
        extent = get_geo_extent_from_globals()
        def _select_indices_by_area(th):
            # Use existing area-based storm detection logic (multiple objects per sample)
            idxs = []
            for i in range(len(tars)):
                events = ResearchVisualizer.identify_storm_events_by_area(
                    tars[i], threshold=th, min_area=min_area
                )
                if len(events) > 0:
                    idxs.append(i)
            return idxs
        out = {}
        for th in thresholds:
            if select_by == 'area':
                idxs = _select_indices_by_area(th)
            else:
                # default: max-based
                idxs = np.where(np.max(tars, axis=(1, 2)) >= th)[0].tolist()
            idxs = deduplicate_keep_order(idxs)
            n = len(idxs)
            if n == 0:
                print(f" composite: th={th} no samples, skip.")
                continue
            # composite mean
            era5_mean = np.mean(tars[idxs], axis=0)
            gfs_mean  = np.mean(gfs[idxs], axis=0)
            mod_mean  = np.mean(preds[idxs], axis=0)
            # frequency maps
            era5_freq = np.mean((tars[idxs] >= th).astype(np.float32), axis=0)
            gfs_freq  = np.mean((gfs[idxs]  >= th).astype(np.float32), axis=0)
            mod_freq  = np.mean((preds[idxs] >= th).astype(np.float32), axis=0)
            # bias maps (mean bias in mm/3h)
            gfs_bias = gfs_mean - era5_mean
            mod_bias = mod_mean - era5_mean
            # ---------- Figure 1: composite mean ----------
            fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
            for ax, field, title in zip(
                axes1,
                [era5_mean, gfs_mean, mod_mean],
                [f"ERA5 mean (events≥{th})",
                 f"GFS mean (events≥{th})",
                 f"Model mean (events≥{th})"]
            ):
                im = ax.imshow(field, cmap=cmap, norm=norm, extent=extent, origin="upper")
                setup_geo_axes(ax, with_grid=False)
                ax.set_title(title, fontweight="bold")
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label("mm/3h", fontweight="bold")
            fig1.suptitle(f"Composite mean precipitation (N={n}, threshold={th} mm/3h)",
                          fontweight="bold", fontsize=14, y=1.02)
            out1 = os.path.join(save_dir, f"composite_mean_th{int(th)}_N{n}")
            save_fig_multi(fig1, out1, dpi=300)
            plt.close(fig1)
            # ---------- Figure 2: frequency maps ----------
            fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
            for ax, field, title in zip(
                axes2,
                [era5_freq, gfs_freq, mod_freq],
                [f"ERA5 freq(P≥{th})",
                 f"GFS freq(P≥{th})",
                 f"Model freq(P≥{th})"]
            ):
                im = ax.imshow(field, cmap="magma", vmin=0.0, vmax=1.0, extent=extent, origin="upper")
                setup_geo_axes(ax, with_grid=False)
                ax.set_title(title, fontweight="bold")
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label("probability", fontweight="bold")
            fig2.suptitle(f"Exceedance frequency maps (N={n}, threshold={th} mm/3h)",
                          fontweight="bold", fontsize=14, y=1.02)
            out2 = os.path.join(save_dir, f"frequency_maps_th{int(th)}_N{n}")
            save_fig_multi(fig2, out2, dpi=300)
            plt.close(fig2)
            # ---------- Figure 3: bias maps ----------
            fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
            for ax, field, title in zip(
                axes3,
                [gfs_bias, mod_bias],
                [f"GFS bias (GFS-ERA5) | th={th}",
                 f"Model bias (Model-ERA5) | th={th}"]
            ):
                vmax = max(1.0, float(np.percentile(np.abs(field), 99)))
                im = ax.imshow(field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=extent, origin="upper")
                setup_geo_axes(ax, with_grid=False)
                ax.set_title(title, fontweight="bold")
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label("mm/3h", fontweight="bold")
            fig3.suptitle(f"Composite bias maps (N={n}, threshold={th} mm/3h)",
                          fontweight="bold", fontsize=14, y=1.02)
            out3 = os.path.join(save_dir, f"bias_maps_th{int(th)}_N{n}")
            save_fig_multi(fig3, out3, dpi=300)
            plt.close(fig3)
            # time coverage info
            if sample_times is not None and n > 0:
                t0 = sample_times[idxs[0]] if idxs[0] < len(sample_times) else None
                t1 = sample_times[idxs[-1]] if idxs[-1] < len(sample_times) else None
            else:
                t0 = t1 = None
            out[th] = {
                "n_samples": n,
                "indices": idxs[:50],
                "time_first": str(t0) if t0 else None,
                "time_last": str(t1) if t1 else None,
                "saved": [out1, out2, out3]
            }
            print(f" composite maps done: th={th}, N={n} -> {save_dir}")
        return out
    @staticmethod
    def analyze_feature_importance(model, test_loader, device, scaling_factor=1.0):
        """
        Feature importance analysis based on final gated prediction (strictly distinguish positive gain from noise)
        """
        print("\n Starting model feature importance analysis...")
        model.eval()
        channel_names = ['CAPE', 'PWAT', 'U-Wind', 'V-Wind', 'V-Velocity', 'GFS-Precip']
        
        th_base = float(GATE_CFG.get("threshold_base", 0.22))
        gate_pow = float(GATE_CFG.get("gate_power", 0.90))
        storm_gate_p = float(GATE_CFG.get("storm_gate_p", 0.30))
        base_mses = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                pred_abs, true_abs, _, _, *rest = get_model_eval_tensors(
                    model=model, inputs=inputs, targets_scaled=targets,
                    scaling_factor=scaling_factor, rain_prob_threshold=th_base,
                    gate_power=gate_pow, min_rain_value=float(GATE_CFG.get("min_rain_value", 0.10)),
                    max_precip=float(GATE_CFG.get("max_precip", 250.0)),
                    adaptive=True, hard_gate=True, storm_gate_p=storm_gate_p
                )
                base_mses.append(F.mse_loss(pred_abs[:, -1], true_abs[:, -1]).item())
        base_mse = float(np.mean(base_mses)) if base_mses else 0.0
        importance_scores = []
        
        for c in range(len(channel_names)):
            perturbed_mses = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs_p = inputs.clone().to(device)
                    targets = targets.to(device)
                    # Perturbation: set to standardized mean 0.0 (effectively remove spatiotemporal variability of that physical feature)
                    inputs_p[:, :, c, :, :] = 0.0
                    pred_abs_p, true_abs_p, _, _, *rest = get_model_eval_tensors(
                        model=model, inputs=inputs_p, targets_scaled=targets,
                        scaling_factor=scaling_factor, rain_prob_threshold=th_base,
                        gate_power=gate_pow, min_rain_value=float(GATE_CFG.get("min_rain_value", 0.10)),
                        max_precip=float(GATE_CFG.get("max_precip", 250.0)),
                        adaptive=True, hard_gate=True, storm_gate_p=storm_gate_p
                    )
                    perturbed_mses.append(F.mse_loss(pred_abs_p[:, -1], true_abs_p[:, -1]).item())
            
            # Compute increase in MSE after perturbation
            importance = float(np.mean(perturbed_mses)) - base_mse
            
            # Optimize log output structure: clearly identify suppressed noise features
            if importance < 0:
                print(f"  - Feature [{channel_names[c]:<10}] attribution: {importance:+.6f}  (appears as background noise, suppressed by adaptive gate)")
                importance_scores.append(0.0)
            else:
                print(f"  - Feature [{channel_names[c]:<10}] attribution: {importance:+.6f}  (positive physical contribution)")
                importance_scores.append(importance)
            
        plt.figure(figsize=(10, 5), dpi=300)
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(channel_names)))
        sorted_idx = np.argsort(importance_scores)[::-1]
        plt.bar(range(len(channel_names)), np.array(importance_scores)[sorted_idx], color=colors)
        plt.xticks(range(len(channel_names)), [channel_names[i] for i in sorted_idx], rotation=45)
        plt.ylabel('Importance (Increase in MSE when removed)', fontweight='bold')
        plt.title('Attribution Analysis: Physical Feature Contribution', fontsize=14, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('feature_importance_wrr_style.png')
        plt.close()
        return importance_scores
    @staticmethod
    def print_scientific_scorecard(metrics):
        """Print Scientific Performance Scorecard"""
        print("\n" + "="*80)
        print(" SCIENTIFIC PERFORMANCE SCORECARD")
        print("="*80)
        print(f"{'Level':<12} | {'Model ETS':<10} | {'GFS ETS':<10} | {'Model POD':<10} | {'GFS POD':<10}")
        print("-"*80)
      
        for level in ['Light', 'Moderate', 'Heavy', 'Storm']:
            if level in metrics['Model'] and level in metrics['GFS']:
                m = metrics['Model'][level]
                g = metrics['GFS'][level]
                print(f"{level:<12} | {m['ETS']:<10.3f} | {g['ETS']:<10.3f} | "
                    f"{m['POD']:<10.3f} | {g['POD']:<10.3f}")
      
        print("="*80)

    @staticmethod
    def create_storm_specific_visualizations(test_metrics_summary, n_cases=3, save_dir='storm_cases', 
                                            sample_times=None, test_start_date=None, test_end_date=None,
                                            specific_indices=None):
        """
        Specialized visualization for storm cases - Enhanced version (includes time information)
      
        Parameters:
            specific_indices: List of sample indices to analyze, if provided, ignores n_cases
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
      
        # 1. Extract data
        predictions = test_metrics_summary['predictions']
        targets = test_metrics_summary['targets']
        gfs_baseline = test_metrics_summary['gfs_baseline']
        dem_features = test_metrics_summary.get('dem_features')
      
        # Ensure correct data dimensions
        if predictions.ndim == 4:  # [B, T, H, W]
            predictions_last = predictions[:, -1]  # Take last time step
            targets_last = targets[:, -1]
            gfs_last = gfs_baseline[:, -1] if gfs_baseline.ndim == 4 else gfs_baseline
        else:
            predictions_last = predictions
            targets_last = targets
            gfs_last = gfs_baseline
      
        # Modification: If specific indices are provided, use those
        if specific_indices is not None:
            storm_indices = specific_indices
        else:
            # 2. Identify storm cases (by maximum precipitation intensity)
            max_precip = np.max(targets_last, axis=(1,2)) if targets_last.ndim == 3 else targets_last
            storm_indices = np.argsort(max_precip)[-n_cases:][::-1]
      
        # New: Get time information
        time_info = []
        if sample_times is not None:
            for i, idx in enumerate(storm_indices):
                if idx < len(sample_times):
                    time_info.append(sample_times[idx])
                else:
                    time_info.append(f"Sample_{idx}")
        else:
            for idx in storm_indices:
                time_info.append(f"Sample_{idx}")
      
        print(f" Analyzing {len(storm_indices)} storm cases:")
        for i, idx in enumerate(storm_indices):
            storm_strength = np.max(targets_last[idx])
            time_str = time_info[i].strftime('%Y-%m-%d %H:%M') if hasattr(time_info[i], 'strftime') else str(time_info[i])
            print(f"  Case {i+1}: Index {idx}, Time {time_str}, Max precipitation {storm_strength:.1f}mm")
      
        # 3. Generate storm case visualizations
        storm_cases_info = []
        for i, idx in enumerate(storm_indices):
            case_info = ResearchVisualizer._create_single_storm_case(
                model_pred=predictions_last[idx],
                gfs_pred=gfs_last[idx],
                era5_truth=targets_last[idx],
                dem_features=dem_features,
                case_idx=idx,
                save_dir=save_dir,
                case_number=i+1,
                case_time=time_info[i],
                test_start_date=test_start_date,
                test_end_date=test_end_date     
            )
            storm_cases_info.append(case_info)
      
        # 4. Generate storm case summary figure
        ResearchVisualizer._create_storm_summary_figure(
            predictions_last, gfs_last, targets_last, 
            storm_indices, save_dir, time_info
        )
      
        return storm_indices, storm_cases_info
  
    @staticmethod
    def identify_all_storm_events(test_metrics_summary, thresholds=None, 
                                 sample_times=None, min_storm_strength=10.0,
                                 use_area_detection=True, min_area=5):
        """
        Identify all storm events in the test set - Improved version
      
        Parameters:
            use_area_detection: Whether to use area detection (True) or point detection (False)
            min_area: Minimum area threshold for area detection
        """
        if thresholds is None:
            thresholds = {
                'Heavy Rain': 10.0,
                'Storm': 20.0,
                'Severe Storm': 50.0,
                'Extreme Storm': 100.0
            }
      
        predictions = test_metrics_summary['predictions']
        targets = test_metrics_summary['targets']
      
        # Get data dimensions
        if targets.ndim == 4:  # [B, T, H, W]
            targets_last = targets[:, -1]
        else:
            targets_last = targets
      
        storm_events = []
      
        # Select detection method
        if use_area_detection:
            print(" Using area detection method for storm events...")
            # Use new area detection method
            for i in range(len(targets_last)):
                # Detect all storm events in current sample
                sample_events = ResearchVisualizer.identify_storm_events_by_area(
                    targets_last[i], 
                    threshold=min_storm_strength,
                    min_area=min_area
                )
              
                # Add time information for each event
                for event in sample_events:
                    event['index'] = i
                    event['time'] = sample_times[i] if sample_times and i < len(sample_times) else None
                    storm_events.append(event)
        else:
            print(" Using point detection method for storm events...")
            # Original method (point detection)
            for i in range(len(targets_last)):
                max_intensity = np.max(targets_last[i])
              
                if max_intensity >= min_storm_strength:
                    # Determine storm level
                    storm_level = 'Heavy Rain'
                    for level, threshold in sorted(thresholds.items(), key=lambda x: x[1], reverse=True):
                        if max_intensity >= threshold:
                            storm_level = level
                            break
                  
                    # Calculate storm area (grid points exceeding 10mm)
                    storm_area = np.sum(targets_last[i] >= min_storm_strength)
                  
                    # Get time information
                    event_time = sample_times[i] if sample_times and i < len(sample_times) else None
                  
                    # Record storm event
                    storm_event = {
                        'index': i,
                        'time': event_time,
                        'max_intensity': float(max_intensity),
                        'storm_level': storm_level,
                        'storm_area': int(storm_area),
                        'avg_intensity': float(np.mean(targets_last[i][targets_last[i] >= 1.0]) if np.any(targets_last[i] >= 1.0) else 0.0),
                        'detection_method': 'point'
                    }
                  
                    storm_events.append(storm_event)
      
        # Sort by intensity
        storm_events.sort(key=lambda x: x['max_intensity'], reverse=True)
      
        # Print statistics
        print(f" Detected {len(storm_events)} storm events")
        if use_area_detection and storm_events:
            areas = [e['area'] for e in storm_events]
            print(f"   Average area: {np.mean(areas):.1f} grid points, Max area: {np.max(areas)} grid points")
      
        return storm_events
  
    @staticmethod
    def identify_storm_events_by_area(targets, threshold=10.0, min_area=5):
        """
        Area-based storm event detection (area identification) - Fixed version
        """
        import numpy as np
        from scipy import ndimage
      
        if targets.ndim == 3:  # [B, H, W]
            storm_events_all = []
            for i in range(targets.shape[0]):
                events = ResearchVisualizer._single_image_storm_detection(
                    targets[i], threshold, min_area, sample_idx=i
                )
                storm_events_all.extend(events)
            return storm_events_all
        else:  # [H, W]
            return ResearchVisualizer._single_image_storm_detection(
                targets, threshold, min_area
            )

    @staticmethod
    def _single_image_storm_detection(image, threshold, min_area, sample_idx=None):
        """Single image storm detection - Fixed version"""
        from scipy import ndimage
      
        # 1. Create binary mask
        binary = image >= threshold
      
        if not binary.any():
            return []
      
        # 2. Connected component analysis
        structure = np.ones((3, 3))
        labeled, num_features = ndimage.label(binary, structure=structure)
      
        storm_events = []
      
        for i in range(1, num_features + 1):
            mask = (labeled == i)
            area = np.sum(mask)
          
            if area >= min_area:  # Area threshold
                # Calculate event properties
                region_data = image[mask]
                max_intensity = np.max(region_data)
                avg_intensity = np.mean(region_data)
              
                # Calculate centroid position
                rows, cols = np.where(mask)
                centroid_lat = int(np.mean(rows))
                centroid_lon = int(np.mean(cols))
              
                # Calculate bounding box
                min_row, max_row = np.min(rows), np.max(rows)
                min_col, max_col = np.min(cols), np.max(cols)
              
                storm_event = {
                    'sample_idx': sample_idx,
                    'storm_area': int(area),
                    'area': int(area),
                    'max_intensity': float(max_intensity),
                    'avg_intensity': float(avg_intensity),
                    'centroid': (centroid_lat, centroid_lon),
                    'bbox': (min_row, max_row, min_col, max_col),
                    'mask': mask
                }
              
                # Determine storm level
                if max_intensity >= 20.0:
                    storm_event['storm_level'] = 'Storm'
                elif max_intensity >= 10.0:
                    storm_event['storm_level'] = 'Heavy Rain'
                elif max_intensity >= 5.0:
                    storm_event['storm_level'] = 'Moderate Rain'
                else:
                    storm_event['storm_level'] = 'Light Rain'
              
                storm_events.append(storm_event)
      
        # Sort by intensity
        storm_events.sort(key=lambda x: x['max_intensity'], reverse=True)
      
        return storm_events
  
    @staticmethod
    def analyze_storm_statistics(storm_events):
        """
        Analyze statistical characteristics of storm events - Fixed version
        """
        if not storm_events:
            return {
                'total_count': 0,
                'intensity_stats': {},
                'temporal_stats': {},
                'level_distribution': {}
            }
      
        intensities = []
        areas = []
        level_counts = {}
      
        for event in storm_events:
            if 'max_intensity' in event:
                intensities.append(event['max_intensity'])
          
            area = 0
            if 'storm_area' in event:
                area = event['storm_area']
            elif 'area' in event:
                area = event['area']
            areas.append(area)
          
            level = event.get('storm_level', 'Unknown')
            level_counts[level] = level_counts.get(level, 0) + 1
      
        if intensities:
            stats = {
                'total_count': len(storm_events),
                'intensity_stats': {
                    'mean': np.mean(intensities),
                    'median': np.median(intensities),
                    'std': np.std(intensities),
                    'max': np.max(intensities),
                    'min': np.min(intensities),
                    'q1': np.percentile(intensities, 25),
                    'q3': np.percentile(intensities, 75)
                },
                'area_stats': {
                    'mean': np.mean(areas),
                    'median': np.median(areas),
                    'max': np.max(areas),
                    'min': np.min(areas)
                } if areas else {},
                'level_distribution': level_counts
            }
        else:
            stats = {
                'total_count': 0,
                'intensity_stats': {},
                'area_stats': {},
                'level_distribution': {}
            }
      
        print(f" Storm Event Statistics:")
        print(f"  Total events: {stats['total_count']}")
      
        if intensities:
            print(f"  Intensity range: {stats['intensity_stats']['min']:.1f} - {stats['intensity_stats']['max']:.1f} mm")
            print(f"  Average intensity: {stats['intensity_stats']['mean']:.1f} ± {stats['intensity_stats']['std']:.1f} mm")
      
        if level_counts:
            print(f"  Storm level distribution:")
            for level, count in level_counts.items():
                percentage = count / stats['total_count'] * 100
                print(f"    {level}: {count} events ({percentage:.1f}%)")
      
        return stats
  
    @staticmethod
    def get_storm_area(event):
        """Unified method to get storm area"""
        if 'storm_area' in event:
            return event['storm_area']
        elif 'area' in event:
            return event['area']
        else:
            return 0
  
    @staticmethod
    def get_storm_intensity(event):
        """Unified method to get storm intensity"""
        if 'max_intensity' in event:
            return event['max_intensity']
        elif 'intensity' in event:
            return event['intensity']
        else:
            return 0.0

    @staticmethod
    def generate_comprehensive_storm_report(storm_events, storm_stats, sample_times, 
                                          detailed_cases, save_path='comprehensive_storm_report.txt'):
        """
        Generate comprehensive storm event analysis report
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" COMPREHENSIVE STORM EVENT ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
          
            # Basic information
            f.write("1. BASIC INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if sample_times:
                f.write(f"Time Range: {sample_times[0].strftime('%Y-%m-%d %H:%M')} to {sample_times[-1].strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Total Samples: {len(sample_times) if sample_times else 'Unknown'}\n")
            f.write(f"Total Storm Events: {storm_stats['total_count']}\n\n")
          
            # Statistical summary
            f.write("2. STATISTICAL SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Maximum Intensity: {storm_stats['intensity_stats']['max']:.1f} mm/3h\n")
            f.write(f"Average Intensity: {storm_stats['intensity_stats']['mean']:.1f} ± {storm_stats['intensity_stats']['std']:.1f} mm/3h\n")
            f.write(f"Median Intensity: {storm_stats['intensity_stats']['median']:.1f} mm/3h\n")
            f.write(f"Intensity Range: {storm_stats['intensity_stats']['min']:.1f} - {storm_stats['intensity_stats']['max']:.1f} mm/3h\n")
            f.write(f"Average Storm Area: {storm_stats['area_stats']['mean']:.0f} grid points\n\n")
          
            # Level distribution
            f.write("3. STORM LEVEL DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            for level, count in storm_stats['level_distribution'].items():
                percentage = count / storm_stats['total_count'] * 100
                f.write(f"{level}: {count} events ({percentage:.1f}%)\n")
            f.write("\n")
          
            # Top 10 strongest storm events
            f.write("4. TOP 10 STRONGEST STORM EVENTS\n")
            f.write("-" * 80 + "\n")
            f.write("Rank | Time | Max Intensity(mm) | Level | Area(Points) | Avg Intensity(mm)\n")
            f.write("-" * 80 + "\n")
          
            top_storms = sorted(storm_events, key=lambda x: x['max_intensity'], reverse=True)[:10]
            for i, storm in enumerate(top_storms, 1):
                time_str = storm['time'].strftime('%Y-%m-%d %H:%M') if storm['time'] else 'Unknown'
                f.write(f"{i:4d} | {time_str} | {storm['max_intensity']:15.1f} | {storm['storm_level']:6} | "
                       f"{ResearchVisualizer.get_storm_area(storm):12d} | {storm.get('mean_intensity', storm.get('avg_intensity', 0)):15.1f}\n")
          
            # Detailed analysis summary
            if detailed_cases:
                f.write("\n5. DETAILED ANALYSIS SUMMARY\n")
                f.write("-" * 40 + "\n")
                for i, case in enumerate(detailed_cases, 1):
                    if 'case_time' in case:
                        f.write(f"\nCase {i}:\n")
                        f.write(f"  Time: {case.get('case_time', 'Unknown')}\n")
                        f.write(f"  Maximum Intensity: {case.get('storm_strength', 0):.1f} mm\n")
                        f.write(f"  Improvement Rate: {case.get('improvement', 0):.1f}%\n")
          
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
      
        print(f" Comprehensive report saved: {save_path}")
  
    @staticmethod
    def _create_single_storm_case(model_pred, gfs_pred, era5_truth, dem_features,
                                  case_idx, save_dir, case_number, case_time=None,
                                  test_start_date=None, test_end_date=None):
        """
        Journal-ready storm case plot:
        - Unified CMA color scale
        - Add lon/lat ticks
        - Unified units mm/3h
        - Concise titles/annotations (avoid text overload)
        """
        from matplotlib.gridspec import GridSpec
        import os
        cmap, norm = ResearchVisualizer.get_professional_labels()
        extent = get_geo_extent_from_globals()
        # Time string
        if case_time is not None and hasattr(case_time, "strftime"):
            time_str = case_time.strftime("%Y-%m-%d %H:%M UTC")
            fname_time = case_time.strftime("%Y%m%d_%H%M")
        else:
            time_str = f"Index {case_idx}"
            fname_time = f"idx{case_idx}"
        storm_strength = float(np.max(era5_truth))
        vmax = max(20.0, min(100.0, storm_strength * 1.2))
        fig = plt.figure(figsize=(20, 12), dpi=300)
        gs = GridSpec(3, 4, figure=fig, hspace=0.28, wspace=0.18)
        # ---------- Row 1: 4 panels ----------
        titles = ["ERA5 (Target)", "GFS (Baseline)", "Model (Corrected)", "Delta (Model - GFS)"]
        fields = [era5_truth, gfs_pred, model_pred, model_pred - gfs_pred]
        for col in range(4):
            ax = fig.add_subplot(gs[0, col])
            if col < 3:
                im = ax.imshow(fields[col], cmap=cmap, norm=norm, extent=extent, origin="upper", interpolation='bicubic')
                setup_geo_axes(ax, with_grid=False)
                ax.set_title(titles[col], fontweight="bold", fontsize=12)
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("mm/3h", fontweight="bold")
            else:
                delta = fields[col]
                vmax_err = max(1.0, float(np.percentile(np.abs(delta), 99)))
                im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax_err, vmax=vmax_err,
                               extent=extent, origin="upper", interpolation='bicubic')
                setup_geo_axes(ax, with_grid=False)
                ax.set_title(titles[col], fontweight="bold", fontsize=12)
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("mm/3h", fontweight="bold")
        # ---------- Row 2: summary bars ----------
        # Max intensity
        ax1 = fig.add_subplot(gs[1, 0])
        intensities = [float(np.max(era5_truth)), float(np.max(gfs_pred)), float(np.max(model_pred))]
        labels = ["ERA5", "GFS", "Model"]
        colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
        bars = ax1.bar(labels, intensities, color=colors)
        ax1.set_ylabel("Max (mm/3h)", fontweight="bold")
        ax1.set_title("Peak Intensity", fontweight="bold")
        ax1.grid(True, axis="y", alpha=0.25, linestyle="--")
        for b, v in zip(bars, intensities):
            ax1.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}", ha="center", va="bottom", fontweight="bold")
        # Area above 20mm
        ax2 = fig.add_subplot(gs[1, 1])
        th = 20.0
        areas = [int(np.sum(era5_truth >= th)), int(np.sum(gfs_pred >= th)), int(np.sum(model_pred >= th))]
        bars2 = ax2.bar(labels, areas, color=colors)
        ax2.set_ylabel(f"Area (grid points ≥ {th}mm)", fontweight="bold")
        ax2.set_title("Storm Area", fontweight="bold")
        ax2.grid(True, axis="y", alpha=0.25, linestyle="--")
        for b, v in zip(bars2, areas):
            ax2.text(b.get_x() + b.get_width()/2, v, f"{v:d}", ha="center", va="bottom", fontweight="bold")
        # MAE
        ax3 = fig.add_subplot(gs[1, 2])
        gfs_mae = float(np.mean(np.abs(gfs_pred - era5_truth)))
        mod_mae = float(np.mean(np.abs(model_pred - era5_truth)))
        bars3 = ax3.bar(["GFS", "Model"], [gfs_mae, mod_mae], color=["#1f77b4", "#ff7f0e"])
        ax3.set_ylabel("MAE (mm/3h)", fontweight="bold")
        ax3.set_title("Mean Absolute Error", fontweight="bold")
        ax3.grid(True, axis="y", alpha=0.25, linestyle="--")
        for b, v in zip(bars3, [gfs_mae, mod_mae]):
            ax3.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
        # Improvement (%)
        ax4 = fig.add_subplot(gs[1, 3])
        imp = (gfs_mae - mod_mae) / (gfs_mae + 1e-8) * 100.0
        bars4 = ax4.bar(["Improvement"], [imp], color=["green" if imp >= 0 else "red"])
        ax4.set_ylabel("ΔMAE / MAE(GFS) (%)", fontweight="bold")
        ax4.set_title("Relative Improvement", fontweight="bold")
        ax4.axhline(0, color="black", linewidth=0.8)
        ax4.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax4.text(bars4[0].get_x() + bars4[0].get_width()/2, imp,
                 f"{imp:.1f}%", ha="center", va="bottom" if imp >= 0 else "top", fontweight="bold")
        # ---------- Row 3: error maps + histogram ----------
        ax5 = fig.add_subplot(gs[2, 0])
        gfs_err = np.abs(gfs_pred - era5_truth)
        im5 = ax5.imshow(gfs_err, cmap="YlOrRd", vmin=0, vmax=max(5.0, np.percentile(gfs_err, 99)),
                         extent=extent, origin="upper")
        setup_geo_axes(ax5, with_grid=False)
        ax5.set_title("Abs Error | GFS", fontweight="bold")
        cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        cbar5.set_label("mm/3h", fontweight="bold")
        ax6 = fig.add_subplot(gs[2, 1])
        mod_err = np.abs(model_pred - era5_truth)
        im6 = ax6.imshow(mod_err, cmap="YlOrRd", vmin=0, vmax=max(5.0, np.percentile(mod_err, 99)),
                         extent=extent, origin="upper")
        setup_geo_axes(ax6, with_grid=False)
        ax6.set_title("Abs Error | Model", fontweight="bold")
        cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        cbar6.set_label("mm/3h", fontweight="bold")
        ax7 = fig.add_subplot(gs[2, 2])
        if dem_features is not None:
            dem2d = dem_features[0] if (hasattr(dem_features, "ndim") and dem_features.ndim == 3) else dem_features
            im7 = ax7.imshow(dem2d, cmap="terrain", extent=extent, origin="upper")
            setup_geo_axes(ax7, with_grid=False)
            ax7.set_title("Topography (DEM)", fontweight="bold")
            cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
            cbar7.set_label("relative", fontweight="bold")
        else:
            ax7.axis("off")
            ax7.text(0.5, 0.5, "DEM not available", ha="center", va="center")
        ax8 = fig.add_subplot(gs[2, 3])
        bins = np.linspace(0, max(30.0, storm_strength * 1.3), 30)
        ax8.hist(era5_truth.ravel(), bins=bins, alpha=0.45, label="ERA5", color="#2ca02c", density=True)
        ax8.hist(gfs_pred.ravel(),  bins=bins, alpha=0.45, label="GFS",  color="#1f77b4", density=True)
        ax8.hist(model_pred.ravel(),bins=bins, alpha=0.45, label="Model", color="#ff7f0e", density=True)
        ax8.set_title("Intensity PDF", fontweight="bold")
        ax8.set_xlabel("mm/3h", fontweight="bold")
        ax8.set_ylabel("density", fontweight="bold")
        ax8.grid(True, alpha=0.25, linestyle="--")
        ax8.legend(frameon=True)
        # Overall title
        if test_start_date and test_end_date:
            range_str = f"Test period: {test_start_date.strftime('%Y-%m-%d')} to {test_end_date.strftime('%Y-%m-%d')}"
        else:
            range_str = ""
        fig.suptitle(f"Storm Case #{case_number} | {time_str} | Peak={storm_strength:.1f} mm/3h\n{range_str}",
                     fontsize=16, fontweight="bold", y=1.02)
        os.makedirs(save_dir, exist_ok=True)
        out_no_ext = os.path.join(save_dir, f"storm_case_{case_number:02d}_{fname_time}_{storm_strength:.1f}mm")
        save_fig_multi(fig, out_no_ext, dpi=300)
        plt.close(fig)
        print(f"  Saved: {out_no_ext}.png/.pdf")
        # Extra object-oriented contour comparison (≥20, ≥10)
        try:
            ResearchVisualizer.plot_storm_contour_comparison(
                era5=era5_truth, gfs=gfs_pred, model=model_pred,
                case_time=case_time,
                threshold=20.0,
                save_dir=save_dir
            )
            ResearchVisualizer.plot_storm_contour_comparison(
                era5=era5_truth, gfs=gfs_pred, model=model_pred,
                case_time=case_time,
                threshold=10.0,
                save_dir=save_dir
            )
        except Exception as _e:
            pass
        return {
            "case_idx": case_idx,
            "case_time": time_str,
            "storm_strength": storm_strength,
            "max_intensities": intensities,
            "storm_areas": areas,
            "errors": [gfs_mae, mod_mae],
            "improvement": imp,
            "filename": out_no_ext + ".png"
        }

    @staticmethod
    def _create_storm_summary_figure(predictions, gfs_preds, targets, storm_indices, save_dir, time_info=None):
        """
        Summary figure (up to 10 cases), unified colorbar + lon/lat
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        cmap, norm = ResearchVisualizer.get_professional_labels()
        extent = get_geo_extent_from_globals()
        display_indices = storm_indices[:10]
        n = len(display_indices)
        if n == 0:
            return
            
        fig, axes = plt.subplots(n, 3, figsize=(22, 5*n), dpi=300)
        
        if n == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i, idx in enumerate(display_indices):
            # time
            if time_info and i < len(time_info) and hasattr(time_info[i], "strftime"):
                tstr = time_info[i].strftime("%Y-%m-%d %H:%M")
            else:
                tstr = f"idx={idx}"
            obs = targets[idx]
            gfs = gfs_preds[idx]
            mod = predictions[idx]
            
            for j, (field, title) in enumerate([(obs, "ERA5"), (gfs, "GFS"), (mod, "Model")]):
                ax = axes[i, j]
                im = ax.imshow(field, cmap=cmap, norm=norm, extent=extent, origin="upper", interpolation='bicubic')
                setup_geo_axes(ax, with_grid=False)
                
                ax.set_title(f"{title} | {tstr} | max={np.max(field):.1f}", fontweight="bold", fontsize=18)
                
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
                cbar.set_label("mm/3h", fontweight="bold", fontsize=16)
                cbar.ax.tick_params(labelsize=14) 
                
        fig.suptitle("Storm Cases Summary (CMA colormap, geo-referenced)", fontsize=26, fontweight="bold", y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=3.5, h_pad=3.0)
        
        out_no_ext = os.path.join(save_dir, "storm_cases_summary_geo")
        save_fig_multi(fig, out_no_ext, dpi=300)
        plt.close(fig)
        print(f"  Summary saved: {out_no_ext}.png/.pdf")
  
    @staticmethod
    def create_threshold_comparison_plot(test_metrics_summary, save_path='threshold_comparison.png',
                                        thresholds=None, min_event_count=30):
        """
        Multi-threshold comparison (journal-friendly), with BIAS subplot emphasizing GFS wet bias and model correction.
        """
        preds = test_metrics_summary['predictions']
        tars  = test_metrics_summary['targets']
        gfs   = test_metrics_summary['gfs_baseline']
        if thresholds is None:
            thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        pred_f = preds.reshape(-1)
        tar_f  = tars.reshape(-1)
        gfs_f  = gfs.reshape(-1)

        def cond_mean(x, th):
            m = x >= th
            if np.sum(m) == 0:
                return np.nan
            return float(np.mean(x[m]))

        def bias_score(pred, obs, th):
            p = pred >= th
            o = obs  >= th
            o_cnt = int(np.sum(o))
            if o_cnt < min_event_count:
                return np.nan, o_cnt
            tp = int(np.sum(p & o))
            fp = int(np.sum(p & ~o))
            fn = int(np.sum((~p) & o))
            b = safe_div(tp + fp, tp + fn, fill=np.nan)
            return b, o_cnt

        cmean_era5, cmean_gfs, cmean_mod = [], [], []
        bias_gfs, bias_mod = [], []
        evt_cnts = []
        for th in thresholds:
            cmean_era5.append(cond_mean(tar_f, th))
            cmean_gfs.append(cond_mean(gfs_f, th))
            cmean_mod.append(cond_mean(pred_f, th))
            b_g, cnt = bias_score(gfs_f, tar_f, th)
            b_m, _   = bias_score(pred_f, tar_f, th)
            bias_gfs.append(b_g)
            bias_mod.append(b_m)
            evt_cnts.append(cnt)

        gfs_err = [abs(g - e) if (np.isfinite(g) and np.isfinite(e)) else np.nan
                for g, e in zip(cmean_gfs, cmean_era5)]
        mod_err = [abs(m - e) if (np.isfinite(m) and np.isfinite(e)) else np.nan
                for m, e in zip(cmean_mod, cmean_era5)]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

        # (1) conditional mean
        ax = axes[0, 0]
        ax.plot(thresholds, cmean_era5, 'o-', lw=2.8, label="ERA5", color="#2ca02c")
        ax.plot(thresholds, cmean_gfs,  's--', lw=2.0, label="GFS",  color="#1f77b4")
        ax.plot(thresholds, cmean_mod,  '^-', lw=2.8, label="Model", color="#ff7f0e")
        ax.set_xlabel("Threshold (mm/3h)", fontweight="bold")
        ax.set_ylabel("Conditional mean (mm/3h)", fontweight="bold")
        ax.set_title("(a) Conditional Mean vs Threshold", fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend()

        # (2) bias – emphasize GFS wet bias and model correction
        ax = axes[0, 1]
        ax.plot(thresholds, bias_gfs, 's--', lw=2.0, label="GFS Bias (wet bias)", color="#1f77b4")
        ax.plot(thresholds, bias_mod, '^-',  lw=2.8, label="Model Bias (corrected)", color="#ff7f0e")
        ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.6, label="Ideal (BIAS=1)")
        ax.set_xlabel("Threshold (mm/3h)", fontweight="bold")
        ax.set_ylabel(f"BIAS (report if events≥{min_event_count})", fontweight="bold")
        ax.set_title("(b) BIAS vs Threshold: GFS wet bias largely removed by model", fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend()

        # (3) conditional mean error
        ax = axes[1, 0]
        x = np.arange(len(thresholds))
        width = 0.35
        ax.bar(x - width/2, gfs_err, width, label="|GFS - ERA5|", color="#1f77b4", alpha=0.75)
        ax.bar(x + width/2, mod_err, width, label="|Model - ERA5|", color="#ff7f0e", alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in thresholds])
        ax.set_xlabel("Threshold (mm/3h)", fontweight="bold")
        ax.set_ylabel("Abs error (mm/3h)", fontweight="bold")
        ax.set_title("(c) Conditional Mean Absolute Error", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.legend()

        # (4) event counts
        ax = axes[1, 1]
        ax.bar([str(t) for t in thresholds], evt_cnts, color="#888888", alpha=0.85)
        ax.set_xlabel("Threshold (mm/3h)", fontweight="bold")
        ax.set_ylabel("Event pixel count (ERA5>=th)", fontweight="bold")
        ax.set_title("(d) Event Sample Size by Threshold", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")

        fig.suptitle("Multi-threshold Diagnostics (robust bias & sample-size aware)",
                    fontsize=16, fontweight="bold", y=1.02)
        base = save_path.replace(".png", "").replace(".pdf", "")
        save_fig_multi(fig, base, dpi=300)
        plt.close(fig)
        print(f" Threshold comparison figure saved: {base}.png/.pdf")
        # text table
        print("\n Threshold summary (NA if insufficient events):")
        print("="*88)
        print(f"{'Th(mm)':<8} {'ERA5Mean':<10} {'GFSMean':<10} {'ModelMean':<10} {'GFSBias':<10} {'ModelBias':<10} {'EvtCnt':<8}")
        print("-"*88)
        for th, e, g, m, bg, bm, cnt in zip(thresholds, cmean_era5, cmean_gfs, cmean_mod, bias_gfs, bias_mod, evt_cnts):
            def fmt(x):
                return "NA" if (x is None or not np.isfinite(x)) else f"{x:.3f}"
            print(f"{th:<8.1f} {fmt(e):<10} {fmt(g):<10} {fmt(m):<10} {fmt(bg):<10} {fmt(bm):<10} {cnt:<8d}")
        return {
            "thresholds": thresholds,
            "cmean_era5": cmean_era5,
            "cmean_gfs": cmean_gfs,
            "cmean_model": cmean_mod,
            "bias_gfs": bias_gfs,
            "bias_model": bias_mod,
            "event_counts": evt_cnts,
            "gfs_errors": gfs_err,
            "model_errors": mod_err
        }
  
class GlobalDataStandardizer:
    def __init__(self, epsilon=1e-6, cache_path="std_params.npy"):
        self.means = None
        self.stds = None
        self.epsilon = epsilon
        self.fitted = False
        self.cache_path = cache_path
      
    def fit(self, dataset_func):
        if os.path.exists(self.cache_path):
            print(f" Standardization parameter cache hit: {self.cache_path}, skip full data scan.")
            try:
                params = np.load(self.cache_path, allow_pickle=True).item()
                self.means, self.stds = params['means'], params['stds']
                self.fitted = True
                return True 
            except: pass
      
        print(" Cache invalid, initializing temporary dataset for statistical analysis...")
        dataset = dataset_func()
      
        # Use streaming incremental calculation to avoid memory explosion
        n = 0
        mean = None
        M2 = None
      
        limit = min(len(dataset), 3000)
        # Use stride=20 to speed up
        for i in tqdm(range(0, limit, 20), desc="Incremental statistics calculation"):
            try:
                x, _ = dataset[i] 
                if isinstance(x, torch.Tensor): x = x.numpy()
              
                # Flatten for statistics: [C, -1]
                x_flat = np.transpose(x, (1, 0, 2, 3)).reshape(x.shape[1], -1)
              
                if mean is None:
                    mean = np.zeros(x_flat.shape[0], dtype=np.float64)
                    M2 = np.zeros(x_flat.shape[0], dtype=np.float64)
              
                batch_data = x_flat
                batch_count = batch_data.shape[1]
                batch_mean = np.mean(batch_data, axis=1)
                batch_m2 = np.sum((batch_data - batch_mean[:, None])**2, axis=1)
              
                delta = batch_mean - mean
                new_n = n + batch_count
              
                mean += delta * batch_count / new_n
                M2 += batch_m2 + delta**2 * n * batch_count / new_n
                n = new_n

            except Exception as e: continue
      
        # Compute final std
        self.means = mean.reshape(1, -1, 1, 1).astype(np.float32)
        variance = M2 / n
        self.stds = np.sqrt(variance).reshape(1, -1, 1, 1).astype(np.float32)
        self.stds = np.where(self.stds < self.epsilon, 1.0, self.stds)
      
        np.save(self.cache_path, {'means': self.means, 'stds': self.stds})
        self.fitted = True
        print(f" Streaming statistics complete. Number of pixels: {n/1e6:.2f}M")
        del dataset
        gc.collect()
        return False
    def transform(self, data):
        if not self.fitted: return data
        is_torch = isinstance(data, torch.Tensor)
        m = torch.from_numpy(self.means).to(data.device).float() if is_torch else self.means
        s = torch.from_numpy(self.stds).to(data.device).float() if is_torch else self.stds
        if data.ndim == 5: m, s = m.unsqueeze(0), s.unsqueeze(0)
        res = (data - m) / s
        return torch.clamp(res, -5.0, 5.0) if is_torch else np.clip(res, -5.0, 5.0)

def process_enhanced_dem_features_fixed(raw_dem_dir=ERA5_DIR, 
                                       cache_file="processed_dem_features_fixed.npy"):
    """
    Optimized solution 6: Fixed terrain feature processing logic
    - Solve the problem of weak slope signal (<1 degree) at low resolution (30km)
    - Fix ValueError: too many values to unpack
    - Inject enhanced terrain lift signal into channels
    """
    # Check cache
    if os.path.exists(cache_file):
        print(f" Loading enhanced DEM features (fixed) from cache: {cache_file}")
        dem_features = np.load(cache_file)
        return torch.FloatTensor(dem_features)
  
    print("=" * 80)
    print(" Starting fixed DEM feature processing (terrain signal enhancement)")
    print("=" * 80)
  
    # GFS grid parameters - Liaohe River Basin range
    gfs_grid_info = {
        'lon_range': (117.0, 126.0),
        'lat_range': (40.0, 46.0),
        'grid_shape': (25, 37),
        'lon_res': 0.277778,
        'lat_res': 0.416667
    }
  
    lon_min, lon_max = gfs_grid_info['lon_range']
    lat_min, lat_max = gfs_grid_info['lat_range']
    target_shape = gfs_grid_info['grid_shape']
  
    # 1. Check and read tiles
    required_tiles = [(60, 3), (61, 3), (60, 4), (61, 4), (60, 5), (61, 5)]
    existing_files = []
    for lon_band, lat_band in required_tiles:
        filename = f"srtm_{lon_band:02d}_{lat_band:02d}.img"
        file_path = os.path.join(raw_dem_dir, filename)
        if os.path.exists(file_path):
            existing_files.append(file_path)

    # 2. Merge data onto target grid
    height, width = target_shape
    lon_grid = np.linspace(lon_min, lon_max, width)
    lat_grid = np.linspace(lat_max, lat_min, height)
    elevation_gfs = np.zeros(target_shape, dtype=np.float32)
  
    for file_path in existing_files:
        try:
            with rasterio.open(file_path) as src:
                elevation_data = src.read(1)
                transform = src.transform
                left, bottom, right, top = src.bounds
                for i in range(height):
                    for j in range(width):
                        if left <= lon_grid[j] <= right and bottom <= lat_grid[i] <= top:
                            col, row = ~transform * (lon_grid[j], lat_grid[i])
                            if 0 <= int(row) < elevation_data.shape[0] and 0 <= int(col) < elevation_data.shape[1]:
                                elev = elevation_data[int(row), int(col)]
                                if elev > -32768: elevation_gfs[i, j] = elev
        except Exception as e:
            print(f"  Error processing file {file_path}: {e}")

    # 3. Handle missing values and smoothing
    zero_mask = elevation_gfs == 0
    if zero_mask.any():
        elevation_gfs = gaussian_filter(elevation_gfs, sigma=0.5)

    # 4. Internal core calculation function (main modification: returns 3 values)
    def calculate_terrain_features_fixed(elevation_array, lon_res_deg, lat_res_deg, lat_min, lat_max):
        # Convert lat/lon resolution to meters
        center_lat = (lat_min + lat_max) / 2
        dy_m = lat_res_deg * 111000
        dx_m = lon_res_deg * 111000 * np.cos(np.radians(center_lat))
      
        # Compute gradient
        dy, dx = np.gradient(elevation_array, dy_m, dx_m)
        slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
      
        # Nonlinear amplification: because grid is too coarse (30km), raw slope is tiny.
        # Use sqrt(x/max) mapping so that 0.1 deg slope yields ~0.3 intensity, greatly enhancing lift perception.
        slope_max = np.max(slope_deg) + 1e-8
        slope_norm_enhanced = np.sqrt(slope_deg / slope_max) 
      
        # Compute aspect
        aspect_rad = np.arctan2(dy, -dx)
        aspect_deg = (np.degrees(aspect_rad) + 360) % 360
      
        return slope_deg, aspect_deg, slope_norm_enhanced

    # Fix unpacking: corresponds to 3 returned values
    slope_raw, aspect_raw, slope_enhanced = calculate_terrain_features_fixed(
        elevation_gfs, 
        gfs_grid_info['lon_res'], 
        gfs_grid_info['lat_res'],
        lat_min,
        lat_max
    )

    # 5. Feature normalization and stacking
    # Elevation normalization (assume max height 1500m)
    elevation_norm = np.clip(elevation_gfs / 1500.0, 0, 1)
    # Aspect normalization
    aspect_norm = aspect_raw / 360.0
  
    # Final feature combination: channel1-elevation, channel2-enhanced slope, channel3-aspect
    dem_features = np.stack([
        elevation_norm,
        slope_enhanced,
        aspect_norm
    ], axis=0)

    # 6. Save and return
    print(f" DEM feature computation complete:")
    print(f"   Elevation range: {elevation_gfs.min():.1f} - {elevation_gfs.max():.1f} m")
    print(f"   Enhanced slope mean: {np.mean(slope_enhanced):.4f}")
    print(f"   Feature shape: {dem_features.shape}")
  
    np.save(cache_file, dem_features)
    return torch.FloatTensor(dem_features)

def create_datasets_with_dem(gfs_base_path, era5_base_path, dem_tensor=None):
    print("=" * 80)
    print(" Integrating DEM features (force CPU storage to prevent DataLoader errors)")
    print("=" * 80)
    datasets_dict = create_strict_isolation_datasets(gfs_base_path, era5_base_path)
  
    # Ensure DEM tensor is on CPU
    dem_cpu = dem_tensor.cpu() if dem_tensor is not None else None
  
    datasets_with_dem = {}
    for name, dataset in datasets_dict.items():
        if name in ['standardizer', 'scaling_factor']:
            datasets_with_dem[name] = dataset; continue
          
        # Inject CPU DEM
        dataset.dem_features = dem_cpu 
      
        if len(dataset) > 0:
            try:
                input_sample, _ = dataset[0]
                print(f"   Dataset {name}: 9-channel integration successful (CPU Shape: {input_sample.shape})")
            except Exception as e:
                print(f"   Dataset {name} diagnosis failed: {str(e)}")
      
        datasets_with_dem[name] = dataset
    return datasets_with_dem

def create_strict_isolation_datasets(gfs_base_path, era5_unused_path):
    """
    Dataset splitting scheme for 2015-2025 data
    Modified: training set 2015-2023, validation 2024, test 2025
    """
    print("\n" + "=" * 80)
    print(" Research-grade data splitting process (2015-2023 train, 2024 val, 2025 test)")
    print("=" * 80)

    # 1. Set ERA5 exact paths
    era5_surface_dir = ERA5_DIR
    era5_pressure_dir = ERA5_DIR
    era5_paths = [era5_surface_dir, era5_pressure_dir]

    # 2. Discover and filter GFS folders
    all_gfs_folders = discover_data_folders(gfs_base_path)
    gfs_precip_path = os.path.join(gfs_base_path, "jiangshui")

    def filter_gfs_by_date(folders, start_dt, end_dt):
        return [f for f in folders if start_dt <= (extract_date_from_filename(os.path.basename(f)) or datetime(1900,1,1)) <= end_dt]

    # Time splits
    train_start, train_end = datetime(2015, 1, 15), datetime(2021, 12, 31)
    val_start, val_end     = datetime(2022, 1, 1),  datetime(2023, 12, 31)
    test_start, test_end   = datetime(2024, 1, 1),  datetime(2025, 12, 31)

    gfs_train_folders = filter_gfs_by_date(all_gfs_folders, train_start, train_end)
    gfs_val_folders  = filter_gfs_by_date(all_gfs_folders, val_start, val_end)
    gfs_test_folders = filter_gfs_by_date(all_gfs_folders, test_start, test_end)

    # 3. Standardizer handling
    global_standardizer = GlobalDataStandardizer(cache_path="std_params.npy")
    if os.path.exists("std_params.npy"):
        global_standardizer.fitted = True
        params = np.load("std_params.npy", allow_pickle=True).item()
        global_standardizer.means = params['means']
        global_standardizer.stds = params['stds']
    else:
        print(" Computing training set distribution parameters...")
        def get_raw_dataset():
            return SelfSupervisedGFSDataset(data_paths=gfs_train_folders, standardizer=None, precip_data_path=gfs_precip_path, max_samples=2000)
        global_standardizer.fit(get_raw_dataset)

    strict_std = StrictStandardizer(global_standardizer)

    # 4. Key config: greatly increase sampling weights for heavy to storm rain
    heavy_rain_priority_weights = {
        'no_precip': 0.5,
        'Light': 2.0,
        'Moderate': 10.0,
        'Heavy': 50.0,
        'Storm': 150.0
    }

    # 5. Create datasets
    print(" Synchronously building paired datasets...")

    # Training set
    gfs_train = SelfSupervisedGFSDataset(
        data_paths=gfs_train_folders, standardizer=strict_std,
        precip_data_path=gfs_precip_path, intensity_weights=heavy_rain_priority_weights, augment=True
    )
    correction_train = PairedGFSEra5ResidualDatasetStrict(
        gfs_folders=gfs_train_folders,
        era5_paths=era5_paths,
        standardizer=strict_std,
        sequence_length=6,
        prediction_horizon=PREDICTION_HORIZON,
        temp_extract_dir="temp_extract",
        precip_data_path=gfs_precip_path,
        start_date=train_start.strftime('%Y-%m-%d'),
        end_date=train_end.strftime('%Y-%m-%d'),
        enable_cleaning=True,
        require_strict_step=True,
        sequence_step_hours=6
    )

    # Validation set
    correction_val = PairedGFSEra5ResidualDatasetStrict(
        gfs_folders=gfs_val_folders,
        era5_paths=era5_paths,
        standardizer=strict_std,
        sequence_length=6,
        prediction_horizon=PREDICTION_HORIZON,
        temp_extract_dir="temp_extract",
        precip_data_path=gfs_precip_path,
        start_date=val_start.strftime('%Y-%m-%d'),
        end_date=val_end.strftime('%Y-%m-%d'),
        enable_cleaning=True,
        require_strict_step=True,
        sequence_step_hours=6
    )

    # Test set
    correction_test = PairedGFSEra5ResidualDatasetStrict(
        gfs_folders=gfs_test_folders,
        era5_paths=era5_paths,
        standardizer=strict_std,
        sequence_length=6,
        prediction_horizon=PREDICTION_HORIZON,
        temp_extract_dir="temp_extract",
        precip_data_path=gfs_precip_path,
        start_date=test_start.strftime('%Y-%m-%d'),
        end_date=test_end.strftime('%Y-%m-%d'),
        enable_cleaning=True,
        require_strict_step=True,
        sequence_step_hours=6
    )

    # Brief dataset size print
    print(f" Training set: {len(correction_train)} samples | Validation set: {len(correction_val)} samples | Test set: {len(correction_test)} samples")
    return {
        'gfs_train': gfs_train,
        'correction_train': correction_train,
        'correction_val': correction_val,
        'correction_test': correction_test,
        'standardizer': global_standardizer,
        'scaling_factor': 1.0
    }
def create_performance_diagram(results_dict, levels=None, save_path='performance_expert_view.png'):
    """
    Performance Diagram (POD vs Success Ratio) + ETS contours
    Default levels aligned with CMA four levels: Light/Moderate/Heavy/Storm
    """
    levels = levels if levels is not None else PRECIP_LEVELS

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    x = np.linspace(0.01, 0.99, 200)  # Success Ratio

    # ETS contours (reference lines)
    for ets_val in [0.1, 0.2, 0.3, 0.4, 0.5]:
        y = ets_val * (1 - x) / (x - ets_val * x + ets_val)
        mask = (y >= 0) & (y <= 1)
        ax.plot(x[mask], y[mask], color='gray', alpha=0.25, linestyle='--', linewidth=1)
        if np.any(mask):
            ax.text(x[mask][-20], y[mask][-20], f'ETS={ets_val}', fontsize=9, alpha=0.6)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['s', 'o', '^', 'D']

    for mi, (name, metrics) in enumerate(results_dict.items()):
        pods, srs = [], []
        for lvl in levels:
            m = metrics.get(lvl, None)
            if not m:
                continue
            pods.append(float(m.get('POD', 0.0)))
            far = float(m.get('FAR', 0.0))
            srs.append(1.0 - far)

        if pods:
            ax.plot(
                srs, pods,
                marker=markers[mi % len(markers)],
                color=colors[mi % len(colors)],
                linewidth=2.5,
                markersize=9,
                label=name
            )

    ax.set_xlabel('Success Ratio (1 - FAR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability of Detection (POD)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Diagram (POD vs Success Ratio)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.15)
    ax.legend(loc='lower left', frameon=True)

    plt.tight_layout()

    base_path = save_path.replace('.png', '').replace('.pdf', '')
    save_fig_multi(fig, base_path, dpi=300)
    plt.close(fig)
def create_scientific_spatial_comparison(predictions, targets, gfs_baseline, dem_features, model_name):
    """Refined research spatial comparison plot: add unit annotations and smoothing"""
    print(f" Generating ultra-smooth research spatial comparison plot (Unit: mm/3h)...")
    plt.rcParams['font.sans-serif'] = ['Arial']

    max_precip_scores = np.max(targets, axis=(1, 2)) if targets.ndim == 3 else np.max(targets, axis=(1, 2, 3))
    extreme_indices = np.argsort(max_precip_scores)[-3:][::-1]

    fig, axes = plt.subplots(len(extreme_indices), 4, figsize=(24, 6 * len(extreme_indices)), dpi=300)
    zoom_f = 8

    for i, idx in enumerate(extreme_indices):
        obs = zoom(targets[idx, -1] if targets.ndim == 4 else targets[idx], zoom_f, order=3)
        gfs = zoom(gfs_baseline[idx, -1] if gfs_baseline.ndim == 4 else gfs_baseline[idx], zoom_f, order=3)
        prd = zoom(predictions[idx, -1] if predictions.ndim == 4 else predictions[idx], zoom_f, order=3)
        delta = prd - gfs

        vmax = 35.0
        cmap_rain = 'YlGnBu'

        titles = ["ERA5 Observed", "Raw GFS Forecast", "Corrected Model", "Correction Delta (M-G)"]
        data_list = [obs, gfs, prd, delta]

        for j in range(4):
            curr_cmap = cmap_rain if j < 3 else 'RdBu_r'
            curr_vmax = vmax if j < 3 else max(abs(delta.min()), abs(delta.max()), 5.0)
            curr_vmin = 0 if j < 3 else -curr_vmax

            im = axes[i, j].imshow(data_list[j], cmap=curr_cmap, vmin=curr_vmin, vmax=curr_vmax, interpolation='bilinear')
            axes[i, j].set_title(f"{titles[j]}\n(Sample {idx})", fontsize=14, fontweight='bold')
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

            cbar = plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)
            cbar.set_label('mm/3h', fontsize=10, fontweight='bold')

    plt.suptitle(f'High-Resolution Spatial Correction Analysis: {model_name}', fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout()

    save_fig_multi(fig, 'scientific_spatial_smooth', dpi=300)
    plt.close(fig)
def staged_training_strategy_with_monitoring(model, train_loader, val_loader, device,
                                             scaling_factor=1.0, epochs=12, second_stage=False):
    """
    Two-stage training strategy, EMA validation, storm monitoring, gate supervision
    """
    criterion = MultiTaskLoss()

    if second_stage:
        for name, param in model.named_parameters():
            if not any(x in name for x in ['res_heads', 'rain_heads', 'storm_head']):
                param.requires_grad = False
        lr = 3e-5
        patience = 6
        print(" Second stage fine-tuning: freeze backbone, train only heads, lr=3e-5")
    else:
        lr = 8e-5
        patience = 5
        print(f" First stage training: lr={lr:.2e}, patience={patience}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=8e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = (torch.cuda.is_available() and (not second_stage))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ema = ModelEMA(model, decay=0.999)
    best_score = -float('inf')
    no_improve_epochs = 0
    min_epochs = 4
    best_pod20 = 0.0
    pod20_no_improve = 0

    history = {
        'stage_c_losses': [], 'stage_c_val_losses': [],
        'storm_ets_15': [], 'storm_pod_15': [], 'storm_far_15': [],
        'storm_ets_20': [], 'storm_pod_20': [], 'storm_far_20': []
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Storm-Focus Training Epoch {epoch+1}")

        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            gfs_base = inputs[:, -1, 5:6, :, :]
            era5_abs = gfs_base + targets / scaling_factor

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                residual, rain_prob, storm_logits, _ = model(
                    inputs, return_residual=True, return_storm_logits=True
                )
                loss = criterion(pred_res=residual, target_abs=era5_abs,
                                 gfs_base=gfs_base, rain_prob=rain_prob,
                                 storm_logits=storm_logits)

            if not torch.isfinite(loss):
                continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            ema.update(model)
            epoch_loss += float(loss.item())
            pbar.set_postfix({'loss': f'{float(loss.item()):.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})

        scheduler.step()
        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        history['stage_c_losses'].append(avg_train_loss)

        # EMA validation
        ema_model = ema.ema
        ema_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v_in, v_tar in val_loader:
                v_in = v_in.to(device, non_blocking=True)
                v_tar = v_tar.to(device, non_blocking=True)
                v_gfs = v_in[:, -1, 5:6, :, :]
                v_abs = v_gfs + v_tar / scaling_factor

                with torch.cuda.amp.autocast(enabled=use_amp):
                    v_res, v_prob, v_storm, _ = ema_model(v_in, return_residual=True, return_storm_logits=True)
                    v_loss = criterion(pred_res=v_res, target_abs=v_abs,
                                       gfs_base=v_gfs, rain_prob=v_prob,
                                       storm_logits=v_storm)
                if torch.isfinite(v_loss):
                    val_loss += float(v_loss.item())
        avg_val_loss = val_loss / max(len(val_loader), 1)
        history['stage_c_val_losses'].append(avg_val_loss)

        # Monitor storm metrics
        storm_metrics = monitor_extreme_event_performance(
            ema_model, val_loader, device,
            thresholds=[15.0, 20.0],
            scaling_factor=scaling_factor,
            storm_gate_p=float(GATE_CFG.get("storm_gate_p", 0.25))
        )

        ets15 = storm_metrics[15.0].get('ETS', 0.0)
        pod15 = storm_metrics[15.0].get('POD', 0.0)
        far15 = storm_metrics[15.0].get('FAR', 1.0)

        ets20 = storm_metrics[20.0].get('ETS', 0.0)
        pod20 = storm_metrics[20.0].get('POD', 0.0)
        far20 = storm_metrics[20.0].get('FAR', 1.0)

        history['storm_ets_15'].append(ets15)
        history['storm_pod_15'].append(pod15)
        history['storm_far_15'].append(far15)
        history['storm_ets_20'].append(ets20)
        history['storm_pod_20'].append(pod20)
        history['storm_far_20'].append(far20)

        # Composite score
        composite_score = (3.0 * ets20 + 1.8 * pod20 - 0.45 * far20 +
                           1.4 * ets15 + 0.9 * pod15 - 0.20 * far15 -
                           0.003 * avg_val_loss)
        pod20_penalty = max(0, 0.20 - pod20) * 2.0
        composite_score -= pod20_penalty

        if pod20 > best_pod20 + 1e-4:
            best_pod20 = pod20
            pod20_no_improve = 0
        else:
            pod20_no_improve += 1

        print(f"\n Validation(EMA) Epoch {epoch+1}: ETS15={ets15:.4f}, POD15={pod15:.4f}, ETS20={ets20:.4f}, POD20={pod20:.4f}, FAR20={far20:.4f}")
        print(f"   CompositeScore={composite_score:.4f} (storm-priority)")

        if composite_score > best_score:
            best_score = composite_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'storm_metrics': storm_metrics,
                'best_score': best_score,
                'stage': 'second' if second_stage else 'first'
            }, 'best_correction_model.pth')
            print(f" Saved best model (EMA): best_score={best_score:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if (epoch + 1) >= min_epochs and (no_improve_epochs >= patience or pod20_no_improve >= patience):
            print(f" Early Stopping: composite score or POD20 no improvement for {patience} epochs")
            break

    return model, history
def monitor_extreme_event_performance(model, data_loader, device,
                                      thresholds=[5.0, 10.0, 20.0],
                                      scaling_factor=1.0,
                                      storm_gate_p=None):
    if storm_gate_p is None:
        storm_gate_p = float(GATE_CFG.get("storm_gate_p", 0.30))

    model.eval()
    results = {th: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for th in thresholds}

    with torch.no_grad():
        for inputs, targets_scaled in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets_scaled = targets_scaled.to(device, non_blocking=True)

            residual, rain_prob, storm_logits, _ = model(
                inputs, return_residual=True, return_storm_logits=True
            )
            gfs_base = inputs[:, -1, 5:6, :, :]

            predictions, _ = compute_gated_precip_prediction(
                gfs_base=gfs_base,
                residual=residual,
                rain_prob=rain_prob,
                adaptive=True,
                hard_gate=True,
                storm_logits=storm_logits,
                storm_gate_p=float(storm_gate_p)
            )

            true_precip = gfs_base.expand(-1, targets_scaled.shape[1], -1, -1) + targets_scaled / scaling_factor

            pred_last = predictions[:, -1].detach().cpu().numpy()
            true_last = true_precip[:, -1].detach().cpu().numpy()

            for th in thresholds:
                pred_binary = (pred_last >= th)
                true_binary = (true_last >= th)

                results[th]['TP'] += int(np.sum(pred_binary & true_binary))
                results[th]['FP'] += int(np.sum(pred_binary & ~true_binary))
                results[th]['FN'] += int(np.sum(~pred_binary & true_binary))
                results[th]['TN'] += int(np.sum(~pred_binary & ~true_binary))

    metrics = {}
    for th in thresholds:
        tp = results[th]['TP']
        fp = results[th]['FP']
        fn = results[th]['FN']
        tn = results[th]['TN']
        total = tp + fp + fn + tn

        pod = tp / (tp + fn + 1e-8)
        far = fp / (tp + fp + 1e-8)
        random_hits = (tp + fp) * (tp + fn) / (total + 1e-8)
        denom = tp + fp + fn - random_hits
        ets = (tp - random_hits) / (denom + 1e-8) if denom > 0 else 0.0

        metrics[th] = {'POD': float(pod), 'FAR': float(far), 'ETS': float(ets),
                       'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)}
    return metrics
def analyze_improvement_by_level(model, test_loader, device, scaling_factor=1.0):
    model.eval()
    level_names = ['Light (0.1-3)', 'Moderate (3-10)', 'Heavy (10-20)', 'Storm (>20)']
    gfs_sq_errors = {name: [] for name in level_names}
    mod_sq_errors = {name: [] for name in level_names}

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            pred_abs, true_abs, gfs_expand, _, _, _, _ = get_model_eval_tensors(
                model=model, inputs=inputs, targets_scaled=targets, scaling_factor=scaling_factor, max_precip=200.0
            )
            pred_last = pred_abs[:, -1]
            obs_last = true_abs[:, -1]
            gfs_last = gfs_expand[:, -1]

            for i, name in enumerate(level_names):
                if i < 3:
                    mask = (obs_last >= PRECIP_THRESHOLDS[i]) & (obs_last < PRECIP_THRESHOLDS[i+1])
                else:
                    mask = (obs_last >= PRECIP_THRESHOLDS[i])
                if mask.any():
                    gfs_err = (gfs_last[mask] - obs_last[mask]) ** 2
                    mod_err = (pred_last[mask] - obs_last[mask]) ** 2
                    gfs_sq_errors[name].extend(gfs_err.cpu().numpy().tolist())
                    mod_sq_errors[name].extend(mod_err.cpu().numpy().tolist())

    summary_results = {}
    for name in level_names:
        if len(gfs_sq_errors[name]) > 0:
            rmse_gfs = np.sqrt(np.mean(gfs_sq_errors[name]))
            rmse_mod = np.sqrt(np.mean(mod_sq_errors[name]))
            improvement = (rmse_gfs - rmse_mod) / (rmse_gfs + 1e-8) * 100
            summary_results[name] = improvement
    return summary_results
def analyze_temporal_performance(model, test_loader, sample_times, device, scaling_factor=1.0):
    """
    Flood/non-flood season evaluation, using gated prediction
    """
    model.eval()

    monthly_stats = {m: {'gfs_se': [], 'mod_se': []} for m in range(1, 13)}
    temporal_stats = {
        'Flood_Season': {'gfs_se': [], 'mod_se': [], 'count': 0},
        'Non_Flood_Season': {'gfs_se': [], 'mod_se': [], 'count': 0}
    }

    sample_idx = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            pred_abs, true_abs, gfs_expand, _, _, _, _ = get_model_eval_tensors(
                model=model,
                inputs=inputs,
                targets_scaled=targets,
                scaling_factor=scaling_factor,
                max_precip=200.0
            )

            pred_last = pred_abs[:, -1]
            obs_last = true_abs[:, -1]
            gfs_last = gfs_expand[:, -1]

            for b in range(inputs.shape[0]):
                if sample_idx >= len(sample_times):
                    break

                month = sample_times[sample_idx].month

                se_gfs = torch.mean((gfs_last[b] - obs_last[b]) ** 2).item()
                se_mod = torch.mean((pred_last[b] - obs_last[b]) ** 2).item()

                monthly_stats[month]['gfs_se'].append(se_gfs)
                monthly_stats[month]['mod_se'].append(se_mod)

                season_key = 'Flood_Season' if month in [6, 7, 8] else 'Non_Flood_Season'
                temporal_stats[season_key]['gfs_se'].append(se_gfs)
                temporal_stats[season_key]['mod_se'].append(se_mod)
                temporal_stats[season_key]['count'] += 1

                sample_idx += 1

    print("\n" + "═" * 60)
    print(f" Liaohe Basin [Flood vs Non-Flood Season] Correction Efficiency Comparison")
    print("-" * 60)
    for season, data in temporal_stats.items():
        if data['count'] > 0:
            rmse_gfs = np.sqrt(np.mean(data['gfs_se']))
            rmse_mod = np.sqrt(np.mean(data['mod_se']))
            imp = (rmse_gfs - rmse_mod) / (rmse_gfs + 1e-8) * 100
            print(f"{season:<18} | samples: {data['count']:>5} | improvement: {imp:>6.2f}%")
    print("═" * 60)

    return monthly_stats
def analyze_correction_sign(predictions, targets, gfs_baseline, eps=1e-3):
    """
    Diagnose whether model correction is positive or negative biased
    predictions/targets/gfs_baseline: [N,H,W]
    """
    delta = predictions - gfs_baseline          # actual model correction
    true_res = targets - gfs_baseline           # true required correction

    def _stat(name, mask):
        d = delta[mask]
        r = true_res[mask]
        if d.size == 0:
            print(f"[{name}] empty samples")
            return
        pos = np.mean(d > eps) * 100
        neg = np.mean(d < -eps) * 100
        neu = 100 - pos - neg
        mean_d = float(np.mean(d))
        # direction agreement (exclude near-zero residual)
        valid = np.abs(r) > eps
        if np.any(valid):
            align = np.mean(np.sign(d[valid]) == np.sign(r[valid])) * 100
        else:
            align = np.nan
        print(f"[{name}] positive correction={pos:.2f}% | negative correction={neg:.2f}% | near zero={neu:.2f}% | mean correction={mean_d:.4f} | direction agreement={align:.2f}%")

    all_mask = np.ones_like(delta, dtype=bool)
    rain_mask = targets >= 0.1
    heavy_mask = targets >= 10.0
    storm_mask = targets >= 20.0

    _stat("All", all_mask)
    _stat("Rain region(>=0.1)", rain_mask)
    _stat("Heavy rain region(>=10)", heavy_mask)
    _stat("Storm region(>=20)", storm_mask)
def run_residual_experiment_enhanced():
    """
    Run residual learning experiment (strict scientific version)
    """
    print("=" * 80)
    print(" GFS(f003,+3h) → ERA5(+3h) residual correction experiment (strict time pairing + unified gated evaluation)")
    print("=" * 80)

    import time as time_module
    start_time = time_module.time()
  
    # ==================== Step 0: Initialize key variables ====================
    test_metrics_summary = {
        'mse': 0.0, 'mae': 0.0, 'rmse': 0.0, 'precip_ratio': 0.0,
        'improvement_vs_gfs': 0.0, 'predictions': None, 'targets': None,
        'gfs_baseline': None, 'dem_features': None,
        'sample_times': None, 'scientific_metrics': None
    }
    summary_printed = False

    # ==================== Step 1: Set device ====================
    device = get_device()

    # ==================== Step 2: Completely disable DEM features ====================
    print("\n Diagnosis: DEM at 0.25° resolution acts as negative noise, globally disabled...")
    dem_tensor = None
    dem_channels = 0
    test_metrics_summary['dem_features'] = None

    # ==================== Step 3: Create datasets (strict pairing) ====================
    gfs_base_path = GFS_DIR
    era5_base_path = ERA5_DIR

    print("\n Creating datasets (pure meteorological variables)...")
    datasets_dict = create_datasets_with_dem(gfs_base_path, era5_base_path, dem_tensor=None)

    # Extract dataset variables
    gfs_train = datasets_dict['gfs_train']
    correction_train = datasets_dict['correction_train']  
    correction_val = datasets_dict['correction_val']
    correction_test = datasets_dict['correction_test']
    scaling_factor = float(datasets_dict.get('scaling_factor', 1.0))

    # Print dimensions
    print(f" DATASET DIMENSIONS")
    print(f"  Train samples: {len(correction_train)}")
    print(f"  Val samples:   {len(correction_val)}")
    print(f"  Test samples:  {len(correction_test)}")
    print(f"  Spatial grid:  {GLOBAL_LATS.shape[0]}x{GLOBAL_LONS.shape[0]}")

    # [Keep original test_metrics_summary['sample_times'] logic]
    if hasattr(correction_test, 'sample_times') and correction_test.sample_times:
        test_metrics_summary['sample_times'] = correction_test.sample_times[:len(correction_test)]
        print(f" Using correction_test.sample_times: {len(test_metrics_summary['sample_times'])} timestamps")
    else:
        print(" correction_test has no sample_times, falling back to synthetic 3-hour sequence")
        test_start = datetime(2024, 1, 1)
        test_metrics_summary['sample_times'] = [test_start + timedelta(hours=i * 3) for i in range(len(correction_test))]

    # ==================== Step 3.5: Configure gate and oversampling ====================
    global GATE_CFG
    GATE_CFG["threshold_base"] = 0.22
    GATE_CFG["threshold_min"] = 0.10
    GATE_CFG["threshold_max"] = 0.40
    GATE_CFG["gate_power"] = 0.90
    GATE_CFG["storm_gate_p"] = 0.30

    dynamic_oversample_ratios = [0.4, 0.8, 1.2, 5.0, 20.0, 45.0, 70.0]
    storm_patch_train = StormPatchWrapper(correction_train, patch=20, storm_th=10.0, storm_prob=1.0)

    print("\n" + "="*50)
    print(" Training configuration")
    print("="*50)
    print(f"Gate: threshold_base={GATE_CFG['threshold_base']:.2f}, gate_power={GATE_CFG['gate_power']:.2f}, storm_gate_p={GATE_CFG['storm_gate_p']:.2f}")
    print(f"Oversampling ratios: no-rain 0.4 | trace 0.8 | light 1.2 | moderate 5.0 | heavy 20.0 | storm 45.0 | torrential 70.0")
    print(f"StormPatchWrapper: patch=20, storm_th=10.0, storm_prob=1.0")
    print("="*50)

    # ==================== Stage 1: Heavy rain event resampling strategy ====================
    print("\n" + "=" * 60)
    print(" Stage 1: Implement intelligent heavy rain event resampling strategy (based on ERA5 absolute precipitation)")
    print("=" * 60)

    intensity_stats = ExtremeEventDataLoader.analyze_dataset_intensity(
        correction_train,
        thresholds=[0.1, 1.0, 5.0, 10.0, 20.0]
    )

    sampled_total = max(1, sum(intensity_stats.values()))
    heavy_rain_count = intensity_stats.get('10.0-20.0mm', 0) + intensity_stats.get('>=20.0mm', 0)
    heavy_rain_ratio = heavy_rain_count / sampled_total

    if heavy_rain_ratio < 0.03:
        oversample_ratio = 10
        print(f" Detected very few heavy rain samples ({heavy_rain_ratio*100:.1f}%), using mild enhanced oversampling: {oversample_ratio}x")
    elif heavy_rain_ratio < 0.08:
        oversample_ratio = 7
        print(f" Detected few heavy rain samples ({heavy_rain_ratio*100:.1f}%), using moderate oversampling: {oversample_ratio}x")
    else:
        oversample_ratio = 5
        print(f" Heavy rain sample proportion acceptable ({heavy_rain_ratio*100:.1f}%), using conservative oversampling: {oversample_ratio}x")

    dynamic_oversample_ratios = [
        0.4,   # 1. No rain (0-0.1mm): further compress background
        0.8,   # 2. Trace (0.1-0.5mm)
        1.2,   # 3. Light (0.5-3.0mm)
        5.0,   # 4. Moderate (3.0-10.0mm)
        20.0,  # 5. Heavy (10.0-20.0mm)
        45.0,  # 6. Storm (20.0-50.0mm)
        70.0   # 7. Torrential (>50.0mm)
    ]

    storm_patch_train = StormPatchWrapper(
        correction_train,
        patch=20,
        storm_th=10.0,
        storm_prob=1.0
    )

    train_loader = ExtremeEventDataLoader.create_adaptive_oversampled_loader(
        dataset=storm_patch_train,
        batch_size=64,
        num_workers=0,
        oversample_ratios=dynamic_oversample_ratios
    )
    # ==================== Step 4: Create validation/test loaders ====================
    print(f"\n Creating validation and test data loaders...")

    def create_fast_dataloader(dataset, shuffle=True):
        config = ultra_fast_training_config()
        loader_kwargs = {
            'batch_size': config['batch_size'],
            'shuffle': shuffle,
            'num_workers': 0,
            'pin_memory': True,
            'drop_last': True,
            'collate_fn': custom_collate_fn,
        }
        return DataLoader(dataset, **loader_kwargs)

    val_loader = create_fast_dataloader(correction_val, shuffle=False)
    test_loader = create_fast_dataloader(correction_test, shuffle=False)

    print(f" Data loaders created:")
    print(f"  Train loader (oversampled Subset): {len(train_loader)} batches")
    print(f"  Validation loader: {len(val_loader)} batches")
    print(f"  Test loader: {len(test_loader)} batches")

    # ==================== Step 5: Initialize model ====================
    input_channels = 6  # pure 6 raw channels, no extra variables

    print(f"\n Initializing pure residual learning model...")
    print(f"  Input channels: {input_channels}")
    print(f"  prediction_horizon: {PREDICTION_HORIZON} (f003 only)")

    print("\n Releasing memory to prepare for model initialization...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time_module.sleep(1)
    RUN_BASELINE_UNET = True  # Set to True for U-Net baseline, False for APCNet
    
    if RUN_BASELINE_UNET:
        print(f"\n Initializing baseline model: Standard U-Net...")
        model = StandardUNet(input_channels=input_channels, hidden_channels=32).to(device)
    else:
        print(f"\n Initializing core model: APCNet...")
        model = AdvancedPrecipCorrectionNet(
            input_channels=input_channels,
            hidden_channels=24,
            sequence_length=6,
            prediction_horizon=PREDICTION_HORIZON,
            spatial_dims=(25, 37),
            dropout_rate=0.1
        ).to(device)
    print(" Model initialization complete.")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "═"*50)
    print(" HYPERPARAMETER & MODEL SUMMARY")
    print("-" * 50)
    print(f"  Model Size: {total_params:,} parameters")
    print(f"  Trainable:  {trainable_params:,} parameters")
    print(f"  Batch Size: {64}")
    print(f"  Sequence:   {6} steps -> Horizon: {1} step")
    print(f"  Input:      {input_channels} channels (Pure Meteo)")
    print(f"  Oversample: {oversample_ratio}x (Heavy Rain Priority)")
    print("═"*50 + "\n")
    # ==================== Step 6: Minimal pretraining ====================
    print("\n" + "=" * 40)
    print(" Stage 2: Minimal pretraining (1 epoch, for initialization)")
    print("=" * 40)

    gfs_pretrain_losses, _ = improved_gfs_pretrain_phase(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=1,          # keep 1 epoch
        learning_rate=3e-5  # reduced learning rate
    )
    if gfs_pretrain_losses:
        print(f" Pretraining complete, final loss: {gfs_pretrain_losses[-1]:.6f}")

    # ==================== Step 7: Staged training ====================
    print("\n" + "=" * 40)
    print(" Stage 3: Residual correction training (EMA validation + storm monitoring)")
    print("=" * 40)
    # First stage training
    model, staged_training_results = staged_training_strategy_with_monitoring(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        scaling_factor=scaling_factor,
        epochs=10,
        second_stage=False
    )
    # Second stage fine-tuning: only if first stage actually learned storm capability (more strict)
    trigger_stage2 = False
    if 'storm_pod_20' in staged_training_results and len(staged_training_results['storm_pod_20']) > 0:
        best_pod20 = float(np.max(staged_training_results['storm_pod_20']))
        best_ets20 = float(np.max(staged_training_results.get('storm_ets_20', [0.0])))
        trigger_stage2 = (best_pod20 >= 0.10 or best_ets20 >= 0.02)

    if trigger_stage2:
        print("\n" + "="*40)
        print(" Second stage fine-tuning (freeze encoder, train only heads)")
        print("="*40)

        best_model_path = 'best_correction_model.pth'
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(" Loaded first stage best model for fine-tuning")

        model, staged_training_results_2 = staged_training_strategy_with_monitoring(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            scaling_factor=scaling_factor,
            epochs=8,
            second_stage=True
        )

        staged_training_results['stage_c_losses'].extend(staged_training_results_2['stage_c_losses'])
        staged_training_results['stage_c_val_losses'].extend(staged_training_results_2['stage_c_val_losses'])
        staged_training_results['storm_ets_20'].extend(staged_training_results_2['storm_ets_20'])
        staged_training_results['storm_pod_20'].extend(staged_training_results_2['storm_pod_20'])
    else:
        print(" Skipping second stage fine-tuning (first stage storm skill insufficient)")

    # Unified: subsequent calibration and evaluation based on best checkpoint (not last epoch)
    best_model_path = 'best_correction_model.pth'
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f" Loaded best model for gate calibration and final evaluation (epoch={ckpt.get('epoch', 'NA')})")

        # ==================== Gate calibration (storm priority, dual threshold) ====================
    old_th = float(GATE_CFG.get("threshold_base", 0.20))
    old_pow = float(GATE_CFG.get("gate_power", 0.80))

    calib = calibrate_gate_threshold_on_val(
        model=model,
        val_loader=val_loader,
        device=device,
        scaling_factor=scaling_factor,
        search=np.linspace(0.12, 0.28, 9),
        power_search=(0.70, 0.80, 0.90, 1.00),
        storm_gate_p_search=[0.40, 0.45, 0.50, 0.55, 0.60],
        min_pod20=0.35,
        min_pod15=0.30,
        max_far20=0.90
    )

    if calib.get("is_valid", False):
        GATE_CFG["threshold_base"] = float(calib["th"])
        GATE_CFG["gate_power"] = float(calib["pow"])
        print(f" Apply calibrated gate: th_base={GATE_CFG['threshold_base']:.3f}, "
              f"gate_power={GATE_CFG['gate_power']:.2f}, POD20={calib.get('pod20',0):.4f}, "
              f"ETS20={calib.get('ets20',0):.4f}, FAR20={calib.get('far20',0):.4f}")
    else:
        GATE_CFG["threshold_base"] = old_th
        GATE_CFG["gate_power"] = old_pow
        print(f" Calibration failed to meet constraints, keep old gate: th_base={old_th:.3f}, gate_power={old_pow:.2f} "
              f"(best_any POD20={calib.get('pod20',0):.4f}, ETS20={calib.get('ets20',0):.4f})")
    print("\n Starting level-wise RMSE improvement evaluation...")
    level_improvements = analyze_improvement_by_level(model, test_loader, device, scaling_factor)
    test_metrics_summary['level_improvements'] = level_improvements

    print("\n Starting flood season specific performance evaluation...")
    monthly_summary = analyze_temporal_performance(
        model=model,
        test_loader=test_loader,
        sample_times=test_metrics_summary['sample_times'],
        device=device,
        scaling_factor=scaling_factor
    )
    test_metrics_summary['monthly_rmse_stats'] = monthly_summary
    ResearchVisualizer.plot_seasonal_comparison_bar(monthly_summary)

    # ==================== Stage 5: Correction effectiveness special evaluation (safe implementation inside this function) ====================
    print("\n" + "=" * 60)
    print(" Special evaluation: Correction effectiveness analysis (Safe version)")
    print("=" * 60)

    def evaluate_correction_effectiveness_safe(model, test_loader, device, scaling_factor=1.0):
        model.eval()
        all_gfs_err_global, all_mod_err_global = [], []
        all_gfs_err_storm, all_mod_err_storm = [], []
        total_storm_pixels = 0

        with torch.no_grad():
            for i, (inputs, targets_scaled) in enumerate(test_loader):
                if i > 100:
                    break
                inputs = inputs.to(device)
                targets_scaled = targets_scaled.to(device)

                pred_abs, true_abs, gfs_expand, _, _, _, _ = get_model_eval_tensors(
                    model=model,
                    inputs=inputs,
                    targets_scaled=targets_scaled,
                    scaling_factor=scaling_factor,
                    max_precip=200.0
                )

                gfs_abs_err = torch.abs(gfs_expand - true_abs)
                mod_abs_err = torch.abs(pred_abs - true_abs)

                all_gfs_err_global.append(gfs_abs_err.mean().item())
                all_mod_err_global.append(mod_abs_err.mean().item())

                storm_mask = (true_abs >= 10.0)
                if storm_mask.any():
                    all_gfs_err_storm.append(gfs_abs_err[storm_mask].detach().cpu().numpy())
                    all_mod_err_storm.append(mod_abs_err[storm_mask].detach().cpu().numpy())
                    total_storm_pixels += int(storm_mask.sum().item())

        out = {
            'total_storm_points': total_storm_pixels,
            'avg_gfs_error': float(np.mean(all_gfs_err_global)) if all_gfs_err_global else 0.0,
            'avg_model_error': float(np.mean(all_mod_err_global)) if all_mod_err_global else 0.0,
        }

        if total_storm_pixels > 0 and all_gfs_err_storm:
            storm_gfs_mae = float(np.concatenate(all_gfs_err_storm).mean())
            storm_mod_mae = float(np.concatenate(all_mod_err_storm).mean())
            storm_imp = (storm_gfs_mae - storm_mod_mae) / (storm_gfs_mae + 1e-8) * 100
            out.update({
                'storm_gfs_mae': storm_gfs_mae,
                'storm_model_mae': storm_mod_mae,
                'storm_improvement': float(storm_imp)
            })
        else:
            out.update({'storm_improvement': 0.0})

        return out

    correction_analysis = evaluate_correction_effectiveness_safe(
        model=model,
        test_loader=test_loader,
        device=device,
        scaling_factor=scaling_factor
    )
    test_metrics_summary['correction_analysis'] = correction_analysis

    print(f" Correction effectiveness summary:")
    print(f"  Global MAE | GFS: {correction_analysis.get('avg_gfs_error', 0):.4f} -> Model: {correction_analysis.get('avg_model_error', 0):.4f}")
    print(f"  Storm area error reduction (>10mm): {correction_analysis.get('storm_improvement', 0):.2f}% (storm_pixels={correction_analysis.get('total_storm_points', 0)})")

    # ==================== Stage 6: Quick storm evaluation ====================
    print("\n" + "=" * 60)
    print(" Quick storm-specific evaluation")
    print("=" * 60)

    storm_stats = quick_storm_evaluation(model, test_loader, device, scaling_factor)
    test_metrics_summary['quick_storm_stats'] = storm_stats

    # ==================== Stage 7: Full traditional metrics evaluation ====================
    print("\n" + "=" * 60)
    print(" Stage 7: Traditional metrics evaluation (unified gate)")
    print("=" * 60)

    test_comprehensive = enhanced_comprehensive_evaluation(
        model=model,
        test_loader=test_loader,
        device=device,
        scaling_factor=scaling_factor
    )

    if test_comprehensive:
        test_metrics_summary['mse'] = float(test_comprehensive.get('mse', 0))
        test_metrics_summary['mae'] = float(test_comprehensive.get('mae', 0))
        test_metrics_summary['rmse'] = float(test_comprehensive.get('rmse', np.sqrt(test_metrics_summary['mse'])))
        test_metrics_summary['precip_ratio'] = float(test_comprehensive.get('precip_ratio', 0))

        test_metrics_summary['predictions'] = test_comprehensive.get('predictions')
        test_metrics_summary['targets'] = test_comprehensive.get('targets')
        test_metrics_summary['gfs_baseline'] = test_comprehensive.get('gfs_baseline')
        test_metrics_summary['probs'] = test_comprehensive.get('probs', None)
        test_metrics_summary['targets_bin'] = test_comprehensive.get('targets_bin', None)
        if 'metrics_by_level' in test_comprehensive:
            test_metrics_summary['metrics_by_level'] = test_comprehensive['metrics_by_level']

    # ==================== Stage 8: GFS baseline comparison ====================
    print("\n" + "=" * 60)
    print(" Stage 8: Comparison with GFS baseline")
    print("=" * 60)

    gfs_comparison = compare_with_gfs_baseline(
        model=model,
        test_loader=test_loader,
        device=device,
        scaling_factor=scaling_factor
    )

    if gfs_comparison:
        test_metrics_summary['improvement_vs_gfs'] = float(gfs_comparison.get('improvement_percentage', 0))
        if 'intensity_improvements' in gfs_comparison:
            test_metrics_summary['intensity_improvements'] = gfs_comparison['intensity_improvements']

    # ==================== Stage 9: Collect test set predictions (strict fix for targets mis-passing bug) ====================
    print("\n" + "=" * 60)
    print(" Stage 9: Collect predictions for subsequent analysis (strict variable passing)")
    print("=" * 60)

    model.eval()
    all_preds, all_targets, all_gfs = [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets_batch) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets_batch = targets_batch.to(device)

            pred_abs, true_abs, gfs_expand, _, _, _, _ = get_model_eval_tensors(
                model=model,
                inputs=inputs,
                targets_scaled=targets_batch,
                scaling_factor=scaling_factor,
                max_precip=200.0
            )

            all_preds.append(pred_abs[:, -1].cpu().numpy())
            all_targets.append(true_abs[:, -1].cpu().numpy())
            all_gfs.append(gfs_expand[:, -1].cpu().numpy())

            if batch_idx % 50 == 0:
                print(f"   Processed {batch_idx+1} batches...")

    if all_preds:
        predictions = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        gfs_predictions = np.concatenate(all_gfs, axis=0)

        test_metrics_summary['predictions'] = predictions
        test_metrics_summary['targets'] = targets
        test_metrics_summary['gfs_baseline'] = gfs_predictions

        if hasattr(correction_test, 'sample_times') and correction_test.sample_times:
            test_metrics_summary['sample_times'] = correction_test.sample_times[:len(predictions)]

        mse_model = float(np.mean((predictions - targets) ** 2))
        mse_gfs = float(np.mean((gfs_predictions - targets) ** 2))
        overall_improvement = (mse_gfs - mse_model) / (mse_gfs + 1e-8) * 100

        print(f"\n Test set prediction statistics:")
        print(f"  Number of samples: {len(predictions)}")
        print(f"  Model MSE: {mse_model:.6f}")
        print(f"  GFS   MSE: {mse_gfs:.6f}")
        print(f"  Improvement relative to GFS: {overall_improvement:.2f}%")

        if not summary_printed:
            verifier = ScientificVerification(thresholds=PRECIP_THRESHOLDS)
            metrics = verifier.evaluate(predictions, targets, gfs_predictions)
            verifier.print_comprehensive_report(metrics, overall_improvement)
            summary_printed = True

            # Build finer threshold binary scores (for figures/tables)
            thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
            labels = ['Trace', 'Light', 'Moderate', 'Heavy', 'VeryHeavy', 'Extreme', 'Torrential']

            model_metrics, gfs_metrics = {}, {}
            total = targets.size

            for i, th in enumerate(thresholds):
                label = labels[i] if i < len(labels) else f'Thresh_{th}mm'
                pred_bin = (predictions >= th)
                target_bin = (targets >= th)

                tp = int(np.sum(pred_bin & target_bin))
                fp = int(np.sum(pred_bin & ~target_bin))
                fn = int(np.sum(~pred_bin & target_bin))
                tn = int(np.sum(~pred_bin & ~target_bin))

                random_hits = (tp + fp) * (tp + fn) / max(total, 1)
                ets = (tp - random_hits) / (tp + fp + fn - random_hits + 1e-8)
                pod = tp / (tp + fn + 1e-8)
                far = fp / (tp + fp + 1e-8)

                model_metrics[label] = {
                    'ETS': max(0.0, float(ets)),
                    'POD': float(pod),
                    'FAR': float(far),
                    'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn
                }

                gfs_bin = (gfs_predictions >= th)
                tp_g = int(np.sum(gfs_bin & target_bin))
                fp_g = int(np.sum(gfs_bin & ~target_bin))
                fn_g = int(np.sum(~gfs_bin & target_bin))
                tn_g = int(np.sum(~gfs_bin & ~target_bin))

                random_hits_g = (tp_g + fp_g) * (tp_g + fn_g) / max(total, 1)
                ets_g = (tp_g - random_hits_g) / (tp_g + fp_g + fn_g - random_hits_g + 1e-8)
                pod_g = tp_g / (tp_g + fn_g + 1e-8)
                far_g = fp_g / (tp_g + fp_g + 1e-8)

                gfs_metrics[label] = {
                    'ETS': max(0.0, float(ets_g)),
                    'POD': float(pod_g),
                    'FAR': float(far_g),
                    'TP': tp_g, 'FP': fp_g, 'FN': fn_g, 'TN': tn_g
                }

            test_metrics_summary['scientific_metrics'] = {
                'Model': model_metrics,
                'GFS': gfs_metrics,
                'verification_full': metrics
            }

            verifier.plot_ts_ets_curves(metrics, save_path='ts_ets_curves.png')
            verifier.plot_comprehensive_analysis(metrics, save_path='comprehensive_analysis.png')
    analyze_correction_sign(
        predictions=test_metrics_summary['predictions'],
        targets=test_metrics_summary['targets'],
        gfs_baseline=test_metrics_summary['gfs_baseline']
    )
    # ==================== Stage 10: Generate research-grade figures ====================
    print("\n" + "=" * 60)
    print(" Stage 10: Generate research-grade figures")
    print("=" * 60)

    if test_metrics_summary.get('scientific_metrics') is not None:
        sci = test_metrics_summary['scientific_metrics']
        if 'Model' in sci and 'GFS' in sci:
            results_dict = {
                'GFS_Baseline': sci['GFS'],
                'Enhanced_Model': sci['Model']
            }
            try:
                create_performance_diagram(results_dict)
            except Exception as e:
                print(f" Performance diagram generation failed: {e}")

    if test_metrics_summary.get('predictions') is not None and test_metrics_summary.get('targets') is not None:
        try:
            create_scientific_spatial_comparison(
                predictions=test_metrics_summary['predictions'],
                targets=test_metrics_summary['targets'],
                gfs_baseline=test_metrics_summary.get('gfs_baseline', None),
                dem_features=dem_tensor.cpu().numpy() if dem_tensor is not None else None,
                model_name="GFS f003(+3h) → ERA5(+3h) Correction Model"
            )
        except Exception as e:
            print(f" Research spatial comparison generation failed: {e}")
    try:
        ResearchVisualizer.plot_reliability_diagram(
            probs=test_metrics_summary.get('probs'),
            targets_bin=test_metrics_summary.get('targets_bin'),
            bins=12,
            save_path="reliability_diagram_rain_occurrence"
        )
    except Exception as e:
        print(f" reliability diagram failed: {e}")
    # ==================== Stage 11: Storm event analysis (keep event identification/cases; skip lead-time analysis) ====================
    print("\n" + "=" * 70)
    print(" Stage 11: Test set storm event analysis (event identification/cases only, no lead-time skill)")
    print("=" * 70)

    if test_metrics_summary.get('predictions') is not None and test_metrics_summary.get('targets') is not None:
        try:
            sample_times = test_metrics_summary.get('sample_times', [])

            storm_thresholds = {
                'Heavy Rain': 10.0,
                'Storm': 20.0,
                'Severe Storm': 50.0,
                'Extreme Storm': 100.0
            }

            all_storm_events = ResearchVisualizer.identify_all_storm_events(
                test_metrics_summary=test_metrics_summary,
                thresholds=storm_thresholds,
                sample_times=sample_times,
                min_storm_strength=10.0,
                use_area_detection=True
            )
            test_metrics_summary['all_storm_events'] = all_storm_events

            if len(all_storm_events) > 0:
                print(f" Identified {len(all_storm_events)} storm events (area-based)")

                # Individual case analysis (Top 5)
                sorted_storms = sorted(all_storm_events, key=lambda x: x['max_intensity'], reverse=True)
                detailed_storm_indices = deduplicate_keep_order([storm['index'] for storm in sorted_storms[:10]])[:5]

                if sample_times:
                    test_start = min(sample_times)
                    test_end   = max(sample_times)
                else:
                    test_start = datetime(2024, 1, 1)
                    test_end   = datetime(2025, 12, 31)

                storm_indices, storm_cases_info = ResearchVisualizer.create_storm_specific_visualizations(
                    test_metrics_summary=test_metrics_summary,
                    n_cases=5,
                    save_dir='detailed_storm_cases',
                    sample_times=sample_times,
                    test_start_date=test_start,
                    test_end_date=test_end,
                    specific_indices=detailed_storm_indices
                )
                test_metrics_summary['detailed_storm_cases'] = storm_cases_info
                test_metrics_summary['storm_indices'] = storm_indices

                storm_stats_summary = ResearchVisualizer.analyze_storm_statistics(all_storm_events)
                ResearchVisualizer.generate_comprehensive_storm_report(
                    all_storm_events,
                    storm_stats_summary,
                    sample_times,
                    storm_cases_info,
                    save_path='comprehensive_storm_report.txt'
                )

            # Threshold comparison plot (frequency/consistency/bias etc.)
            threshold_stats = ResearchVisualizer.create_threshold_comparison_plot(
                test_metrics_summary=test_metrics_summary,
                save_path='threshold_comparison_analysis.png'
            )
            test_metrics_summary['threshold_stats'] = threshold_stats

        except Exception as e:
            print(f" Storm event analysis failed: {e}")
            import traceback
            traceback.print_exc()
    # Event composite spatial maps
    try:
        ResearchVisualizer.create_event_composite_maps(
            test_metrics_summary=test_metrics_summary,
            thresholds=(10.0, 20.0),
            select_by='max',
            min_area=5,
            save_dir='storm_composites'
        )
    except Exception as e:
        print(f" composite maps failed: {e}")
    # ==================== Stage 12: Storm case RMSE improvement ranking (keep) ====================
    print("\n" + "=" * 70)
    print(" Stage 12: Storm event RMSE improvement analysis (sorted by improvement)")
    print("=" * 70)

    predictions = test_metrics_summary.get('predictions')
    targets = test_metrics_summary.get('targets')
    gfs_predictions = test_metrics_summary.get('gfs_baseline')
    sample_times = test_metrics_summary.get('sample_times', [])

    if predictions is not None and targets is not None and gfs_predictions is not None:
        storm_event_details = []
        for i in range(len(targets)):
            max_p = float(np.max(targets[i]))
            if max_p >= 10.0:
                rmse_gfs = float(np.sqrt(np.mean((gfs_predictions[i] - targets[i]) ** 2)))
                rmse_mod = float(np.sqrt(np.mean((predictions[i] - targets[i]) ** 2)))
                rmse_imp = (rmse_gfs - rmse_mod) / (rmse_gfs + 1e-8) * 100
                time_str = sample_times[i].strftime('%Y-%m-%d %H:%M') if i < len(sample_times) else f"Idx_{i}"
                storm_event_details.append({
                    'index': i,
                    'time': time_str,
                    'max_p': max_p,
                    'rmse_imp': float(rmse_imp),
                    'rmse_gfs': rmse_gfs,
                    'rmse_mod': rmse_mod
                })

        if storm_event_details:
            storm_event_details.sort(key=lambda x: x['rmse_imp'], reverse=True)
            test_metrics_summary['storm_rmse_improvements'] = storm_event_details
            ContinuousStormEventAnalyzer.create_top_rank_gallery(test_metrics_summary, top_n=10)

    # ==================== Stage 13: Generate research-grade experiment report ====================
    print("\n" + "=" * 60)
    print(" Stage 13: Generate research-grade experiment report")
    print("=" * 60)

    total_time = time_module.time() - start_time
    training_history = {
        'strategy': 'Strict paired dataset (valid_time match) + residual learning + gated evaluation + EMA',
        'total_time': float(total_time),
        'pretrain_losses': gfs_pretrain_losses if 'gfs_pretrain_losses' in locals() else [],
        'stage_c_losses': staged_training_results.get('stage_c_losses', []),
        'stage_c_val_losses': staged_training_results.get('stage_c_val_losses', []),
        'storm_pod_5': staged_training_results.get('storm_pod_5', []),
        'storm_pod_10': staged_training_results.get('storm_pod_10', []),
        'storm_pod_20': staged_training_results.get('storm_pod_20', []),
        'best_val_loss': float(min(staged_training_results.get('stage_c_val_losses', [float('inf')]))),
        'scaling_factor': float(scaling_factor),
        'oversample_ratio': int(oversample_ratio),
        'dem_channels': int(dem_channels),
        'device': str(device)
    }
    print("\n" + " VISUALIZATION ARTIFACTS GENERATED")
    print("-" * 50)
    artifacts = [
        "taylor_diagram.png", "training_convergence.png", 
        "spatial_bias_gain.png", "density_scatter_comparison.png",
        "scientific_spatial_smooth.png", "comprehensive_storm_report.txt"
    ]
    for art in artifacts:
        if os.path.exists(art):
            print(f"  [DONE] -> {art}")
    # ==================== Stage 14: Generate Taylor diagram, training curve, spatial bias map ====================
    if test_metrics_summary.get('predictions') is not None:
        try:
            ResearchVisualizer.plot_taylor_diagram(
                preds=test_metrics_summary['predictions'],
                obs=test_metrics_summary['targets'],
                gfs=test_metrics_summary['gfs_baseline'],
                save_path='taylor_diagram.png'
            )
            print(" Taylor diagram saved: taylor_diagram.png")
            ResearchVisualizer.plot_spatial_bias_map(
                preds=test_metrics_summary['predictions'],
                obs=test_metrics_summary['targets'],
                gfs=test_metrics_summary['gfs_baseline'],
                save_path='spatial_bias_gain.png'
            )
            print(" Spatial bias gain: spatial_bias_gain.png")
        except Exception as e:
            print(f" Taylor diagram/spatial bias map generation failed: {e}")

    # Training curve using staged_training_results
    if 'staged_training_results' in locals() and staged_training_results:
        history = {
            'stage_c_losses': staged_training_results.get('stage_c_losses', []),
            'stage_c_val_losses': staged_training_results.get('stage_c_val_losses', [])
        }
        if len(history['stage_c_losses']) > 0:
            ResearchVisualizer.plot_training_history(history, save_path='training_convergence.png')
            print(" Spatial bias gain: training_convergence.png")
    # ==================== Temporary file cleanup ====================
    print("\n Cleaning temporary files...")
    for temp_dir in ["temp_extract", "cache", "./data_cache"]:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"   Cleaned: {temp_dir}")
            except Exception:
                pass

    # ==================== Post-training comprehensive evaluation: feature importance + terrain region ====================
    print("\n" + "=" * 70)
    print(" Post-training comprehensive performance evaluation")
    print("=" * 70)

    print("\n Generating feature importance ranking plot...")
    try:
        importance_scores = ResearchVisualizer.analyze_feature_importance(
            model=model,
            test_loader=test_loader,
            device=device,
            scaling_factor=scaling_factor
        )
        test_metrics_summary['feature_importance'] = importance_scores
    except Exception as e:
        print(f" Feature importance analysis failed: {e}")

    print("\n Performing terrain subregion evaluation...")
    try:
        subregion_results = evaluate_subregion_performance(
            model=model,
            test_loader=test_loader,
            device=device,
            dem_tensor=dem_tensor,
            scaling_factor=scaling_factor
        )
        test_metrics_summary['subregion_results'] = subregion_results
    except Exception as e:
        print(f" Terrain subregion evaluation failed: {e}")

    # ==================== Save final model ====================
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_metrics_summary': test_metrics_summary,
        'training_history': training_history,
        'dem_channels': dem_channels,
        'scaling_factor': scaling_factor,
        'oversample_ratio': oversample_ratio,
        'task': 'GFS f003(+3h) -> ERA5(+3h) residual correction (strict valid_time pairing)'
    }, 'final_intelligent_storm_oversampling_model.pth')
    print(" Final model saved: final_intelligent_storm_oversampling_model.pth")

    print("\n" + "=" * 80)
    print(" Experiment complete (strict scientific version)")
    print("=" * 80)

    return {
        'model': model,
        'training_history': training_history,
        'test_metrics_summary': test_metrics_summary,
        'dem_channels': dem_channels,
        'scaling_factor': scaling_factor,
        'oversample_ratio': oversample_ratio,
        'staged_training_results': staged_training_results
    }
def improved_gfs_pretrain_phase(model, train_loader, device, epochs=1, learning_rate=5e-5):
    """
    Modification: greatly shorten pretraining, only 1 epoch
    """
    print(f"\n Short pretraining ({epochs} epoch), only for model initialization")
  
    if epochs == 0:
        print("   Skipping pretraining, go directly to residual learning")
        return [], []
  
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
  
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Pretraining Epoch {epoch+1}")
      
        for inputs, _ in pbar:
            if inputs is None or inputs.shape[0] == 0: 
                continue
          
            inputs = inputs.to(device)
            # Pretraining target is GFS's own precipitation
            gfs_truth = inputs[:, -1, 5:6, :, :].expand(-1, model.ph, -1, -1).to(device)
          
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                pred_precip = model(inputs, return_residual=False)
                loss = criterion(pred_precip, gfs_truth)
          
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
          
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
          
        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)
        print(f"  Epoch {epoch+1} Average Loss: {avg_epoch_loss:.6f}")
  
    return losses, []
class MultiTaskLoss(nn.Module):
    """
    Gen-5 ultimate: Smooth Asymmetric Loss
    - Eliminate hard cutoffs causing FAR increase, use smooth transition penalties.
    """
    def __init__(self, window_size=5, gamma=2.0, lambda_rain=0.2, lambda_storm=0.1, lambda_ets=1.5):
        super().__init__()
        self.window_size = window_size
        self.gamma = gamma
        self.pool = nn.AvgPool2d(window_size, stride=1, padding=window_size//2)
        self.lambda_rain = lambda_rain
        self.lambda_storm = lambda_storm
        self.lambda_ets = lambda_ets

        storm_class_weights = torch.tensor([1.0, 1.2, 2.0, 4.0, 8.0])
        self.ce = nn.CrossEntropyLoss(weight=storm_class_weights, ignore_index=-1)

    def _precip_weight(self, target_abs):
        w = torch.ones_like(target_abs)
        w = torch.where(target_abs < 0.1,  w * 1.2, w)
        w = torch.where(target_abs >= 0.1,  w * 2.0, w)
        w = torch.where(target_abs >= 3.0,  w * 2.5, w)
        w = torch.where(target_abs >= 10.0, w * 3.5, w)
        w = torch.where(target_abs >= 20.0, w * 5.0, w)
        return w

    def _soft_ets_loss(self, pred, target, threshold):
        p_prob = torch.sigmoid((pred - threshold) * 2.0)
        t_bin = (target >= threshold).float()
        
        tp = torch.sum(p_prob * t_bin)
        fp = torch.sum(p_prob * (1.0 - t_bin))
        fn = torch.sum((1.0 - p_prob) * t_bin)
        
        total = torch.numel(p_prob)
        random_hits = (tp + fp) * (tp + fn) / (total + 1e-8)
        ets = (tp - random_hits) / (tp + fp + fn - random_hits + 1e-8)
        return 1.0 - torch.clamp(ets, min=0.0, max=1.0)

    def forward(self, pred_res, target_abs, gfs_base, rain_prob, storm_logits):
        if gfs_base.dim() == 4 and gfs_base.shape[1] == 1:
            gfs_base = gfs_base.expand(-1, pred_res.shape[1], -1, -1)

        pred_abs = gfs_base + pred_res
        diff = pred_abs - target_abs

        # Smooth asymmetric weight
        asymmetric_weight = torch.ones_like(diff)
        
        # Miss penalty: when target heavy rain and prediction low, weight increases smoothly (up to 2.5)
        miss_ratio = torch.clamp((target_abs - pred_abs) / 10.0, 0.0, 1.5)
        miss_mask = (target_abs >= 5.0) & (diff < 0.0)
        asymmetric_weight = torch.where(miss_mask, 1.0 + miss_ratio, asymmetric_weight)
        
        # False alarm penalty: smooth penalty for overprediction in no/trace rain areas
        fa_ratio = torch.clamp(pred_abs / 2.0, 0.0, 1.5)
        fa_mask = (target_abs < 0.5) & (diff > 0.0)
        asymmetric_weight = torch.where(fa_mask, 1.0 + fa_ratio, asymmetric_weight)

        precip_w = self._precip_weight(target_abs)
        
        # Focal weight
        normalized_diff = torch.abs(diff) / (target_abs + 2.0)
        focal_w = (normalized_diff ** self.gamma).detach()
        weight_map = precip_w * (1.0 + focal_w) * asymmetric_weight
        
        reg_loss = torch.mean(torch.abs(diff) * weight_map)
        mse_loss = torch.mean((diff ** 2) * precip_w * asymmetric_weight) * 0.5 

        # Spatial consistency (FSS)
        p_bin_10 = torch.sigmoid((pred_abs - 10.0) * 2.0)
        t_bin_10 = (target_abs >= 10.0).float()
        fss_10 = torch.mean((self.pool(p_bin_10) - self.pool(t_bin_10))**2) / (
                 torch.mean(self.pool(p_bin_10)**2 + self.pool(t_bin_10)**2) + 1e-6)

        p_bin_20 = torch.sigmoid((pred_abs - 20.0) * 2.0)
        t_bin_20 = (target_abs >= 20.0).float()
        fss_20 = torch.mean((self.pool(p_bin_20) - self.pool(t_bin_20))**2) / (
                 torch.mean(self.pool(p_bin_20)**2 + self.pool(t_bin_20)**2) + 1e-6)

        true_rain = (target_abs > 0.1).float()
        with torch.cuda.amp.autocast(enabled=False):
            loss_rain = F.binary_cross_entropy(rain_prob.float(), true_rain.float())

        storm_class = torch.zeros_like(target_abs, dtype=torch.long)
        storm_class[(target_abs >= 0.1) & (target_abs < 3.0)] = 1
        storm_class[(target_abs >= 3.0) & (target_abs < 10.0)] = 2
        storm_class[(target_abs >= 10.0) & (target_abs < 20.0)] = 3
        storm_class[target_abs >= 20.0] = 4

        valid_mask = target_abs >= 0.1
        if valid_mask.any():
            valid_mask_flat = valid_mask.squeeze(1)
            storm_logits_reshaped = storm_logits.permute(0, 2, 3, 1)
            storm_logits_valid = storm_logits_reshaped[valid_mask_flat]
            storm_class_valid = storm_class.squeeze(1)[valid_mask_flat]
            
            if self.ce.weight.device != storm_logits_valid.device:
                self.ce.weight = self.ce.weight.to(storm_logits_valid.device)
                
            loss_storm = self.ce(storm_logits_valid, storm_class_valid)
        else:
            loss_storm = torch.tensor(0.0, device=storm_logits.device)

        batch_max = torch.max(target_abs)
        dynamic_storm_multiplier = torch.clamp(batch_max / 20.0, 1.0, 3.0)
        
        loss_ets_10 = self._soft_ets_loss(pred_abs, target_abs, 10.0)
        loss_ets_20 = self._soft_ets_loss(pred_abs, target_abs, 20.0)

        total = (reg_loss + mse_loss +
                 1.0 * fss_10 + (2.0 * dynamic_storm_multiplier) * fss_20 +
                 self.lambda_rain * loss_rain + self.lambda_storm * loss_storm +
                 self.lambda_ets * (loss_ets_10 + dynamic_storm_multiplier * loss_ets_20))
                 
        return total
class ASPPBlock(nn.Module):
    """Multi-scale perception module: enhance longer lead-time capture capability"""
    def __init__(self, in_channels, out_channels):
        super(ASPPBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.atrous_conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.atrous_conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Conv2d(in_channels, out_channels, 1)
        self.fuse = nn.Conv2d(out_channels * 4, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.atrous_conv2(x)
        x3 = self.atrous_conv3(x)
        x4 = F.interpolate(self.conv_pool(self.global_pool(x)), size=x.shape[2:], mode='bilinear', align_corners=True)
        return self.fuse(torch.cat([x1, x2, x3, x4], dim=1))
def compute_gated_precip_prediction(
        gfs_base, residual, rain_prob,
        storm_logits=None,
        threshold_base=0.22,
        gate_power=0.85,
        min_rain_value=0.10,
        max_precip=250.0,
        adaptive=True,
        hard_gate=True,
        storm_gate_p=0.40,
        **kwargs           # absorb any extra keyword arguments
):
    """
    Complete gated residual fusion
    """
    # Ensure dimension alignment
    if gfs_base.dim() == 4 and gfs_base.shape[1] == 1:
        gfs_base = gfs_base.expand(-1, residual.shape[1], -1, -1)

    # 1. Obtain storm probability from storm_logits (if provided)
    if storm_logits is not None:
        storm_probs = torch.softmax(storm_logits, dim=1)[:, -1:, :, :]   # last class is storm probability
    else:
        storm_probs = torch.zeros_like(rain_prob)

    # 2. Base gate factor (rain probability)
    if adaptive:
        gate = torch.pow(rain_prob, gate_power)
    else:
        gate = (rain_prob >= threshold_base).float()

    # 3. Hard cutoff to suppress drizzle over-forecasting
    hard_cutoff = threshold_base * 0.7
    gate = torch.where(rain_prob < hard_cutoff, torch.zeros_like(gate), gate)

    # 4. Storm special handling: fully pass residual when storm probability high
    storm_boost = (storm_probs >= storm_gate_p).float()
    if adaptive or hard_gate:
        gate = torch.where(storm_boost.bool(), torch.ones_like(gate), gate)
    # 4. Final precipitation = GFS + gate * residual
    pred = gfs_base + gate * residual

    # 5. Physical constraints: non-negative, max precipitation limit
    pred = torch.clamp(pred, min=0.0, max=max_precip)
    pred = torch.where(pred < min_rain_value, torch.zeros_like(pred), pred)

    return pred, gate
def get_model_eval_tensors(model, inputs, targets_scaled, scaling_factor=1.0,
                           rain_prob_threshold=None,
                           gate_power=None,
                           min_rain_value=0.10,
                           max_precip=250.0,
                           adaptive=True,
                           hard_gate=True,
                           storm_gate_p=None):
    """
    Unified evaluation interface (including scientific isolation for pure data-driven baseline U-Net)
    """
    if rain_prob_threshold is None:
        rain_prob_threshold = float(GATE_CFG.get("threshold_base", 0.22))
    if gate_power is None:
        gate_power = float(GATE_CFG.get("gate_power", 0.90))
    if storm_gate_p is None:
        storm_gate_p = float(GATE_CFG.get("storm_gate_p", 0.30))

    residual, rain_prob, storm_logits, _ = model(inputs, return_residual=True, return_storm_logits=True)

    gfs_base = inputs[:, -1, 5:6, :, :]
    gfs_expand = gfs_base.expand(-1, targets_scaled.shape[1], -1, -1)

    # Scientific patch: detect if model is pure data-driven U-Net baseline
    is_unet = False
    model_name = model.__class__.__name__
    if model_name == 'StandardUNet':
        is_unet = True
    elif model_name == 'ModelEMA' and hasattr(model, 'ema') and model.ema.__class__.__name__ == 'StandardUNet':
        is_unet = True

    if is_unet:
        # If U-Net, force disable physical gate (simulate traditional AI black-box mapping)
        # Allow all residuals to pass unconditionally, to highlight importance of gate mechanism
        pred_abs = gfs_base + residual
        pred_abs = torch.clamp(pred_abs, min=0.0, max=float(max_precip))
        pred_abs = torch.where(pred_abs < float(min_rain_value), torch.zeros_like(pred_abs), pred_abs)
        rain_gate = torch.ones_like(residual) # record gate fully open
    else:
        # If APCNet, enable advanced adaptive gate
        pred_abs, rain_gate = compute_gated_precip_prediction(
            gfs_base=gfs_base,
            residual=residual,
            rain_prob=rain_prob,
            rain_prob_threshold=float(rain_prob_threshold),
            gate_power=float(gate_power),
            min_rain_value=float(min_rain_value),
            max_precip=float(max_precip),
            adaptive=bool(adaptive),
            hard_gate=bool(hard_gate),
            storm_logits=storm_logits,
            storm_gate_p=float(storm_gate_p)
        )

    true_abs = gfs_expand + targets_scaled / scaling_factor
    return pred_abs, true_abs, gfs_expand, rain_gate, residual, rain_prob, storm_logits
class FiLMLayer(nn.Module):
    """
    Gen-5 tuning: introduce Hardsigmoid absolute cutoff mechanism
    - Replace Sigmoid with Hardsigmoid.
    - Meteorological interpretation: For physical fields with very low SNR in GFS (e.g., coarse-resolution CAPE and vertical velocity),
      Hardsigmoid can output absolute 0 weights, achieving "hard pruning" of physical channels,
      eliminating weak negative noise leakage, ensuring only high-confidence physical quantities (like PWAT) remain.
    """
    def __init__(self, phys_channels, feat_channels):
        super().__init__()

        # 1. Physical feature adaptive filter (SE structure - hard cutoff version)
        self.phys_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(phys_channels, max(4, phys_channels // 2), 1),
            nn.SiLU(),
            nn.Conv2d(max(4, phys_channels // 2), phys_channels, 1),
            nn.Hardsigmoid()  
        )

        # 2. Spatial convolution (preserve thermodynamic spatial structure)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(phys_channels, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.SiLU(),
            nn.Conv2d(feat_channels, feat_channels * 2, kernel_size=3, padding=1)
        )

        # 3. Auxiliary channel attention
        self.channel_conv = nn.Sequential(
            nn.Conv2d(phys_channels, feat_channels * 2, kernel_size=1)
        )

    def forward(self, x, phys):
        # Core: Hardsigmoid dynamic suppression, keep weights of negative features at 0
        attn_weights = self.phys_attention(phys)  
        phys_filtered = phys * attn_weights         

        spatial_stats = self.spatial_conv(phys_filtered)   
        channel_stats = self.channel_conv(phys_filtered)   

        stats = spatial_stats + channel_stats
        gamma, beta = torch.chunk(stats, 2, dim=1)  

        return x * (1.0 + torch.tanh(gamma)) + beta
class TemporalAttention(nn.Module):
    """
    Temporal attention: let the model learn which timesteps are most important for current prediction.
    For f003 correction, recent timesteps are usually more important, but heavy rain events may have precursor signals.
    """
    def __init__(self, channels, num_timesteps):
        super().__init__()
        # Fix 1: safely compute hidden channels, avoid zero when input channels <4
        hidden_channels = max(1, channels // 4)
        
        self.query = nn.Conv2d(channels, hidden_channels, 1)
        self.key   = nn.Conv2d(channels, hidden_channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
        # Fix 2: use safe hidden_channels for scale factor, completely eliminate division by zero error
        self.scale = hidden_channels ** -0.5
        self.num_timesteps = num_timesteps

    def forward(self, x):
        """
        x: (B, T, C, H, W) -> output (B, C, H, W) weighted aggregation
        """
        b, t, c, h, w = x.shape
        # Take last timestep as query
        q = self.query(x[:, -1]).view(b, -1, h * w)          # (B, hidden_channels, HW)
        # All timesteps as key/value
        x_flat = x.view(b * t, c, h, w)
        k = self.key(x_flat).view(b, t, -1, h * w)           # (B, T, hidden_channels, HW)
        v = self.value(x_flat).view(b, t, -1, h * w)         # (B, T, C, HW)
        
        # Attention scores
        attn = torch.einsum('bch, btch -> bt', q, k) * self.scale  # (B, T)
        attn = torch.softmax(attn, dim=1)                          # (B, T)
        
        # Weighted aggregation
        out = torch.einsum('bt, btch -> bch', attn, v)             # (B, C, HW)
        return out.view(b, c, h, w), attn
class SpatialAttention(nn.Module):
    """
    Spatial attention: let model focus on key precipitation areas (fronts, low-pressure centers, etc.)
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, 1, 7, padding=3),  # large receptive field
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.conv(x)  # (B, 1, H, W)
        return x * attn, attn
class StandardUNet(nn.Module):
    """
    Standard U-Net baseline model (for AIES comparison experiment)
    Pure data-driven, no physical thermodynamic decoupling or gate mechanism
    """
    def __init__(self, input_channels=6, hidden_channels=32, prediction_horizon=1):
        super().__init__()
        self.ph = prediction_horizon
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
            
        self.enc1 = conv_block(input_channels, hidden_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(hidden_channels, hidden_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = conv_block(hidden_channels * 2, hidden_channels * 4)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = conv_block(hidden_channels * 6, hidden_channels * 2)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = conv_block(hidden_channels * 3, hidden_channels)
        
        self.res_head = nn.Conv2d(hidden_channels, self.ph, 1)
        self.rain_prob_head = nn.Sequential(nn.Conv2d(hidden_channels, 1, 1), nn.Sigmoid())
        self.storm_head = nn.Conv2d(hidden_channels, 5, 1)

    def forward(self, x, return_residual=True, return_storm_logits=False):
        b, t, c, h, w = x.shape
        x_last = x[:, -1, :, :, :] # [B, 6, H, W]
        
        e1 = self.enc1(x_last)
        e2 = self.enc2(self.pool1(e1))
        bn = self.bottleneck(self.pool2(e2))
        
        u2 = self.up2(bn)
        if u2.shape[2:] != e2.shape[2:]:
            u2 = F.interpolate(u2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        if u1.shape[2:] != e1.shape[2:]:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        residual = self.res_head(d1) * 10.0
        rain_prob = self.rain_prob_head(d1).expand(-1, self.ph, -1, -1)
        storm_logits = self.storm_head(d1)
        
        if return_residual:
            if return_storm_logits:
                return residual, rain_prob, storm_logits, None
            else:
                return residual, rain_prob, None
        else:
            # Fix error point: handle return_residual=False case
            gfs_base = x[:, -1, 5:6, :, :]
            pred_abs = torch.clamp(gfs_base + residual, min=0.0, max=250.0)
            if return_storm_logits:
                return pred_abs, rain_prob
            else:
                return pred_abs
class AdvancedPrecipCorrectionNet(nn.Module):
    """
    Gen-5 ultimate: Kinematic-Thermodynamic Decoupling architecture
    - Backbone: receives 3 channels [U-Wind, V-Wind, GFS-Precip], uses wind field to provide continuous spatial dynamics topology, saving Spatial CC.
    - FiLM layer: receives 5 channels [CAPE, PWAT, U, V, VVEL] as thermodynamic triggers, dynamically regulating storm peaks.
    """
    def __init__(self, input_channels=6, hidden_channels=24, prediction_horizon=1,
                 sequence_length=6, spatial_dims=(25, 37), dropout_rate=0.1):
        super().__init__()
        self.ph = prediction_horizon
        hc = hidden_channels  

        # ========== Backbone receives 3 channels (U, V, GFS-precip) ==========
        self.temporal_attn = TemporalAttention(3, sequence_length)

        self.enc_stage1 = nn.Sequential(
            nn.Conv2d(3, hc, 3, padding=1),
            nn.BatchNorm2d(hc), nn.SiLU(),
            nn.Conv2d(hc, hc * 2, 3, padding=1),
            nn.BatchNorm2d(hc * 2), nn.SiLU()
        )
        self.pool = nn.MaxPool2d(2)
        self.enc_stage2 = nn.Sequential(
            ASPPBlock(hc * 2, hc * 4),
            nn.Conv2d(hc * 4, hc * 4, 3, padding=1),
            nn.BatchNorm2d(hc * 4), nn.SiLU()
        )

        self.bottleneck = ASPPBlock(hc * 4, hc * 8)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_stage1 = nn.Sequential(
            nn.Conv2d(hc * 10, hc * 4, 3, padding=1),   
            nn.BatchNorm2d(hc * 4), nn.SiLU(),
            nn.Conv2d(hc * 4, hc * 4, 3, padding=1),
            nn.BatchNorm2d(hc * 4), nn.SiLU()
        )

        self.spatial_attn = SpatialAttention(hc * 4)

        # ========== Physical auxiliary modulation (5 channels: CAPE/PWAT/U/V/VV) ==========
        self.physics_film1 = FiLMLayer(phys_channels=5, feat_channels=hc * 4)
        self.physics_film2 = FiLMLayer(phys_channels=5, feat_channels=hc * 4)

        self.res_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hc * 4, 1, 1)) for _ in range(self.ph)
        ])
        self.rain_prob_head = nn.Sequential(
            nn.Conv2d(hc * 4, 1, 1),
            nn.Sigmoid()
        )
        self.storm_head = nn.Conv2d(hc * 4, 5, 1)

    def forward(self, x, return_residual=True, return_storm_logits=False):
        b, t, c, h, w = x.shape
        x_processed = x.clone()

        if self.training:
            # Keep 10% mild dropout, only on precipitation channel, not disturbing wind field structure
            drop_prob = 0.10
            mask = (torch.rand(b, 1, 1, 1, 1, device=x.device) > drop_prob).float()
            x_processed[:, :, 5:6, :, :] = (x_processed[:, :, 5:6, :, :] * mask) / (1.0 - drop_prob)

        # Extract kinematic combination: [U-Wind(2), V-Wind(3), GFS-Precip(5)] into backbone
        kinematic_input = x_processed[:, :, [2, 3, 5], :, :]  # [B, T, 3, H, W]
        x_ta, _ = self.temporal_attn(kinematic_input)         # (B, 3, H, W)

        e1 = self.enc_stage1(x_ta)          
        e2 = self.enc_stage2(self.pool(e1)) 
        bn = self.bottleneck(e2)            

        up_feat = self.up(bn)               
        if up_feat.shape[2:] != e1.shape[2:]:
            up_feat = F.interpolate(up_feat, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec_stage1(torch.cat([up_feat, e1], dim=1))  

        d1_sa, _ = self.spatial_attn(d1)

        # Thermodynamic global modulation: extract [CAPE(0), PWAT(1), U(2), V(3), VVEL(4)]
        phys_context = x_processed[:, -1, 0:5, :, :] 
        d1_film1 = self.physics_film1(d1_sa, phys_context)
        d1_film2 = self.physics_film2(d1_film1, phys_context)   

        residual_list = []
        for i in range(self.ph):
            res = self.res_heads[i](d1_film2)
            residual_list.append(res * 10.0)
        residual = torch.cat(residual_list, dim=1)  

        rain_prob = self.rain_prob_head(d1_film2).expand(-1, self.ph, -1, -1)  
        storm_logits = self.storm_head(d1_film2)    

        if return_residual:
            if return_storm_logits:
                return residual, rain_prob, storm_logits, None
            else:
                return residual, rain_prob, None
        else:
            gfs_base = x[:, -1, 5:6, :, :]   
            pred_abs, _ = compute_gated_precip_prediction(
                gfs_base=gfs_base,
                residual=residual,
                rain_prob=rain_prob,
                storm_logits=storm_logits,
                **GATE_CFG
            )
            if return_storm_logits:
                return pred_abs, rain_prob
            else:
                return pred_abs
class MeteorologicalEvaluator:
    """Unified research metric evaluation class - solve zero value and duplicate display issues"""
    def __init__(self, thresholds=PRECIP_THRESHOLDS, levels=PRECIP_LEVELS):
        self.thresholds = thresholds
        self.levels = levels


    def evaluate(self, pred, obs, gfs=None):
        metrics = {'Model': {}, 'GFS': {}}
        p_f, o_f = pred.flatten(), obs.flatten()
        if gfs is not None: g_f = gfs.flatten()


        for i, th in enumerate(self.thresholds):
            lvl = self.levels[i]
            metrics['Model'][lvl] = self._get_scores(p_f >= th, o_f >= th)
            if gfs is not None:
                metrics['GFS'][lvl] = self._get_scores(g_f >= th, o_f >= th)
        return metrics

    def _get_scores(self, p_bin, o_bin):
        tp = np.sum(p_bin & o_bin)
        fp = np.sum(p_bin & ~o_bin)
        fn = np.sum(~p_bin & o_bin)
        tn = np.sum(~p_bin & ~o_bin)
        total = len(o_bin)
        r_hits = (tp + fp) * (tp + fn) / total if total > 0 else 0
        ets = (tp - r_hits) / (tp + fp + fn - r_hits + 1e-8)
        pod = tp / (tp + fn + 1e-8)
        far = fp / (tp + fp + 1e-8)
        return {'ETS': max(0.0, ets), 'POD': pod, 'FAR': far}
  
    def calculate_comprehensive_meteorological_scores(self, predictions, targets, thresholds=None):
        """Compute complete meteorological score metrics"""
      
        if thresholds is None:
            thresholds = self.thresholds
      
        results = {}
      
        for threshold in thresholds:
            # Binarize
            pred_binary = (predictions >= threshold).astype(int)
            target_binary = (targets >= threshold).astype(int)
          
            # Confusion matrix
            tp = np.sum((pred_binary == 1) & (target_binary == 1))
            fp = np.sum((pred_binary == 1) & (target_binary == 0))
            fn = np.sum((pred_binary == 0) & (target_binary == 1))
            tn = np.sum((pred_binary == 0) & (target_binary == 0))
            total = tp + fp + fn + tn
          
            # Compute metrics
            pod = tp / max(tp + fn, 1)  # hit rate / probability of detection
            far = fp / max(tp + fp, 1)  # false alarm ratio
            csi = tp / max(tp + fp + fn, 1)  # critical success index
            bias = (tp + fp) / max(tp + fn, 1)  # bias score
          
            # ETS (equitable threat score)
            random_hits = (tp + fp) * (tp + fn) / max(total, 1)
            if tp + fp + fn - random_hits > 0:
                ets = (tp - random_hits) / (tp + fp + fn - random_hits)
            else:
                ets = 0.0
          
            # HSS (Heidke skill score)
            expected_correct_random = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / max(total, 1)
            if total - expected_correct_random > 0:
                hss = (tp + tn - expected_correct_random) / (total - expected_correct_random)
            else:
                hss = 0.0
          
            # PSS (Peirce skill score)
            pss = (tp / max(tp + fn, 1)) - (fp / max(fp + tn, 1))
          
            results[f'Threshold_{threshold}mm'] = {
                'ETS': ets,
                'CSI': csi,
                'HSS': hss,
                'PSS': pss,
                'POD': pod,
                'FAR': far,
                'BIAS': bias,
                'TP': int(tp),
                'FP': int(fp),
                'FN': int(fn),
                'TN': int(tn),
                'Total': int(total)
            }
      
        return results
  
    def print_meteorological_scorecard(self, results):
        """Print meteorological scorecard"""
      
        print("\n" + "=" * 120)
        print(" Meteorological Scorecard")
        print("=" * 120)
      
        # Header
        headers = ['Threshold(mm)', 'ETS', 'CSI', 'HSS', 'PSS', 'POD', 'FAR', 'BIAS', 'Hit', 'FalseAlarm', 'Miss', 'CorrectNeg', 'Total']
        header_format = "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}"
        print(header_format.format(*headers))
        print("-" * 120)
      
        # Data rows
        for threshold_key, metrics in results.items():
            if isinstance(threshold_key, str) and threshold_key.startswith('Threshold_'):
                threshold_value = float(threshold_key.replace('Threshold_', '').replace('mm', ''))
                if threshold_value in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
                    row = [
                        f"{threshold_value}mm",
                        f"{metrics['ETS']:.4f}",
                        f"{metrics['CSI']:.4f}",
                        f"{metrics['HSS']:.4f}",
                        f"{metrics['PSS']:.4f}",
                        f"{metrics['POD']:.4f}",
                        f"{metrics['FAR']:.4f}",
                        f"{metrics['BIAS']:.2f}",
                        f"{metrics['TP']}",
                        f"{metrics['FP']}",
                        f"{metrics['FN']}",
                        f"{metrics['TN']}",
                        f"{metrics['Total']}"
                    ]
                    print(header_format.format(*row))
      
        print("=" * 120)
      
        # Compute average scores
        strong_precip_keys = ['Threshold_5.0mm', 'Threshold_10.0mm', 'Threshold_20.0mm']
        strong_precip_scores = []
      
        for key in strong_precip_keys:
            if key in results:
                strong_precip_scores.append(results[key]['ETS'])
      
        if strong_precip_scores:
            avg_ets = np.mean(strong_precip_scores)
            print(f" Average ETS for heavy rain (≥5mm): {avg_ets:.4f}")
          
            # Performance rating
            if avg_ets > 0.3:
                print(" ETS score: Excellent")
            elif avg_ets > 0.2:
                print(" ETS score: Good")
            elif avg_ets > 0.1:
                print(" ETS score: Fair")
            else:
                print(" ETS score: Needs improvement")
      
        # Bias score analysis
        bias_scores = []
        for key, metrics in results.items():
            if 'BIAS' in metrics:
                bias_scores.append(metrics['BIAS'])
      
        if bias_scores:
            avg_bias = np.mean(bias_scores)
            print(f" Average bias score: {avg_bias:.2f}")
            if 0.8 < avg_bias < 1.2:
                print(" Bias score: Ideal range")
            elif 0.5 < avg_bias < 2.0:
                print(" Bias score: Acceptable range")
            else:
                print(" Bias score: Needs adjustment")

class ScientificEvaluator:
    @staticmethod
    def calculate_fss(pred, obs, threshold=0.1, window_size=5):
        """Calculate FSS (Fractional Skill Score) for spatial consistency"""
        pred_bin = (pred >= threshold).float()
        obs_bin = (obs >= threshold).float()
      
        # Use average pooling for neighborhood frequency
        padding = window_size // 2
        pool = nn.AvgPool2d(window_size, stride=1, padding=padding)
      
        p_frac = pool(pred_bin)
        o_frac = pool(obs_bin)
      
        mse_frac = torch.mean((p_frac - o_frac)**2)
        ref_frac = torch.mean(p_frac**2 + o_frac**2)
      
        if ref_frac < 1e-8: return 1.0
        return 1.0 - (mse_frac / ref_frac)

class ScientificVerification:
    def __init__(self, thresholds=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], levels=None):
        self.thresholds = thresholds
        if levels is None:
            self.labels = ['Trace', 'Light', 'Mod', 'Heavy', 'Storm', 'Extr', 'Torr'][:len(thresholds)]
        else:
            self.labels = levels

    def _calc_cc(self, a, b):
        """Calculate spatial correlation coefficient"""
        a_mean, b_mean = np.mean(a), np.mean(b)
        numerator = np.sum((a - a_mean) * (b - b_mean))
        denominator = np.sqrt(np.sum((a - a_mean)**2)) * np.sqrt(np.sum((b - b_mean)**2))
        return numerator / (denominator + 1e-8)

    def evaluate(self, pred, obs, gfs):
        """Evaluate model and GFS baseline - extended: includes TS, RMSE, CC and more"""
        metrics = {'Model': {}, 'GFS': {}}

        pred_f, obs_f, gfs_f = pred.flatten(), obs.flatten(), gfs.flatten()

        # Continuous variable scores (global)
        mse_model = np.mean((pred_f - obs_f) ** 2)
        rmse_model = np.sqrt(mse_model)
        mae_model = np.mean(np.abs(pred_f - obs_f))
        cc_model = self._calc_cc(pred_f, obs_f)

        mse_gfs = np.mean((gfs_f - obs_f) ** 2)
        rmse_gfs = np.sqrt(mse_gfs)
        mae_gfs = np.mean(np.abs(gfs_f - obs_f))
        cc_gfs = self._calc_cc(gfs_f, obs_f)

        metrics['Model']['Continuous'] = {
            'MSE': mse_model,
            'RMSE': rmse_model,
            'MAE': mae_model,
            'CC': cc_model
        }

        metrics['GFS']['Continuous'] = {
            'MSE': mse_gfs,
            'RMSE': rmse_gfs,
            'MAE': mae_gfs,
            'CC': cc_gfs
        }

        for i, th in enumerate(self.thresholds):
            label = self.labels[i] if i < len(self.labels) else f'Thresh_{th}mm'

            # Model evaluation
            model_metrics = self._calculate_comprehensive_scores(pred_f >= th, obs_f >= th, pred_f, obs_f, th)
            metrics['Model'][label] = model_metrics

            # GFS evaluation
            gfs_metrics = self._calculate_comprehensive_scores(gfs_f >= th, obs_f >= th, gfs_f, obs_f, th)
            metrics['GFS'][label] = gfs_metrics

        return metrics
    def evaluate_with_ci(self, pred, obs, gfs, n_bootstrap=500):
        """Extend evaluate method with CI calculation"""
        metrics = self.evaluate(pred, obs, gfs)
        print(f" Computing Bootstrap confidence intervals for {len(self.thresholds)} levels (n={n_bootstrap})...")
        
        for i, th in enumerate(self.thresholds):
            label = self.labels[i]
            # Compute CI for model
            m_pod_ci, m_far_ci, m_ets_ci = self.bootstrap_metrics_ci(pred, obs, th, n_bootstrap)
            metrics['Model'][label]['POD_CI'] = m_pod_ci
            metrics['Model'][label]['FAR_CI'] = m_far_ci
            metrics['Model'][label]['ETS_CI'] = m_ets_ci
            
            # Compute CI for GFS
            g_pod_ci, g_far_ci, g_ets_ci = self.bootstrap_metrics_ci(gfs, obs, th, n_bootstrap)
            metrics['GFS'][label]['POD_CI'] = g_pod_ci
            metrics['GFS'][label]['FAR_CI'] = g_far_ci
            metrics['GFS'][label]['ETS_CI'] = g_ets_ci
            
        return metrics
    @staticmethod
    def bootstrap_metrics_ci(pred, obs, threshold, n_bootstrap=1000, ci=0.95):
        """
        Use bootstrap to compute confidence intervals for POD, FAR, ETS.
        Parameters:
            pred: prediction precipitation array (any shape)
            obs: observed precipitation array (same shape as pred)
            threshold: precipitation threshold (mm/3h)
            n_bootstrap: number of resamples
            ci: confidence level (default 0.95)
        Returns:
            (pod_lower, pod_upper), (far_lower, far_upper), (ets_lower, ets_upper)
        """
        pred_f = pred.flatten()
        obs_f = obs.flatten()
        n = len(obs_f)
        pod_list = []
        far_list = []
        ets_list = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            p = pred_f[idx]
            o = obs_f[idx]

            p_bin = p >= threshold
            o_bin = o >= threshold

            tp = np.sum(p_bin & o_bin)
            fp = np.sum(p_bin & ~o_bin)
            fn = np.sum(~p_bin & o_bin)

            pod = tp / (tp + fn + 1e-8)
            far = fp / (tp + fp + 1e-8)

            total = len(o)
            random_hits = (tp + fp) * (tp + fn) / total if total > 0 else 0
            ets = (tp - random_hits) / (tp + fp + fn - random_hits + 1e-8) if (tp + fp + fn - random_hits) > 0 else 0.0

            pod_list.append(pod)
            far_list.append(far)
            ets_list.append(max(0.0, ets))

        lower = (1 - ci) / 2 * 100
        upper = (1 + ci) / 2 * 100
        pod_ci = (np.percentile(pod_list, lower), np.percentile(pod_list, upper))
        far_ci = (np.percentile(far_list, lower), np.percentile(far_list, upper))
        ets_ci = (np.percentile(ets_list, lower), np.percentile(ets_list, upper))

        return pod_ci, far_ci, ets_ci
    def _calculate_comprehensive_scores(self, pred_bin, obs_bin, pred_cont, obs_cont, threshold):
        """Calculate comprehensive score metrics"""
        pred_f, obs_f = pred_bin.flatten(), obs_bin.flatten()
        pred_c, obs_c = pred_cont.flatten(), obs_cont.flatten()

        tp = np.sum(pred_f & obs_f)
        fp = np.sum(pred_f & ~obs_f)
        fn = np.sum(~pred_bin & obs_bin)
        tn = np.sum(~pred_bin & ~obs_bin)
        total = len(obs_f)

        # Random hit expectation
        random_hits = (tp + fp) * (tp + fn) / total if total > 0 else 0

        # TS (Threat Score / Critical Success Index)
        if (tp + fp + fn) > 0:
            ts = tp / (tp + fp + fn)
        else:
            ts = 0.0

        # ETS (Equitable Threat Score)
        denom = tp + fp + fn - random_hits
        ets = (tp - random_hits) / denom if denom > 0 else 0

        # POD (Probability of Detection / Hit Rate)
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0

        # FAR (False Alarm Ratio)
        far = fp / (tp + fp) if (tp + fp) > 0 else 0

        # CSI (Critical Success Index, same as TS)
        csi = ts

        # Bias score
        bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0

        # MSE and RMSE for samples exceeding threshold
        mask = obs_c >= threshold
        if np.sum(mask) > 0:
            mse_thresh = np.mean((pred_c[mask] - obs_c[mask]) ** 2)
            rmse_thresh = np.sqrt(mse_thresh)
        else:
            mse_thresh = 0.0
            rmse_thresh = 0.0

        return {
            'TS': ts,
            'ETS': ets,
            'POD': pod,
            'FAR': far,
            'CSI': csi,
            'BIAS': bias,
            'MSE': mse_thresh,
            'RMSE': rmse_thresh,
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'Total': int(total)
        }

    def print_comprehensive_report(self, metrics, overall_improvement):
        """Print merged academic scorecard"""
        print("\n" + "="*130)
        print(f"#{' UNIFIED SCIENTIFIC PERFORMANCE SCORECARD (GFS vs. MODEL) ':^128}#")
        print("="*130)
        headers = ["Level", "TS_G", "TS_M", "ETS_G", "ETS_M", "POD_G", "POD_M", "FAR_G", "FAR_M", "RMSE_Imp%"]
        header_format = "{:<10} | {:>7} {:>7} | {:>7} {:>7} | {:>7} {:>7} | {:>7} {:>7} | {:>10}"
        print(header_format.format(*headers))
        print("-" * 130)

        for i, th in enumerate(self.thresholds):
            label = self.labels[i] if i < len(self.labels) else f'{th}mm'
            m, g = metrics['Model'].get(label, {}), metrics['GFS'].get(label, {})
            if not m: continue

            # Compute RMSE improvement for this level
            rmse_imp = (g.get('RMSE', 0) - m.get('RMSE', 0)) / (g.get('RMSE', 0) + 1e-8) * 100

            row = [label,
                   f"{g.get('TS',0):.3f}", f"{m.get('TS',0):.3f}",
                   f"{g.get('ETS',0):.3f}", f"{m.get('ETS',0):.3f}",
                   f"{g.get('POD',0):.3f}", f"{m.get('POD',0):.3f}",
                   f"{g.get('FAR',0):.3f}", f"{m.get('FAR',0):.3f}",
                   f"{rmse_imp:.2f}%"]
            print(header_format.format(*row))

        print("-" * 130)
        print(f"OVERALL MSE IMPROVEMENT: {overall_improvement:.2f}%")
        print("="*130 + "\n")

        # Continuous variable metrics
        if 'Continuous' in metrics['Model']:
            print(f"{'CONTINUOUS METRICS':^120}")
            print("-" * 120)
            cont_headers = ["Source", "MSE", "RMSE", "MAE", "CC", "Improvement%"]
            cont_format = "{:<10} | {:>12} | {:>10} | {:>10} | {:>8} | {:>12}"
            print(cont_format.format(*cont_headers))
            print("-" * 120)

            m_cont = metrics['Model']['Continuous']
            g_cont = metrics['GFS']['Continuous']

            mse_improvement = (g_cont['MSE'] - m_cont['MSE']) / g_cont['MSE'] * 100 if g_cont['MSE'] > 0 else 0
            rmse_improvement = (g_cont['RMSE'] - m_cont['RMSE']) / g_cont['RMSE'] * 100 if g_cont['RMSE'] > 0 else 0
            mae_improvement = (g_cont['MAE'] - m_cont['MAE']) / g_cont['MAE'] * 100 if g_cont['MAE'] > 0 else 0
            cc_improvement = (m_cont['CC'] - g_cont['CC']) / (abs(g_cont['CC']) + 1e-8) * 100

            print(cont_format.format(
                "Model",
                f"{m_cont['MSE']:.6f}",
                f"{m_cont['RMSE']:.4f}",
                f"{m_cont['MAE']:.4f}",
                f"{m_cont['CC']:.4f}",
                "-"
            ))

            print(cont_format.format(
                "GFS",
                f"{g_cont['MSE']:.6f}",
                f"{g_cont['RMSE']:.4f}",
                f"{g_cont['MAE']:.4f}",
                f"{g_cont['CC']:.4f}",
                "-"
            ))

            print("-" * 120)
            print(cont_format.format(
                "Improvement",
                f"{mse_improvement:.1f}%",
                f"{rmse_improvement:.1f}%",
                f"{mae_improvement:.1f}%",
                f"{cc_improvement:.1f}%",
                f"{overall_improvement:.1f}%"
            ))

        print("="*120 + "\n")

    # New: plot TS and ETS curves vs threshold
    def plot_ts_ets_curves(self, metrics, save_path='ts_ets_curves.png'):
        """Plot TS and ETS curves vs precipitation threshold"""
        thresholds = self.thresholds
        labels = self.labels[:len(thresholds)]

        # Extract data
        ts_model = []
        ets_model = []
        ts_gfs = []
        ets_gfs = []

        for i, th in enumerate(thresholds):
            label = labels[i] if i < len(labels) else f'Thresh_{th}mm'
            if label in metrics['Model']:
                ts_model.append(metrics['Model'][label]['TS'])
                ets_model.append(metrics['Model'][label]['ETS'])
                ts_gfs.append(metrics['GFS'][label]['TS'])
                ets_gfs.append(metrics['GFS'][label]['ETS'])

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

        # TS curve
        ax1.plot(thresholds, ts_model, 'o-', linewidth=3, markersize=8,
                label='Enhanced Model', color='#ff7f0e')
        ax1.plot(thresholds, ts_gfs, 's--', linewidth=2, markersize=8,
                label='GFS Baseline', color='#1f77b4')
        ax1.set_xlabel('Precipitation Threshold (mm/3h)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Threat Score (TS)', fontsize=12, fontweight='bold')
        ax1.set_title('TS vs. Precipitation Threshold', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(thresholds)
        ax1.set_xticklabels([f'{t}' for t in thresholds], rotation=45)

        # ETS curve
        ax2.plot(thresholds, ets_model, 'o-', linewidth=3, markersize=8,
                label='Enhanced Model', color='#ff7f0e')
        ax2.plot(thresholds, ets_gfs, 's--', linewidth=2, markersize=8,
                label='GFS Baseline', color='#1f77b4')
        ax2.set_xlabel('Precipitation Threshold (mm/3h)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Equitable Threat Score (ETS)', fontsize=12, fontweight='bold')
        ax2.set_title('ETS vs. Precipitation Threshold', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(thresholds)
        ax2.set_xticklabels([f'{t}' for t in thresholds], rotation=45)

        plt.suptitle('TS and ETS Performance Across Precipitation Thresholds',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f" TS/ETS curves saved to: {save_path}")

    def plot_comprehensive_analysis(self, metrics, save_path='comprehensive_analysis.png'):
        """Plot comprehensive analysis of POD, FAR, TS, ETS with subplot letters (a)-(d)"""
        thresholds = self.thresholds
        labels = self.labels[:len(thresholds)]

        # Extract data
        pod_model = []
        far_model = []
        ts_model = []
        ets_model = []

        for i, th in enumerate(thresholds):
            label = labels[i] if i < len(labels) else f'Thresh_{th}mm'
            if label in metrics['Model']:
                pod_model.append(metrics['Model'][label]['POD'])
                far_model.append(metrics['Model'][label]['FAR'])
                ts_model.append(metrics['Model'][label]['TS'])
                ets_model.append(metrics['Model'][label]['ETS'])

        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

        # 1. POD vs Threshold
        ax1.plot(thresholds, pod_model, 'o-', linewidth=3, markersize=8,
                color='#2ca02c', label='Model POD')
        ax1.set_xlabel('Threshold (mm/3h)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Probability of Detection (POD)', fontsize=11, fontweight='bold')
        ax1.set_title('(a) POD Across Different Thresholds', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.set_xticks(thresholds)
        ax1.set_xticklabels([f'{t}' for t in thresholds], rotation=45)

        # 2. FAR vs Threshold
        ax2.plot(thresholds, far_model, 's--', linewidth=3, markersize=8,
                color='#d62728', label='Model FAR')
        ax2.set_xlabel('Threshold (mm/3h)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('False Alarm Ratio (FAR)', fontsize=11, fontweight='bold')
        ax2.set_title('(b) FAR Across Different Thresholds', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.set_xticks(thresholds)
        ax2.set_xticklabels([f'{t}' for t in thresholds], rotation=45)

        # 3. TS vs Threshold
        ax3.plot(thresholds, ts_model, '^-', linewidth=3, markersize=8,
                color='#9467bd', label='Model TS')
        ax3.set_xlabel('Threshold (mm/3h)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Threat Score (TS)', fontsize=11, fontweight='bold')
        ax3.set_title('(c) TS Across Different Thresholds', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')
        ax3.set_xticks(thresholds)
        ax3.set_xticklabels([f'{t}' for t in thresholds], rotation=45)

        # 4. ETS vs Threshold
        ax4.plot(thresholds, ets_model, 'D-', linewidth=3, markersize=8,
                color='#8c564b', label='Model ETS')
        ax4.set_xlabel('Threshold (mm/3h)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Equitable Threat Score (ETS)', fontsize=11, fontweight='bold')
        ax4.set_title('(d) ETS Across Different Thresholds', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best')
        ax4.set_xticks(thresholds)
        ax4.set_xticklabels([f'{t}' for t in thresholds], rotation=45)

        plt.suptitle('Comprehensive Analysis: POD, FAR, TS, ETS vs Precipitation Thresholds',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f" Comprehensive analysis figure saved to: {save_path}")

        # Interpretation
        print("\n Metrics analysis interpretation:")
        print("-" * 50)

        # Find best threshold
        if ts_model:
            best_ts_idx = np.argmax(ts_model)
            best_ts_threshold = thresholds[best_ts_idx]
            print(f"1. Best TS: {ts_model[best_ts_idx]:.3f} (threshold={best_ts_threshold}mm)")

        if ets_model:
            best_ets_idx = np.argmax(ets_model)
            best_ets_threshold = thresholds[best_ets_idx]
            print(f"2. Best ETS: {ets_model[best_ets_idx]:.3f} (threshold={best_ets_threshold}mm)")

        # Trend analysis
        if len(pod_model) >= 2:
            pod_trend = "decreasing" if pod_model[-1] < pod_model[0] else "increasing"
            print(f"3. POD trend: {pod_trend} with increasing threshold")

        if len(far_model) >= 2:
            far_trend = "decreasing" if far_model[-1] < far_model[0] else "increasing"
            print(f"4. FAR trend: {far_trend} with increasing threshold")

        # Overall evaluation
        avg_ts = np.mean(ts_model)
        avg_ets = np.mean(ets_model)
        print(f"5. Average TS: {avg_ts:.3f}, Average ETS: {avg_ets:.3f}")

        if avg_ts > 0.3 and avg_ets > 0.2:
            print("6. Overall evaluation:  Model performance is good")
        elif avg_ts > 0.2 or avg_ets > 0.15:
            print("6. Overall evaluation:  Model performance is moderate, needs improvement")
        else:
            print("6. Overall evaluation:  Model performance needs significant improvement")
class ContinuousStormEventAnalyzer:
    """
    Continuous storm event analyzer - merge temporally continuous storm events into segments and compute different lead times
    (Research-enhanced version: lead times unified to 24h/72h/120h, automatically generate multi-period comparison plots)
    """

    @staticmethod
    def identify_continuous_storm_events(storm_events, max_gap_hours=6):
        """
        Identify continuous storm events and merge them into segments
        """
        if not storm_events:
            return []
      
        # Sort by time
        sorted_events = sorted(storm_events, key=lambda x: x.get('time', datetime(1900,1,1)))
      
        storm_segments = []
        current_segment = []
      
        for i, event in enumerate(sorted_events):
            if not event.get('time'):
                continue
          
            if not current_segment:
                current_segment.append(event)
                continue
          
            # Check time gap with previous event
            prev_event = current_segment[-1]
            prev_time = prev_event.get('time')
            curr_time = event.get('time')
          
            if not prev_time or not curr_time:
                current_segment.append(event)
                continue
          
            time_diff_hours = (curr_time - prev_time).total_seconds() / 3600
          
            if time_diff_hours <= max_gap_hours:
                # Time gap within threshold, merge into current segment
                current_segment.append(event)
            else:
                # Time gap exceeds threshold, end current segment and start new one
                if current_segment:
                    storm_segments.append(ContinuousStormEventAnalyzer._create_storm_segment(current_segment))
                current_segment = [event]
      
        # Process last segment
        if current_segment:
            storm_segments.append(ContinuousStormEventAnalyzer._create_storm_segment(current_segment))
      
        print(f" Identified {len(sorted_events)} original storm events, merged into {len(storm_segments)} continuous segments")
      
        return storm_segments
  
    @staticmethod
    def _create_storm_segment(events):
        """Create a storm event segment"""
        if not events:
            return None
      
        # Calculate segment basic information
        times = [e.get('time') for e in events if e.get('time')]
        intensities = [e.get('max_intensity', 0) for e in events]
        areas = [e.get('storm_area', 0) for e in events]
      
        start_time = min(times) if times else None
        end_time = max(times) if times else None
      
        # Calculate segment duration (hours)
        duration_hours = 0
        if start_time and end_time:
            duration_hours = (end_time - start_time).total_seconds() / 3600
      
        # Calculate average and maximum intensity
        avg_intensity = np.mean(intensities) if intensities else 0
        max_intensity = max(intensities) if intensities else 0
      
        # Determine storm level
        if max_intensity >= 50.0:
            storm_level = 'Extreme Storm'
        elif max_intensity >= 20.0:
            storm_level = 'Storm'
        elif max_intensity >= 10.0:
            storm_level = 'Heavy Rain'
        elif max_intensity >= 5.0:
            storm_level = 'Moderate Rain'
        else:
            storm_level = 'Light Rain'
      
        # Create segment
        segment = {
            'events': events,
            'start_time': start_time,
            'end_time': end_time,
            'duration_hours': duration_hours,
            'avg_intensity': avg_intensity,
            'max_intensity': max_intensity,
            'storm_level': storm_level,
            'event_count': len(events),
            'sample_indices': [e.get('index', 0) for e in events],
            'time_str': f"{start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%Y-%m-%d %H:%M')}" 
                        if start_time and end_time else "Unknown"
        }
      
        return segment

    @staticmethod
    def analyze_storm_segments(test_metrics_summary, sample_times=None, 
                            max_gap_hours=6, min_segment_duration=3):
        """
        Storm event segment in-depth analysis
        - Identify continuous events
        - Generate statistical consistency proof plot (frequency histogram)
        - Compute forecast performance at 24h/72h/120h lead times
        - Generate multi-period improvement comparison plot
        """
        print("\n" + "="*70)
        print(" In-depth analysis of continuous storm event segments (including statistical consistency verification)")
        print("="*70)
      
        storm_events = test_metrics_summary.get('all_storm_events', [])
        if not storm_events:
            print(" No storm event data found")
            return []
      
        # 1. Merge continuous events
        storm_segments = ContinuousStormEventAnalyzer.identify_continuous_storm_events(
            storm_events, max_gap_hours=max_gap_hours
        )
      
        # 2. Filter by duration (>=3h to capture local sudden storms)
        valid_segments = [s for s in storm_segments if s['duration_hours'] >= min_segment_duration]
        print(f" Obtained {len(valid_segments)} valid storm segments (duration≥{min_segment_duration}h)")
      
        # 3. Generate precipitation intensity frequency matching plot (statistical consistency proof)
        try:
            ContinuousStormEventAnalyzer.plot_statistical_consistency(
                test_metrics_summary['predictions'],
                test_metrics_summary['targets'],
                test_metrics_summary['gfs_baseline']
            )
        except Exception as e:
            print(f" Statistical consistency plot generation failed: {e}")

    @staticmethod
    def plot_statistical_consistency(preds, targets, gfs, save_path='storm_intensity_consistency.png'):
        """Histogram comparison to prove model frequency of heavy rain is closer to observation"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 7), dpi=300)

        threshold = 10.0
        p_flat = preds[preds > threshold].flatten()
        t_flat = targets[targets > threshold].flatten()
        g_flat = gfs[gfs > threshold].flatten()

        bins = np.linspace(threshold, 50, 25)

        plt.hist(t_flat, bins=bins, alpha=0.3, label='ERA5 Truth (Target)', color='green', density=True, histtype='stepfilled')
        plt.hist(g_flat, bins=bins, alpha=0.4, label='GFS Baseline', color='blue', density=True, histtype='step', linewidth=2)
        plt.hist(p_flat, bins=bins, alpha=0.5, label='Model Corrected', color='red', density=True, histtype='step', linewidth=2.5)

        plt.title("Statistical Consistency Analysis: Frequency Matching for Heavy Rain (>10mm)", fontsize=14, fontweight='bold')
        plt.xlabel("Precipitation Intensity (mm/3h)", fontsize=12)
        plt.ylabel("Probability Density", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.3)

        fig = plt.gcf()
        base_path = save_path.replace('.png', '').replace('.pdf', '')
        save_fig_multi(fig, base_path, dpi=300)
        plt.close(fig)
    @staticmethod
    def create_segment_visualizations(storm_segments, test_metrics_summary, save_dir='storm_segments'):
        """
        Create visualizations for continuous storm event segments
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
      
        print(f"\n Generating visualizations for continuous storm event segments...")
      
        # Get data
        predictions = test_metrics_summary.get('predictions')
        targets = test_metrics_summary.get('targets')
        gfs_baseline = test_metrics_summary.get('gfs_baseline')
      
        for i, segment in enumerate(storm_segments):
            print(f"   Analyzing segment {i+1}/{len(storm_segments)}: {segment['time_str']}")
          
            # Create segment chart
            ContinuousStormEventAnalyzer._create_single_segment_chart(
                segment=segment,
                predictions=predictions,
                targets=targets,
                gfs_baseline=gfs_baseline,
                segment_number=i+1,
                save_dir=save_dir
            )
      
        print(f" Continuous storm event segment analysis complete, results saved to: {save_dir}")
  
    @staticmethod
    def _create_single_segment_chart(segment, predictions, targets, gfs_baseline, 
                                     segment_number, save_dir):
        sample_indices = segment.get('sample_indices', [])
        if not sample_indices: return
      
        # 1. Identify peak precipitation time
        max_intensity_idx = sample_indices[0]
        max_val = 0
        for idx in sample_indices:
            if idx < len(targets):
                curr_max = np.max(targets[idx])
                if curr_max > max_val:
                    max_val = curr_max
                    max_intensity_idx = idx

        # 2. Perform 8x spatial smoothing (similar to scientific_spatial_smooth)
        from scipy.ndimage import zoom
        zoom_f = 8
        obs_raw = targets[max_intensity_idx]
        gfs_raw = gfs_baseline[max_intensity_idx]
        mod_raw = predictions[max_intensity_idx]
      
        obs_smooth = zoom(obs_raw, zoom_f, order=3)
        gfs_smooth = zoom(gfs_raw, zoom_f, order=3)
        mod_smooth = zoom(mod_raw, zoom_f, order=3)
      
        # 3. Plot configuration
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        fig, axes = plt.subplots(2, 3, figsize=(20, 14), dpi=300)
        cmap, norm = ResearchVisualizer.get_professional_labels()
        vmax = max(30.0, max_val * 1.1)

        # A. Top-left: precipitation evolution (keep original detailed line plot)
        ContinuousStormEventAnalyzer._plot_event_evolution(axes[0, 0], segment, predictions, targets, gfs_baseline, sample_indices)

        # B. Top-middle: high-res observation heatmap
        ax_obs = axes[0, 1]
        im1 = ax_obs.imshow(obs_smooth, cmap=cmap, norm=norm, interpolation='bilinear')
        ax_obs.set_title(f'ERA5 Observation (Peak Phase)\nMax: {max_val:.1f} mm/3h', fontweight='bold', fontsize=13)
        plt.colorbar(im1, ax=ax_obs, label='mm/3h', fraction=0.046, pad=0.04)
        ax_obs.axis('off')

        # D. Bottom-left: intensity distribution histogram
        ax_hist = axes[1, 0]
        intensities = [e.get('max_intensity', 0) for e in segment.get('events', [])]
        ax_hist.hist(intensities, bins=10, alpha=0.7, color='steelblue', edgecolor='black')
        ax_hist.set_title('Event Intensity Distribution', fontweight='bold')
        ax_hist.set_xlabel('mm/3h')

        # E. Bottom-middle: smoothed GFS raw forecast
        ax_gfs = axes[1, 1]
        im2 = ax_gfs.imshow(gfs_smooth, cmap=cmap, norm=norm, interpolation='bilinear')
        ax_gfs.set_title(f'GFS Baseline Forecast', fontweight='bold')
        plt.colorbar(im2, ax=ax_gfs, fraction=0.046, pad=0.04)
        ax_gfs.axis('off')

        # F. Bottom-right: smoothed model corrected forecast
        ax_mod = axes[1, 2]
        im3 = ax_mod.imshow(mod_smooth, cmap=cmap, norm=norm, interpolation='bilinear')
        ax_mod.set_title(f'Model Corrected Forecast', fontweight='bold', color='red')
        plt.colorbar(im3, ax=ax_mod, fraction=0.046, pad=0.04)
        ax_mod.axis('off')

        plt.suptitle(f"Storm Segment Analysis #{segment_number} | {segment['time_str']}\nSpatial Correlation and Peak Intensity Correction", 
                     fontsize=20, fontweight='bold', y=0.98)
      
        plt.tight_layout()
        filename = f"storm_segment_{segment_number}_{segment['start_time'].strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _plot_event_evolution(ax, segment, predictions, targets, gfs_baseline, sample_indices):
        """
        Improved case time series plot, clearly showing model's correction of precipitation peaks
        """
        times = []
        obs_intensities = []
        gfs_intensities = []
        model_intensities = []
      
        # 1. Extract sequence data
        for idx in sample_indices:
            if idx < len(targets):
                times.append(idx)
                obs_intensities.append(np.max(targets[idx]))
                if gfs_baseline is not None:
                    gfs_intensities.append(np.max(gfs_baseline[idx]))
                if predictions is not None:
                    model_intensities.append(np.max(predictions[idx]))
      
        if not times: return

        # 2. Plot with distinct colors and line styles
        ax.plot(range(len(times)), obs_intensities, 'o-', color='#2ca02c', linewidth=2.5, markersize=7, label='ERA5 Observed')
        ax.plot(range(len(times)), gfs_intensities, 's--', color='#1f77b4', linewidth=1.5, markersize=5, label='GFS Baseline', alpha=0.7)
        ax.plot(range(len(times)), model_intensities, '^-', color='#d62728', linewidth=2.5, markersize=8, label='Model Corrected')
      
        # 3. Adjust Y-axis range dynamically
        all_vals = obs_intensities + gfs_intensities + model_intensities
        y_min = max(0, min(all_vals) * 0.9)
        y_max = max(all_vals) * 1.25
        ax.set_ylim(y_min, y_max)

        # 4. Annotation
        ax.set_xlabel('Time Steps (Relative)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Max Intensity (mm/3h)', fontsize=10, fontweight='bold')
        ax.set_title('Intensity Peak Correction Evolution', fontsize=12, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)
      
        # Annotate max improvement
        err_gfs = np.abs(np.array(gfs_intensities) - np.array(obs_intensities))
        err_mod = np.abs(np.array(model_intensities) - np.array(obs_intensities))
        if np.mean(err_gfs) > 0:
            imp = (np.mean(err_gfs) - np.mean(err_mod)) / np.mean(err_gfs) * 100
            ax.text(0.05, 0.9, f'MAE Imp: {imp:.1f}%', transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'), fontweight='bold', color='red')

    @staticmethod
    def _plot_accumulated_precipitation(ax, segment, targets, sample_indices):
        """Plot accumulated precipitation"""
      
        if not sample_indices:
            return
      
        accumulated = 0
        accumulated_list = []
      
        for idx in sample_indices:
            if idx < len(targets):
                region_avg = np.mean(targets[idx][targets[idx] > 0.1]) if np.any(targets[idx] > 0.1) else 0
                accumulated += region_avg
                accumulated_list.append(accumulated)
      
        if accumulated_list:
            ax.plot(range(len(accumulated_list)), accumulated_list, 'o-', 
                   linewidth=2, markersize=6, color='#d62728')
            ax.set_xlabel('Time Step', fontweight='bold')
            ax.set_ylabel('Accumulated Precipitation (mm)', fontweight='bold')
            ax.set_title('Accumulated Precipitation Evolution', fontweight='bold')
            ax.grid(True, alpha=0.3)
          
            total_accumulated = accumulated_list[-1] if accumulated_list else 0
            ax.text(0.98, 0.02, f'Total: {total_accumulated:.1f}mm', 
                   transform=ax.transAxes, fontsize=10, 
                   horizontalalignment='right', verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
  
    @staticmethod
    def _generate_segments_report(storm_segments, save_path='storm_segments_report.txt'):
        """Generate specialized report for event segments"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" Continuous Storm Event Segment Analysis Report\n")
            f.write("=" * 80 + "\n\n")
          
            f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Segments: {len(storm_segments)}\n\n")
          
            for i, segment in enumerate(storm_segments):
                f.write(f"Segment {i+1}:\n")
                f.write(f"  Time Range: {segment['time_str']}\n")
                f.write(f"  Duration: {segment['duration_hours']:.1f} hours\n")
                f.write(f"  Max Intensity: {segment['max_intensity']:.1f} mm\n")
                f.write(f"  Avg Intensity: {segment['avg_intensity']:.1f} mm\n")
                f.write(f"  Storm Level: {segment['storm_level']}\n")
                f.write(f"  Event Count: {segment['event_count']}\n")
              
                lead_perf = segment.get('lead_time_performance', {})
                if lead_perf:
                    f.write("  Lead Time Performance:\n")
                    for lead_time, perf in lead_perf.items():
                        f.write(f"    {lead_time}: Model Error={perf['Model Avg Error']:.2f}mm, "
                               f"GFS Error={perf['GFS Avg Error']:.2f}mm, "
                               f"Improvement Rate={perf['Improvement Rate']:.1f}%\n")
              
                f.write("\n")
          
            # Summary statistics
            f.write("\n" + "=" * 50 + "\n")
            f.write("Summary Statistics:\n")
            f.write("=" * 50 + "\n")
          
            durations = [s['duration_hours'] for s in storm_segments]
            intensities = [s['max_intensity'] for s in storm_segments]
          
            if durations:
                f.write(f"Average Duration: {np.mean(durations):.1f} hours\n")
                f.write(f"Shortest Duration: {np.min(durations):.1f} hours\n")
                f.write(f"Longest Duration: {np.max(durations):.1f} hours\n")
          
            if intensities:
                f.write(f"Average Max Intensity: {np.mean(intensities):.1f} mm\n")
                f.write(f"Strongest Event Intensity: {np.max(intensities):.1f} mm\n")
                f.write(f"Weakest Event Intensity: {np.min(intensities):.1f} mm\n")
      
        print(f" Segment-specific report saved: {save_path}")

    # ========== New: ranking case plot (with axis format fixed) ==========
    @staticmethod
    def create_top_rank_gallery(test_metrics_summary, top_n=10, save_dir='top_rank_cases'):
        """
        Generate ranking case time series plots: show three curves ERA5, GFS, Model.
        """
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import os
        import numpy as np

        os.makedirs(save_dir, exist_ok=True)

        preds = test_metrics_summary.get('predictions')
        targets = test_metrics_summary.get('targets')
        gfs_baseline = test_metrics_summary.get('gfs_baseline')
        rank_list = test_metrics_summary.get('storm_rmse_improvements', [])
        sample_times = test_metrics_summary.get('sample_times', [])

        print(f" Generating triple-line top {top_n} storm event plots (including GFS baseline)...")

        for rank, entry in enumerate(rank_list[:top_n], 1):
            idx = entry['index']
            start_win = max(0, idx - 4)
            end_win = min(len(targets), idx + 5)

            time_win = sample_times[start_win:end_win]
            obs_win = [np.max(targets[i]) for i in range(start_win, end_win)]
            gfs_win = [np.max(gfs_baseline[i]) for i in range(start_win, end_win)]
            mod_win = [np.max(preds[i]) for i in range(start_win, end_win)]

            fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

            ax.plot(time_win, obs_win, 'g-s', lw=2.5, markersize=8, label='ERA5 Observation', alpha=0.8)
            ax.plot(time_win, gfs_win, 'b--d', lw=1.5, markersize=6, label='GFS Baseline (Raw)', alpha=0.6)
            ax.plot(time_win, mod_win, 'r-^', lw=3.0, markersize=9, label='Model Corrected', alpha=0.9)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
            plt.xticks(rotation=30, fontsize=10)
            ax.set_ylabel("3-hour Cumulative Precipitation (mm/3h)", fontweight='bold', fontsize=12)
            ax.set_xlabel("Storm Event Time (UTC)", fontweight='bold', fontsize=12)

            ax.set_title(f"Rank {rank}: Peak Intensity Correction Analysis\n"
                        f"RMSE Improvement: {entry['rmse_imp']:.2f}% | Max: {np.max(obs_win):.1f}mm",
                        fontweight='bold', fontsize=14, pad=15)

            ax.legend(loc='upper left', frameon=True, shadow=True)
            ax.grid(True, ls=':', alpha=0.5)

            peak_idx = np.argmax(obs_win)
            ax.annotate(f'Peak: {obs_win[peak_idx]:.1f}', xy=(time_win[peak_idx], obs_win[peak_idx]),
                        xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'))

            base_path = f"{save_dir}/rank_{rank:02d}_triple_line_comparison"
            save_fig_multi(fig, base_path, dpi=300)
            plt.close(fig)

        print(f" Top rank triple-line comparison plots saved to {save_dir}")
def calibrate_gate_threshold_on_val(model, val_loader, device, scaling_factor=1.0,
                                    search=np.linspace(0.12, 0.28, 9),
                                    power_search=(0.70, 0.80, 0.90, 1.00),
                                    storm_gate_p_search=None,
                                    min_pod20=0.35,
                                    min_pod15=0.30,
                                    max_far20=0.90):
    """
    Calibrate gate threshold and power on validation set, using model's actual rain_prob and storm_logits outputs
    """
    if storm_gate_p_search is None:
        storm_gate_p_list = [float(GATE_CFG.get("storm_gate_p", 0.40))]
    else:
        storm_gate_p_list = storm_gate_p_search

    model.eval()
    best_valid = {"th": None, "pow": None, "gate_p": None, "score": -1e9}
    best_any = {"th": None, "pow": None, "gate_p": None, "score": -1e9,
                "pod20": 0.0, "far20": 1.0, "ets20": -1.0}

    with torch.no_grad():
        for th_base in search:
            for gp in power_search:
                for sg in storm_gate_p_list:
                    stat = {
                        15.0: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
                        20.0: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
                    }

                    for inputs, targets_scaled in val_loader:
                        inputs = inputs.to(device, non_blocking=True)
                        targets_scaled = targets_scaled.to(device, non_blocking=True)

                        residual, rain_prob, storm_logits, _ = model(
                            inputs, return_residual=True, return_storm_logits=True
                        )
                        gfs_base = inputs[:, -1, 5:6, :, :]

                        pred_abs, _ = compute_gated_precip_prediction(
                            gfs_base=gfs_base,
                            residual=residual,
                            rain_prob=rain_prob,
                            storm_logits=storm_logits,
                            threshold_base=float(th_base),
                            gate_power=float(gp),
                            min_rain_value=float(GATE_CFG.get("min_rain_value", 0.10)),
                            max_precip=float(GATE_CFG.get("max_precip", 250.0)),
                            adaptive=True,
                            hard_gate=True,
                            storm_gate_p=float(sg)
                        )

                        true_abs = gfs_base.expand(-1, targets_scaled.shape[1], -1, -1) + targets_scaled / scaling_factor
                        pred_last = pred_abs[:, -1].cpu().numpy()
                        true_last = true_abs[:, -1].cpu().numpy()

                        for th in [15.0, 20.0]:
                            p = (pred_last >= th)
                            o = (true_last >= th)
                            stat[th]['TP'] += int(np.sum(p & o))
                            stat[th]['FP'] += int(np.sum(p & ~o))
                            stat[th]['FN'] += int(np.sum(~p & o))
                            stat[th]['TN'] += int(np.sum(~p & ~o))

                    def calc_ets_pod_far(s):
                        TP, FP, FN, TN = s['TP'], s['FP'], s['FN'], s['TN']
                        total = TP + FP + FN + TN
                        rh = (TP + FP) * (TP + FN) / max(total, 1)
                        ets = (TP - rh) / (TP + FP + FN - rh + 1e-8)
                        pod = TP / (TP + FN + 1e-8)
                        far = FP / (TP + FP + 1e-8)
                        return float(ets), float(pod), float(far)

                    ets15, pod15, far15 = calc_ets_pod_far(stat[15.0])
                    ets20, pod20, far20 = calc_ets_pod_far(stat[20.0])

                    score = (5.0 * pod20 + 3.0 * pod15 + 2.0 * ets20 + 1.0 * ets15 - 0.1 * far20)

                    if score > best_any["score"]:
                        best_any = {
                            "th": float(th_base), "pow": float(gp), "gate_p": float(sg),
                            "score": float(score),
                            "ets20": ets20, "pod20": pod20, "far20": far20,
                            "ets15": ets15, "pod15": pod15, "far15": far15
                        }

                    valid = (pod20 >= min_pod20) and (pod15 >= min_pod15) and (far20 <= max_far20)
                    if valid and score > best_valid["score"]:
                        best_valid = {
                            "th": float(th_base), "pow": float(gp), "gate_p": float(sg),
                            "score": float(score),
                            "ets20": ets20, "pod20": pod20, "far20": far20,
                            "ets15": ets15, "pod15": pod15, "far15": far15
                        }

    if best_valid["th"] is not None:
        out = dict(best_valid)
        out["is_valid"] = True
        print(f" Gate calibration(valid): th={out['th']:.3f}, power={out['pow']:.2f}, gate_p={out['gate_p']:.2f}, "
              f"POD20={out['pod20']:.4f}, ETS20={out['ets20']:.4f}")
        return out
    else:
        out = dict(best_any)
        out["is_valid"] = False
        print(f" Gate calibration(fallback to best POD): th={out['th']:.3f}, power={out['pow']:.2f}, gate_p={out['gate_p']:.2f}, "
              f"POD20={out['pod20']:.4f}, ETS20={out['ets20']:.4f}")
        return out
def enhanced_comprehensive_evaluation(model, test_loader, device, scaling_factor=1.0, storm_gate_p=None):
    """
    Research-grade full metrics evaluation (using gate mechanism)
    """
    print(" Starting research-grade full metrics evaluation (including gate)...")
    model.eval()

    if storm_gate_p is None:
        storm_gate_p = float(GATE_CFG.get("storm_gate_p", 0.30))
    th_base = float(GATE_CFG.get("threshold_base", 0.22))
    gate_pow = float(GATE_CFG.get("gate_power", 0.90))

    all_preds, all_obs, all_gfs = [], [], []
    all_probs, all_targets_bin = [], []

    thresholds = PRECIP_THRESHOLDS
    levels = PRECIP_LEVELS
    metrics_by_level = {lvl: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for lvl in levels}

    total_fss = 0.0
    batch_count = 0

    with torch.no_grad():
        for inputs, targets_scaled in test_loader:
            inputs = inputs.to(device)
            targets_scaled = targets_scaled.to(device)

            residual, rain_prob, storm_logits, _ = model(
                inputs, return_residual=True, return_storm_logits=True
            )
            gfs_base = inputs[:, -1, 5:6, :, :]

            pred_abs, gate = compute_gated_precip_prediction(
                gfs_base=gfs_base,
                residual=residual,
                rain_prob=rain_prob,
                storm_logits=storm_logits,
                threshold_base=th_base,
                gate_power=gate_pow,
                min_rain_value=float(GATE_CFG.get("min_rain_value", 0.10)),
                max_precip=float(GATE_CFG.get("max_precip", 250.0)),
                adaptive=True,
                hard_gate=True,
                storm_gate_p=float(storm_gate_p)
            )

            true_abs = gfs_base.expand(-1, targets_scaled.shape[1], -1, -1) + targets_scaled / scaling_factor
            gfs_expand = gfs_base.expand(-1, targets_scaled.shape[1], -1, -1)

            pred_np = pred_abs.cpu().numpy()
            obs_np = true_abs.cpu().numpy()
            gfs_np = gfs_expand.cpu().numpy()
            prob_np = gate.cpu().numpy()      # gate can serve as proxy for precipitation probability

            all_preds.append(pred_np)
            all_obs.append(obs_np)
            all_gfs.append(gfs_np)
            all_probs.append(prob_np)
            all_targets_bin.append((obs_np > 0.1).astype(np.float32))

            # Compute FSS using final precipitation
            try:
                fss_val = ScientificEvaluator.calculate_fss(pred_abs[:, -1], true_abs[:, -1], threshold=0.1)
                total_fss += float(fss_val)
                batch_count += 1
            except Exception:
                pass

            # Statistics by level
            for i, th in enumerate(thresholds):
                lvl = levels[i]
                p_bin = (pred_np[:, -1] >= th)
                o_bin = (obs_np[:, -1] >= th)

                metrics_by_level[lvl]['TP'] += int(np.sum(p_bin & o_bin))
                metrics_by_level[lvl]['FP'] += int(np.sum(p_bin & ~o_bin))
                metrics_by_level[lvl]['FN'] += int(np.sum(~p_bin & o_bin))
                metrics_by_level[lvl]['TN'] += int(np.sum(~p_bin & ~o_bin))

    full_preds = np.concatenate(all_preds, axis=0)
    full_obs   = np.concatenate(all_obs, axis=0)
    full_gfs   = np.concatenate(all_gfs, axis=0)
    full_probs = np.concatenate(all_probs, axis=0)
    full_targets_bin = np.concatenate(all_targets_bin, axis=0)

    mse = float(np.mean((full_preds - full_obs) ** 2))
    mae = float(np.mean(np.abs(full_preds - full_obs)))
    precip_ratio = float(np.mean(full_obs >= 0.1))

    final_metrics = {}
    for lvl in levels:
        m = metrics_by_level[lvl]
        total = m['TP'] + m['FP'] + m['FN'] + m['TN']
        r_hits = (m['TP'] + m['FP']) * (m['TP'] + m['FN']) / total if total > 0 else 0
        ets = (m['TP'] - r_hits) / (m['TP'] + m['FP'] + m['FN'] - r_hits + 1e-8)
        pod = m['TP'] / (m['TP'] + m['FN'] + 1e-8)
        far = m['FP'] / (m['TP'] + m['FP'] + 1e-8)
        final_metrics[lvl] = {
            'ETS': max(0.0, float(ets)),
            'POD': float(pod),
            'FAR': float(far),
            'TP': int(m['TP']),
            'FP': int(m['FP']),
            'FN': int(m['FN']),
            'TN': int(m['TN'])
        }

    fss = float(total_fss / (batch_count + 1e-8))
    print(f" Evaluation complete: MSE={mse:.4f}, MAE={mae:.4f}, FSS={fss:.4f}")

    return {
        'mse': mse,
        'mae': mae,
        'rmse': float(np.sqrt(mse)),
        'precip_ratio': precip_ratio,
        'fss': fss,
        'metrics_by_level': final_metrics,
        'probs': full_probs,
        'targets_bin': full_targets_bin,
        'predictions': full_preds[:, -1],
        'targets': full_obs[:, -1],
        'gfs_baseline': full_gfs[:, -1]
    }
def evaluate_subregion_performance(model, test_loader, device, dem_tensor, scaling_factor=1.0):
    """
    Terrain subregion evaluation (optimized version)
    """
    # Fix: if DEM is disabled, skip evaluation to avoid NoneType error
    if dem_tensor is None:
        print("\n Current model does not use DEM features, skipping terrain subregion performance evaluation.")
        return None

    model.eval()
    slope = dem_tensor[1].cpu().numpy()
    slope_threshold = np.percentile(slope, 65)
    mountain_mask = slope >= slope_threshold
    plain_mask = ~mountain_mask

    mountain_err_gfs, mountain_err_mod = [], []
    plain_err_gfs, plain_err_mod = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            pred_abs, true_abs, gfs_expand, _, _, _, _ = get_model_eval_tensors(
                model=model, inputs=inputs, targets_scaled=targets,
                scaling_factor=scaling_factor, max_precip=200.0
            )
            pred_last = pred_abs[:, -1].cpu().numpy()
            obs_last = true_abs[:, -1].cpu().numpy()
            gfs_last = gfs_expand[:, -1].cpu().numpy()

            for b in range(pred_last.shape[0]):
                rainy_mask = obs_last[b] >= 0.1
                mountain_rainy = mountain_mask & rainy_mask
                plain_rainy = plain_mask & rainy_mask

                if np.any(mountain_rainy):
                    mountain_err_gfs.append(np.mean(np.abs(gfs_last[b][mountain_rainy] - obs_last[b][mountain_rainy])))
                    mountain_err_mod.append(np.mean(np.abs(pred_last[b][mountain_rainy] - obs_last[b][mountain_rainy])))

                if np.any(plain_rainy):
                    plain_err_gfs.append(np.mean(np.abs(gfs_last[b][plain_rainy] - obs_last[b][plain_rainy])))
                    plain_err_mod.append(np.mean(np.abs(pred_last[b][plain_rainy] - obs_last[b][plain_rainy])))

    mountain_imp = (np.mean(mountain_err_gfs) - np.mean(mountain_err_mod)) / (np.mean(mountain_err_gfs) + 1e-8) * 100 if mountain_err_gfs else 0.0
    plain_imp = (np.mean(plain_err_gfs) - np.mean(plain_err_mod)) / (np.mean(plain_err_gfs) + 1e-8) * 100 if plain_err_gfs else 0.0

    print("\n" + "=" * 50)
    print(" Terrain subregion correction performance report (Rainy Pixels Only)")
    print(f"Mountain threshold (slope percentile 65%): {slope_threshold:.4f}")
    print(f"Mountain area improvement: {mountain_imp:.2f}%")
    print(f"Plain area improvement: {plain_imp:.2f}%")
    print("=" * 50)

    return {
        'mountain_improvement': mountain_imp, 'plain_improvement': plain_imp,
        'mountain_gfs_mae': np.mean(mountain_err_gfs), 'mountain_model_mae': np.mean(mountain_err_mod)
    }
def compare_with_gfs_baseline(model, test_loader, device, scaling_factor=1.0, storm_gate_p=None):
    print(" Comparison with GFS baseline (unified gate interface)...")
    model.eval()

    if storm_gate_p is None:
        storm_gate_p = float(GATE_CFG.get("storm_gate_p", 0.30))

    th_base = float(GATE_CFG.get("threshold_base", 0.22))
    gate_pow = float(GATE_CFG.get("gate_power", 0.90))

    all_gfs_errors = []
    all_model_errors = []
    all_target_values = []

    with torch.no_grad():
        for inputs, targets_scaled in test_loader:
            if inputs is None or targets_scaled is None:
                continue

            inputs = inputs.to(device)
            targets_scaled = targets_scaled.to(device)

            pred_abs, true_abs, gfs_expand, _, *rest = get_model_eval_tensors(
                model=model,
                inputs=inputs,
                targets_scaled=targets_scaled,
                scaling_factor=scaling_factor,
                rain_prob_threshold=th_base,
                gate_power=gate_pow,
                min_rain_value=float(GATE_CFG.get("min_rain_value", 0.10)),
                max_precip=float(GATE_CFG.get("max_precip", 250.0)),
                adaptive=True,
                hard_gate=True,
                storm_gate_p=float(storm_gate_p)
            )

            model_last = pred_abs[:, -1].cpu().numpy()
            gfs_last = gfs_expand[:, -1].cpu().numpy()
            target_last = true_abs[:, -1].cpu().numpy()

            gfs_error = np.abs(gfs_last - target_last)
            model_error = np.abs(model_last - target_last)

            all_gfs_errors.append(gfs_error.flatten())
            all_model_errors.append(model_error.flatten())
            all_target_values.append(target_last.flatten())

    if not all_gfs_errors:
        print(" No valid comparison data")
        return {}

    gfs_errors = np.concatenate(all_gfs_errors)
    model_errors = np.concatenate(all_model_errors)
    target_values = np.concatenate(all_target_values)

    gfs_mse = np.mean(gfs_errors ** 2)
    model_mse = np.mean(model_errors ** 2)
    improvement_percentage = (gfs_mse - model_mse) / gfs_mse * 100 if gfs_mse > 0 else 0

    print(f"   GFS baseline MSE: {gfs_mse:.6f}")
    print(f"   Model corrected MSE: {model_mse:.6f}")
    print(f"   Overall improvement: {improvement_percentage:.2f}%")

    intensity_bins = [0, 0.1, 3.0, 10.0, 20.0, 100.0]
    intensity_labels = ['No Rain', 'Light', 'Moderate', 'Heavy', 'Storm']
    intensity_improvements = {}

    for i in range(len(intensity_bins) - 1):
        mask = (target_values >= intensity_bins[i]) & (target_values < intensity_bins[i + 1])
        if np.sum(mask) > 0:
            g_mse = np.mean(gfs_errors[mask] ** 2)
            m_mse = np.mean(model_errors[mask] ** 2)
            intensity_improvements[intensity_labels[i]] = {
                'improvement_rate_mse': (g_mse - m_mse) / (g_mse + 1e-8) * 100
            }

    return {
        'improvement_percentage': improvement_percentage,
        'intensity_improvements': intensity_improvements
    }
def quick_storm_evaluation(model, test_loader, device, scaling_factor=1.0, storm_gate_p=None):
    if storm_gate_p is None:
        storm_gate_p = float(GATE_CFG.get("storm_gate_p", 0.30))

    print("\n Executing quick multi-level precipitation capture capability (POD) assessment (unified gate interface)...")
    model.eval()

    eval_thresholds = [5.0, 10.0, 20.0]
    storm_stats = {th: [] for th in eval_thresholds}

    th_base = float(GATE_CFG.get("threshold_base", 0.22))
    gate_pow = float(GATE_CFG.get("gate_power", 0.90))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if batch_idx >= 50:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            pred_abs, true_abs, _, _, *rest = get_model_eval_tensors(
                model=model,
                inputs=inputs,
                targets_scaled=targets,
                scaling_factor=scaling_factor,
                rain_prob_threshold=th_base,
                gate_power=gate_pow,
                min_rain_value=float(GATE_CFG.get("min_rain_value", 0.10)),
                max_precip=float(GATE_CFG.get("max_precip", 250.0)),
                adaptive=True,
                hard_gate=True,
                storm_gate_p=float(storm_gate_p)
            )

            for th in eval_thresholds:
                p_bin = (pred_abs[:, -1] >= th)
                t_bin = (true_abs[:, -1] >= th)
                if t_bin.sum() > 0:
                    pod = (p_bin & t_bin).sum().item() / (t_bin.sum().item() + 1e-8)
                    storm_stats[th].append(pod)

    for th, pods in storm_stats.items():
        if pods:
            print(f"    [≥{th}mm] POD: {np.mean(pods):.4f}")
        else:
            print(f"    [≥{th}mm] POD: insufficient samples")

    return storm_stats
# ==================== Main function: generate paper-ready figures ====================
if __name__ == '__main__':
    # Set global random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    results = run_residual_experiment_enhanced()
  
    if results:
        eval_results = results.get('test_metrics_summary', {})
        preds = eval_results.get('predictions')
        targets = eval_results.get('targets')
        gfs_base = eval_results.get('gfs_baseline')

        if preds is not None and targets is not None:
            # Single authoritative evaluator to eliminate metric inconsistency
            verifier = ScientificVerification(thresholds=PRECIP_THRESHOLDS, levels=PRECIP_LEVELS)
            final_metrics = verifier.evaluate_with_ci(preds, targets, gfs_base, n_bootstrap=500)
            
            p_f, o_f, g_f = preds.flatten(), targets.flatten(), gfs_base.flatten()
            
            def calc_cc(a, b):
                return np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else 0.0

            mse_m, mse_g = np.mean((p_f - o_f)**2), np.mean((g_f - o_f)**2)
            mae_m, mae_g = np.mean(np.abs(p_f - o_f)), np.mean(np.abs(g_f - o_f))
            cc_m, cc_g = calc_cc(p_f, o_f), calc_cc(g_f, o_f)

            print("\n\n" + "="*80)
            print(" PAPER-READY TABLES GENERATED (Copy to Word)")
            print("="*80 + "\n")

            # ---------------- TABLE 1: Continuous metrics ----------------
            print("### Table 1: Global Continuous Verification Metrics (2024-2025)\n")
            print("| Metric | Raw GFS | Corrected Model | Improvement (%) |")
            print("| :--- | :---: | :---: | :---: |")
            print(f"| MSE (mm²/3h) | {mse_g:.4f} | {mse_m:.4f} | +{(mse_g-mse_m)/mse_g*100:.2f}% |")
            print(f"| RMSE (mm/3h) | {np.sqrt(mse_g):.4f} | {np.sqrt(mse_m):.4f} | +{(np.sqrt(mse_g)-np.sqrt(mse_m))/np.sqrt(mse_g)*100:.2f}% |")
            print(f"| MAE (mm/3h) | {mae_g:.4f} | {mae_m:.4f} | +{(mae_g-mae_m)/mae_g*100:.2f}% |")
            print(f"| Spatial CC | {cc_g:.4f} | {cc_m:.4f} | +{(cc_m-cc_g)/cc_g*100:.2f}% |")
            print("\n")

            # ---------------- TABLE 2: Categorical performance ----------------
            print("### Table 2: Categorical Performance across Precipitation Intensities\n")
            print("| Intensity Level | Threshold | POD (GFS / Model) | FAR (GFS / Model) | ETS (GFS / Model) |")
            print("| :--- | :---: | :---: | :---: | :---: |")
            for lvl, th in zip(PRECIP_LEVELS, PRECIP_THRESHOLDS):
                m = final_metrics['Model'].get(lvl, {})
                g = final_metrics['GFS'].get(lvl, {})
                
                def fmt_ci(val, ci_tuple):
                    return f"{val:.3f} ({ci_tuple[0]:.3f}-{ci_tuple[1]:.3f})"

                if m and g:
                    print(f"| {lvl} | >= {th}mm | "
                        f"{fmt_ci(g['POD'], g['POD_CI'])} / **{fmt_ci(m['POD'], m['POD_CI'])}** | "
                        f"{fmt_ci(g['FAR'], g['FAR_CI'])} / **{fmt_ci(m['FAR'], m['FAR_CI'])}** | "
                        f"{fmt_ci(g['ETS'], g['ETS_CI'])} / **{fmt_ci(m['ETS'], m['ETS_CI'])}** |")
            print("\n")

            # ---------------- TABLE 3: Attribution analysis ----------------
            print("### Table 3: Physical Feature Attribution\n")
            importance = eval_results.get('feature_importance', [])
            if importance:
                channels = ['CAPE', 'PWAT', 'U-Wind', 'V-Wind', 'V-Velocity', 'GFS-Precip']
                total_imp = sum([abs(x) for x in importance]) + 1e-8
                print("| Physical Feature | Absolute Importance | Relative Contribution (%) |")
                print("| :--- | :---: | :---: |")
                for i, c in enumerate(channels):
                    print(f"| {c} | {importance[i]:.5f} | {abs(importance[i])/total_imp*100:.1f}% |")
            print("\n" + "="*80)

            print("\n Generating final streamlined high-resolution figures...")
            # Only keep useful figures for paper
            ResearchVisualizer.plot_density_scatter(preds, targets, gfs_base)
            ResearchVisualizer.plot_high_res_spatial_pro(preds, targets, gfs_base, idx=0)
            
            # Integrated figure generation
            results_dict = {'GFS_Baseline': final_metrics['GFS'], 'Enhanced_Model': final_metrics['Model']}
            create_performance_diagram(results_dict)