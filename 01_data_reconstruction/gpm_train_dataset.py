# -*- coding: utf-8 -*-
"""
gpm_train_dataset.py — GPM IMERG 观测直接训练数据集（敏感性对照实验）
=====================================================================
与主实验（ERA5 训练）的差异仅一处：标签从 ERA5 3h 累积换成 GPM IMERG 3h 累积。
GFS 输入完全一致（6 时步 × 8 通道，sequence_step_hours=6，03/09/15/21Z）。

口径（与 verify_gpm_independent.py 严格一致）：
  - 样本时刻 t0 (UTC) 代表 3h 累积窗 [t0-3h, t0]
  - GPM 半小时降水率 (mm/hr)，6 bin 累加 -> mm
  - 双线性插值 GPM 0.1° -> 主网格 0.25° (25x37)，coverage mask

划分（GPM 数据 2018-2025）：
  训练 2018-01~2021-12 / 验证 2022-01~2023-12 / 测试 2024-01~2025-12
（与主实验 test 2024-2025 对齐，双产品验证可复用）
"""
import glob
import importlib
import os
import random
import sys
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, r"C:\Users\yg181\Desktop\论文三\13.0修复重跑")
# 符号来源优先级：主程序以 __main__ 运行时直接用其符号（patch 路径，避免二次 exec）；
# 独立入口时用 spec 加载一次并注册。
if "__main__" in sys.modules and hasattr(sys.modules["__main__"], "EnhancedDataProcessor"):
    M = sys.modules["__main__"]
elif "main13" in sys.modules:
    M = sys.modules["main13"]
else:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("main13", os.path.join(r"C:\Users\yg181\Desktop\论文三\13.0修复重跑", "13.0_main.py"))
    M = _ilu.module_from_spec(_spec)
    sys.modules["main13"] = M
    _spec.loader.exec_module(M)

# ---------------- GPM 标签基础函数（与 verify_gpm_independent.py 一致） ----------------
GPM_ROOT = r"C:\Users\yg181\Downloads\GPM_IMERG_test"
GPM_ROOT_D = r"D:\liaohe\GPM_IMERG"
GPM_ROOTS = [GPM_ROOT, GPM_ROOT_D]  # Downloads: 2018-01~2023-12；D盘: 2024-01~2025-09
GLOBAL_LATS = M.GLOBAL_LATS if hasattr(M, "GLOBAL_LATS") else np.linspace(46.0, 40.0, 25)
GLOBAL_LONS = M.GLOBAL_LONS if hasattr(M, "GLOBAL_LONS") else np.linspace(117.0, 126.0, 37)
BIN_H = 0.5
N_BINS = 6


def gpm_file_for(dt, root=None):
    """多根目录查找：返回第一个存在的 (path, True)；全部缺失返回 (首候选, False)。"""
    if root is None:
        roots = GPM_ROOTS
    elif isinstance(root, str):
        roots = [root]
    else:
        roots = root
    d = dt.date()
    ym = d.strftime("%Y%m")
    first = None
    for r in roots:
        p = os.path.join(r, f"imerg_{ym}", f"imerg_{d.strftime('%Y%m%d')}_{dt.strftime('%H%M%S')}.nc4")
        if first is None:
            first = p
        if os.path.exists(p):
            return p, True
    return first, False


def read_gpm_rate(path):
    import netCDF4 as nc
    ds = nc.Dataset(path)
    g = ds.groups["Grid"]
    lat = g.variables["lat"][:].astype(np.float64)
    lon = g.variables["lon"][:].astype(np.float64)
    rate = g.variables["precipitation"][0, :, :].astype(np.float64)  # (lon, lat)
    ds.close()
    return rate, lon, lat


def gpm_3h_accum(t0, cache=None, root=None):
    """t0 代表 [t0-3h, t0] 3h 累积；6 个半小时 bin 累加。
    任一 bin 缺失 -> ok=False"""
    if cache is None:
        cache = {}
    bins = []
    for k in range(N_BINS):
        bt = t0 - timedelta(hours=3) + timedelta(minutes=int(30 * k))
        key = bt.strftime("%Y%m%d_%H%M%S")
        if key in cache:
            r = cache[key]
            if r is None:
                return None, None, None, False
        else:
            p, exists = gpm_file_for(bt, root=root)
            if not exists:
                cache[key] = None
                return None, None, None, False
            try:
                r, lon, lat = read_gpm_rate(p)
            except Exception:
                cache[key] = None
                return None, None, None, False
            cache[key] = (r, lon, lat)
        bins.append(cache[key])
    lon, lat = bins[0][1], bins[0][2]
    acc = np.zeros_like(bins[0][0])
    for r, _, _ in bins:
        acc += r * BIN_H  # mm/hr * 0.5h -> mm
    return acc, lon, lat, True


def regrid_to_main(acc_lonlat, lon, lat):
    """双线性插值 GPM (91,60) -> 主网格 (25,37)。返回 (field, mask)。"""
    from scipy.interpolate import RegularGridInterpolator
    lon2d, lat2d = np.meshgrid(GLOBAL_LONS, GLOBAL_LATS)
    pts = np.stack([lon2d.ravel(), lat2d.ravel()], axis=-1)
    interp = RegularGridInterpolator(
        (lon, lat), acc_lonlat, method="linear", bounds_error=False, fill_value=np.nan
    )
    vals = interp(pts).reshape(25, 37)
    mask = ~np.isnan(vals)
    vals = np.where(mask, vals, 0.0)
    return vals, mask


class PairedGFSGPMResidualDatasetStrict(Dataset):
    """GPM 标签版成对数据集（GFS 输入装配与主实验 PairedGFSEra5ResidualDatasetStrict 相同）。"""

    def __init__(self,
                 gfs_folders,
                 gpm_root=GPM_ROOTS,
                 standardizer=None,
                 sequence_length=6,
                 prediction_horizon=1,
                 temp_extract_dir="temp_extract_gpm",
                 precip_data_path=None,
                 start_date=None,
                 end_date=None,
                 dem_features=None,
                 enable_cleaning=True,
                 require_strict_step=True,
                 sequence_step_hours=None,
                 augment=False,
                 min_mask_frac=0.8):
        assert prediction_horizon == 1, "只有 f003 时 prediction_horizon 必须为 1"
        self.gpm_root = gpm_root
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
        self.min_mask_frac = min_mask_frac
        self.start_dt = None
        self.end_dt = None
        try:
            if start_date:
                self.start_dt = pd_start(start_date)
            if end_date:
                self.end_dt = pd_end(end_date)
        except Exception:
            self.start_dt, self.end_dt = None, None

        self.dem_features = dem_features.cpu() if dem_features is not None else None
        os.makedirs(self.temp_extract_dir, exist_ok=True)

        # 1) GFS map（与父类相同）
        self.gfs_map = self._build_gfs_time_map(gfs_folders)

        # 2) GPM map：对每个 GFS 时次构建 (field25x37, mask25x37)
        print("📡 构建 GPM 3h 标签 map ...")
        self.gpm_map = {}
        self.gpm_mask = {}
        _cache = {}
        _t0 = datetime.now()
        for t in sorted(self.gfs_map.keys()):
            acc, lon, lat, ok = gpm_3h_accum(t, cache=_cache, root=self.gpm_root)
            if not ok:
                continue
            field, mask = regrid_to_main(acc, lon, lat)
            frac = float(mask.mean())
            if frac < self.min_mask_frac:
                continue
            self.gpm_map[t] = field
            self.gpm_mask[t] = mask
        print(f"  GPM 覆盖时次 {len(self.gpm_map)}/{len(self.gfs_map)} "
              f"(耗时 {(datetime.now()-_t0).total_seconds():.0f}s)")

        # 3) 公共时刻
        common_times = sorted(set(self.gfs_map.keys()).intersection(set(self.gpm_map.keys())))
        if len(common_times) < self.sequence_length:
            raise RuntimeError(f"GPM 公共时次不足: {len(common_times)}")

        # 4) 序列步长
        step_hours = self.sequence_step_hours or self._infer_step_hours(common_times)
        self.inferred_step_hours = step_hours
        dt = timedelta(hours=int(step_hours))

        # 5) 构造样本
        self.sample_times = []
        self.sequences = []
        common_set = set(common_times)
        for t0 in common_times:
            seq_times = [t0 - dt * (self.sequence_length - 1 - k) for k in range(self.sequence_length)]
            if any(t not in common_set for t in seq_times):
                continue
            if self.require_strict_step:
                ok = all((seq_times[k] - seq_times[k - 1]) == dt for k in range(1, len(seq_times)))
                if not ok:
                    continue
            x_list, raw_precip_list = [], []
            for t in seq_times:
                arr = self.gfs_map[t]
                x_list.append(arr)
                raw_precip_list.append(arr[-1])
            x = torch.FloatTensor(np.stack(x_list, axis=0))
            if self.standardizer is not None:
                raw_precip = torch.FloatTensor(np.stack(raw_precip_list, axis=0))
                x = self.standardizer.transform(x)
                x[:, -1, :, :] = raw_precip
            # 标签 residual = GPM_abs(t0) - GFS_base(t0)
            gpm_abs = torch.FloatTensor(self.gpm_map[t0])   # [H,W]
            gfs_base = x[-1, -1, :, :]
            y_res = (gpm_abs - gfs_base).unsqueeze(0)       # [1,H,W]
            abs_max = float(torch.max(gpm_abs).item())
            self.sequences.append((x, y_res, abs_max))
            self.sample_times.append(t0)

        print(f"✅ GPM Strict Dataset ready: {len(self.sequences)} samples "
              f"(step={self.inferred_step_hours}h, 输入窗 {self.sequence_length}x{step_hours}h)")
        if len(self.sequences) == 0:
            raise RuntimeError("GPM dataset 0 samples")

    # ---- 复用父类逻辑 ----
    def _in_time_window(self, t):
        if t is None:
            return False
        if self.start_dt is not None and t < self.start_dt:
            return False
        if self.end_dt is not None and t > self.end_dt:
            return False
        return True

    def _build_gfs_time_map(self, gfs_folders):
        """GFS map：与主实验完全一致（进程池 process_archive_batch_parallel_with_times）。
        主实验已验证该路径稳定；线程池读取 netCDF4/HDF5 在 Windows 会 0xC0000005 崩溃。"""
        gfs_map = {}
        for folder in gfs_folders:
            archive_files = (
                glob.glob(os.path.join(folder, "*.zip")) +
                glob.glob(os.path.join(folder, "*.tar")) +
                glob.glob(os.path.join(folder, "*.nc"))
            )
            archive_files.sort()
            proc = M.EnhancedDataProcessor(base_path=folder, precip_path=self.precip_data_path, end_date=None)
            pairs = proc.process_archive_batch_parallel_with_times(archive_files, self.temp_extract_dir)
            for t, arr in pairs:
                if arr is None or t is None:
                    continue
                if not self._in_time_window(t):
                    continue
                if self.enable_cleaning:
                    arr = M.advanced_precip_cleaning(arr)
                gfs_map[t] = arr
        return gfs_map

    def _infer_step_hours(self, times):
        from collections import Counter
        diffs = []
        for i in range(1, len(times)):
            dh = int(round((times[i] - times[i - 1]).total_seconds() / 3600.0))
            if dh > 0:
                diffs.append(dh)
        if not diffs:
            return M.TIME_STEP_HOURS
        c = Counter(diffs)
        step = c.most_common(1)[0][0]
        if step not in (1, 3, 6, 12, 24):
            step = M.TIME_STEP_HOURS
        return step

    def get_target_max_precip(self, idx):
        return self.sequences[idx][2]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y_res, abs_max = self.sequences[idx]
        is_storm = abs_max >= 20.0
        if self.augment:
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-1])
                y_res = torch.flip(y_res, dims=[-1])
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-2])
                y_res = torch.flip(y_res, dims=[-2])
            # 注：网格 25x37 非正方形，rot90 会交换空间维导致 batch 内形状不一致，
            # 故仅保留 flip 增强（与主实验 PairedGFSEra5ResidualDatasetStrict 一致）。
        if is_storm and self.augment:
            noise = torch.randn_like(x[:, -1]) * 0.5
            mask = (x[:, -1] > 0).float()
            x[:, -1] = x[:, -1] + noise * mask * 0.3
            x = torch.clamp(x, min=0.0)
        if is_storm and self.augment and random.random() > 0.7:
            other_idx = random.randint(0, len(self.sequences) - 1)
            other_x, other_y_res, other_abs_max = self.sequences[other_idx]
            if other_abs_max >= 20.0:
                lam = random.betavariate(0.5, 0.5)
                x = lam * x + (1 - lam) * other_x
                y_res = lam * y_res + (1 - lam) * other_y_res
        return x, y_res


def pd_start(s):
    import pandas as pd
    return pd.to_datetime(s).to_pydatetime()


def pd_end(s):
    import pandas as pd
    e = pd.to_datetime(s).to_pydatetime()
    return e + timedelta(hours=23, minutes=59, seconds=59)


# ---------------- 主入口：构建与主实验同构的 GPM 数据集 ----------------
def create_gpm_strict_datasets(gfs_base_path, gpm_root=GPM_ROOTS):
    """GPM 观测训练版数据集：训练 2018-2021 / 验证 2022-2023 / 测试 2024-2025。
    返回结构与 create_datasets_with_dem 一致，可直接替换进 run_residual_experiment_enhanced。"""
    gfs_precip_path = os.path.join(gfs_base_path, "jiangshui")
    all_gfs_folders = M.discover_data_folders(gfs_base_path)

    def filter_by_date(folders, start_dt, end_dt):
        return [f for f in folders
                if start_dt <= (M.extract_date_from_filename(os.path.basename(f)) or datetime(1900, 1, 1)) <= end_dt]

    train_start, train_end = datetime(2018, 1, 1), datetime(2021, 12, 31)
    val_start, val_end = datetime(2022, 1, 1), datetime(2023, 12, 31)
    test_start, test_end = datetime(2024, 1, 1), datetime(2025, 12, 31)

    gfs_train_folders = filter_by_date(all_gfs_folders, train_start, train_end)
    gfs_val_folders = filter_by_date(all_gfs_folders, val_start, val_end)
    gfs_test_folders = filter_by_date(all_gfs_folders, test_start, test_end)
    print(f"GFS 文件夹: 训练{len(gfs_train_folders)} 验证{len(gfs_val_folders)} 测试{len(gfs_test_folders)}")

    # 标准器：基于 GPM 训练期 GFS（2018-2021）拟合
    global_standardizer = M.GlobalDataStandardizer(cache_path="std_params_gpm.npy")

    def get_raw_dataset():
        return M.SelfSupervisedGFSDataset(
            data_paths=gfs_train_folders, standardizer=None,
            precip_data_path=gfs_precip_path, max_samples=2000)
    global_standardizer.fit(get_raw_dataset)
    strict_std = M.StrictStandardizer(global_standardizer)

    def _ds(folders, sd, ed, augment=False, temp="temp_extract_gpm"):
        return PairedGFSGPMResidualDatasetStrict(
            gfs_folders=folders, gpm_root=gpm_root, standardizer=strict_std,
            sequence_length=6, prediction_horizon=M.PREDICTION_HORIZON,
            temp_extract_dir=temp, precip_data_path=gfs_precip_path,
            start_date=sd.strftime("%Y-%m-%d"), end_date=ed.strftime("%Y-%m-%d"),
            enable_cleaning=True, require_strict_step=True,
            sequence_step_hours=6, augment=augment)

    correction_train = _ds(gfs_train_folders, train_start, train_end, augment=True)
    correction_val = _ds(gfs_val_folders, val_start, val_end, augment=False)
    correction_test = _ds(gfs_test_folders, test_start, test_end, augment=False)
    gfs_train = M.SelfSupervisedGFSDataset(
        data_paths=gfs_train_folders, standardizer=strict_std,
        precip_data_path=gfs_precip_path, augment=True)

    print(f"✅ GPM 训练集 {len(correction_train)} | 验证集 {len(correction_val)} | 测试集 {len(correction_test)}")
    return {
        "gfs_train": gfs_train,
        "correction_train": correction_train,
        "correction_val": correction_val,
        "correction_test": correction_test,
        "standardizer": global_standardizer,
        "scaling_factor": 1.0,
    }
