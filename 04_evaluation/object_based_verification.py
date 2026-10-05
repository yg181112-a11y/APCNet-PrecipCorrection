# -*- coding: utf-8 -*-
"""
Object-based verification metrics for precipitation forecasts.
Uses connected-component analysis to identify precipitation objects,
then matches forecast objects to observation objects and computes
position, intensity, and area errors.
"""
import numpy as np
from scipy import ndimage
import pickle, os, json

WORK = r'D:\liaohe\校正优化过程\第三阶段\12优化\manuscript_work'

# Load data
pred = np.load(os.path.join(WORK, 'predictions_apcnet.npy'))
target = np.load(os.path.join(WORK, 'targets_test.npy'))
gfs = np.load(os.path.join(WORK, 'gfs_test.npy'))
with open(os.path.join(WORK, 'sample_times_test.pkl'), 'rb') as f:
    times = pickle.load(f)

print(f"Data loaded: pred={pred.shape}, target={target.shape}, gfs={gfs.shape}")

def detect_objects(field, threshold, min_area=3):
    """Detect precipitation objects using connected-component labeling."""
    binary = field >= threshold
    labeled, num = ndimage.label(binary)
    objects = []
    for i in range(1, num + 1):
        mask = labeled == i
        area = mask.sum()
        if area < min_area:
            continue
        coords = np.argwhere(mask)
        centroid = coords.mean(axis=0)  # (row, col) = (lat, lon)
        intensity = field[mask].mean()
        max_intensity = field[mask].max()
        objects.append({
            'area': int(area),
            'centroid': centroid.tolist(),
            'mean_intensity': float(intensity),
            'max_intensity': float(max_intensity),
            'mask': mask,
        })
    return objects

def match_objects(fcst_objs, obs_objs, max_dist=5.0):
    """Match forecast objects to observation objects by centroid distance.
    Returns list of (fcst_idx, obs_idx, distance) matched pairs,
    plus unmatched forecast and observation indices.
    """
    if not fcst_objs or not obs_objs:
        return [], list(range(len(fcst_objs))), list(range(len(obs_objs)))
    
    # Compute distance matrix
    n_f, n_o = len(fcst_objs), len(obs_objs)
    dist_mat = np.zeros((n_f, n_o))
    for i, fo in enumerate(fcst_objs):
        for j, oo in enumerate(obs_objs):
            dist_mat[i, j] = np.sqrt(
                (fo['centroid'][0] - oo['centroid'][0])**2 +
                (fo['centroid'][1] - oo['centroid'][1])**2
            )
    
    # Greedy matching (closest first)
    matched = []
    used_f, used_o = set(), set()
    flat = []
    for i in range(n_f):
        for j in range(n_o):
            flat.append((dist_mat[i, j], i, j))
    flat.sort()
    
    for d, i, j in flat:
        if d > max_dist:
            break
        if i not in used_f and j not in used_o:
            matched.append((i, j, float(d)))
            used_f.add(i)
            used_o.add(j)
    
    unmatched_f = [i for i in range(n_f) if i not in used_f]
    unmatched_o = [j for j in range(n_o) if j not in used_o]
    return matched, unmatched_f, unmatched_o

def compute_object_metrics(fcst_field, obs_field, threshold, min_area=3, max_dist=5.0):
    """Compute object-based metrics for a single forecast-observation pair."""
    fcst_objs = detect_objects(fcst_field, threshold, min_area)
    obs_objs = detect_objects(obs_field, threshold, min_area)
    matched, unmatched_f, unmatched_o = match_objects(fcst_objs, obs_objs, max_dist)
    
    # Basic counts
    n_fcst = len(fcst_objs)
    n_obs = len(obs_objs)
    n_matched = len(matched)
    n_false_alarm = len(unmatched_f)
    n_missed = len(unmatched_o)
    
    # Position error (centroid distance in grid units)
    pos_errors = [d for _, _, d in matched]
    
    # Intensity error (relative difference in mean intensity)
    intensity_errors = []
    area_errors = []
    for fi, oi, _ in matched:
        fi_mean = fcst_objs[fi]['mean_intensity']
        oi_mean = obs_objs[oi]['mean_intensity']
        if oi_mean > 0:
            intensity_errors.append(abs(fi_mean - oi_mean) / oi_mean)
        fi_area = fcst_objs[fi]['area']
        oi_area = obs_objs[oi]['area']
        if oi_area > 0:
            area_errors.append(abs(fi_area - oi_area) / oi_area)
    
    # POD/FAR for objects
    pod = n_matched / n_obs if n_obs > 0 else 0.0
    far = n_false_alarm / n_fcst if n_fcst > 0 else 0.0
    
    return {
        'n_fcst': n_fcst,
        'n_obs': n_obs,
        'n_matched': n_matched,
        'n_false_alarm': n_false_alarm,
        'n_missed': n_missed,
        'pod': pod,
        'far': far,
        'mean_pos_error': float(np.mean(pos_errors)) if pos_errors else 0.0,
        'mean_intensity_error': float(np.mean(intensity_errors)) if intensity_errors else 0.0,
        'mean_area_error': float(np.mean(area_errors)) if area_errors else 0.0,
        'mean_fcst_area': float(np.mean([o['area'] for o in fcst_objs])) if fcst_objs else 0.0,
        'mean_obs_area': float(np.mean([o['area'] for o in obs_objs])) if obs_objs else 0.0,
    }

# Run for both thresholds
results = {}
for threshold_name, threshold in [('>=10mm', 10.0), ('>=20mm', 20.0)]:
    print(f"\n{'='*60}")
    print(f"Object-based verification: {threshold_name}")
    print(f"{'='*60}")
    
    gfs_metrics = []
    apc_metrics = []
    
    for t in range(len(times)):
        gm = compute_object_metrics(gfs[t], target[t], threshold)
        am = compute_object_metrics(pred[t], target[t], threshold)
        gfs_metrics.append(gm)
        apc_metrics.append(am)
    
    # Aggregate
    def agg(metrics_list, key):
        vals = [m[key] for m in metrics_list if m['n_obs'] > 0 or key in ('n_fcst','n_obs','n_matched','n_false_alarm','n_missed')]
        return float(np.mean(vals)) if vals else 0.0
    
    gfs_agg = {k: agg(gfs_metrics, k) for k in gfs_metrics[0].keys()}
    apc_agg = {k: agg(apc_metrics, k) for k in apc_metrics[0].keys()}
    
    # Total counts
    gfs_total_obs = sum(m['n_obs'] for m in gfs_metrics)
    gfs_total_fcst = sum(m['n_fcst'] for m in gfs_metrics)
    gfs_total_matched = sum(m['n_matched'] for m in gfs_metrics)
    apc_total_obs = sum(m['n_obs'] for m in apc_metrics)
    apc_total_fcst = sum(m['n_fcst'] for m in apc_metrics)
    apc_total_matched = sum(m['n_matched'] for m in apc_metrics)
    
    print(f"\n--- GFS ---")
    print(f"  Total objects: obs={gfs_total_obs}, fcst={gfs_total_fcst}, matched={gfs_total_matched}")
    print(f"  Object POD: {gfs_total_matched/gfs_total_obs:.4f}" if gfs_total_obs else "  POD: N/A")
    print(f"  Object FAR: {1-gfs_total_matched/gfs_total_fcst:.4f}" if gfs_total_fcst else "  FAR: N/A")
    print(f"  Mean pos error: {gfs_agg['mean_pos_error']:.3f} grid")
    print(f"  Mean intensity error: {gfs_agg['mean_intensity_error']:.4f}")
    print(f"  Mean area error: {gfs_agg['mean_area_error']:.4f}")
    print(f"  Mean fcst area: {gfs_agg['mean_fcst_area']:.1f} grid pts")
    print(f"  Mean obs area: {gfs_agg['mean_obs_area']:.1f} grid pts")
    
    print(f"\n--- APCNet ---")
    print(f"  Total objects: obs={apc_total_obs}, fcst={apc_total_fcst}, matched={apc_total_matched}")
    print(f"  Object POD: {apc_total_matched/apc_total_obs:.4f}" if apc_total_obs else "  POD: N/A")
    print(f"  Object FAR: {1-apc_total_matched/apc_total_fcst:.4f}" if apc_total_fcst else "  FAR: N/A")
    print(f"  Mean pos error: {apc_agg['mean_pos_error']:.3f} grid")
    print(f"  Mean intensity error: {apc_agg['mean_intensity_error']:.4f}")
    print(f"  Mean area error: {apc_agg['mean_area_error']:.4f}")
    print(f"  Mean fcst area: {apc_agg['mean_fcst_area']:.1f} grid pts")
    print(f"  Mean obs area: {apc_agg['mean_obs_area']:.1f} grid pts")
    
    results[threshold_name] = {
        'gfs': {
            'total_obs': gfs_total_obs, 'total_fcst': gfs_total_fcst,
            'total_matched': gfs_total_matched,
            'pod': gfs_total_matched/gfs_total_obs if gfs_total_obs else 0,
            'far': 1-gfs_total_matched/gfs_total_fcst if gfs_total_fcst else 0,
            'pos_error': gfs_agg['mean_pos_error'],
            'intensity_error': gfs_agg['mean_intensity_error'],
            'area_error': gfs_agg['mean_area_error'],
            'mean_fcst_area': gfs_agg['mean_fcst_area'],
            'mean_obs_area': gfs_agg['mean_obs_area'],
        },
        'apcnet': {
            'total_obs': apc_total_obs, 'total_fcst': apc_total_fcst,
            'total_matched': apc_total_matched,
            'pod': apc_total_matched/apc_total_obs if apc_total_obs else 0,
            'far': 1-apc_total_matched/apc_total_fcst if apc_total_fcst else 0,
            'pos_error': apc_agg['mean_pos_error'],
            'intensity_error': apc_agg['mean_intensity_error'],
            'area_error': apc_agg['mean_area_error'],
            'mean_fcst_area': apc_agg['mean_fcst_area'],
            'mean_obs_area': apc_agg['mean_obs_area'],
        }
    }

# Save results
with open(os.path.join(WORK, 'object_based_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to object_based_results.json")
