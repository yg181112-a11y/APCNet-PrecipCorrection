# -*- coding: utf-8 -*-
"""M1 分支A：真实任务 MSE-选点重训（三种子 42/40/41）。
严格对照主实验：数据、过采样、StormPatch、pretrain、复合损失、EMA、AMP、两阶段触发、
门控校准全部不变；唯一改动 = 验证期残差 MSE 最小选点（替代复合 S 准则）。
输出：每种子目录 best_correction_model.pth + history.json + gate_calib.json + 日志。
"""
import sys, os, json, time, gc, shutil, datetime
import numpy as np
import torch

MAIN_DIR = r'D:\liaohe\论文三\03_重建成稿代_2026_R3全链主实验'
OUT_ROOT = r'D:\liaohe\_waf_review\m1_diagnostic\mse_select'
# 用合法模块名 main13（=13.0_main.py 的副本，本脚本同目录），保证 Windows spawn
# 子进程（ProcessPoolExecutor 多进程提取）可 re-import 该模块以 pickle 绑定方法。
sys.path.insert(0, MAIN_DIR)
# 让 spawn 子进程也能解析 main13（子进程继承 PYTHONPATH + cwd）
os.environ['PYTHONPATH'] = MAIN_DIR + ';' + os.path.dirname(os.path.abspath(__file__)) + ';' + os.environ.get('PYTHONPATH', '')
import main13 as M

SEEDS = [42, 40, 41]
GFS_BASE = r"D:\liaohe\GFS-data\gfs.0p25.2015-2025.f003"
ERA5_BASE = r"D:\liaohe\ERA5-data"

def staged_mse_selection(model, train_loader, val_loader, device, scaling_factor,
                         epochs, second_stage, ckpt_path, history_bag, tag):
    """复刻 M.staged_training_strategy_with_monitoring，选点改为验证残差 MSE 最小。"""
    criterion = M.MultiTaskLoss()
    if second_stage:
        for name, param in model.named_parameters():
            if not any(x in name for x in ['res_heads', 'rain_heads', 'storm_head']):
                param.requires_grad = False
        lr = 3e-5; patience = 6
        print(f"[{tag}] stage2 finetune: freeze trunk, lr=3e-5, patience=6", flush=True)
    else:
        lr = 8e-5; patience = 8
        print(f"[{tag}] stage1: lr=8e-5, patience=8 (MSE selection)", flush=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=8e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    use_amp = (torch.cuda.is_available() and (not second_stage))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = M.ModelEMA(model, decay=0.999)

    best_val_loss = float('inf')
    no_improve_epochs = 0
    min_epochs = 4

    history = {'stage_c_losses': [], 'stage_c_val_losses': [],
               'storm_ets_15': [], 'storm_pod_15': [], 'storm_far_15': [],
               'storm_ets_20': [], 'storm_pod_20': [], 'storm_far_20': [],
               'composite_scores': [], 'saved_epoch': None}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = M.tqdm(train_loader, desc=f"{tag} Epoch {epoch+1}")
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            gfs_base = inputs[:, -1, -1:, :, :]
            era5_abs = gfs_base + targets / scaling_factor
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                residual, rain_prob, storm_logits, _ = model(
                    inputs, return_residual=True, return_storm_logits=True)
                loss = criterion(pred_res=residual, target_abs=era5_abs,
                                 gfs_base=gfs_base, rain_prob=rain_prob,
                                 storm_logits=storm_logits)
            if not torch.isfinite(loss):
                continue
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            ema.update(model)
            epoch_loss += float(loss.item())
            pbar.set_postfix({'loss': f'{float(loss.item()):.4f}'})
        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        history['stage_c_losses'].append(avg_train_loss)

        ema_model = ema.ema
        ema_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v_in, v_tar in val_loader:
                v_in = v_in.to(device, non_blocking=True)
                v_tar = v_tar.to(device, non_blocking=True)
                v_gfs = v_in[:, -1, -1:, :, :]
                v_abs = v_gfs + v_tar / scaling_factor
                with torch.cuda.amp.autocast(enabled=use_amp):
                    v_res, v_prob, v_storm, _ = ema_model(v_in, return_residual=True, return_storm_logits=True)
                    v_loss = criterion(pred_res=v_res, target_abs=v_abs, gfs_base=v_gfs,
                                       rain_prob=v_prob, storm_logits=v_storm)
                if torch.isfinite(v_loss):
                    val_loss += float(v_loss.item())
        avg_val_loss = val_loss / max(len(val_loader), 1)
        scheduler.step(avg_val_loss)
        history['stage_c_val_losses'].append(avg_val_loss)

        storm_metrics = M.monitor_extreme_event_performance(
            ema_model, val_loader, device, thresholds=[15.0, 20.0],
            scaling_factor=scaling_factor,
            storm_gate_p=float(M.GATE_CFG.get("storm_gate_p", 0.25)))
        ets15 = storm_metrics[15.0].get('ETS', 0.0); pod15 = storm_metrics[15.0].get('POD', 0.0)
        far15 = storm_metrics[15.0].get('FAR', 1.0)
        ets20 = storm_metrics[20.0].get('ETS', 0.0); pod20 = storm_metrics[20.0].get('POD', 0.0)
        far20 = storm_metrics[20.0].get('FAR', 1.0)
        for k, v in [('storm_ets_15', ets15), ('storm_pod_15', pod15), ('storm_far_15', far15),
                     ('storm_ets_20', ets20), ('storm_pod_20', pod20), ('storm_far_20', far20)]:
            history[k].append(v)
        cs = (3.0*ets20 + 1.8*pod20 - 0.45*far20 + 1.4*ets15 + 0.9*pod15 - 0.20*far15 - 0.003*avg_val_loss)
        cs -= max(0, 0.20 - pod20) * 2.0
        history['composite_scores'].append(cs)

        print(f"[{tag}] E{epoch+1}: val_loss={avg_val_loss:.6f} ETS20={ets20:.4f} POD20={pod20:.4f} "
              f"S={cs:.4f} best_val={best_val_loss:.6f}", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'epoch': epoch + 1, 'model_state_dict': ema_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss, 'train_loss': avg_train_loss,
                        'storm_metrics': storm_metrics, 'best_val_loss': best_val_loss,
                        'selection': 'mse', 'stage': 'second' if second_stage else 'first'},
                       ckpt_path)
            history['saved_epoch'] = epoch + 1
            no_improve_epochs = 0
            print(f"[{tag}]  save MSE-best @ E{epoch+1} (val_loss={best_val_loss:.6f})", flush=True)
        else:
            no_improve_epochs += 1

        if (epoch + 1) >= min_epochs and no_improve_epochs >= patience:
            print(f"[{tag}]  early stop @ E{epoch+1} (val_loss no-improve {patience})", flush=True)
            break

    history_bag[tag] = history
    return model

def run_seed(seed, datasets_dict):
    out_dir = os.path.join(OUT_ROOT, f'seed{seed}')
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'best_correction_model.pth')
    torch.manual_seed(seed); np.random.seed(seed); __import__('random').seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*70}\n[seed {seed}] 训练 (MSE selection, 数据集共享)\n{'='*70}", flush=True)
    correction_train = datasets_dict['correction_train']
    correction_val = datasets_dict['correction_val']
    correction_test = datasets_dict['correction_test']
    scaling_factor = float(datasets_dict.get('scaling_factor', 1.0))
    print(f"[seed {seed}] train={len(correction_train)} val={len(correction_val)} test={len(correction_test)}", flush=True)

    # 门控配置：与主实验完全一致
    M.GATE_CFG["adaptive"] = True; M.GATE_CFG["hard_gate"] = True
    M.GATE_CFG["threshold_base"] = 0.22; M.GATE_CFG["threshold_min"] = 0.10
    M.GATE_CFG["threshold_max"] = 0.40; M.GATE_CFG["gate_power"] = 0.90
    M.GATE_CFG["storm_gate_p"] = 0.25

    intensity_stats = M.ExtremeEventDataLoader.analyze_dataset_intensity(
        correction_train, thresholds=[0.1, 1.0, 5.0, 10.0, 20.0])
    sampled_total = max(1, sum(intensity_stats.values()))
    heavy = intensity_stats.get('10.0-20.0mm', 0) + intensity_stats.get('>=20.0mm', 0)
    hrr = heavy / sampled_total
    oversample_ratio = 10 if hrr < 0.03 else (7 if hrr < 0.08 else 5)
    print(f"[seed {seed}] heavy-rain ratio {hrr*100:.2f}% -> oversample {oversample_ratio}", flush=True)
    ratios = [0.4, 0.8, 1.2, 5.0, 20.0, 45.0, 70.0]
    storm_patch_train = M.StormPatchWrapper(correction_train, patch=20, storm_th=10.0, storm_prob=1.0)
    train_loader = M.ExtremeEventDataLoader.create_adaptive_oversampled_loader(
        dataset=storm_patch_train, batch_size=64, num_workers=0, oversample_ratios=ratios)

    cfg = M.ultra_fast_training_config()
    def fast_loader(ds, shuffle, drop_last):
        return M.DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle, num_workers=0,
                            pin_memory=True, drop_last=drop_last, collate_fn=M.custom_collate_fn)
    val_loader = fast_loader(correction_val, False, True)
    test_loader = fast_loader(correction_test, False, False)

    device = M.get_device()
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    model = M.AdvancedPrecipCorrectionNet(
        input_channels=8, hidden_channels=24, sequence_length=6,
        prediction_horizon=M.PREDICTION_HORIZON, spatial_dims=(25, 37), dropout_rate=0.1).to(device)
    print(f"[seed {seed}] model params {sum(p.numel() for p in model.parameters()):,}", flush=True)

    _, _ = M.improved_gfs_pretrain_phase(model=model, train_loader=train_loader,
                                         device=device, epochs=1, learning_rate=3e-5)

    history_bag = {}
    model = staged_mse_selection(model, train_loader, val_loader, device, scaling_factor,
                                 epochs=10, second_stage=False, ckpt_path=ckpt_path,
                                 history_bag=history_bag, tag='S1')

    # 两阶段触发：与主实验一致（stage1 暴雨能力判定）
    trigger_stage2 = False
    if len(history_bag['S1'].get('storm_pod_20', [])) > 0:
        bp20 = float(np.max(history_bag['S1']['storm_pod_20']))
        be20 = float(np.max(history_bag['S1'].get('storm_ets_20', [0.0])))
        trigger_stage2 = (bp20 >= 0.10 or be20 >= 0.02)
        print(f"[seed {seed}] stage1 best POD20={bp20:.4f} ETS20={be20:.4f} -> stage2={'Y' if trigger_stage2 else 'N'}", flush=True)

    if trigger_stage2 and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        model = staged_mse_selection(model, train_loader, val_loader, device, scaling_factor,
                                     epochs=8, second_stage=True, ckpt_path=ckpt_path,
                                     history_bag=history_bag, tag='S2')

    # 门控校准（与主实验一致，在最终 MSE-best 权重上校准推理门控）
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    old_th = float(M.GATE_CFG.get("threshold_base", 0.22)); old_pow = float(M.GATE_CFG.get("gate_power", 0.85))
    try:
        calib = M.calibrate_gate_threshold_on_val(
            model=model, val_loader=val_loader, device=device, scaling_factor=scaling_factor,
            search=np.linspace(0.08, 0.24, 9), power_search=(0.80, 0.90, 1.00),
            storm_gate_p_search=[0.10, 0.15, 0.20, 0.25, 0.30],
            min_pod20=0.25, min_pod15=0.25, max_far20=0.999, min_ets20=0.005)
        if calib.get("is_valid", False):
            M.GATE_CFG["threshold_base"] = float(calib["th"]); M.GATE_CFG["gate_power"] = float(calib["pow"])
            M.GATE_CFG["storm_gate_p"] = float(calib["gate_p"])
            print(f"[seed {seed}] gate calibrated th={calib['th']:.3f} pow={calib['pow']:.2f} p={calib['gate_p']:.2f} "
                  f"POD20={calib.get('pod20',0):.4f} ETS20={calib.get('ets20',0):.4f}", flush=True)
        elif calib.get("ets20", -1) >= 0.005 and calib.get("pod20", 0) >= 0.20:
            M.GATE_CFG["threshold_base"] = float(calib["th"]); M.GATE_CFG["gate_power"] = float(calib["pow"])
            M.GATE_CFG["storm_gate_p"] = float(calib["gate_p"])
            print(f"[seed {seed}] gate fallback best_any th={calib['th']:.3f}", flush=True)
        else:
            M.GATE_CFG["threshold_base"] = old_th; M.GATE_CFG["gate_power"] = old_pow
            print(f"[seed {seed}] gate calib rejected, keep old gate", flush=True)
    except Exception as e:
        print(f"[seed {seed}] gate calib exception: {e}", flush=True)

    # 保存产物
    hist_all = {'seed': seed, 'selected_epoch': ck.get('epoch'),
                'selected_val_loss': ck.get('val_loss'),
                'stage1': history_bag.get('S1'), 'stage2': history_bag.get('S2'),
                'gate': {k: M.GATE_CFG.get(k) for k in ('threshold_base','threshold_min','threshold_max','gate_power','storm_gate_p')},
                'trigger_stage2': trigger_stage2}
    with open(os.path.join(out_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(hist_all, f, ensure_ascii=False, indent=1)
    # 记录评估用口径
    with open(os.path.join(out_dir, 'scaling_factor.txt'), 'w') as f:
        f.write(str(scaling_factor))
    print(f"[seed {seed}] DONE selected_epoch={ck.get('epoch')} val_loss={ck.get('val_loss'):.6f} -> {out_dir}", flush=True)

def main():
    print(f"=== M1 branch-A MSE-selection retrain, seeds={SEEDS} ===", flush=True)
    t0 = time.time()
    # 数据集构建一次（含 11 年 GFS 多进程提取 + ERA5 加载），三种子共享
    print("[data] 构建数据集（train 2015-2021 / val 2022-2023 / test 2024-2025）...", flush=True)
    datasets_dict = M.create_datasets_with_dem(GFS_BASE, ERA5_BASE, dem_tensor=None)
    print(f"[data] OK train={len(datasets_dict['correction_train'])} val={len(datasets_dict['correction_val'])} "
          f"test={len(datasets_dict['correction_test'])} scaling={datasets_dict.get('scaling_factor')}", flush=True)
    for seed in SEEDS:
        run_seed(seed, datasets_dict)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"ALL DONE in {(time.time()-t0)/3600:.2f} h", flush=True)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
