#!/usr/bin/env python3
"""
Fix NaN BatchNorm running statistics in EMA checkpoint.

Problem: FP16 training caused some BatchNorm running_mean/running_var to become NaN.
Solution: Reset NaN stats, recalibrate BN via forward pass, fix DFL conv weight.

Usage:
    uv run python fix_checkpoint_nan.py --checkpoint weights/ava_yolon_mae_large_cross_freeze/best.pth
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import copy

sys.path.insert(0, str(Path(__file__).parent))

from YOWOFormer.model.TSN.YOWOFormer import build_yowoformer
from YOWOFormer.utils.collate import collate_fn


def fix_checkpoint(checkpoint_path, data_root, dataset,
                   calibration_batches=100, batch_size=8):

    print("=" * 60)
    print("Fixing NaN BatchNorm stats in checkpoint")
    print("=" * 60)

    # Step 1: Load and report NaN
    print(f"\nLoading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    saved_config = checkpoint.get('config', {})

    nan_keys = [k for k, v in state_dict.items()
                if v.is_floating_point() and torch.isnan(v).any()]
    print(f"\nFound {len(nan_keys)} keys with NaN:")
    for k in nan_keys:
        print(f"  {k}")

    if not nan_keys:
        print("No NaN found. Checkpoint is clean.")
        return checkpoint_path

    # Step 2: Reset NaN running stats
    print("\nResetting NaN running stats...")
    for k in nan_keys:
        if 'running_mean' in k:
            state_dict[k] = torch.zeros_like(state_dict[k])
        elif 'running_var' in k:
            state_dict[k] = torch.ones_like(state_dict[k])

    # Step 3: Fix DFL conv weight
    dfl_key = 'detection_head.dfl.conv.weight'
    if dfl_key in state_dict:
        expected = torch.arange(16, dtype=torch.float).view(1, 16, 1, 1)
        if not torch.allclose(state_dict[dfl_key], expected):
            print("Fixing DFL conv weight (EMA drift)")
            state_dict[dfl_key] = expected

    # Step 4: Build model
    print("\nBuilding model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'dataset': dataset,
        'data_root': data_root,
        'num_classes': 24 if dataset == 'ucf' else 80,
        'backbone2D': saved_config.get('backbone2D', 'yolov11_n'),
        'backbone3D': 'videomae',
        'videomae_variant': saved_config.get('videomae_variant', 'large'),
        'videomae_method': saved_config.get('videomae_method', 'cross'),
        'fusion_module': saved_config.get('fusion_module', 'CrossAttention'),
        'mode': 'decoupled',
        'interchannels': saved_config.get('interchannels', [256, 256, 256]),
        'img_size': 224,
        'clip_length': saved_config.get('clip_length', 16),
        'sampling_rate': saved_config.get('sampling_rate', 1),
        'pretrain_2d': True,
        'pretrain_3d': True,
        'freeze_bb2D': False,
        'freeze_bb3D': False,
        'pretrain_path': None,
        'BACKBONE3D': {
            'TYPE': 'videomae',
            'VARIANT': saved_config.get('videomae_variant', 'large'),
            'METHOD': saved_config.get('videomae_method', 'cross'),
            'FREEZE': False,
            'TARGET_CHANNELS': 1024,
            'TARGET_SPATIAL_SIZE': 7,
            'DROPOUT': 0.2,
        },
        'LOSS': saved_config.get('LOSS', {
            'TAL': {
                'top_k': 10, 'alpha': 1.0, 'beta': 6.0, 'radius': 2.5,
                'iou_type': 'ciou', 'iou_weight': 2.0,
                'scale_cls_loss': 0.5, 'scale_box_loss': 7.5, 'scale_dfl_loss': 1.5,
                'soft_label': True
            }
        }),
        'active_checker': False,
    }

    model = build_yowoformer(config)
    model = model.to(device)
    model.load_state_dict(state_dict, strict=False)

    # Step 5: BN Calibration
    print(f"\nBN Calibration ({calibration_batches} batches)...")
    model.eval()

    # Enable train mode only for affected BN layers
    bn_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if any(name in k for k in nan_keys):
                module.train()
                module.running_mean.zero_()
                module.running_var.fill_(1.0)
                module.num_batches_tracked.zero_()
                module.momentum = None  # cumulative average = more stable
                bn_names.append(name)
                print(f"  Calibrating: {name}")

    # Build dataloader
    if dataset == 'ava':
        from YOWOFormer.cus_datasets.ava.load_data import build_ava_dataset
        train_dataset = build_ava_dataset(config, phase='train')
    else:
        from YOWOFormer.cus_datasets.ucf.load_data import build_ucf_dataset
        train_dataset = build_ucf_dataset(config, phase='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )

    with torch.no_grad():
        for i, (clips, boxes, labels) in enumerate(tqdm(
            train_loader, total=calibration_batches, desc='Calibrating'
        )):
            if i >= calibration_batches:
                break
            clips = clips.to(device)
            _ = model(clips)

    # Verify
    print("\nVerifying...")
    all_clean = True
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and name in bn_names:
            has_nan = torch.isnan(module.running_mean).any() or torch.isnan(module.running_var).any()
            if has_nan:
                print(f"  FAIL: {name} still NaN")
                all_clean = False
            else:
                print(f"  OK: {name}")

    # Step 6: Save
    model.eval()
    fixed_checkpoint = copy.deepcopy(checkpoint)
    fixed_checkpoint['model_state_dict'] = model.state_dict()
    fixed_checkpoint['nan_fix'] = {
        'original_nan_keys': nan_keys,
        'calibration_batches': calibration_batches,
    }

    save_path = Path(checkpoint_path)
    fixed_path = save_path.parent / f"{save_path.stem}_fixed{save_path.suffix}"
    torch.save(fixed_checkpoint, fixed_path)
    print(f"\nSaved: {fixed_path}")

    # Step 7: Quick test
    print("\nInference test...")
    dummy = torch.randn(1, 3, config['clip_length'], 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
    box_nan = torch.isnan(out[:, :4, :]).any().item()
    cls_nan = torch.isnan(out[:, 4:, :]).any().item()
    print(f"  Box NaN: {box_nan}, Cls NaN: {cls_nan}")

    if not box_nan and not cls_nan and all_clean:
        print("\nFixed successfully! Re-run evaluation with:")
        print(f"  uv run python evaluate_ava_fixed.py --checkpoint {fixed_path} --save_json")
    else:
        print("\nWARNING: Still has issues. May need retraining with FP32.")

    print("=" * 60)
    return str(fixed_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='weights/ava_yolon_mae_large_cross_freeze/best.pth')
    parser.add_argument('--data_root', type=str, default='data/AVA_Dataset')
    parser.add_argument('--dataset', type=str, default='ava')
    parser.add_argument('--calibration_batches', type=int, default=100,
                        help='Number of batches for BN recalibration')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    fix_checkpoint(
        args.checkpoint,
        data_root=args.data_root,
        dataset=args.dataset,
        calibration_batches=args.calibration_batches,
        batch_size=args.batch_size,
    )
