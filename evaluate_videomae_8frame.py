#!/usr/bin/env python3
"""
VideoMAE + YOWOFormer Evaluation Script
Supports both 8-frame and 16-frame models with GFLOPs and FPS measurement

Features:
- Auto-detect model configuration from checkpoint
- GFLOPs measurement
- FPS measurement (synthetic and real data)
- mAP evaluation for UCF101-24
"""

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import argparse
from pathlib import Path
import sys
import time
from tqdm import tqdm
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from YOWOFormer.model.TSN.YOWOFormer import build_yowoformer
from YOWOFormer.utils.collate import collate_fn


def detect_model_config(checkpoint_path):
    """Auto-detect model configuration from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})

    detected = {
        'backbone2D': config.get('backbone2D', 'yolov11_n'),
        'backbone3D': config.get('backbone3D', 'videomae'),
        'clip_length': config.get('clip_length', 16),
        'sampling_rate': config.get('sampling_rate', 1),
        'interchannels': config.get('interchannels', [256, 256, 256]),
        'fusion_module': config.get('fusion_module', 'Simple'),
        'mode': config.get('mode', 'decoupled'),
        'videomae_variant': config.get('videomae_variant', 'base'),
        'videomae_method': config.get('videomae_method', 'cross'),
        'num_classes': config.get('num_classes', 24),
        'dataset': config.get('dataset', 'ucf'),
        'BACKBONE3D': config.get('BACKBONE3D', {}),
        'epoch': checkpoint.get('epoch', 'unknown'),
        'train_loss': checkpoint.get('train_loss', None),
    }

    return detected, checkpoint


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_gflops(model, input_shape, device):
    """
    Measure GFLOPs using multiple methods
    """
    gflops = None
    method_used = None

    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    # Method 1: fvcore
    try:
        from fvcore.nn import FlopCountAnalysis

        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy_input)
            gflops = flops.total() / 1e9
            method_used = "fvcore"
            return gflops, method_used

    except ImportError:
        print("   fvcore not installed, trying ptflops...")
    except Exception as e:
        print(f"   fvcore failed: {str(e)[:50]}")

    # Method 2: ptflops
    try:
        from ptflops import get_model_complexity_info

        def input_constructor(input_res):
            return torch.randn(1, *input_res).to(device)

        input_res = input_shape[1:]
        macs, _ = get_model_complexity_info(
            model,
            input_res,
            input_constructor=input_constructor,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        gflops = macs / 1e9
        method_used = "ptflops"
        return gflops, method_used

    except ImportError:
        print("   ptflops not installed, trying thop...")
    except Exception as e:
        print(f"   ptflops failed: {str(e)[:50]}")

    # Method 3: thop
    try:
        from thop import profile

        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
            gflops = macs / 1e9
            method_used = "thop"
            return gflops, method_used

    except ImportError:
        print("   thop not installed")
    except Exception as e:
        print(f"   thop failed: {str(e)[:50]}")

    # Fallback: estimation
    total_params, _ = count_parameters(model)
    gflops = total_params * 2 / 1e9
    method_used = "estimated"
    print(f"   Install fvcore for accurate GFLOPs: pip install fvcore")

    return gflops, method_used


def measure_fps_synthetic(model, input_shape, device, num_warmup=10, num_runs=100):
    """Measure FPS with synthetic input"""
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Measure
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(dummy_input)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            timings.append(time.perf_counter() - start)

    timings = np.array(timings)
    latency_mean = timings.mean() * 1000
    latency_std = timings.std() * 1000
    fps = 1.0 / timings.mean()
    fps_std = fps * (latency_std / latency_mean)

    return fps, fps_std, latency_mean, latency_std


def measure_fps_dataloader(model, dataloader, device, max_batches=50):
    """Measure FPS with real data"""
    model.eval()
    total_time = 0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (clips, boxes, labels) in enumerate(tqdm(dataloader, desc="FPS (real data)")):
            if batch_idx >= max_batches:
                break

            clips = clips.to(device, non_blocking=True)
            batch_size = clips.shape[0]

            if device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(clips)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            total_time += (time.perf_counter() - start)
            num_samples += batch_size

    fps = num_samples / total_time if total_time > 0 else 0
    return fps, total_time, num_samples


def evaluate_map(model, config, device):
    """Evaluate mAP on UCF101-24"""
    try:
        from conceptcode.evaluate_map import YOWOFormerEvaluator

        evaluator = YOWOFormerEvaluator(config)

        def custom_load():
            evaluator.model = model
            evaluator.model.eval()

        evaluator.load_model = custom_load
        evaluator.load_model()
        evaluator.setup_dataset()
        results = evaluator.evaluate()

        return results

    except Exception as e:
        print(f"   mAP evaluation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='VideoMAE Evaluation with GFLOPs/FPS')

    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (.pth)')

    # Dataset
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset (auto-detect if not specified)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Dataset root path')

    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--conf_thres', type=float, default=0.005)
    parser.add_argument('--nms_thres', type=float, default=0.5)
    parser.add_argument('--iou_type', type=str, default='0.5', choices=['0.5', 'all'])
    parser.add_argument('--num_workers', type=int, default=4)

    # Performance measurement
    parser.add_argument('--fps_warmup', type=int, default=10)
    parser.add_argument('--fps_runs', type=int, default=100)
    parser.add_argument('--fps_with_data', action='store_true', default=False,
                        help='Also measure FPS with real data')
    parser.add_argument('--fps_data_batches', type=int, default=50)

    # Options
    parser.add_argument('--skip_map', action='store_true', default=False,
                        help='Skip mAP evaluation')
    parser.add_argument('--skip_gflops', action='store_true', default=False,
                        help='Skip GFLOPs measurement')
    parser.add_argument('--skip_fps', action='store_true', default=False,
                        help='Skip FPS measurement')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("VideoMAE Evaluation - GFLOPs & FPS Measurement")
    print("="*60)

    # Check checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\nCheckpoint not found: {checkpoint_path}")
        return

    # Load and detect config
    print(f"\nLoading checkpoint: {checkpoint_path.name}")
    detected_config, checkpoint = detect_model_config(checkpoint_path)

    # Override dataset if specified
    if args.dataset:
        detected_config['dataset'] = args.dataset
        detected_config['num_classes'] = 24 if args.dataset == 'ucf' else 80

    # Print configuration
    print(f"\n{'='*60}")
    print("Detected Configuration")
    print(f"{'='*60}")
    print(f"  Backbone 2D:      {detected_config['backbone2D']}")
    print(f"  Backbone 3D:      {detected_config['backbone3D']}")
    print(f"  Clip Length:      {detected_config['clip_length']} frames")
    print(f"  Sampling Rate:    {detected_config['sampling_rate']}")
    print(f"  VideoMAE Variant: {detected_config['videomae_variant']}")
    print(f"  VideoMAE Method:  {detected_config['videomae_method']}")
    print(f"  Interchannels:    {detected_config['interchannels']}")
    print(f"  Dataset:          {detected_config['dataset']}")
    print(f"  Num Classes:      {detected_config['num_classes']}")
    print(f"  Epoch:            {detected_config['epoch']}")
    if detected_config['train_loss']:
        print(f"  Train Loss:       {detected_config['train_loss']:.4f}")
    print(f"{'='*60}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Build config
    config = {
        'dataset': detected_config['dataset'],
        'data_root': args.data_root or (
            f'data/{detected_config["dataset"].upper()}101-24'
            if detected_config['dataset'] == 'ucf' else 'data/AVA_Dataset'
        ),
        'num_classes': detected_config['num_classes'],

        'backbone2D': detected_config['backbone2D'],
        'backbone3D': detected_config['backbone3D'],
        'fusion_module': detected_config['fusion_module'],
        'mode': detected_config['mode'],
        'interchannels': detected_config['interchannels'],

        'videomae_variant': detected_config['videomae_variant'],
        'videomae_method': detected_config['videomae_method'],
        'BACKBONE3D': detected_config['BACKBONE3D'],

        'img_size': 224,
        'clip_length': detected_config['clip_length'],
        'sampling_rate': detected_config['sampling_rate'],

        'batch_size': args.batch_size,
        'conf_thres': args.conf_thres,
        'nms_thres': args.nms_thres,
        'iou_type': args.iou_type,
        'num_workers': args.num_workers,

        'LOSS': {
            'TAL': {
                'top_k': 10, 'alpha': 1.0, 'beta': 6.0, 'radius': 2.5,
                'iou_type': 'ciou', 'iou_weight': 2.0,
                'scale_cls_loss': 1.0, 'scale_box_loss': 7.5, 'scale_dfl_loss': 1.5,
                'soft_label': False
            }
        },
        'active_checker': False,
        'pretrain_2d': True,
        'pretrain_3d': True,
        'freeze_bb2D': False,
        'freeze_bb3D': False,
        'pretrain_path': None
    }

    # Build model
    print(f"\nBuilding model...")
    model = build_yowoformer(config)
    model = model.to(device)

    # Load weights
    if 'model_state_dict' in checkpoint:
        missing, unexpected = model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        print(f"  Checkpoint loaded successfully")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # Count parameters
    total_params, trainable_params = count_parameters(model)

    # Results dictionary
    results = {
        'checkpoint': str(checkpoint_path),
        'epoch': detected_config['epoch'],
        'train_loss': detected_config['train_loss'],
        'model': {
            'backbone2D': detected_config['backbone2D'],
            'backbone3D': detected_config['backbone3D'],
            'videomae_variant': detected_config['videomae_variant'],
            'videomae_method': detected_config['videomae_method'],
            'clip_length': detected_config['clip_length'],
            'sampling_rate': detected_config['sampling_rate'],
            'interchannels': detected_config['interchannels'],
            'total_params': total_params,
            'total_params_M': round(total_params / 1e6, 2),
        },
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    print(f"\n{'='*60}")
    print("Model Statistics")
    print(f"{'='*60}")
    print(f"  Total parameters:     {total_params/1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")

    # Measure GFLOPs
    if not args.skip_gflops:
        print(f"\n{'='*60}")
        print("GFLOPs Measurement")
        print(f"{'='*60}")

        input_shape = (1, 3, config['clip_length'], config['img_size'], config['img_size'])
        print(f"  Input shape: {input_shape}")

        gflops, method = measure_gflops(model, input_shape, device)

        results['gflops'] = {
            'value': round(gflops, 2),
            'method': method,
        }

        print(f"  GFLOPs: {gflops:.2f} ({method})")

    # Measure FPS (synthetic)
    if not args.skip_fps:
        print(f"\n{'='*60}")
        print("FPS Measurement (Synthetic Input)")
        print(f"{'='*60}")

        input_shape = (1, 3, config['clip_length'], config['img_size'], config['img_size'])
        print(f"  Input shape: {input_shape}")
        print(f"  Warmup: {args.fps_warmup} iterations")
        print(f"  Runs: {args.fps_runs} iterations")

        fps, fps_std, latency_ms, latency_std = measure_fps_synthetic(
            model, input_shape, device,
            num_warmup=args.fps_warmup,
            num_runs=args.fps_runs
        )

        results['fps'] = {
            'synthetic': {
                'fps': round(fps, 2),
                'fps_std': round(fps_std, 2),
                'latency_ms': round(latency_ms, 2),
                'latency_std_ms': round(latency_std, 2),
            }
        }

        print(f"\n  FPS:     {fps:.2f} +/- {fps_std:.2f}")
        print(f"  Latency: {latency_ms:.2f} +/- {latency_std:.2f} ms")

    # Measure FPS with real data
    if not args.skip_fps and args.fps_with_data:
        print(f"\n{'='*60}")
        print("FPS Measurement (Real Data)")
        print(f"{'='*60}")

        data_path = Path(config['data_root'])
        if data_path.exists():
            if config['dataset'] == 'ucf':
                from YOWOFormer.cus_datasets.ucf.load_data import build_ucf_dataset
                test_dataset = build_ucf_dataset(config, phase='test')
            elif config['dataset'] == 'ava':
                from YOWOFormer.cus_datasets.ava.load_data import build_ava_dataset
                test_dataset = build_ava_dataset(config, phase='test')

            test_loader = data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn
            )

            fps_real, total_time, num_samples = measure_fps_dataloader(
                model, test_loader, device, max_batches=args.fps_data_batches
            )

            results['fps']['real_data'] = {
                'fps': round(fps_real, 2),
                'total_time': round(total_time, 2),
                'num_samples': num_samples,
            }

            print(f"\n  FPS (real data): {fps_real:.2f}")
            print(f"  Samples: {num_samples} in {total_time:.2f}s")
        else:
            print(f"  Dataset not found: {data_path}")

    # mAP Evaluation
    if not args.skip_map:
        print(f"\n{'='*60}")
        print("mAP Evaluation")
        print(f"{'='*60}")

        data_path = Path(config['data_root'])
        if data_path.exists():
            eval_results = evaluate_map(model, config, device)

            if eval_results:
                results['mAP'] = eval_results
                print(f"\n  mAP evaluation complete!")
                if 'mAP@0.5' in eval_results:
                    print(f"  mAP@0.5: {eval_results['mAP@0.5']:.4f}")
        else:
            print(f"  Dataset not found: {data_path}")
            print(f"  Skipping mAP evaluation...")

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:      {detected_config['backbone3D']}-{detected_config['videomae_variant']}")
    print(f"  Input:      {config['clip_length']} frames @ {config['img_size']}x{config['img_size']}")
    print(f"  Parameters: {total_params/1e6:.2f}M")

    if 'gflops' in results:
        print(f"  GFLOPs:     {results['gflops']['value']:.2f}")

    if 'fps' in results:
        print(f"  FPS:        {results['fps']['synthetic']['fps']:.2f}")
        print(f"  Latency:    {results['fps']['synthetic']['latency_ms']:.2f}ms")

    if 'mAP' in results and results['mAP']:
        if 'mAP@0.5' in results['mAP']:
            print(f"  mAP@0.5:    {results['mAP']['mAP@0.5']:.4f}")

    print(f"{'='*60}")

    # Save results
    def to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return [to_serializable(i) for i in obj]
        return obj

    results = to_serializable(results)

    results_file = checkpoint_path.parent / f'eval_{checkpoint_path.stem}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
