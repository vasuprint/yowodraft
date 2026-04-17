#!/usr/bin/env python3
"""
Model Statistics Measurement Tool
Measures GFLOPs, Parameters, and FPS for any checkpoint

Usage:
    # Single checkpoint
    uv run python measure_model_stats.py --checkpoint weights/model/best.pth

    # Multiple checkpoints
    uv run python measure_model_stats.py --checkpoints weights/model1/best.pth weights/model2/best.pth

    # All checkpoints in a directory
    uv run python measure_model_stats.py --dir weights/

    # Skip specific measurements
    uv run python measure_model_stats.py --checkpoint weights/model/best.pth --skip_gflops
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import time
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent))


def load_checkpoint_config(checkpoint_path):
    """Load config from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})

    return {
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
        'img_size': config.get('img_size', 224),
        'epoch': checkpoint.get('epoch', 'unknown'),
        'train_loss': checkpoint.get('train_loss', None),
    }, checkpoint


def build_model(config, checkpoint, device):
    """Build model and load weights"""
    from YOWOFormer.model.TSN.YOWOFormer import build_yowoformer

    full_config = {
        **config,
        'data_root': 'data/UCF101-24',
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

    model = build_yowoformer(full_config)
    model = model.to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model


def count_parameters(model):
    """Count parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Breakdown by component
    breakdown = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        breakdown[name] = params

    return {
        'total': total,
        'trainable': trainable,
        'total_M': round(total / 1e6, 2),
        'trainable_M': round(trainable / 1e6, 2),
        'breakdown': {k: round(v / 1e6, 2) for k, v in breakdown.items()}
    }


def measure_gflops(model, input_shape, device):
    """Measure GFLOPs"""
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    # Try fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy_input)
            gflops = flops.total() / 1e9
        return round(gflops, 2), "fvcore"
    except:
        pass

    # Try ptflops
    try:
        from ptflops import get_model_complexity_info
        def input_constructor(input_res):
            return torch.randn(1, *input_res).to(device)
        macs, _ = get_model_complexity_info(
            model, input_shape[1:],
            input_constructor=input_constructor,
            as_strings=False, print_per_layer_stat=False, verbose=False
        )
        return round(macs / 1e9, 2), "ptflops"
    except:
        pass

    # Try thop
    try:
        from thop import profile
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return round(macs / 1e9, 2), "thop"
    except:
        pass

    return None, "failed"


def measure_fps(model, input_shape, device, warmup=10, runs=100):
    """Measure FPS.  Uses FP16 autocast on CUDA for realistic throughput."""
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    use_amp = device.type == 'cuda'

    # Warmup — let CUDA kernels JIT-compile & caches fill
    with torch.no_grad():
        for _ in range(warmup):
            if use_amp:
                with torch.amp.autocast('cuda'):
                    _ = model(dummy_input)
                torch.cuda.synchronize()
            else:
                _ = model(dummy_input)

    # Measure
    timings = []
    with torch.no_grad():
        for _ in range(runs):
            if use_amp:
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.amp.autocast('cuda'):
                    _ = model(dummy_input)
                torch.cuda.synchronize()
            else:
                start = time.perf_counter()
                _ = model(dummy_input)
            timings.append(time.perf_counter() - start)

    timings = np.array(timings)
    fps = 1.0 / timings.mean()
    latency_ms = timings.mean() * 1000

    return {
        'fps': round(fps, 2),
        'latency_ms': round(latency_ms, 2),
        'fps_std': round(fps * timings.std() / timings.mean(), 2),
        'device': str(device),
        'amp': use_amp,
    }


def measure_checkpoint(checkpoint_path, device, skip_gflops=False, skip_fps=False,
                       fps_warmup=10, fps_runs=100):
    """Measure all stats for a checkpoint"""
    print(f"\n{'='*60}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"{'='*60}")

    # Load config
    config, checkpoint = load_checkpoint_config(checkpoint_path)

    print(f"\nConfig:")
    print(f"  Backbone 2D:  {config['backbone2D']}")
    print(f"  Backbone 3D:  {config['backbone3D']}")
    print(f"  Clip Length:  {config['clip_length']} frames")
    print(f"  Method:       {config.get('videomae_method', 'N/A')}")
    print(f"  Dataset:      {config['dataset']}")
    print(f"  Epoch:        {config['epoch']}")

    # Build model
    print(f"\nBuilding model...")
    model = build_model(config, checkpoint, device)

    results = {
        'checkpoint': str(checkpoint_path),
        'config': {
            'backbone2D': config['backbone2D'],
            'backbone3D': config['backbone3D'],
            'clip_length': config['clip_length'],
            'videomae_method': config.get('videomae_method', 'N/A'),
            'dataset': config['dataset'],
            'epoch': config['epoch'],
        }
    }

    # Parameters
    print(f"\n--- Parameters ---")
    params = count_parameters(model)
    results['parameters'] = params
    print(f"  Total:     {params['total_M']:.2f}M")
    print(f"  Trainable: {params['trainable_M']:.2f}M")
    print(f"  Breakdown:")
    for name, count in params['breakdown'].items():
        print(f"    {name}: {count:.2f}M")

    # GFLOPs
    if not skip_gflops:
        print(f"\n--- GFLOPs ---")
        input_shape = (1, 3, config['clip_length'], config['img_size'], config['img_size'])
        print(f"  Input: {input_shape}")
        gflops, method = measure_gflops(model, input_shape, device)
        if gflops:
            results['gflops'] = {'value': gflops, 'method': method}
            print(f"  GFLOPs: {gflops} ({method})")
        else:
            print(f"  GFLOPs: Failed to measure")

    # FPS
    if not skip_fps:
        print(f"\n--- FPS ---")
        input_shape = (1, 3, config['clip_length'], config['img_size'], config['img_size'])
        print(f"  Input: {input_shape}")
        print(f"  Warmup: {fps_warmup}, Runs: {fps_runs}")
        fps_results = measure_fps(model, input_shape, device, fps_warmup, fps_runs)
        results['fps'] = fps_results
        print(f"  FPS:     {fps_results['fps']:.2f} +/- {fps_results['fps_std']:.2f}")
        print(f"  Latency: {fps_results['latency_ms']:.2f}ms")

    # Cleanup
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description='Measure Model GFLOPs, Parameters, FPS')

    # Input options
    parser.add_argument('--checkpoint', type=str, help='Single checkpoint path')
    parser.add_argument('--checkpoints', type=str, nargs='+', help='Multiple checkpoint paths')
    parser.add_argument('--dir', type=str, help='Directory containing checkpoints')
    parser.add_argument('--pattern', type=str, default='**/best.pth',
                        help='Glob pattern for finding checkpoints in dir')

    # Measurement options
    parser.add_argument('--skip_gflops', action='store_true', help='Skip GFLOPs measurement')
    parser.add_argument('--skip_fps', action='store_true', help='Skip FPS measurement')
    parser.add_argument('--fps_warmup', type=int, default=10, help='FPS warmup iterations')
    parser.add_argument('--fps_runs', type=int, default=100, help='FPS measurement iterations')

    # Output
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')

    args = parser.parse_args()

    # Collect checkpoints
    checkpoints = []
    if args.checkpoint:
        checkpoints.append(Path(args.checkpoint))
    if args.checkpoints:
        checkpoints.extend([Path(p) for p in args.checkpoints])
    if args.dir:
        checkpoints.extend(Path(args.dir).glob(args.pattern))

    if not checkpoints:
        print("No checkpoints specified!")
        print("Use --checkpoint, --checkpoints, or --dir")
        return

    # Filter existing
    checkpoints = [p for p in checkpoints if p.exists()]
    print(f"\nFound {len(checkpoints)} checkpoint(s)")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDNN benchmark: enabled")

    # Measure each checkpoint
    all_results = []
    for checkpoint_path in checkpoints:
        try:
            results = measure_checkpoint(
                checkpoint_path, device,
                skip_gflops=args.skip_gflops,
                skip_fps=args.skip_fps,
                fps_warmup=args.fps_warmup,
                fps_runs=args.fps_runs
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nError measuring {checkpoint_path}: {e}")

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        print(f"{'Checkpoint':<30} {'Params':<10} {'GFLOPs':<10} {'FPS':<10} {'Latency':<10}")
        print(f"{'-'*80}")
        for r in all_results:
            name = Path(r['checkpoint']).parent.name
            params = r.get('parameters', {}).get('total_M', 'N/A')
            gflops = r.get('gflops', {}).get('value', 'N/A')
            fps = r.get('fps', {}).get('fps', 'N/A')
            latency = r.get('fps', {}).get('latency_ms', 'N/A')
            print(f"{name:<30} {params:<10} {gflops:<10} {fps:<10} {latency:<10}")
        print(f"{'='*80}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('model_stats.json')

    with open(output_path, 'w') as f:
        json.dump({
            'measurement_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device),
            'results': all_results
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
