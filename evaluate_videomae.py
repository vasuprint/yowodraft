#!/usr/bin/env python3
"""
VideoMAE + YOWOFormer Evaluation Script
Specifically designed for models trained with train_videomae.py
"""

import torch
import torch.utils.data as data
import numpy as np
import argparse
from pathlib import Path
import sys
from tqdm import tqdm

# Add YOWOFormer to path
sys.path.append(str(Path(__file__).parent))

from YOWOFormer.model.TSN.YOWOFormer import build_yowoformer
from YOWOFormer.utils.box import non_max_suppression, box_iou
from YOWOFormer.utils.collate import collate_fn
from conceptcode.evaluate_map import compute_ap, YOWOFormerEvaluator


def detect_yolo_version(checkpoint_path):
    """Auto-detect YOLOv11 version from checkpoint weights"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # First check if config has the backbone2D info
    if 'config' in checkpoint and 'backbone2D' in checkpoint['config']:
        detected_version = checkpoint['config']['backbone2D']
        print(f"🔍 Detected YOLOv11 version from config: {detected_version}")
        return detected_version

    # Otherwise try to detect from model weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Check multiple layers to identify the model more accurately
    key_checks = {
        'net2D.backbone.2.cv2.conv.weight': (256, 192),  # m has 192 input channels
        'net2D.backbone.4.cv2.conv.weight': (512, 384),  # m has 384 input channels
        'net2D.backbone.6.cv2.conv.weight': (512, 768),  # m has 768 input channels
    }

    matches_m = True
    for key, expected_shape in key_checks.items():
        if key in state_dict:
            shape = state_dict[key].shape[:2]
            if shape != expected_shape:
                matches_m = False
                break

    if matches_m:
        print(f"🔍 Auto-detected YOLOv11 version from weight shapes: yolov11_m")
        return 'yolov11_m'

    # Check first conv layer as fallback
    first_conv_key = 'net2D.backbone.0.conv.weight'
    if first_conv_key in state_dict:
        out_channels = state_dict[first_conv_key].shape[0]
        if out_channels == 48:
            return 'yolov11_m'
        elif out_channels == 64:
            return 'yolov11_m'

    return None


def load_videomae_config_from_checkpoint(checkpoint):
    """Extract VideoMAE configuration from checkpoint"""
    config = checkpoint.get('config', {})

    # Extract VideoMAE specific settings
    videomae_config = {
        'backbone3D': config.get('backbone3D', 'videomae'),
        'videomae_variant': config.get('videomae_variant', 'base'),
        'videomae_method': config.get('videomae_method', 'simple'),
        'freeze_videomae': config.get('freeze_videomae', False),  # For eval, always unfrozen
    }

    # Extract BACKBONE3D config if available
    if 'BACKBONE3D' in config:
        backbone3d_config = config['BACKBONE3D']
        videomae_config['BACKBONE3D'] = backbone3d_config
    else:
        # Create default BACKBONE3D config
        videomae_config['BACKBONE3D'] = {
            'TYPE': 'videomae',
            'VARIANT': videomae_config['videomae_variant'],
            'METHOD': videomae_config['videomae_method'],
            'FREEZE': False,  # Always unfrozen for evaluation
            'TARGET_CHANNELS': 1024,
            'TARGET_SPATIAL_SIZE': 7,
            'DROPOUT': 0.2,
            'I3D': {
                'PRETRAIN': {
                    'default': None
                }
            }
        }

    return videomae_config


def main():
    parser = argparse.ArgumentParser(description='VideoMAE + YOWOFormer Evaluation')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--dataset', type=str, default='ucf',
                        choices=['ucf', 'ava'],
                        help='Dataset to evaluate on')

    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--conf_thres', type=float, default=0.005,
                        help='Confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.5,
                        help='NMS IoU threshold')
    parser.add_argument('--iou_type', type=str, default='0.5',
                        choices=['0.5', 'all'],
                        help='IoU threshold type (0.5 only or 0.5:0.95)')

    # Dataset paths
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override default dataset path')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🎯 VideoMAE + YOWOFormer Evaluation")
    print("="*60)

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        return

    # Load checkpoint
    print(f"\n📦 Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Auto-detect YOLOv11 version
    detected_backbone2d = detect_yolo_version(checkpoint_path)
    if not detected_backbone2d:
        print("❌ Could not detect YOLOv11 version from checkpoint!")
        return

    # Load VideoMAE configuration
    videomae_config = load_videomae_config_from_checkpoint(checkpoint)
    print(f"🔍 Detected VideoMAE configuration:")
    print(f"   Variant: {videomae_config['videomae_variant']}")
    print(f"   Method: {videomae_config['videomae_method']}")

    # Build configuration
    config = {
        # Dataset
        'dataset': args.dataset,
        'data_root': args.data_root or (f'data/{args.dataset.upper()}101-24' if args.dataset == 'ucf' else 'data/AVA_Dataset'),
        'num_classes': 24 if args.dataset == 'ucf' else 80,

        # Model - Use detected versions
        'backbone2D': detected_backbone2d,
        'backbone3D': videomae_config['backbone3D'],
        'fusion_module': checkpoint.get('config', {}).get('fusion_module', 'Simple'),
        'mode': checkpoint.get('config', {}).get('mode', 'decoupled'),
        'interchannels': checkpoint.get('config', {}).get('interchannels', [256, 256, 256]),

        # VideoMAE specific
        'videomae_variant': videomae_config['videomae_variant'],
        'videomae_method': videomae_config['videomae_method'],
        'BACKBONE3D': videomae_config['BACKBONE3D'],

        # Input
        'img_size': 224,
        'clip_length': 16,
        'sampling_rate': 1,

        # Evaluation
        'checkpoint_path': args.checkpoint,
        'batch_size': args.batch_size,
        'conf_thres': args.conf_thres,
        'nms_thres': args.nms_thres,
        'iou_type': args.iou_type,
        'num_workers': args.num_workers,

        # Loss config (needed for model building)
        'LOSS': {
            'TAL': {
                'top_k': 10,
                'alpha': 1.0,
                'beta': 6.0,
                'radius': 2.5,
                'iou_type': 'ciou',
                'iou_weight': 2.0,
                'scale_cls_loss': 1.0,
                'scale_box_loss': 7.5,
                'scale_dfl_loss': 1.5,
                'soft_label': False
            }
        },
        'active_checker': False,

        # Pretrain flags (for model building, not used in eval)
        'pretrain_2d': True,
        'pretrain_3d': True,
        'freeze_bb2D': False,
        'freeze_bb3D': False,  # Always unfrozen for evaluation
        'pretrain_path': None
    }

    # Override with checkpoint config if available
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        for key in ['interchannels', 'fusion_module', 'mode', 'num_classes']:
            if key in saved_config:
                config[key] = saved_config[key]

    print("\n" + "="*60)
    print("Model Configuration")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'train_loss' in checkpoint:
        print(f"Training Loss: {checkpoint['train_loss']:.4f}")
    print(f"\nDataset: {config['dataset'].upper()}")
    print(f"Num Classes: {config['num_classes']}")
    print(f"\nModel Architecture:")
    print(f"  2D Backbone: {detected_backbone2d}")
    print(f"  3D Backbone: VideoMAE-{videomae_config['videomae_variant']}")
    print(f"  VideoMAE Method: {videomae_config['videomae_method']}")
    print(f"  Fusion Module: {config['fusion_module']}")
    print(f"  Mode: {config['mode']}")
    print(f"\nEvaluation Settings:")
    print(f"  Confidence Threshold: {config['conf_thres']}")
    print(f"  NMS Threshold: {config['nms_thres']}")
    print(f"  IoU Type: {config['iou_type']}")
    print(f"  Batch Size: {config['batch_size']}")
    print("="*60)

    # Check dataset
    data_path = Path(config['data_root'])
    if not data_path.exists():
        print(f"\n❌ Dataset not found at: {data_path}")
        print("   Please download the dataset first.")
        return

    # Run evaluation
    evaluator = YOWOFormerEvaluator(config)

    # Override load_model to use our detected config
    original_load = evaluator.load_model
    def custom_load():
        print(f"\n🔧 Building YOWOFormer model with VideoMAE...")
        print(f"📦 Loading checkpoint from: {checkpoint_path}")

        # Build model with correct config
        evaluator.model = build_yowoformer(config)
        evaluator.model = evaluator.model.to(evaluator.device)

        # Load model state with strict=False to handle architecture mismatches
        if 'model_state_dict' in checkpoint:
            missing_keys, unexpected_keys = evaluator.model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
            print(f"   ✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'train_loss' in checkpoint:
                print(f"   Training loss: {checkpoint['train_loss']:.4f}")

            # Report any loading issues
            if missing_keys:
                print(f"   ⚠️  Missing keys: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"      - {key}")
                else:
                    for key in missing_keys[:3]:
                        print(f"      - {key}")
                    print(f"      ... and {len(missing_keys)-3} more")

            if unexpected_keys:
                print(f"   ⚠️  Unexpected keys: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"      - {key}")
                else:
                    for key in unexpected_keys[:3]:
                        print(f"      - {key}")
                    print(f"      ... and {len(unexpected_keys)-3} more")
        else:
            missing_keys, unexpected_keys = evaluator.model.load_state_dict(
                checkpoint, strict=False
            )

        evaluator.model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in evaluator.model.parameters())
        print(f"   Total params: {total_params/1e6:.2f}M")

        # VideoMAE specific info
        if hasattr(evaluator.model, 'net3D') and hasattr(evaluator.model.net3D, 'videomae'):
            videomae_params = sum(p.numel() for p in evaluator.model.net3D.videomae.parameters())
            print(f"   VideoMAE encoder: {videomae_params/1e6:.2f}M params")

    evaluator.load_model = custom_load
    evaluator.load_model()
    evaluator.setup_dataset()
    results = evaluator.evaluate()

    if results:
        print("\n✅ Evaluation complete!")

        # Save results
        import json
        results_file = checkpoint_path.parent / f'evaluation_results_videomae.json'

        # Add metadata
        results['metadata'] = {
            'checkpoint': str(checkpoint_path),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'train_loss': float(checkpoint.get('train_loss', 0)) if 'train_loss' in checkpoint else None,
            'model_architecture': {
                'backbone2D': detected_backbone2d,
                'backbone3D': 'videomae',
                'videomae_variant': videomae_config['videomae_variant'],
                'videomae_method': videomae_config['videomae_method'],
                'fusion_module': config['fusion_module']
            },
            'evaluation_settings': {
                'conf_thres': config['conf_thres'],
                'nms_thres': config['nms_thres'],
                'iou_type': config['iou_type']
            }
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   Results saved to: {results_file}")


if __name__ == "__main__":
    main()
