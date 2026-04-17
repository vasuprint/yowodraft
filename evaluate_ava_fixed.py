#!/usr/bin/env python3
"""
Fixed AVA Evaluation Script for YOWOFormer
Correctly handles NMS output format: [x1, y1, x2, y2, score, class_idx]
"""

import sys
from pathlib import Path
import torch
import argparse
import logging

sys.path.append(str(Path(__file__).parent))

from YOWOFormer.evaluator.ava_evaluator_fixed import AVAEvaluator


def detect_yolo_version(checkpoint_path):
    """Auto-detect YOLO version from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'config' in checkpoint and 'backbone2D' in checkpoint['config']:
        return checkpoint['config']['backbone2D']

    # Try to detect from weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Simple detection
    first_conv_key = 'net2D.backbone.0.conv.weight'
    if first_conv_key in state_dict:
        out_channels = state_dict[first_conv_key].shape[0]
        if out_channels == 48:
            return 'yolov11_n'
        elif out_channels == 64:
            return 'yolov11_m'

    return 'yolov11_n'  # default


def main():
    parser = argparse.ArgumentParser(description='Fixed AVA Evaluation for YOWOFormer')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--conf_thres', type=float, default=0.01,
                        help='Confidence threshold (AVA standard: 0.01)')
    parser.add_argument('--nms_thres', type=float, default=0.5,
                        help='NMS IoU threshold (standard: 0.5)')
    parser.add_argument('--data_root', type=str, default='data/AVA_Dataset',
                        help='Path to AVA dataset')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of dataloader workers')
    parser.add_argument('--save_json', action='store_true',
                        help='Save results to JSON file')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🎯 YOWOFormer AVA Evaluation (FIXED VERSION)")
    print("="*60)
    print("📝 This version correctly handles NMS output format")
    print("   NMS output: [x1, y1, x2, y2, score, class_idx]")
    print("="*60)

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Check checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    # Load checkpoint to get configuration
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Detect backbone version
    backbone2d = detect_yolo_version(checkpoint_path)
    print(f"🔍 Detected YOLOv11 version: {backbone2d}")

    # Extract VideoMAE config
    videomae_variant = 'base'
    videomae_method = 'cross'

    if 'config' in checkpoint:
        videomae_variant = checkpoint['config'].get('videomae_variant', 'base')
        videomae_method = checkpoint['config'].get('videomae_method', 'cross')

    print(f"🔍 VideoMAE variant: {videomae_variant}")
    print(f"🔍 VideoMAE method: {videomae_method}")

    # Build configuration
    config = {
        # Dataset
        'dataset': 'ava',
        'data_root': args.data_root,
        'num_classes': 80,

        # Model
        'backbone2D': backbone2d,
        'backbone3D': 'videomae',
        'videomae_variant': videomae_variant,
        'videomae_method': videomae_method,
        'fusion_module': checkpoint.get('config', {}).get('fusion_module', 'CrossAttention'),
        'mode': 'decoupled',
        'interchannels': [256, 256, 256],

        # Input
        'img_size': 224,
        'clip_length': 16,
        'sampling_rate': 1,

        # Evaluation
        'batch_size': args.batch_size,
        'conf_thres': args.conf_thres,
        'nms_thres': args.nms_thres,
        'num_workers': args.num_workers,

        # File paths
        'labelmap': f'{args.data_root}/annotations/ava_v2.2/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
        'groundtruth': f'{args.data_root}/annotations/ava_v2.2/ava_val_v2.2.csv',
        'detections': str(checkpoint_path.parent / 'ava_detections_fixed.csv'),

        # VideoMAE config
        'BACKBONE3D': {
            'TYPE': 'videomae',
            'VARIANT': videomae_variant,
            'METHOD': videomae_method,
            'FREEZE': False,
            'TARGET_CHANNELS': 1024,
            'TARGET_SPATIAL_SIZE': 7,
            'DROPOUT': 0.2,
            'I3D': {
                'PRETRAIN': {
                    'default': None
                }
            }
        },

        # Loss config (needed for model building)
        'LOSS': {
            'TAL': {
                'top_k': 10,
                'alpha': 1.0,
                'beta': 6.0,
                'radius': 2.5,
                'iou_type': 'ciou',
                'iou_weight': 2.0,
                'scale_cls_loss': 0.5,
                'scale_box_loss': 7.5,
                'scale_dfl_loss': 1.5,
                'soft_label': True
            }
        },

        # Other required configs
        'active_checker': False,
        'pretrain_2d': True,
        'pretrain_3d': True,
        'freeze_bb2D': False,
        'freeze_bb3D': False,
        'pretrain_path': None
    }

    # Override with checkpoint config if available
    if 'config' in checkpoint:
        for key in ['interchannels', 'fusion_module']:
            if key in checkpoint['config']:
                config[key] = checkpoint['config'][key]

    # Create evaluator
    evaluator = AVAEvaluator(config)

    # Load model
    evaluator.load_model(checkpoint_path)

    # Setup dataset
    evaluator.setup_dataset()

    # Generate detections
    num_detections = evaluator.generate_detections()
    print(f"\n✅ Generated {num_detections} detections")

    # Run evaluation
    results = evaluator.evaluate()

    # Save results
    if results and args.save_json:
        import json
        results_file = checkpoint_path.parent / 'ava_evaluation_results_fixed.json'

        # Add metadata
        results['metadata'] = {
            'checkpoint': str(checkpoint_path),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'train_loss': float(checkpoint.get('train_loss', 0)) if 'train_loss' in checkpoint else None,
            'num_detections': num_detections,
            'evaluation_type': 'official_ava_fixed',
            'conf_thres': args.conf_thres,
            'nms_thres': args.nms_thres,
            'note': 'Fixed version that correctly handles NMS output format'
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {results_file}")

    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print(f"📊 mAP@0.5: {results['mAP@0.5']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()