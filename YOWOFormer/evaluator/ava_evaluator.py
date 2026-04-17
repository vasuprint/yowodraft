#!/usr/bin/env python3
"""
AVA Dataset Evaluator for YOWOFormer
Using official AVA evaluation metrics for reliable paper results
"""

import torch
import torch.utils.data as data
import numpy as np
import csv
import os
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import logging
import json
import pprint

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from YOWOFormer.model.TSN.YOWOFormer import build_yowoformer
from YOWOFormer.utils.box import non_max_suppression
from YOWOFormer.utils.collate import collate_fn
from YOWOFormer.cus_datasets.ava.load_data import build_ava_dataset

# Import official AVA evaluation
from .ava_official import object_detection_evaluation
from .ava_official import standard_fields


class AVAEvaluator:
    """
    Official AVA evaluation for YOWOFormer models.
    Produces results compatible with AVA benchmark and papers.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # AVA specific settings (following standard evaluation protocol)
        self.num_classes = 80
        # Standard AVA confidence threshold used in most papers (SlowFast, YOWOv2, etc.)
        self.conf_threshold = config.get('conf_thres', 0.01)  # AVA standard: 0.01
        self.nms_threshold = config.get('nms_thres', 0.5)  # NMS for duplicate removal

        # Black list classes (not evaluated in AVA)
        self.black_list = [2, 16, 18, 19, 21, 23, 25, 31, 32, 33, 35, 39, 40, 42, 44, 50, 53, 55, 71, 75]

        # File paths
        self.labelmap_file = config.get('labelmap', 'data/AVA_Dataset/annotations/ava_v2.2/ava_action_list_v2.2_for_activitynet_2019.pbtxt')
        self.groundtruth_file = config.get('groundtruth', 'data/AVA_Dataset/annotations/ava_v2.2/ava_val_v2.2.csv')
        self.detection_file = config.get('detections', 'ava_detections.csv')

        print(f"🖥️ Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name()}")

    def load_model(self, checkpoint_path):
        """Load YOWOFormer model from checkpoint"""
        print(f"\n🔧 Loading YOWOFormer model...")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract configuration from checkpoint
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            # Update model configuration
            for key in ['backbone2D', 'backbone3D', 'videomae_variant',
                       'videomae_method', 'fusion_module', 'interchannels']:
                if key in saved_config:
                    self.config[key] = saved_config[key]

        # Build model
        self.model = build_yowoformer(self.config)
        self.model = self.model.to(self.device)

        # Load weights (support both regular and EMA checkpoints)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("   Loading model weights...")
        elif 'ema_state_dict' in checkpoint:
            state_dict = checkpoint['ema_state_dict']
            print("   Loading EMA weights...")
        else:
            state_dict = checkpoint

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        print(f"   ✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'train_loss' in checkpoint:
            print(f"   Training loss: {checkpoint['train_loss']:.4f}")

        if missing:
            print(f"   ⚠️ Missing keys: {len(missing)}")
        if unexpected:
            print(f"   ⚠️ Unexpected keys: {len(unexpected)}")

        self.model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Total params: {total_params/1e6:.2f}M")

    def setup_dataset(self):
        """Setup AVA test dataset"""
        print(f"\n📊 Setting up AVA test dataset...")

        # Custom collate function for AVA
        def ava_collate_fn(batch):
            clips = []
            video_names = []
            timestamps = []

            for b in batch:
                clips.append(b[0])
                # For AVA, we need video name and timestamp
                if len(b) > 3:
                    video_names.append(b[3])  # video name
                    timestamps.append(b[4])   # timestamp
                else:
                    # Fallback if format is different
                    video_names.append("unknown")
                    timestamps.append(0.0)

            clips = torch.stack(clips, dim=0)
            return clips, video_names, timestamps

        self.test_dataset = build_ava_dataset(self.config, phase='test')

        # Create dataloader
        self.test_loader = data.DataLoader(
            self.test_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            collate_fn=ava_collate_fn,
            num_workers=self.config.get('num_workers', 6),
            pin_memory=True,
            drop_last=False
        )

        print(f"   Test samples: {len(self.test_dataset):,}")
        print(f"   Test batches: {len(self.test_loader):,}")

    @torch.no_grad()
    def generate_detections(self):
        """Generate detection results in AVA CSV format"""
        print(f"\n🚀 Generating detections...")
        print("="*60)

        results = []

        # Progress bar
        pbar = tqdm(self.test_loader, desc='Processing')

        for clips, video_names, timestamps in pbar:
            # Move to device
            clips = clips.to(self.device)

            # Model inference
            outputs = self.model(clips)
            outputs = outputs.cpu()

            # Process each image's detections
            for output, video_name, timestamp in zip(outputs, video_names, timestamps):
                # Apply NMS
                output = output.unsqueeze(0)
                output = non_max_suppression(output, self.conf_threshold, self.nms_threshold)[0]

                if output is None or len(output) == 0:
                    continue

                # Normalize coordinates to [0, 1]
                H = W = self.config['img_size']
                output[:, 0] /= W  # x1
                output[:, 1] /= H  # y1
                output[:, 2] /= W  # x2
                output[:, 3] /= H  # y2

                # Process each detection
                for detection in output:
                    x1, y1, x2, y2, conf = detection[:5].tolist()

                    # Multi-label: check all class probabilities
                    if len(detection) > 5:
                        class_scores = detection[5:].tolist()

                        # Save detection for each class with score > threshold
                        for class_idx in range(min(len(class_scores), self.num_classes)):
                            class_id = class_idx + 1  # AVA classes are 1-indexed

                            # Skip black-listed classes
                            if class_id in self.black_list:
                                continue

                            # Combined score (objectness * class probability)
                            final_score = conf * class_scores[class_idx]

                            if final_score > self.conf_threshold:
                                results.append([
                                    video_name,
                                    float(timestamp),
                                    round(x1, 3),
                                    round(y1, 3),
                                    round(x2, 3),
                                    round(y2, 3),
                                    class_id,
                                    round(final_score, 3)
                                ])
                    else:
                        # Single class detection (fallback)
                        class_id = int(detection[5].item()) + 1 if len(detection) > 5 else 1
                        if class_id not in self.black_list:
                            results.append([
                                video_name,
                                float(timestamp),
                                round(x1, 3),
                                round(y1, 3),
                                round(x2, 3),
                                round(y2, 3),
                                class_id,
                                round(conf, 3)
                            ])

        # Save detections to CSV
        print(f"\n💾 Saving {len(results)} detections to {self.detection_file}")
        with open(self.detection_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results)

        return len(results)

    def evaluate(self):
        """Run official AVA evaluation"""
        print(f"\n📈 Running official AVA evaluation...")
        print("="*60)

        # Read label map
        categories, class_whitelist = self.read_labelmap(self.labelmap_file)
        logging.info("CATEGORIES (%d):\n%s", len(categories),
                    pprint.pformat(categories, indent=2))

        # Create evaluator
        pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(categories)

        # Load ground truth
        print("Loading ground truth...")
        boxes, labels, _, included_keys = self.read_csv(
            self.groundtruth_file, class_whitelist, capacity=0
        )

        # Add ground truth to evaluator
        for image_key in boxes:
            pascal_evaluator.add_single_ground_truth_image_info(
                image_key, {
                    standard_fields.InputDataFields.groundtruth_boxes:
                        np.array(boxes[image_key], dtype=float),
                    standard_fields.InputDataFields.groundtruth_classes:
                        np.array(labels[image_key], dtype=int),
                    standard_fields.InputDataFields.groundtruth_difficult:
                        np.zeros(len(boxes[image_key]), dtype=bool)
                })

        # Load detections
        print("Loading detections...")
        det_boxes, det_labels, det_scores, _ = self.read_csv(
            self.detection_file, class_whitelist, capacity=50
        )

        # Add detections to evaluator
        for image_key in det_boxes:
            if image_key not in included_keys:
                continue
            pascal_evaluator.add_single_detected_image_info(
                image_key, {
                    standard_fields.DetectionResultFields.detection_boxes:
                        np.array(det_boxes[image_key], dtype=float),
                    standard_fields.DetectionResultFields.detection_classes:
                        np.array(det_labels[image_key], dtype=int),
                    standard_fields.DetectionResultFields.detection_scores:
                        np.array(det_scores[image_key], dtype=float)
                })

        # Run evaluation
        print("\nComputing metrics...")
        metrics = pascal_evaluator.evaluate()

        # Print results
        print("\n" + "="*60)
        print("🎯 EVALUATION RESULTS")
        print("="*60)
        pprint.pprint(metrics, indent=2)

        # Extract key metrics
        result_dict = {
            'mAP@0.5': float(metrics.get('PascalBoxes_Precision/mAP@0.5IOU', 0)),
            'PerformanceByCategory': {}
        }

        # Per-category AP
        for category in categories:
            cat_name = category['name']
            cat_id = category['id']
            key = f'PascalBoxes_PerformanceByCategory/AP@0.5IOU/{cat_name}'
            if key in metrics:
                result_dict['PerformanceByCategory'][cat_name] = float(metrics[key])

        return result_dict

    def read_labelmap(self, labelmap_file):
        """Read label map file"""
        labelmap = []
        class_ids = set()

        with open(labelmap_file, 'r') as f:
            name = ""
            class_id = ""
            for line in f:
                if line.startswith("  name:"):
                    name = line.split('"')[1]
                elif line.startswith("  id:") or line.startswith("  label_id:"):
                    class_id = int(line.strip().split(" ")[-1])
                    labelmap.append({"id": class_id, "name": name})
                    class_ids.add(class_id)

        return labelmap, class_ids

    def read_csv(self, csv_file, class_whitelist=None, capacity=0):
        """Read CSV file in AVA format"""
        from collections import defaultdict
        import heapq

        entries = defaultdict(list)
        boxes = defaultdict(list)
        labels = defaultdict(list)
        scores = defaultdict(list)
        all_keys = set()

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) not in [2, 7, 8]:
                    continue

                # Create image key
                image_key = f"{row[0]},{float(row[1]):.6f}"
                all_keys.add(image_key)

                if len(row) == 2:
                    continue

                x1, y1, x2, y2 = [float(n) for n in row[2:6]]
                action_id = int(row[6])

                if class_whitelist and action_id not in class_whitelist:
                    continue

                score = 1.0
                if len(row) == 8:
                    score = float(row[7])

                if capacity < 1 or len(entries[image_key]) < capacity:
                    heapq.heappush(entries[image_key], (score, action_id, y1, x1, y2, x2))
                elif score > entries[image_key][0][0]:
                    heapq.heapreplace(entries[image_key], (score, action_id, y1, x1, y2, x2))

        for image_key in entries:
            # Sort by score (descending)
            entry = sorted(entries[image_key], key=lambda x: -x[0])
            for item in entry:
                score, action_id, y1, x1, y2, x2 = item
                boxes[image_key].append([y1, x1, y2, x2])
                labels[image_key].append(action_id)
                scores[image_key].append(score)

        return boxes, labels, scores, all_keys


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
    parser = argparse.ArgumentParser(description='AVA Evaluation for YOWOFormer')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--conf_thres', type=float, default=0.01,
                        help='Confidence threshold (AVA standard: 0.01, used by SlowFast, YOWOv2, etc.)')
    parser.add_argument('--nms_thres', type=float, default=0.5,
                        help='NMS IoU threshold (standard: 0.5 for duplicate removal)')
    parser.add_argument('--data_root', type=str, default='data/AVA_Dataset',
                        help='Path to AVA dataset')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of dataloader workers')
    parser.add_argument('--save_json', action='store_true',
                        help='Save results to JSON file')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🎯 YOWOFormer AVA Evaluation (Official Metrics)")
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
        'detections': str(checkpoint_path.parent / 'ava_detections.csv'),

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
                'scale_cls_loss': 1.0,
                'scale_box_loss': 7.5,
                'scale_dfl_loss': 1.5,
                'soft_label': False
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
        results_file = checkpoint_path.parent / 'ava_evaluation_results.json'

        # Add metadata
        results['metadata'] = {
            'checkpoint': str(checkpoint_path),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'train_loss': float(checkpoint.get('train_loss', 0)) if 'train_loss' in checkpoint else None,
            'num_detections': num_detections,
            'evaluation_type': 'official_ava',
            'conf_thres': args.conf_thres,
            'nms_thres': args.nms_thres
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