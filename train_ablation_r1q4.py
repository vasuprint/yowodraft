#!/usr/bin/env python3
"""
Ablation Study: R1-Q4 - SQA without self-attention
Based on train_videomae_stable.py but with hardcoded config to avoid argparse issues.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import logging
import gc
import os
import psutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

sys.path.append(str(Path(__file__).parent))

from train_videomae_stable import VideoMAEYOWOTrainerStable, clear_gpu_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ucf', choices=['ucf', 'ava'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()

    # ============================================================
    # HARDCODED CONFIG - R1-Q4 Ablation: SQA no self-attention
    # ============================================================
    config = {
        # Dataset
        'dataset': args.dataset,
        'data_root': f'data/{args.dataset.upper()}101-24' if args.dataset == 'ucf' else 'data/AVA_Dataset',
        'num_classes': 24 if args.dataset == 'ucf' else 80,

        # Model
        'backbone2D': 'yolov11_n',
        'backbone3D': 'videomae',
        'fusion_module': 'Simple',
        'mode': 'decoupled',
        'interchannels': [256, 256, 256],

        # VideoMAE config - KEY: USE_SELF_ATTENTION = False
        'BACKBONE3D': {
            'TYPE': 'videomae',
            'VARIANT': 'base',
            'METHOD': 'cross',
            'FREEZE': True,
            'TARGET_CHANNELS': 1024,
            'TARGET_SPATIAL_SIZE': 7,
            'DROPOUT': 0.2,
            'USE_SELF_ATTENTION': False  # <--- ABLATION: disabled
        },

        # Training
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': 0.00033,
        'lr': 0.00033,
        'optimizer_type': 'adamw',
        'weight_decay': 0.0001,
        'gradient_clip': 0.5,
        'gradient_accumulation': 1,
        'use_warmup': True,

        # Scheduler
        'scheduler_type': 'cosine',
        'cosine_t_max': args.epochs,
        'cosine_eta_min': 0.000001,
        'multistep_milestones': None,
        'multistep_gamma': 0.5,

        # VideoMAE
        'freeze_videomae': True,
        'unfreeze_epoch': 6,
        'videomae_lr_ratio': 0.1,
        'use_fp16': True,
        'videomae_variant': 'base',
        'videomae_method': 'cross',

        # EMA
        'use_ema': True,
        'ema_decay': 0.999,

        # Input
        'img_size': 224,
        'clip_length': 16,
        'sampling_rate': 1,

        # System
        'num_workers': args.num_workers,
        'save_dir': f'weights/{args.dataset}_sqa_no_selfattn',
        'resume': None,

        # Pretrain
        'pretrain_2d': True,
        'pretrain_3d': True,
        'freeze_bb2D': False,
        'freeze_bb3D': True,
        'pretrain_path': None,

        # Loss config
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
        'active_checker': False,
        'max_step_warmup': 500
    }

    # Print key settings
    print("=" * 60)
    print("R1-Q4 ABLATION: SQA without self-attention")
    print("=" * 60)
    print(f"  Dataset: {args.dataset}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  USE_SELF_ATTENTION: {config['BACKBONE3D']['USE_SELF_ATTENTION']}")
    print(f"  Weight Decay: {config['weight_decay']}")
    print(f"  Unfreeze Epoch: {config['unfreeze_epoch']}")
    print(f"  Save dir: {config['save_dir']}")
    print("=" * 60)

    trainer = VideoMAEYOWOTrainerStable(config)
    trainer.train()


if __name__ == "__main__":
    main()
