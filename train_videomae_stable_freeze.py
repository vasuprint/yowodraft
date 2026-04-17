#!/usr/bin/env python3
"""
VideoMAE + YOWOFormer Training Script - FREEZE BACKBONES VERSION
Train ONLY Fusion Module and Detection Head

Frozen (NOT trained):
- YOLOv11 (2D Backbone)
- VideoMAE encoder + adapter (cross/token/hybrid/simple)

Trained:
- Fusion Module (Simple/CrossAttention/CFAM/etc.)
- Detection Head

Benefits:
- 2-4x faster training
- ~50% less GPU memory
- Faster convergence (5-10 epochs)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
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
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Enable CUDNN optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

sys.path.append(str(Path(__file__).parent))


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_memory_info():
    """Get current memory usage"""
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            'gpu_allocated': gpu_allocated,
            'gpu_reserved': gpu_reserved,
            'gpu_total': gpu_total,
            'gpu_free': gpu_total - gpu_allocated
        }
    return None


class FreezeBackboneTrainer:
    """Trainer that freezes both backbones and trains only Fusion + Head"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Clear GPU memory at start
        clear_gpu_memory()

        # Training history
        self.max_history_size = 100
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # Best model tracking
        self.best_loss = float('inf')
        self.best_epoch = 0

        # Force EMA for stable training
        self.config['use_ema'] = True

        # Setup logging
        self.setup_logging()

        # Log system info
        self.logger.info("="*60)
        self.logger.info("FREEZE BACKBONE Training")
        self.logger.info("Train ONLY: Fusion Module + Detection Head")
        self.logger.info("="*60)
        self.logger.info(f"Device: {self.device}")
        if self.device.type == 'cuda':
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        self.logger.info("="*60)

    def setup_logging(self):
        """Setup logging system"""
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"freeze_fusion_{self.config['dataset']}_{self.config['fusion_module']}_{timestamp}.log"
        log_path = save_dir / log_filename

        self.logger = logging.getLogger('FreezeBackbone')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("="*60)
        self.logger.info("Training Configuration - FREEZE BACKBONE")
        self.logger.info("="*60)
        self.log_config()
        self.save_config_json()

    def log_config(self):
        """Log configuration"""
        self.logger.info("\nFrozen Components (NOT trained):")
        self.logger.info(f"  2D Backbone: {self.config['backbone2D']} [FROZEN]")
        self.logger.info(f"  3D Backbone: VideoMAE-{self.config.get('videomae_variant', 'base')} [FROZEN]")
        self.logger.info(f"  VideoMAE Adapter: {self.config.get('videomae_method', 'cross')} [FROZEN]")

        self.logger.info("\nTrained Components:")
        self.logger.info(f"  Fusion Module: {self.config['fusion_module']} [TRAINED]")
        self.logger.info(f"  Detection Head: [TRAINED]")

        self.logger.info("\nTraining Parameters:")
        self.logger.info(f"  Epochs: {self.config['epochs']}")
        self.logger.info(f"  Batch Size: {self.config['batch_size']} x {self.config['gradient_accumulation']}")
        self.logger.info(f"  Learning Rate: {self.config['learning_rate']}")
        self.logger.info(f"  Optimizer: {self.config['optimizer_type']}")
        self.logger.info(f"  Scheduler: {self.config.get('scheduler_type', 'cosine')}")
        self.logger.info("="*60)

    def save_config_json(self):
        """Save configuration"""
        config_path = Path(self.config['save_dir']) / 'training_config.json'
        config_to_save = {}
        for key, value in self.config.items():
            if isinstance(value, Path):
                config_to_save[key] = str(value)
            elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                config_to_save[key] = value
            else:
                config_to_save[key] = str(value)

        config_to_save['metadata'] = {
            'training_start': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pytorch_version': torch.__version__,
            'training_mode': 'freeze_backbone',
            'frozen_components': ['backbone2D', 'backbone3D', 'videomae_adapter'],
            'trained_components': ['fusion_module', 'detection_head'],
            'checkpoint_policy': 'ema_only'
        }

        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=4)

    def setup_model(self):
        """Setup model with frozen backbones"""
        self.logger.info("\nSetting up model...")
        from YOWOFormer.model.TSN.YOWOFormer import build_yowoformer

        self.model = build_yowoformer(self.config)
        self.model = self.model.to(self.device)

        # Count parameters BEFORE freezing
        total_params = sum(p.numel() for p in self.model.parameters())

        # ============================================
        # FREEZE BACKBONES
        # ============================================
        self.logger.info("\n" + "="*60)
        self.logger.info("FREEZING BACKBONES...")
        self.logger.info("="*60)

        frozen_params = 0
        trainable_params = 0

        for name, param in self.model.named_parameters():
            # Check if parameter belongs to backbone
            is_backbone = any([
                'net2D' in name,      # YOLOv11 2D backbone
                'net3D' in name,      # VideoMAE 3D backbone (includes adapter)
                'backbone2D' in name,
                'backbone3D' in name,
            ])

            if is_backbone:
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                # This includes: fusion module, detection head, etc.
                param.requires_grad = True
                trainable_params += param.numel()

        # Log frozen/trainable breakdown
        self.logger.info(f"\nParameter Summary:")
        self.logger.info(f"  Total: {total_params/1e6:.2f}M")
        self.logger.info(f"  Frozen: {frozen_params/1e6:.2f}M ({frozen_params/total_params*100:.1f}%)")
        self.logger.info(f"  Trainable: {trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.1f}%)")

        # Log trainable components
        self.logger.info("\nTrainable Components:")
        trainable_components = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Get top-level component name
                component = name.split('.')[0]
                trainable_components.add(component)

        for comp in sorted(trainable_components):
            comp_params = sum(p.numel() for n, p in self.model.named_parameters()
                            if p.requires_grad and n.startswith(comp))
            self.logger.info(f"  {comp}: {comp_params/1e6:.2f}M")

        self.logger.info("="*60)

        # Load backbone from existing checkpoint if specified
        if self.config.get('load_backbone_from'):
            self.load_backbone_weights(self.config['load_backbone_from'])

        # Setup EMA
        from YOWOFormer.utils.EMA import EMA
        self.ema_model = EMA(self.model, decay=self.config.get('ema_decay', 0.999))
        self.logger.info(f"EMA enabled with decay: {self.config.get('ema_decay', 0.999)}")

        return self.model

    def load_backbone_weights(self, checkpoint_path):
        """Load backbone weights (net2D, net3D) and optionally head from checkpoint"""
        self.logger.info("\n" + "="*60)
        self.logger.info(f"Loading weights from: {checkpoint_path}")
        self.logger.info("="*60)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Categorize keys
        backbone_keys = {}
        head_keys = {}
        fusion_keys = {}

        for key, value in state_dict.items():
            if key.startswith('net2D.') or key.startswith('net3D.'):
                backbone_keys[key] = value
            elif 'fusion' in key.lower():
                fusion_keys[key] = value
            else:
                head_keys[key] = value

        self.logger.info(f"  Checkpoint contains:")
        self.logger.info(f"    - Backbone (net2D/net3D): {len(backbone_keys)} keys")
        self.logger.info(f"    - Fusion: {len(fusion_keys)} keys")
        self.logger.info(f"    - Head/Other: {len(head_keys)} keys")

        # Load backbone + head (skip fusion - we want new fusion module)
        load_dict = {}
        load_dict.update(backbone_keys)
        load_dict.update(head_keys)

        self.logger.info(f"\n  Loading: Backbone + Head ({len(load_dict)} keys)")
        self.logger.info(f"  Skipping: Fusion ({len(fusion_keys)} keys) - will use new {self.config['fusion_module']}")

        # Load with strict=False
        missing_keys, unexpected_keys = self.model.load_state_dict(
            load_dict, strict=False
        )

        # Analyze missing keys
        fusion_missing = [k for k in missing_keys if 'fusion' in k.lower()]
        other_missing = [k for k in missing_keys if k not in fusion_missing]

        self.logger.info(f"\n  Results:")
        self.logger.info(f"    - Missing (fusion, expected): {len(fusion_missing)}")
        if other_missing:
            self.logger.info(f"    - Missing (other, check!): {len(other_missing)}")
            for k in other_missing[:3]:
                self.logger.info(f"        {k}")
        self.logger.info(f"    - Unexpected: {len(unexpected_keys)}")

        self.logger.info("\n  Backbone + Head loaded successfully!")
        self.logger.info(f"  New Fusion Module ({self.config['fusion_module']}) will be trained from scratch")
        self.logger.info("="*60)

    def setup_data(self):
        """Setup data loaders"""
        self.logger.info("\nSetting up datasets...")
        from YOWOFormer.utils.collate import collate_fn

        if self.config['dataset'] == 'ucf':
            from YOWOFormer.cus_datasets.ucf.load_data import build_ucf_dataset
            self.train_dataset = build_ucf_dataset(self.config, phase='train')
            self.val_dataset = build_ucf_dataset(self.config, phase='test')
        elif self.config['dataset'] == 'ava':
            from YOWOFormer.cus_datasets.ava.load_data import build_ava_dataset
            self.train_dataset = build_ava_dataset(self.config, phase='train')
            self.val_dataset = build_ava_dataset(self.config, phase='test')

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            prefetch_factor=2,
            persistent_workers=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=True
        )

        self.logger.info(f"Datasets loaded!")
        self.logger.info(f"  Train samples: {len(self.train_dataset):,}")
        self.logger.info(f"  Val samples: {len(self.val_dataset):,}")

    def setup_training(self):
        """Setup optimizer and scheduler for trainable parameters only"""
        self.logger.info("\nSetting up training components...")

        # Get only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.logger.info(f"Optimizing {len(trainable_params)} parameter groups")

        # Proper weight decay grouping
        params_with_wd = []
        params_without_wd = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # No weight decay for bias and normalization
            if any(nd in name.lower() for nd in ['bias', 'norm', 'bn', 'ln']):
                params_without_wd.append(param)
            else:
                params_with_wd.append(param)

        self.logger.info(f"  WITH weight decay: {len(params_with_wd)}")
        self.logger.info(f"  WITHOUT weight decay: {len(params_without_wd)}")

        # Create optimizer
        if self.config['optimizer_type'] == 'sgd':
            self.optimizer = torch.optim.SGD([
                {"params": params_with_wd, "lr": self.config['learning_rate'],
                 "weight_decay": self.config['weight_decay']},
                {"params": params_without_wd, "lr": self.config['learning_rate'],
                 "weight_decay": 0.0}
            ], momentum=0.9)
        else:  # AdamW
            self.optimizer = torch.optim.AdamW([
                {"params": params_with_wd, "lr": self.config['learning_rate'],
                 "weight_decay": self.config['weight_decay']},
                {"params": params_without_wd, "lr": self.config['learning_rate'],
                 "weight_decay": 0.0}
            ])

        # Setup scheduler
        scheduler_type = self.config.get('scheduler_type', 'cosine')

        if scheduler_type == 'cosine':
            t_max = self.config.get('cosine_t_max', self.config['epochs'])
            eta_min = self.config.get('cosine_eta_min', self.config['learning_rate'] * 0.01)

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min
            )
            self.logger.info(f"CosineAnnealingLR: T_max={t_max}, eta_min={eta_min:.6f}")

        elif scheduler_type == 'multistep':
            milestones = self.config.get('multistep_milestones', [3, 5, 7])
            gamma = self.config.get('multistep_gamma', 0.5)

            self.scheduler = MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=gamma
            )
            self.logger.info(f"MultiStepLR: milestones={milestones}, gamma={gamma}")

        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config['learning_rate'] * 0.01
            )

        # Setup warmup
        if self.config.get('use_warmup', True):
            from YOWOFormer.utils.warmup_lr import LinearWarmup
            self.warmup = LinearWarmup(self.config)
            self.logger.info("Warmup enabled")

        # Loss function
        from YOWOFormer.utils.loss import TAL
        self.criterion = TAL(self.model, self.config)

        # Mixed Precision
        if self.config.get('use_fp16', True):
            self.scaler = torch.amp.GradScaler('cuda')
            self.logger.info("Mixed precision enabled")

        self.logger.info("Training setup complete!")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()

        # Keep backbones in eval mode for frozen layers
        if hasattr(self.model, 'net2D'):
            self.model.net2D.eval()
        if hasattr(self.model, 'net3D'):
            self.model.net3D.eval()

        epoch_loss = 0
        batch_count = 0
        accumulation_steps = self.config['gradient_accumulation']

        cnt_param_update = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')

        self.optimizer.zero_grad()
        clear_gpu_memory()

        try:
            for batch_idx, (clips, boxes, labels) in enumerate(pbar):
                clips = clips.to(self.device, non_blocking=True)

                # Forward pass with mixed precision
                if self.config.get('use_fp16', True):
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(clips)
                else:
                    outputs = self.model(clips)

                # Prepare targets
                targets = self.prepare_targets(boxes, labels)

                # Calculate loss
                loss_output = self.criterion(outputs, targets)

                if isinstance(loss_output, dict):
                    loss = loss_output.get('total_loss', sum(loss_output.values()))
                else:
                    loss = loss_output

                loss = loss / accumulation_steps

                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"Invalid loss at batch {batch_idx}, skipping...")
                    self.optimizer.zero_grad()
                    continue

                # Backward pass
                if self.config.get('use_fp16', True) and hasattr(self, 'scaler'):
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    cnt_param_update += 1

                    # Warmup
                    if epoch == 0 and self.config.get('use_warmup', True):
                        if hasattr(self, 'warmup'):
                            self.warmup(self.optimizer, cnt_param_update)

                    # Gradient clipping
                    if self.config.get('gradient_clip', 0) > 0:
                        if self.config.get('use_fp16', True) and hasattr(self, 'scaler'):
                            self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            max_norm=self.config['gradient_clip']
                        )

                    # Optimizer step
                    if self.config.get('use_fp16', True) and hasattr(self, 'scaler'):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Update EMA
                    self.ema_model.update(self.model)

                epoch_loss += loss.item() * accumulation_steps
                batch_count += 1

                # Update progress bar
                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({
                        'loss': f'{loss.item() * accumulation_steps:.4f}',
                        'avg': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.6f}',
                        'mode': 'fusion_only'
                    })

                # Memory cleanup
                if batch_idx % 50 == 0 and batch_idx > 0:
                    del clips, outputs, targets
                    clear_gpu_memory()

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            clear_gpu_memory()
            raise e

        clear_gpu_memory()
        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        return avg_epoch_loss

    def prepare_targets(self, boxes, labels):
        """Prepare targets for loss calculation"""
        targets = []
        num_classes = self.config['num_classes']

        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]

            if isinstance(box, torch.Tensor):
                box = box.to(self.device, non_blocking=True)
            else:
                box = torch.tensor(box).to(self.device, non_blocking=True)

            if isinstance(label, torch.Tensor):
                label = label.to(self.device, non_blocking=True)
            else:
                label = torch.tensor(label).to(self.device, non_blocking=True)

            if len(box) > 0:
                batch_idx = torch.ones(len(box), 1).to(self.device, non_blocking=True) * i

                if label.dim() == 1:
                    label_one_hot = torch.zeros(len(box), num_classes).to(self.device, non_blocking=True)
                    label_long = label.long()
                    for j in range(len(box)):
                        if 0 <= label_long[j] < num_classes:
                            label_one_hot[j, label_long[j]] = 1.0
                    label = label_one_hot
                elif label.dim() == 2:
                    if label.shape[1] != num_classes:
                        new_label = torch.zeros(len(box), num_classes).to(self.device, non_blocking=True)
                        new_label[:, :min(label.shape[1], num_classes)] = label[:, :min(label.shape[1], num_classes)]
                        label = new_label

                target = torch.cat([batch_idx, box, label], dim=1)
                targets.append(target)

        if targets:
            targets = torch.cat(targets, 0)
        else:
            targets = torch.zeros((0, 5 + num_classes)).to(self.device, non_blocking=True)

        return targets

    def save_checkpoint(self, epoch, train_loss):
        """Save EMA checkpoint"""
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"epoch_{epoch+1}.pth"
        ema_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.ema_model.ema.state_dict(),
            'train_loss': train_loss,
            'train_losses': self.train_losses,
            'learning_rates': self.learning_rates,
            'config': self.config,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'is_ema': True,
            'training_mode': 'freeze_backbone'
        }

        torch.save(ema_checkpoint, save_path)
        self.logger.info(f"Checkpoint saved: {save_path.name}")

        # Save best model
        if train_loss < self.best_loss:
            best_path = save_dir / "best.pth"
            torch.save(ema_checkpoint, best_path)
            self.logger.info(f"New best model! (loss: {train_loss:.4f})")
            self.best_loss = train_loss
            self.best_epoch = epoch

    def train(self):
        """Main training loop"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Starting FREEZE BACKBONE Training")
        self.logger.info("="*60)

        self.setup_model()
        self.setup_data()
        self.setup_training()

        start_epoch = 0
        total_epochs = self.config['epochs']

        self.logger.info(f"\nTraining for {total_epochs} epochs...")
        self.logger.info("="*60)

        start_time = time.time()

        for epoch in range(start_epoch, total_epochs):
            epoch_start = time.time()

            self.logger.info(f"\nEpoch {epoch+1}/{total_epochs}")
            self.logger.info("-"*40)

            # Memory info
            mem_info = get_memory_info()
            if mem_info:
                self.logger.info(f"GPU Memory: {mem_info['gpu_allocated']:.2f}GB / {mem_info['gpu_total']:.2f}GB")

            # Train
            train_loss = self.train_epoch(epoch)

            # Update history
            self.train_losses.append(train_loss)
            if len(self.train_losses) > self.max_history_size:
                self.train_losses = self.train_losses[-self.max_history_size:]

            # Step scheduler
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # Save checkpoint
            self.save_checkpoint(epoch, train_loss)

            # Summary
            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch+1} Summary:")
            self.logger.info(f"  Train Loss: {train_loss:.4f}")
            self.logger.info(f"  Learning Rate: {current_lr:.6f}")
            self.logger.info(f"  Epoch Time: {epoch_time/60:.2f} min")

        total_time = time.time() - start_time
        self.logger.info("\n" + "="*60)
        self.logger.info("Training Complete!")
        self.logger.info("="*60)
        self.logger.info(f"Total Time: {total_time/3600:.2f} hours")
        self.logger.info(f"Best Loss: {self.best_loss:.4f} (Epoch {self.best_epoch+1})")
        self.logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train Fusion Module Only (Freeze Backbones)')

    # Dataset
    parser.add_argument('--dataset', type=str, default='ucf', choices=['ucf', 'ava'])
    parser.add_argument('--data_root', type=str, default=None)

    # Model
    parser.add_argument('--yolo_version', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--videomae_variant', type=str, default='base',
                        choices=['small', 'base', 'large', 'huge'])
    parser.add_argument('--videomae_method', type=str, default='cross',
                        choices=['simple', 'token', 'cross', 'hybrid'])

    # FUSION MODULE - Main focus of this script!
    parser.add_argument('--fusion_module', type=str, default='CrossAttention',
                        choices=['CFAM', 'Simple', 'CrossAttention', 'CrossAttentionV2'])

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)  # Higher LR for fusion only
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adamw'])
    parser.add_argument('--gradient_accumulation', type=int, default=2)

    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'multistep'])
    parser.add_argument('--cosine_t_max', type=int, default=None)
    parser.add_argument('--cosine_eta_min', type=float, default=None)
    parser.add_argument('--multistep_milestones', type=str, default='3,5,7')
    parser.add_argument('--multistep_gamma', type=float, default=0.5)

    # Regularization
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--use_warmup', action='store_true', default=True)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--use_fp16', action='store_true', default=True)

    # System
    parser.add_argument('--save_dir', type=str, default='weights/fusion_experiment')
    parser.add_argument('--num_workers', type=int, default=4)

    # Load pretrained backbone from existing checkpoint
    parser.add_argument('--load_backbone_from', type=str, default=None,
                        help='Path to checkpoint to load backbone weights (net2D, net3D) only')

    args = parser.parse_args()

    # Parse milestones
    if args.scheduler == 'multistep':
        milestones = [int(x) for x in args.multistep_milestones.split(',')]
    else:
        milestones = None

    # Configuration
    config = {
        # Dataset
        'dataset': args.dataset,
        'data_root': args.data_root or (f'data/{args.dataset.upper()}101-24' if args.dataset == 'ucf' else 'data/AVA_Dataset'),
        'num_classes': 24 if args.dataset == 'ucf' else 80,

        # Model
        'backbone2D': f'yolov11_{args.yolo_version}',
        'backbone3D': 'videomae',
        'fusion_module': args.fusion_module,
        'mode': 'decoupled',
        'interchannels': [256, 256, 256],

        # VideoMAE config
        'BACKBONE3D': {
            'TYPE': 'videomae',
            'VARIANT': args.videomae_variant,
            'METHOD': args.videomae_method,
            'FREEZE': True,  # Always frozen in this script
            'TARGET_CHANNELS': 1024,
            'TARGET_SPATIAL_SIZE': 7,
            'DROPOUT': 0.2
        },

        # Training
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'lr': args.lr,
        'optimizer_type': args.optimizer,
        'weight_decay': args.weight_decay,
        'gradient_clip': args.gradient_clip,
        'gradient_accumulation': args.gradient_accumulation,
        'use_warmup': args.use_warmup,

        # Scheduler
        'scheduler_type': args.scheduler,
        'cosine_t_max': args.cosine_t_max or args.epochs,
        'cosine_eta_min': args.cosine_eta_min or (args.lr * 0.01),
        'multistep_milestones': milestones,
        'multistep_gamma': args.multistep_gamma,

        # VideoMAE settings
        'freeze_videomae': True,  # Always frozen
        'videomae_variant': args.videomae_variant,
        'videomae_method': args.videomae_method,
        'use_fp16': args.use_fp16,

        # EMA
        'use_ema': True,
        'ema_decay': args.ema_decay,

        # Input
        'img_size': 224,
        'clip_length': 16,
        'sampling_rate': 1,

        # System
        'num_workers': args.num_workers,
        'save_dir': args.save_dir,

        # Load backbone from existing checkpoint
        'load_backbone_from': args.load_backbone_from,

        # Pretrain - Load pretrained but freeze
        'pretrain_2d': True,
        'pretrain_3d': True,
        'freeze_bb2D': True,   # FROZEN!
        'freeze_bb3D': True,   # FROZEN!
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

    # Print summary
    print("\n" + "="*60)
    print("FREEZE BACKBONE Training Mode")
    print("="*60)
    print(f"Frozen: YOLOv11-{args.yolo_version}, VideoMAE-{args.videomae_variant} ({args.videomae_method})")
    print(f"Training: {args.fusion_module} Fusion Module")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    if args.load_backbone_from:
        print(f"Load backbone from: {args.load_backbone_from}")
    print("="*60 + "\n")

    # Create trainer and train
    trainer = FreezeBackboneTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
