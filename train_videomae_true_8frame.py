#!/usr/bin/env python3
"""
VideoMAE + YOWOFormer Training Script - TRUE 8-FRAME VERSION
Uses actual 8-frame input with interpolated positional embeddings

Key Differences from train_videomae_stable_8frame.py:
1. clip_length=8 (actual 8 frames, not 16)
2. sampling_rate=1 (consecutive frames)
3. Uses videomae_8frame backbone with interpolated pos_embed
4. ~50% less GFLOPs, ~2x faster inference

Expected Benefits:
- GFLOPs: ~90G (vs ~180G for 16 frames)
- FPS: ~25-30 (vs ~15 for 16 frames)
- GPU Memory: ~5GB (vs ~8GB for 16 frames)
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
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

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


class VideoMAE8FrameTrainer:
    """Trainer for true 8-frame VideoMAE model"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        clear_gpu_memory()

        self.max_history_size = 100
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        self.best_loss = float('inf')
        self.best_epoch = 0

        self.videomae_frozen = config.get('freeze_videomae', True)
        self.unfreeze_epoch = config.get('unfreeze_epoch', 5)
        self.videomae_already_unfrozen = False

        self.config['use_ema'] = True

        self.setup_logging()

        self.logger.info("="*60)
        self.logger.info("VideoMAE Training - TRUE 8-FRAME VERSION")
        self.logger.info("  clip_length: 8 (actual 8 frames)")
        self.logger.info("  sampling_rate: 1 (consecutive frames)")
        self.logger.info("  backbone: videomae_8frame (interpolated pos_embed)")
        self.logger.info("  Expected: ~50% less GFLOPs, ~2x faster FPS")
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
        log_name_parts = [
            f"{self.config['dataset']}",
            f"yolo{self.config['backbone2D'].split('_')[1]}",
            "videomae_8frame",
            f"{self.config.get('videomae_variant', 'base')}",
            f"{self.config.get('videomae_method', 'cross')}",
            timestamp
        ]

        log_filename = "_".join(log_name_parts) + ".log"
        log_path = save_dir / log_filename

        self.logger = logging.getLogger('VideoMAE_8Frame')
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
        self.logger.info("Training Configuration - TRUE 8-FRAME")
        self.logger.info("="*60)
        self.log_config()
        self.save_config_json()

    def log_config(self):
        """Log configuration"""
        self.logger.info("Model Architecture:")
        self.logger.info(f"  2D: {self.config['backbone2D']}")
        self.logger.info(f"  3D: VideoMAE-8Frame-{self.config.get('videomae_variant', 'base')}")
        self.logger.info(f"  Method: {self.config.get('videomae_method', 'cross')}")

        self.logger.info("\nInput Configuration:")
        self.logger.info(f"  clip_length: {self.config['clip_length']} (TRUE 8 frames)")
        self.logger.info(f"  sampling_rate: {self.config['sampling_rate']}")
        self.logger.info(f"  img_size: {self.config['img_size']}")

        self.logger.info("\nLearning Rate Schedule:")
        scheduler_type = self.config.get('scheduler_type', 'cosine')
        self.logger.info(f"  Type: {scheduler_type}")
        self.logger.info(f"  Initial LR: {self.config['learning_rate']}")

        self.logger.info("\nTraining Parameters:")
        self.logger.info(f"  Epochs: {self.config['epochs']}")
        self.logger.info(f"  Batch Size: {self.config['batch_size']} x {self.config['gradient_accumulation']}")
        self.logger.info(f"  Optimizer: {self.config['optimizer_type']}")

        self.logger.info("\nVideoMAE-8Frame:")
        self.logger.info(f"  Frozen Initially: {self.config.get('freeze_videomae', True)}")
        self.logger.info(f"  Unfreeze Epoch: {self.config.get('unfreeze_epoch', 5)}")
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
            'version': 'true_8frame',
            'checkpoint_policy': 'ema_only'
        }

        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=4)

    def setup_model(self):
        """Setup model"""
        self.logger.info("\nSetting up model...")
        from YOWOFormer.model.TSN.YOWOFormer import build_yowoformer

        self.model = build_yowoformer(self.config)
        self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Model created successfully!")
        self.logger.info(f"  Total parameters: {total_params/1e6:.2f}M")
        self.logger.info(f"  Trainable parameters: {trainable_params/1e6:.2f}M")

        from YOWOFormer.utils.EMA import EMA
        self.ema_model = EMA(self.model, decay=self.config.get('ema_decay', 0.999))
        self.logger.info(f"EMA enabled with decay: {self.config.get('ema_decay', 0.999)}")

        return self.model

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

        self.logger.info(f"Datasets loaded successfully!")
        self.logger.info(f"  Train samples: {len(self.train_dataset):,}")
        self.logger.info(f"  Val samples: {len(self.val_dataset):,}")

    def setup_training(self):
        """Setup optimizer and scheduler"""
        self.logger.info("\nSetting up training components...")

        params_with_wd = []
        params_without_wd = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if any(nd in name.lower() for nd in [
                'bias', 'norm', 'bn', 'ln', 'embedding', 'embed',
                'positional', 'pos_embed', 'cls_token', 'temporal_embed',
            ]):
                params_without_wd.append(param)
            else:
                params_with_wd.append(param)

        self.logger.info(f"Parameter groups:")
        self.logger.info(f"  WITH weight decay: {len(params_with_wd)}")
        self.logger.info(f"  WITHOUT weight decay: {len(params_without_wd)}")

        if self.config['optimizer_type'] == 'sgd':
            self.optimizer = torch.optim.SGD([
                {"params": params_with_wd, "lr": self.config['learning_rate'],
                 "weight_decay": self.config['weight_decay']},
                {"params": params_without_wd, "lr": self.config['learning_rate'],
                 "weight_decay": 0.0}
            ], momentum=0.9)
        else:
            self.optimizer = torch.optim.AdamW([
                {"params": params_with_wd, "lr": self.config['learning_rate'],
                 "weight_decay": self.config['weight_decay']},
                {"params": params_without_wd, "lr": self.config['learning_rate'],
                 "weight_decay": 0.0}
            ])

        scheduler_type = self.config.get('scheduler_type', 'cosine')

        if scheduler_type == 'cosine':
            t_max = self.config.get('cosine_t_max', self.config['epochs'])
            eta_min = self.config.get('cosine_eta_min', self.config['learning_rate'] * 0.01)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
        else:
            milestones = self.config.get('multistep_milestones', [3, 5, 7, 9])
            gamma = self.config.get('multistep_gamma', 0.5)
            self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        if self.config.get('use_warmup', True):
            from YOWOFormer.utils.warmup_lr import LinearWarmup
            self.warmup = LinearWarmup(self.config)

        from YOWOFormer.utils.loss import TAL
        self.criterion = TAL(self.model, self.config)

        if self.config.get('use_fp16', True):
            self.scaler = torch.amp.GradScaler('cuda')

        self.logger.info("Training setup complete!")

    def unfreeze_videomae(self):
        """Unfreeze VideoMAE"""
        if self.config['backbone3D'] == 'videomae_8frame' and hasattr(self.model, 'net3D'):
            self.logger.info("\n" + "="*60)
            self.logger.info("Unfreezing VideoMAE-8Frame encoder...")
            self.logger.info("="*60)

            if hasattr(self.model.net3D, 'unfreeze_encoder'):
                self.model.net3D.unfreeze_encoder(last_n_layers=6)
            else:
                for param in self.model.net3D.videomae.parameters():
                    param.requires_grad = True

            current_lr = self.optimizer.param_groups[0]['lr']

            videomae_params_with_wd = []
            videomae_params_no_wd = []

            for name, param in self.model.net3D.named_parameters():
                if not param.requires_grad:
                    continue

                if 'videomae' in name.lower():
                    if any(nd in name.lower() for nd in [
                        'bias', 'norm', 'layernorm', 'ln', 'embed',
                        'positional', 'pos_embed', 'cls_token', 'temporal_embed'
                    ]):
                        videomae_params_no_wd.append(param)
                    else:
                        videomae_params_with_wd.append(param)

            videomae_lr_ratio = self.config.get('videomae_lr_ratio', 0.1)
            videomae_lr = current_lr * videomae_lr_ratio

            if videomae_params_with_wd:
                self.optimizer.add_param_group({
                    "params": videomae_params_with_wd,
                    "lr": videomae_lr,
                    "weight_decay": self.config['weight_decay']
                })

            if videomae_params_no_wd:
                self.optimizer.add_param_group({
                    "params": videomae_params_no_wd,
                    "lr": videomae_lr,
                    "weight_decay": 0.0
                })

            self.logger.info(f"  VideoMAE LR: {videomae_lr:.6f}")
            self.videomae_frozen = False
            self.videomae_already_unfrozen = True
            self.logger.info("  VideoMAE-8Frame unfrozen successfully")
            self.logger.info("="*60)

            clear_gpu_memory()

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        batch_count = 0
        accumulation_steps = self.config['gradient_accumulation']

        self.current_training_epoch = epoch

        if self.videomae_frozen and epoch >= self.unfreeze_epoch and not self.videomae_already_unfrozen:
            self.unfreeze_videomae()

        cnt_param_update = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')

        self.optimizer.zero_grad()
        clear_gpu_memory()

        try:
            for batch_idx, (clips, boxes, labels) in enumerate(pbar):
                clips = clips.to(self.device, non_blocking=True)

                if self.config.get('use_fp16', True):
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(clips)
                else:
                    outputs = self.model(clips)

                targets = self.prepare_targets(boxes, labels)

                loss_output = self.criterion(outputs, targets)

                if isinstance(loss_output, dict):
                    loss = loss_output.get('total_loss', sum(loss_output.values()))
                else:
                    loss = loss_output

                loss = loss / accumulation_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"Invalid loss at batch {batch_idx}, skipping...")
                    self.optimizer.zero_grad()
                    continue

                if self.config.get('use_fp16', True) and hasattr(self, 'scaler'):
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    cnt_param_update += 1

                    if epoch == 0 and self.config.get('use_warmup', True):
                        if hasattr(self, 'warmup'):
                            self.warmup(self.optimizer, cnt_param_update)

                    if self.config.get('gradient_clip', 0) > 0:
                        if self.config.get('use_fp16', True) and hasattr(self, 'scaler'):
                            self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['gradient_clip'])

                    if self.config.get('use_fp16', True) and hasattr(self, 'scaler'):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.ema_model.update(self.model)

                epoch_loss += loss.item() * accumulation_steps
                batch_count += 1

                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    current_lr = self.optimizer.param_groups[0]["lr"]

                    pbar.set_postfix({
                        'loss': f'{loss.item() * accumulation_steps:.4f}',
                        'avg': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.6f}',
                        'vmae': 'frozen' if self.videomae_frozen else 'unfrozen'
                    })

                if batch_idx % 50 == 0 and batch_idx > 0:
                    del clips, outputs, targets
                    clear_gpu_memory()

                if self.config.get('max_batches_per_epoch') and batch_idx >= self.config['max_batches_per_epoch']:
                    break

        except Exception as e:
            self.logger.error(f"Error during training epoch {epoch+1}: {str(e)}")
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
            'videomae_frozen': self.videomae_frozen,
            'videomae_already_unfrozen': self.videomae_already_unfrozen,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'is_ema': True
        }

        torch.save(ema_checkpoint, save_path)
        self.logger.info(f"EMA checkpoint saved: {save_path.name}")

        if train_loss < self.best_loss:
            best_path = save_dir / "best.pth"
            torch.save(ema_checkpoint, best_path)
            self.logger.info(f"New best model saved (loss: {train_loss:.4f})")
            self.best_loss = train_loss
            self.best_epoch = epoch

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        self.logger.info(f"\nLoading checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.ema_model.ema.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                self.logger.warning("Could not load optimizer state")

        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                self.logger.warning("Could not load scheduler state")

        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'learning_rates' in checkpoint:
            self.learning_rates = checkpoint['learning_rates']

        if 'videomae_frozen' in checkpoint:
            self.videomae_frozen = checkpoint['videomae_frozen']
        if 'videomae_already_unfrozen' in checkpoint:
            self.videomae_already_unfrozen = checkpoint['videomae_already_unfrozen']

        start_epoch = checkpoint.get('epoch', 0)
        self.logger.info(f"Resuming from epoch: {start_epoch}")

        return start_epoch

    def train(self):
        """Main training loop"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Starting Training - TRUE 8-FRAME VERSION")
        self.logger.info("="*60)

        self.setup_model()
        self.setup_data()
        self.setup_training()

        start_epoch = 0
        if self.config.get('resume'):
            checkpoint_path = Path(self.config['resume'])
            if checkpoint_path.exists():
                start_epoch = self.load_checkpoint(str(checkpoint_path))

                for _ in range(start_epoch):
                    self.scheduler.step()

                if self.config['backbone3D'] == 'videomae_8frame':
                    self.current_training_epoch = start_epoch
                    if self.videomae_frozen and start_epoch >= self.unfreeze_epoch and not self.videomae_already_unfrozen:
                        self.unfreeze_videomae()

        total_epochs = self.config['epochs']
        remaining_epochs = total_epochs - start_epoch
        self.logger.info(f"\nTraining for {remaining_epochs} epochs...")
        self.logger.info("="*60)

        start_time = time.time()

        for epoch in range(start_epoch, self.config['epochs']):
            epoch_start = time.time()

            self.logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            self.logger.info("-"*40)

            mem_info = get_memory_info()
            if mem_info:
                self.logger.info(f"GPU Memory: {mem_info['gpu_allocated']:.2f}GB / {mem_info['gpu_total']:.2f}GB")

            train_loss = self.train_epoch(epoch)

            self.train_losses.append(train_loss)
            if len(self.train_losses) > self.max_history_size:
                self.train_losses = self.train_losses[-self.max_history_size:]

            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            self.save_checkpoint(epoch, train_loss)

            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch+1} Summary:")
            self.logger.info(f"  Train Loss: {train_loss:.4f}")
            self.logger.info(f"  Learning Rate: {current_lr:.6f}")
            self.logger.info(f"  VideoMAE: {'Frozen' if self.videomae_frozen else 'Unfrozen'}")
            self.logger.info(f"  Epoch Time: {epoch_time/60:.2f} min")

        total_time = time.time() - start_time
        self.logger.info("\n" + "="*60)
        self.logger.info("Training Complete!")
        self.logger.info("="*60)
        self.logger.info(f"Total Time: {total_time/3600:.2f} hours")
        self.logger.info(f"Best Loss: {self.best_loss:.4f} (Epoch {self.best_epoch+1})")
        self.logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='VideoMAE Training - TRUE 8-FRAME VERSION')

    # Dataset
    parser.add_argument('--dataset', type=str, default='ucf', choices=['ucf', 'ava'])
    parser.add_argument('--data_root', type=str, default=None)

    # Model
    parser.add_argument('--yolo_version', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--videomae_variant', type=str, default='base', choices=['base', 'large'])
    parser.add_argument('--videomae_method', type=str, default='cross',
                        choices=['simple', 'token', 'cross', 'hybrid'])
    parser.add_argument('--fusion_module', type=str, default='Simple',
                        choices=['CFAM', 'Simple', 'CrossAttention', 'CrossAttentionV2', 'CBAM', 'MultiHead'])

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adamw'])
    parser.add_argument('--gradient_accumulation', type=int, default=2)

    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'multistep'])
    parser.add_argument('--cosine_t_max', type=int, default=None)
    parser.add_argument('--cosine_eta_min', type=float, default=None)
    parser.add_argument('--multistep_milestones', type=str, default='3,5,7,9')
    parser.add_argument('--multistep_gamma', type=float, default=0.5)

    # VideoMAE
    parser.add_argument('--freeze_videomae', action='store_true', default=True)
    parser.add_argument('--unfreeze_epoch', type=int, default=5)
    parser.add_argument('--videomae_lr_ratio', type=float, default=0.1)
    parser.add_argument('--use_fp16', action='store_true', default=True)

    # Regularization
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--use_warmup', action='store_true', default=True)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    # System
    parser.add_argument('--save_dir', type=str, default='weights/videomae_true_8frame')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    if args.scheduler == 'multistep':
        milestones = [int(x) for x in args.multistep_milestones.split(',')]
    else:
        milestones = None

    config = {
        # Dataset
        'dataset': args.dataset,
        'data_root': args.data_root or (f'data/{args.dataset.upper()}101-24' if args.dataset == 'ucf' else 'data/AVA_Dataset'),
        'num_classes': 24 if args.dataset == 'ucf' else 80,

        # Model - Using videomae_8frame backbone
        'backbone2D': f'yolov11_{args.yolo_version}',
        'backbone3D': 'videomae_8frame',  # Use 8-frame version
        'fusion_module': args.fusion_module,
        'mode': 'decoupled',
        'interchannels': [128, 128, 128],

        # VideoMAE-8Frame config
        'BACKBONE3D': {
            'TYPE': 'videomae_8frame',
            'VARIANT': args.videomae_variant,
            'METHOD': args.videomae_method,
            'FREEZE': args.freeze_videomae,
            'TARGET_CHANNELS': 512,
            'TARGET_SPATIAL_SIZE': 7,
            'DROPOUT': 0.2,
            'NUM_FRAMES': 8  # TRUE 8 frames
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

        # VideoMAE
        'freeze_videomae': args.freeze_videomae,
        'unfreeze_epoch': args.unfreeze_epoch,
        'videomae_lr_ratio': args.videomae_lr_ratio,
        'use_fp16': args.use_fp16,
        'videomae_variant': args.videomae_variant,
        'videomae_method': args.videomae_method,

        # EMA
        'use_ema': True,
        'ema_decay': args.ema_decay,

        # Input - TRUE 8 FRAMES
        'img_size': 224,
        'clip_length': 8,      # TRUE 8 frames (not 16!)
        'sampling_rate': 1,    # Consecutive frames

        # System
        'num_workers': args.num_workers,
        'save_dir': args.save_dir,
        'resume': args.resume,

        # Pretrain
        'pretrain_2d': True,
        'pretrain_3d': True,
        'freeze_bb2D': False,
        'freeze_bb3D': args.freeze_videomae,
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

    trainer = VideoMAE8FrameTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
