"""
Compare SimpleAdapter (new, spatial-token-based) vs Original Simple (CLS-based)
and all other adapters in videomae_advanced.py

Measures: Total Params, Trainable Params, GFLOPs, FPS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from einops import rearrange

# ============================================================
# 1) New SimpleAdapter (spatial-token-based, for fair comparison)
# ============================================================
class SimpleAdapter(nn.Module):
    """
    Simplest possible adapter using spatial tokens (like advanced adapters).
    Just: Linear projection per token → reshape → adaptive pool → small refine.
    No attention, no cross-attention, no 3D conv.
    """

    def __init__(self,
                 hidden_size: int = 768,
                 target_channels: int = 1024,
                 target_spatial_size: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_channels = target_channels
        self.target_spatial_size = target_spatial_size

        # Simple per-token linear projection
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_size, target_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Lightweight spatial refinement (1 conv only)
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(target_channels, target_channels, 3, 1, 1),
            nn.BatchNorm2d(target_channels),
            nn.GELU(),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (target_spatial_size, target_spatial_size)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N, D) all tokens from VideoMAE
        Returns:
            (B, C, 1, H, W)
        """
        B = features.shape[0]

        # Use spatial tokens only (skip CLS)
        spatial_tokens = features[:, 1:, :]  # (B, N-1, 768)

        # Project each token
        projected = self.token_proj(spatial_tokens)  # (B, N-1, 1024)

        # Treat as 1D sequence → AdaptiveAvgPool1d to target_spatial_size²
        projected = projected.permute(0, 2, 1)  # (B, 1024, N-1)
        target_len = self.target_spatial_size * self.target_spatial_size  # 49
        pooled = F.adaptive_avg_pool1d(projected, target_len)  # (B, 1024, 49)

        # Reshape to 2D spatial
        spatial = pooled.view(B, self.target_channels,
                              self.target_spatial_size,
                              self.target_spatial_size)  # (B, 1024, 7, 7)

        # Light refinement
        spatial = self.spatial_refine(spatial) + spatial  # residual

        # Add temporal dim
        return spatial.unsqueeze(2)  # (B, 1024, 1, 7, 7)


# ============================================================
# 2) Import existing adapters
# ============================================================
from YOWOFormer.model.backbone3D.videomae_advanced import (
    TokenAdapter, CrossAttentionAdapter, HybridAdapter, VideoMAEBackboneAdvanced
)
from YOWOFormer.model.backbone3D.videomae import VideoMAEBackbone


# ============================================================
# 3) Wrapper: Advanced backbone + SimpleAdapter
# ============================================================
class VideoMAEWithSimpleAdapter(nn.Module):
    """VideoMAE backbone using the new SimpleAdapter (spatial-token-based)"""

    def __init__(self, model_variant="base", freeze_encoder=True, dropout=0.1):
        super().__init__()
        from transformers import VideoMAEModel

        model_map = {
            "base": "MCG-NJU/videomae-base-finetuned-kinetics",
            "large": "MCG-NJU/videomae-large-finetuned-kinetics",
        }
        model_name = model_map.get(model_variant, model_variant)

        self.videomae = VideoMAEModel.from_pretrained(model_name)
        hidden_size = self.videomae.config.hidden_size

        if freeze_encoder:
            for param in self.videomae.parameters():
                param.requires_grad = False

        self.adapter = SimpleAdapter(
            hidden_size=hidden_size,
            target_channels=1024,
            target_spatial_size=7,
            dropout=dropout,
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        with torch.amp.autocast('cuda', enabled=False):
            outputs = self.videomae(x, return_dict=True)
        hidden_states = outputs.last_hidden_state
        return self.adapter(hidden_states)

    def load_pretrain(self):
        pass


# ============================================================
# 4) Measurement functions
# ============================================================
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_adapter_params(adapter_module):
    return sum(p.numel() for p in adapter_module.parameters())


def measure_gflops(model, input_tensor):
    """Measure GFLOPs using torch profiler"""
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, input_tensor)
        return flops.total() / 1e9
    except ImportError:
        pass

    try:
        from thop import profile
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return flops / 1e9
    except ImportError:
        pass

    # Manual estimation fallback
    print("  [WARNING] fvcore/thop not available, using torch profiler")
    model.eval()
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            with_flops=True
        ) as prof:
            model(input_tensor)

    total_flops = 0
    for event in prof.key_averages():
        if event.flops:
            total_flops += event.flops
    return total_flops / 1e9


def measure_fps(model, input_tensor, warmup=10, runs=50):
    """Measure FPS with warmup"""
    model.eval()
    device = input_tensor.device

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    fps = input_tensor.shape[0] / avg_time  # batch_size / time
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    fps_std = input_tensor.shape[0] * std_time / (avg_time ** 2)
    return fps, fps_std


# ============================================================
# 5) Main: build all models and measure
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 80)

    # Test input (batch=1 for GFLOPs, batch=8 for FPS)
    dummy_gflops = torch.randn(1, 3, 16, 224, 224).to(device)
    dummy_fps = torch.randn(8, 3, 16, 224, 224).to(device)

    # Define all models to test
    configs = [
        ("Simple (OLD, CLS-based)", "old_simple"),
        ("Simple (NEW, spatial-token)", "new_simple"),
        ("Token", "token"),
        ("SQA (Cross-Attention)", "cross"),
        ("Hybrid", "hybrid"),
    ]

    results = []

    for name, method in configs:
        print(f"\n{'─' * 60}")
        print(f"  Building: {name}")
        print(f"{'─' * 60}")

        # Build model
        if method == "old_simple":
            model = VideoMAEBackbone(
                model_variant="base",
                freeze_encoder=True,
                target_channels=1024,
                target_spatial_size=7,
                dropout_rate=0.2
            )
        elif method == "new_simple":
            model = VideoMAEWithSimpleAdapter(
                model_variant="base",
                freeze_encoder=True,
                dropout=0.1
            )
        else:
            model = VideoMAEBackboneAdvanced(
                model_variant="base",
                method=method,
                freeze_encoder=True,
                target_channels=1024,
                target_spatial_size=7,
                dropout=0.1
            )

        model = model.to(device)
        model.eval()

        # Count params
        total, trainable = count_params(model)

        # Adapter-only params
        if method == "old_simple":
            adapter_p = total - sum(p.numel() for p in model.videomae.parameters())
        elif method == "new_simple":
            adapter_p = count_adapter_params(model.adapter)
        else:
            adapter_p = count_adapter_params(model.adapter)

        print(f"  Total params:     {total / 1e6:.2f}M")
        print(f"  Trainable params: {trainable / 1e6:.2f}M")
        print(f"  Adapter params:   {adapter_p / 1e6:.2f}M")

        # GFLOPs
        print(f"  Measuring GFLOPs...")
        gflops = measure_gflops(model, dummy_gflops)
        print(f"  GFLOPs: {gflops:.2f}")

        # FPS
        print(f"  Measuring FPS (batch=8)...")
        try:
            fps, fps_std = measure_fps(model, dummy_fps, warmup=5, runs=30)
            print(f"  FPS: {fps:.2f} ± {fps_std:.2f}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                # Retry with batch=1
                dummy_fps_small = torch.randn(1, 3, 16, 224, 224).to(device)
                fps, fps_std = measure_fps(model, dummy_fps_small, warmup=5, runs=30)
                fps = fps  # batch=1 FPS
                print(f"  FPS (batch=1, OOM at 8): {fps:.2f} ± {fps_std:.2f}")
            else:
                raise

        results.append({
            'name': name,
            'total_M': total / 1e6,
            'trainable_M': trainable / 1e6,
            'adapter_M': adapter_p / 1e6,
            'gflops': gflops,
            'fps': fps,
            'fps_std': fps_std,
        })

        # Free memory
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ============================================================
    # 6) Print summary table
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Adapter':<30} {'Total(M)':>10} {'Train(M)':>10} {'Adapt(M)':>10} {'GFLOPs':>10} {'FPS':>12}"
    print(header)
    print("─" * len(header))
    for r in results:
        print(f"{r['name']:<30} {r['total_M']:>10.2f} {r['trainable_M']:>10.2f} "
              f"{r['adapter_M']:>10.2f} {r['gflops']:>10.2f} "
              f"{r['fps']:>7.2f}±{r['fps_std']:.2f}")

    # Highlight the difference
    print("\n" + "=" * 80)
    print("KEY COMPARISON: Simple OLD vs Simple NEW")
    print("=" * 80)
    old = next(r for r in results if 'OLD' in r['name'])
    new = next(r for r in results if 'NEW' in r['name'])
    print(f"  Total params:   {old['total_M']:.2f}M → {new['total_M']:.2f}M  (Δ {new['total_M'] - old['total_M']:+.2f}M)")
    print(f"  Adapter params: {old['adapter_M']:.2f}M → {new['adapter_M']:.2f}M  (Δ {new['adapter_M'] - old['adapter_M']:+.2f}M)")
    print(f"  GFLOPs:         {old['gflops']:.2f} → {new['gflops']:.2f}")
    print(f"  FPS:            {old['fps']:.2f} → {new['fps']:.2f}")


if __name__ == "__main__":
    main()
