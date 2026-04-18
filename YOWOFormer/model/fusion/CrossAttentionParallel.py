"""
Parallel Cross-Attention Fusion Module for YOWOFormer (R1-Q5 Ablation)
Same as CrossAttention.py but uses PARALLEL bidirectional attention
instead of SEQUENTIAL.

Difference:
  Sequential: 2D→3D first, then 3D→(updated 2D)
  Parallel:   2D→3D and 3D→2D simultaneously using original features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, return_attention=False):
        B, N_q, C = query.shape
        N_kv = key.shape[1]

        Q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, N_q, C)
        output = self.out_proj(attn_output)

        if return_attention:
            return output, attn_weights
        return output


class ParallelBidirectionalCrossAttentionBlock(nn.Module):
    """
    PARALLEL Bidirectional Cross-Attention Block
    Both directions use ORIGINAL features (not updated ones)
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, mlp_ratio=4.0):
        super().__init__()

        self.cross_attn_2d_to_3d = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.cross_attn_3d_to_2d = MultiHeadCrossAttention(embed_dim, num_heads, dropout)

        self.norm1_2d = nn.LayerNorm(embed_dim)
        self.norm1_3d = nn.LayerNorm(embed_dim)
        self.norm2_2d = nn.LayerNorm(embed_dim)
        self.norm2_3d = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp_2d = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.mlp_3d = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, feat_2d, feat_3d):
        # ============================================================
        # PARALLEL: both attend to ORIGINAL features simultaneously
        # ============================================================
        feat_2d_attended = self.cross_attn_2d_to_3d(
            query=feat_2d, key=feat_3d, value=feat_3d
        )
        feat_3d_attended = self.cross_attn_3d_to_2d(
            query=feat_3d, key=feat_2d, value=feat_2d  # uses ORIGINAL feat_2d
        )

        feat_2d = self.norm1_2d(feat_2d + feat_2d_attended)
        feat_3d = self.norm1_3d(feat_3d + feat_3d_attended)

        # Feed-forward networks
        feat_2d = feat_2d + self.mlp_2d(self.norm2_2d(feat_2d))
        feat_3d = feat_3d + self.mlp_3d(self.norm2_3d(feat_3d))

        return feat_2d, feat_3d


class CrossAttentionParallelFusionBlock(nn.Module):
    def __init__(self, channels_2d, channels_3d, inter_channels, num_heads=8, dropout=0.1):
        super().__init__()

        self.reduce_2d = nn.Sequential(
            nn.Conv2d(channels_2d, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.reduce_3d = nn.Sequential(
            nn.Conv2d(channels_3d, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # Use PARALLEL version
        self.cross_attention = ParallelBidirectionalCrossAttentionBlock(
            embed_dim=inter_channels,
            num_heads=num_heads,
            dropout=dropout
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(inter_channels * 2, inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_2d, feat_3d):
        B, _, H, W = feat_2d.shape

        feat_2d_reduced = self.reduce_2d(feat_2d)
        feat_3d_reduced = self.reduce_3d(feat_3d)
        C = feat_2d_reduced.shape[1]

        feat_2d_flat = feat_2d_reduced.flatten(2).transpose(1, 2)
        feat_3d_flat = feat_3d_reduced.flatten(2).transpose(1, 2)

        feat_2d_attended, feat_3d_attended = self.cross_attention(feat_2d_flat, feat_3d_flat)

        feat_2d_attended = feat_2d_attended.transpose(1, 2).reshape(B, C, H, W)
        feat_3d_attended = feat_3d_attended.transpose(1, 2).reshape(B, C, H, W)

        fused = torch.cat([feat_2d_attended, feat_3d_attended], dim=1)
        output = self.output_conv(fused)
        return output


class CrossAttentionParallelFusion(nn.Module):
    def __init__(self, channels_2D, channels_3D, interchannels, mode='decoupled'):
        super().__init__()
        assert mode in ['coupled', 'decoupled']
        self.mode = mode

        if mode == 'coupled':
            self.fusion_blocks = nn.ModuleList()
            for channels_2d in channels_2D:
                self.fusion_blocks.append(
                    CrossAttentionParallelFusionBlock(
                        channels_2d=channels_2d + channels_3D,
                        channels_3d=channels_3D,
                        inter_channels=interchannels,
                        num_heads=8, dropout=0.1
                    )
                )
        elif mode == 'decoupled':
            self.box_blocks = nn.ModuleList()
            self.cls_blocks = nn.ModuleList()
            for channels_2d in channels_2D:
                self.box_blocks.append(
                    CrossAttentionParallelFusionBlock(
                        channels_2d=channels_2d[0], channels_3d=channels_3D,
                        inter_channels=interchannels, num_heads=8, dropout=0.1
                    )
                )
                self.cls_blocks.append(
                    CrossAttentionParallelFusionBlock(
                        channels_2d=channels_2d[1], channels_3d=channels_3D,
                        inter_channels=interchannels, num_heads=8, dropout=0.1
                    )
                )

    def forward(self, ft_2D, ft_3D):
        _, C_3D, H_3D, W_3D = ft_3D.shape
        fused_features = []

        if self.mode == 'coupled':
            for idx, feat_2d in enumerate(ft_2D):
                _, _, H_2D, W_2D = feat_2d.shape
                scale_factor = H_2D / H_3D
                if scale_factor != 1:
                    upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
                    ft_3D_upsampled = upsampler(ft_3D)
                else:
                    ft_3D_upsampled = ft_3D
                fused = self.fusion_blocks[idx](feat_2d, ft_3D_upsampled)
                fused_features.append(fused)

        elif self.mode == 'decoupled':
            for idx, feat_2d in enumerate(ft_2D):
                _, _, H_2D, W_2D = feat_2d[0].shape
                scale_factor = H_2D / H_3D
                if scale_factor != 1:
                    upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
                    ft_3D_upsampled = upsampler(ft_3D)
                else:
                    ft_3D_upsampled = ft_3D
                box_fused = self.box_blocks[idx](feat_2d[0], ft_3D_upsampled)
                cls_fused = self.cls_blocks[idx](feat_2d[1], ft_3D_upsampled)
                fused_features.append([box_fused, cls_fused])

        return fused_features
