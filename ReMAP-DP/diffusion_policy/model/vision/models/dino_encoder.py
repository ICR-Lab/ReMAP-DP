"""
DINOv2 Encoder for multi-view projection images.

This module provides a wrapper around the DINOv2 ViT-S/14 model for encoding
projection images and extracting visual tokens for cross-view correspondence learning.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union
from termcolor import cprint
import numpy as np

class DINOv2Encoder(nn.Module):
    """
    DINOv2 ViT-S/14 encoder wrapper.
    
    Args:
        model_name: Name of the DINOv2 model variant
        input_size: Input image size (assumes square images)
        freeze_backbone: Whether to freeze the backbone weights
        use_cls_token: Whether to include the CLS token in output
    """
    
    def __init__(
        self,
        model_name: str = 'dinov2_vits14',
        input_size: int = 224,
        freeze_backbone: bool = False,
        use_cls_token: bool = True,
        out_dim: int = 256,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.input_size = input_size
        self.freeze_backbone = freeze_backbone
        self.use_cls_token = use_cls_token
        
        # Load DINOv2 model
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # Get model specs
        self.embed_dim = self.backbone.embed_dim  # 384 for ViT-S
        self.patch_size = self.backbone.patch_embed.patch_size[0]  # 14
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            dummy_features = self.backbone.get_intermediate_layers(dummy_input, n=1)[0]
            # features shape: (B, 1 + num_patches + num_register, D)
            self.num_patches = dummy_features.shape[1] - 1  # Exclude CLS token
        
        # Output dimension
        self.out_dim = self.embed_dim
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            cprint(f"[DINOv2Encoder] Backbone frozen", "yellow")
        else:
            cprint(f"[DINOv2Encoder] Backbone trainable (full finetune)", "green")
        
        cprint(f"[DINOv2Encoder] Model: {model_name}", "cyan")
        cprint(f"[DINOv2Encoder] Embed dim: {self.embed_dim}", "cyan")
        cprint(f"[DINOv2Encoder] Patch size: {self.patch_size}", "cyan")
        cprint(f"[DINOv2Encoder] Num patches (detected): {self.num_patches}", "cyan")
        
        # Image normalization (ImageNet stats for DINOv2)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(self.embed_dim, out_dim)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input images with ImageNet stats."""
        return (x - self.mean) / self.std
    
    def forward(
        self, 
        x: torch.Tensor,
        return_patches: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through DINOv2.
        
        Args:
            x: Input images, shape (B, 3, H, W), values in [0, 1]
            return_patches: Whether to return spatial patch tokens
            
        Returns:
            feature_vec: (B, D)
            If return_patches=True:
                patch_features: (B, D, h, w)
        """
        # Ensure input is in correct range
        if x.max() > 1.0:
            x = x / 255.0
        
        B, C, H, W = x.shape
        
        # Normalize
        x = self.normalize(x)
        
        # Get intermediate features (last layer)
        # Output: (B, N+1, D) where N is num_patches, +1 for CLS token
        features = self.backbone.get_intermediate_layers(x, n=1)[0]
        
        cls_token = features[:, 0]  # (B, D)
        patch_tokens = features[:, 1:]  # (B, N, D)
        
        feature_vec = self.fc(cls_token)
        
        if return_patches:
            # Reshape patch tokens to spatial map
            # N = H_p * W_p
            h_p = H // self.patch_size
            w_p = W // self.patch_size
            
            # (B, N, D) -> (B, D, h_p, w_p)
            patch_features = patch_tokens.permute(0, 2, 1).reshape(B, self.embed_dim, h_p, w_p)
            return feature_vec, patch_features
        
        return feature_vec
    
class DINOv2EncoderSpatialAttn(nn.Module):
    def __init__(
            self,
            model_name: str = 'dinov2_vits14',
            freeze_backbone: bool = True,
            out_dim=256
        ):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.embed_dim = 384
        self.patch_size = 14
        self.attn_pool = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Softmax(dim=1)
        )
        self.proj = nn.Linear(self.embed_dim, out_dim)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            cprint(f"[DINOv2Encoder] Backbone frozen", "yellow")
        else:
            cprint(f"[DINOv2Encoder] Backbone trainable (full finetune)", "green")
        
        # set dinov2 normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input images with ImageNet stats."""
        return (x - self.mean) / self.std    

    def forward(self, x, return_attention=False, return_patches=False):
        if x.max() > 1.0: x = x / 255.0
        
        B, C, H, W = x.shape
        x = self.normalize(x)

        features = self.backbone.get_intermediate_layers(x, n=1)[0]
        
        # Determine token layout
        n_tokens = features.shape[1]
        h_p, w_p = H // self.patch_size, W // self.patch_size
        n_patches = h_p * w_p
        
        if n_tokens == n_patches:
            # No CLS token
            patch_tokens = features
        elif n_tokens == n_patches + 1:
            # CLS token at 0
            patch_tokens = features[:, 1:]
        elif n_tokens == n_patches + 5:
             # CLS + 4 Registers (DINOv2 with registers)
             patch_tokens = features[:, -n_patches:]
        else:
            # Fallback: assume CLS at 0 and take rest, but warn if shape mismatch
            patch_tokens = features[:, 1:]

        if return_patches:
            return patch_tokens
    
class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module (from arXiv:1807.06521)"""
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_att(x)
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        return x