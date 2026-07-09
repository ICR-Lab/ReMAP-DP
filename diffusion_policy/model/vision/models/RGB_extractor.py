"""ResNet18-based RGB feature extractor.

Provides a small wrapper around torchvision.models.resnet18 that returns
intermediate feature maps and a global pooled feature vector suitable for
downstream use (or for use as a standalone feature extractor).
"""
import numpy as np
from typing import Tuple, Optional, List, Type
import torch
import torch.nn as nn
import copy
from torchvision import models

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

class PlainConv(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_dim=256,
        pool_feature_map=False,
        last_act=True,  # True for ConvBody, False for CNN
    ):
        super().__init__()
        # assume input image size is 128x128

        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(128, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=last_act)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4 * 4, [out_dim], last_act=last_act)
        
        # FusionEncoder expects out_dim to be the feature map channel dimension
        # PlainConv always outputs 128 channels in the last conv layer
        self.out_dim = 128

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        feature_map = x # (B, C, H, W)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x, feature_map

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, weights=None, freeze_backbone=True, out_dim=256):
        super().__init__()
        self.resnet = models.resnet18(weights=weights)
        if freeze_backbone:
            for p in self.resnet.parameters():
                p.requires_grad = False
        self.embed_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc = nn.Linear(self.embed_dim, out_dim)
        # ResNet18 has 512 channels in the last conv layer
        self.out_dim = 512

    def nomalize(self, x):
        # ResNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, x, return_patches=False):
        if x.max() > 1.0:
            x = x / 255.0
        x = self.nomalize(x)
        
        feature_map = self.resnet.conv1(x)
        feature_map = self.resnet.bn1(feature_map)
        feature_map = self.resnet.relu(feature_map)
        feature_map = self.resnet.maxpool(feature_map)

        feature_map = self.resnet.layer1(feature_map)
        feature_map = self.resnet.layer2(feature_map)
        feature_map = self.resnet.layer3(feature_map)
        feature_map = self.resnet.layer4(feature_map)

        pooled_feature = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)
        feature_vec = self.fc(pooled_feature)

        if return_patches:
            # mimic DINOv2EncoderSpatialAttn behavior: return flattened patch tokens
            # (B, C, H, W) -> (B, H*W, C)
            B, C, H, W = feature_map.shape 
            return feature_map.flatten(2).transpose(1, 2)
        
        return feature_vec

class DINOFeatureExtractor(nn.Module):
    def __init__(self, model_name='dino_vits16', freeze_backbone=True, out_dim=256, checkpoint_path=None):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', model_name)
        
        # Load custom checkpoint if provided (e.g., fine-tuned weights)
        if checkpoint_path is not None:
            import os
            if os.path.exists(checkpoint_path):
                print(f"[DINOFeatureExtractor] Loading weights from: {checkpoint_path}")
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                # Handle different checkpoint formats
                if 'dino_backbone' in ckpt:
                    state_dict = ckpt['dino_backbone']
                elif 'model_state_dict' in ckpt:
                    state_dict = ckpt['model_state_dict']
                elif 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                else:
                    state_dict = ckpt
                # Load with partial matching
                msg = self.model.load_state_dict(state_dict, strict=False)
                print(f"[DINOFeatureExtractor] Loaded weights: {msg}")
            else:
                print(f"[DINOFeatureExtractor] Warning: checkpoint not found: {checkpoint_path}")
        
        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
        self.embed_dim = self.model.embed_dim
        self.fc = nn.Linear(self.embed_dim, out_dim)
        self.patch_size = self.model.patch_embed.patch_size
        # FusionEncoder expects out_dim to be the feature map channel dimension
        self.out_dim = self.embed_dim

    def forward(self, x):
        # x: (B, 3, H, W)
        # DINO expects normalized images.
        # We assume x is already normalized [0,1].
        
        # get_intermediate_layers returns list of tensors. n=1 gives the last layer output.
        # Output shape: (B, N_patches + 1, D)
        features = self.model.get_intermediate_layers(x, n=1)[0] 
        cls_token = features[:, 0]
        patch_tokens = features[:, 1:]
        
        # Reshape patch tokens to feature map
        B, N, D = patch_tokens.shape
        H, W = x.shape[2], x.shape[3]
        P = self.patch_size
        h, w = H // P, W // P
        
        # Handle case where input size is not perfectly divisible or different from training
        # DINO ViT handles arbitrary sizes by interpolation usually, but get_intermediate_layers might return what it computed.
        # If N != h*w, we might need to be careful.
        # For now assume standard sizes.
        
        if N == h * w:
            feature_map = patch_tokens.transpose(1, 2).reshape(B, D, h, w)
        else:
            # Fallback: interpolate to square
            side = int(np.sqrt(N))
            feature_map = patch_tokens.transpose(1, 2).reshape(B, D, side, side)
            if side != h or side != w:
                feature_map = torch.nn.functional.interpolate(feature_map, size=(h, w), mode='bilinear', align_corners=False)

        feature_vec = self.fc(cls_token)
        
        return feature_vec

class CompositeFeatureExtractor(nn.Module):
    def __init__(self, encoders: List[nn.Module]):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.out_dim = sum([e.out_dim for e in encoders])

    def forward(self, x):
        vecs = []
        maps = []
        for e in self.encoders:
            v, m = e(x)
            vecs.append(v)
            maps.append(m)
        
        # Concatenate vectors
        feature_vec = torch.cat(vecs, dim=-1)
        
        # Resize all to the size of the first map
        target_h, target_w = maps[0].shape[2], maps[0].shape[3]
        resized_maps = [maps[0]]
        for m in maps[1:]:
            if m.shape[2] != target_h or m.shape[3] != target_w:
                m = torch.nn.functional.interpolate(m, size=(target_h, target_w), mode='bilinear', align_corners=False)
            resized_maps.append(m)
            
        feature_map = torch.cat(resized_maps, dim=1)
        
        return feature_vec, feature_map

