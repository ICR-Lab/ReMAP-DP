import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

class SpatialSoftmax(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, feature):
        # feature: [B, C, H, W]
        B, C, H, W = feature.shape
        feature = feature.view(B, C, -1) # [B, C, H*W]
        softmax_attention = F.softmax(feature / self.temperature, dim=-1) # [B, C, H*W]
        
        # Create meshgrid
        device = feature.device
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        pos_x = x.reshape(-1)
        pos_y = y.reshape(-1)
        
        expected_x = torch.sum(pos_x * softmax_attention, dim=-1, keepdim=True)
        expected_y = torch.sum(pos_y * softmax_attention, dim=-1, keepdim=True)
        
        return torch.cat([expected_x, expected_y], dim=-1).reshape(B, -1) # [B, C*2]

class PointMapPositionEmbedding(nn.Module):
    """处理(B, H, W, 3)形状的pointmap的位置嵌入"""
    
    def __init__(self, in_channels=3, num_pos_feats=768, n_freqs=8, logscale=True, 
                 norm_center=[-0.1, 0.0, 0.5], norm_scale=1000.0, pooling='max'):
        super(PointMapPositionEmbedding, self).__init__()
        self.n_freqs = n_freqs
        self.freq_out_channels = in_channels * (2 * n_freqs + 1)
        self.pooling = pooling
        
        if logscale:
            freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (n_freqs - 1), n_freqs)
        
        # Allow custom normalization
        center = torch.tensor(norm_center).float()
        if center.numel() == 3 and in_channels > 3:
             center = center.repeat(in_channels // 3)
        
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.register_buffer("center", center, persistent=False)
        self.register_buffer("scale", torch.tensor(norm_scale).float(), persistent=False)
        
        # 修改MLP以处理展平后的特征
        self.position_embedding_head = nn.Sequential(
            nn.Linear(self.freq_out_channels, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats),
        )
        
        # 简单的卷积头，用于聚合局部特征
        self.conv_head = nn.Sequential(
            nn.Conv2d(num_pos_feats, num_pos_feats, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_pos_feats, num_pos_feats, kernel_size=3, padding=1),
            nn.ReLU()
        )

        if pooling == 'spatial_softmax':
            self.spatial_softmax = SpatialSoftmax()
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)
    
    @torch.no_grad()
    def frequency_encoding(self, xyz):
        """处理(B, H, W, 3)形状的输入"""
        # 1. 保持原始形状，但要确保在正确的维度上进行频率编码
        B, H, W, C = xyz.shape
        
        # 2. 展平前两个空间维度，变为(B*H*W, C)
        xyz_flat = xyz.reshape(-1, C)
        
        # 3. 归一化（根据你的数据范围调整）
        xyz_n = ((xyz_flat - self.center) / self.scale).to(self.freq_bands.dtype)
        
        # 4. 频率编码
        xyz_feq = xyz_n.unsqueeze(-1) * self.freq_bands  # (B*H*W, C, n_freqs)
        sin_xyz, cos_xyz = torch.sin(xyz_feq), torch.cos(xyz_feq)
        
        # 5. 拼接并展平
        encoding = torch.cat([xyz_n.unsqueeze(-1), sin_xyz, cos_xyz], -1)
        encoding = encoding.reshape(B, H, W, -1)  # (B, H, W, C*(2*n_freqs+1))
        
        return encoding
    
    def forward(self, xyz, return_spatial=False):
        """
        输入: xyz (B, H, W, 3) - pointmap
        输出: 
            - 全局特征: (B, num_pos_feats)
            - 可选: 空间特征图 (B, num_pos_feats, H, W)
        """
        B, H, W, C = xyz.shape
        
        # 1. 频率编码
        freq_encoding = self.frequency_encoding(xyz)  # (B, H, W, freq_out_channels)
        
        # 2. 展平空间维度以便MLP处理
        freq_flat = freq_encoding.reshape(B * H * W, -1)
        
        # 3. MLP处理
        pos_embedding = self.position_embedding_head(freq_flat)  # (B*H*W, num_pos_feats)
        
        # 4. 恢复空间形状
        pos_embedding = pos_embedding.reshape(B, H, W, -1)  # (B, H, W, num_pos_feats)
        
        # 5. 转换为通道优先格式 (B, C, H, W)
        pos_embedding = pos_embedding.permute(0, 3, 1, 2)  # (B, num_pos_feats, H, W)
        
        # 6. 应用卷积头聚合局部特征
        pos_embedding = self.conv_head(pos_embedding)

        if return_spatial:
            # 返回空间特征图
            return pos_embedding
        else:
            # 生成全局特征（多种池化策略）
            if self.pooling == 'max':
                global_feat = F.adaptive_max_pool2d(pos_embedding, (1, 1)).squeeze(-1).squeeze(-1)
            elif self.pooling == 'avg':
                global_feat = F.adaptive_avg_pool2d(pos_embedding, (1, 1)).squeeze(-1).squeeze(-1)
            elif self.pooling == 'spatial_softmax':
                global_feat = self.spatial_softmax(pos_embedding)
            else:
                raise ValueError(f"Unknown pooling type: {self.pooling}")
            
            return global_feat  # (B, out_dim)

class PointMapResNet(nn.Module):
    """
    Treat PointMap as a standard image and process with ResNet.
    Input: (B, H, W, 3)
    """
    def __init__(self, in_channels=3, out_dim=256, use_pretrained=False, norm_type='none'):
        super().__init__()
        # Use ResNet18 as a lightweight backbone
        # Note: pretrained weights are for ImageNet (RGB), might not be optimal for XYZ but often better than random
        self.model = torchvision.models.resnet18(pretrained=use_pretrained)
        
        # Modify input layer if channels != 3
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # Replace the fully connected layer to match out_dim
        self.model.fc = nn.Linear(self.model.fc.in_features, out_dim)
        
        self.norm_type = norm_type
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, return_spatial=False):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        
        # Permute to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Optional Normalization
        if self.norm_type == 'imagenet':
            # Assumes x is in [0, 1] or similar range. 
            # If x is raw coordinates (e.g. meters), this might be wrong without pre-scaling.
            x = (x - self.imagenet_mean) / self.imagenet_std
        elif self.norm_type == 'instance':
            # Instance normalization per sample
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True) + 1e-5
            x = (x - mean) / std
            
        if return_spatial:
            # Manual forward for spatial features
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x) # (B, 512, H/32, W/32)
            return x
            
        # Use standard ResNet forward pass
        return self.model(x)


class PointMapPlainConv(nn.Module):
    """
    Simple CNN for PointMap processing.
    Input: (B, H, W, 3)
    """
    def __init__(self, in_channels=3, out_dim=256, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2), # H/2
            
            # Layer 2
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.MaxPool2d(2), # H/4
            
            # Layer 3
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
            nn.MaxPool2d(2), # H/8
            
            # Layer 4
            nn.Conv2d(hidden_dim*4, hidden_dim*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*8),
            nn.ReLU(),
            nn.MaxPool2d(2), # H/16
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(hidden_dim*8, out_dim)

    def forward(self, x, return_spatial=False):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        
        # Permute to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Forward pass
        feat_map = self.net(x) # (B, 512, H/16, W/16)
        
        if return_spatial:
            return feat_map
            
        # Global pooling
        x = self.pool(feat_map) # (B, 512, 1, 1)
        x = torch.flatten(x, 1) # (B, 512)
        x = self.proj(x) # (B, out_dim)
        
        return x

class PointNetEncoder(nn.Module):
    """
    Simple PointNet-like encoder for PointMap.
    Treats the (H, W) grid as a set of points, ignoring grid structure.
    Robust to sparsity and zeros.
    """
    def __init__(self, in_channels=3, out_dim=256, hidden_dim=128, use_norm=True, num_points=1024):
        super().__init__()
        self.num_points = num_points
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim) if use_norm else nn.Identity(),
            nn.ReLU()
        )
    
    def forward(self, x, mask_zero=True, return_spatial=False):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        
        # Flatten to (B, N, C)
        x = x.reshape(B, -1, C)
        
        if return_spatial:
            # Return as (B, out_dim, H, W) for potential fusion
            feat = self.mlp(x)
            return feat.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Sampling logic
        if self.num_points > 0:
            sampled_data = []
            for i in range(B):
                pts = x[i]
                if mask_zero:
                    # Create mask for valid points (assuming 0,0,0 is invalid)
                    mask = (torch.abs(pts).sum(dim=-1) > 1e-6)
                    valid_pts = pts[mask]
                else:
                    valid_pts = pts
                
                n_valid = valid_pts.shape[0]
                
                if n_valid > 0:
                    if n_valid >= self.num_points:
                        # Random sample without replacement
                        idx = torch.randperm(n_valid, device=x.device)[:self.num_points]
                        selected = valid_pts[idx]
                    else:
                        # Random sample with replacement (padding)
                        idx = torch.randint(0, n_valid, (self.num_points,), device=x.device)
                        selected = valid_pts[idx]
                else:
                    # No valid points, return zeros
                    selected = torch.zeros((self.num_points, C), device=x.device)
                
                sampled_data.append(selected)
            
            x = torch.stack(sampled_data, dim=0) # (B, num_points, C)

        # MLP: (B, N, out_dim)
        feat = self.mlp(x)
        
        # Global Max Pooling
        # (B, out_dim)
        global_feat = torch.max(feat, dim=1)[0]
        
        return global_feat

class PointMapViT(nn.Module):
    """
    ViT for PointMap.
    """
    def __init__(self, in_channels=3, out_dim=256, img_size=76, patch_size=8, embed_dim=128, depth=4, num_heads=4, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Calculate number of patches
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio), dropout=drop_rate, activation='gelu', batch_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_dim)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_spatial=False):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)
        
        # Patch Embedding
        x = self.patch_embed(x) # (B, E, H', W')
        H_prime, W_prime = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2) # (B, N, E)
        
        # Add CLS token and Positional Embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Handle pos_embed interpolation if input size is different from training size
        if x.shape[1] != self.pos_embed.shape[1]:
             # Simple interpolation for pos_embed if needed
             pass 
        
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.pos_drop(x)
        
        # Transformer Blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        if return_spatial:
            # Remove cls token
            x = x[:, 1:, :]
            # Return patch tokens directly (B, N, E)
            return x

        # Global feature (CLS token)
        x = x[:, 0]
        x = self.head(x)
        return x


