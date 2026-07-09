
# import sys
# sys.path.append('/home/icrlab/imitation/diffusion_policy_3d/diffusion_policy_3d/model/vision/')
import torch
from torch import nn
from typing import Dict, List, Type, Optional, Union, Tuple
from termcolor import cprint
import numpy as np
import torchvision
import torch.nn.functional as F
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
import copy
from diffusion_policy.common.utils import replace_submodules
from diffusion_policy.model.vision.models.pointmap_encoder import PointMapPositionEmbedding, PointMapResNet, PointMapPlainConv, PointNetEncoder, PointMapViT
from diffusion_policy.model.vision.model_getter import get_rgb_model

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

def img_Bhwc_to_Bchw(img: torch.Tensor) -> torch.Tensor:
    """Convert image from HWC to CHW format.

    Args:
        img (torch.Tensor): Image tensor in HWC format.

    Returns:
        torch.Tensor: Image tensor in CHW format.
    """
    if img.max() > 1.0:
        img = img / 255.0
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    return img.permute(0, 3, 1, 2)

# if we have pointmap and rgb image
class SpatialEncoder(nn.Module):
    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str,nn.Module]],
        share_rgb_model: bool = True,
        imagenet_norm: bool = True,
        use_group_norm: bool = False,
        random_crop: bool = False,
        crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
        resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
        use_spatial_attention: bool = False,
        pointmap_cfg: Dict = None,
        pretrained_rgb_path: str = None,
        freeze_rgb: bool = False,
        **kwargs
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.shape_meta = shape_meta
        self.use_spatial_attention = use_spatial_attention
        self.pointmap_cfg = pointmap_cfg if pointmap_cfg is not None else {}
        
        rgb_keys = list()
        pointmap_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        
        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model
        
        #handle obs keys
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dimx')
            key_shape_map[key] = shape
            
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
                self.rgb_dim = kwargs.get('rgb_dim', 384) # Default to DINOv2 ViT-S output dim
            
            elif type == 'pointmap':
                pointmap_keys.append(key)
                # Use config from pointmap_cfg or defaults
                pm_cfg = copy.deepcopy(self.pointmap_cfg)
                model_type = pm_cfg.pop('model_type', 'embedding') # Default to embedding
                
                if model_type == 'resnet':
                    cprint(f"[SpatialEncoder] Using ResNet for pointmap encoding", 'green')
                    # ResNet specific args
                    # Filter relevant args or pass all remaining
                    resnet_kwargs = {
                        'in_channels': 3,
                        'out_dim': 256,
                        'use_pretrained': False,
                        'norm_type': 'none'
                    }
                    resnet_kwargs.update(pm_cfg)
                    key_model_map[key] = PointMapResNet(**resnet_kwargs)
                    self.pointmap_dim = 512
                elif model_type == 'plain_conv':
                    # Plain Conv specific args
                    plain_kwargs = {
                        'in_channels': 3,
                        'out_dim': 256,
                        'hidden_dim': 64
                    }
                    plain_kwargs.update(pm_cfg)
                    key_model_map[key] = PointMapPlainConv(**plain_kwargs)
                    self.pointmap_dim = plain_kwargs.get('hidden_dim', 64) * 8
                elif model_type == 'pointnet':
                    # PointNet specific args
                    pointnet_kwargs = {
                        'in_channels': 3,
                        'out_dim': 256,
                        'hidden_dim': 128,
                        'use_norm': True,
                        'num_points': 1024
                    }
                    pointnet_kwargs.update(pm_cfg)
                    key_model_map[key] = PointNetEncoder(**pointnet_kwargs)
                    self.pointmap_dim = pointnet_kwargs.get('out_dim', 256)
                elif model_type == 'vit':
                    cprint(f"[SpatialEncoder] Using ViT for pointmap encoding", 'green')
                    # ViT specific args
                    vit_kwargs = {
                        'in_channels': 3,
                        'out_dim': 256,
                        'img_size': 76, # Default, should be overridden by config
                        'patch_size': 8,
                        'embed_dim': 128,
                        'depth': 4,
                        'num_heads': 4
                    }
                    # Update with config, handling potential shape info if available
                    # If resize_shape is set for this key, use it for img_size
                    if resize_shape is not None:
                        if isinstance(resize_shape, dict):
                            h, w = resize_shape[key]
                        else:
                            h, w = resize_shape
                        vit_kwargs['img_size'] = h # Assume square or take H
                    elif shape is not None:
                         # shape is usually (C, H, W) or (H, W, C)
                         # We assume H is the second dimension if len=3 and first is 3
                         if len(shape) == 3:
                             if shape[0] == 3:
                                 vit_kwargs['img_size'] = shape[1]
                             else:
                                 vit_kwargs['img_size'] = shape[0]
                         elif len(shape) == 2:
                             vit_kwargs['img_size'] = shape[0]

                    vit_kwargs.update(pm_cfg)
                    key_model_map[key] = PointMapViT(**vit_kwargs)
                    self.pointmap_dim = vit_kwargs.get('out_dim', 256)
                    self.pointmap_spatial_dim = vit_kwargs.get('embed_dim', 128)
                else:
                    # Embedding specific args
                    pm_kwargs = {
                        'in_channels': 3,
                        'num_pos_feats': 256,
                        **pm_cfg
                    }
                    key_model_map[key] = PointMapPositionEmbedding(**pm_kwargs)
                    self.pointmap_dim = pm_kwargs.get('num_pos_feats', 256)
                
                # configure resize/crop for pointmap
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                
                this_transform = nn.Sequential(this_resizer, this_randomizer)
                key_transform_map[key] = this_transform
            
            elif type == 'low_dimx':
                low_dim_keys.append(key)
                # Configure MLP for low_dim if requested
                if kwargs.get('use_state_mlp', False): # Default to False as per request
                    input_dim = shape[0]
                    output_dim = kwargs.get('state_mlp_dim', 128) # Default 128
                    # Simple MLP: Linear -> ReLU -> Linear
                    key_model_map[key] = nn.Sequential(
                        nn.Linear(input_dim, output_dim),
                        nn.ReLU(),
                        nn.Linear(output_dim, output_dim)
                    )
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        rgb_keys = sorted(rgb_keys)
        pointmap_keys = sorted(pointmap_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.pointmap_keys = pointmap_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        
        # Ensure pointmap_spatial_dim is set (defaults to pointmap_dim if not set by ViT)
        if not hasattr(self, 'pointmap_spatial_dim'):
            self.pointmap_spatial_dim = getattr(self, 'pointmap_dim', 256)
        
        # Attention fusion layers
        if self.use_spatial_attention:
            # Multi-Modal Transformer Fusion
            # We project both RGB and Pointmap tokens to a shared dimension
            # and process them with a Transformer Encoder.
            
            self.fusion_dim = 256
            
            # RGB Projection (DINOv2 ViT-S is 384)
            self.rgb_proj = nn.Linear(self.rgb_dim, self.fusion_dim)
            
            # Pointmap Projection (PointMapViT is 128)
            self.pmp_proj = nn.Linear(self.pointmap_spatial_dim, self.fusion_dim)
            
            # Modality Embeddings
            if kwargs.get('use_modality_embed', True):
                self.modality_embed = nn.Parameter(torch.zeros(1, 2, self.fusion_dim)) # 0: RGB, 1: PMP
                nn.init.trunc_normal_(self.modality_embed, std=0.02) 
            else:
                cprint('[SpatialEncoder] Not using modality embeddings', 'yellow')
                self.modality_embed = nn.Parameter(torch.zeros(1, 2, self.fusion_dim), requires_grad=False)
                
            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.fusion_dim, 
                nhead=4, 
                dim_feedforward=1024, 
                dropout=0.1, 
                activation='gelu', 
                batch_first=True
            )
            self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            
            # Final Pooling / Projection
            # We can use a CLS token or Global Average Pooling
            # No extra projection needed if fusion_dim matches desired output dim?
            # But SpatialEncoder usually expects to output concatenated features.
            # Here we are fusing them into ONE feature vector per pair.
            # So output dim will be fusion_dim.
            
            pass

        # Load pretrained RGB weights if provided
        if pretrained_rgb_path is not None:
            cprint(f"[SpatialEncoder] Loading pretrained RGB weights from {pretrained_rgb_path}", "yellow")
            state_dict = torch.load(pretrained_rgb_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Filter and load RGB weights
            prefix = 'policy.obs_encoder.key_model_map.'
            
            # Identify which keys in key_model_map correspond to RGB models
            target_keys = set()
            if self.share_rgb_model:
                target_keys.add('rgb')
            else:
                for k in self.rgb_keys:
                    target_keys.add(k)
            
            for target_key in target_keys:
                if target_key in self.key_model_map:
                    model = self.key_model_map[target_key]
                    model_prefix = prefix + target_key + '.'
                    model_state = {}
                    for k, v in state_dict.items():
                        if k.startswith(model_prefix):
                            local_k = k[len(model_prefix):]
                            model_state[local_k] = v
                    
                    if len(model_state) > 0:
                        msg = model.load_state_dict(model_state, strict=False)
                        cprint(f"[SpatialEncoder] Loaded {len(model_state)} keys for {target_key}: {msg}", "green")
                    else:
                        cprint(f"[SpatialEncoder] Warning: No weights found for {target_key} with prefix {model_prefix}", "red")

        # Freeze RGB models if requested
        if freeze_rgb:
            cprint("[SpatialEncoder] Freezing RGB models", "yellow")
            target_keys = set()
            if self.share_rgb_model:
                target_keys.add('rgb')
            else:
                for k in self.rgb_keys:
                    target_keys.add(k)
            
            for target_key in target_keys:
                if target_key in self.key_model_map:
                    # Freeze parameters
                    for param in self.key_model_map[target_key].parameters():
                        param.requires_grad = False
                    # Important: set encoder to eval mode to disable stochastic layers (e.g., DropPath/Dropout)
                    # This ensures frozen RGB attention remains consistent across stages.
                    self.key_model_map[target_key].eval()

        cprint(f"[SpatialEncoder] Initialized with rgb_keys: {self.rgb_keys}, pointmap_keys: {self.pointmap_keys}, low_dim_keys: {self.low_dim_keys}", "cyan")
        if self.use_spatial_attention and len(self.rgb_keys) == len(self.pointmap_keys):
            cprint('use_spatial_attention activated in SpatialEncoder, training transformer', 'green')

    def _fuse_rgb_pointmap(self, rgb_attn_map, pmap_feat, pmap_rel, valid_mask):
        """
        Deprecated. Logic moved to forward().
        """
        pass

    def forward(self, obs_dict) -> torch.Tensor:
        batch_size = None
        features = list()
        
        # Check if we can perform spatial attention
        # Requirements: 
        # 1. use_spatial_attention is True
        # 2. We have matching RGB and Pointmap keys (simple 1-to-1 assumption for now)
        # 3. We have agent_pos for relative coordinates
        do_spatial_attn = (
            self.use_spatial_attention 
            and len(self.rgb_keys) == len(self.pointmap_keys)
            and 'agent_pos' in obs_dict
        )

        if do_spatial_attn:
            agent_pos = obs_dict['agent_pos'][:, 18:21] # (B, 3)
            
            for i, rgb_key in enumerate(self.rgb_keys):
                # 1. Process RGB -> Get Patch Tokens
                img = obs_dict[rgb_key]
                if batch_size is None: batch_size = img.shape[0]
                
                img = img_Bhwc_to_Bchw(img)
                img = self.key_transform_map[rgb_key](img)
                
                # Get RGB Tokens (B, N_rgb, 384)
                # Note: DINOv2EncoderSpatialAttn.forward now supports return_patches
                
                rgb_tokens = self.key_model_map[rgb_key](img, return_patches=True)
                
                # 2. Process Pointmap -> Get Patch Tokens
                pmap_key = self.pointmap_keys[i] # Assume aligned
                pmap = obs_dict[pmap_key]
                
                # Ensure (B, H, W, 3)
                if pmap.shape[1] == 3 and len(pmap.shape) == 4:
                    pmap = pmap.permute(0, 2, 3, 1)
                
                # Apply transforms
                if pmap_key in self.key_transform_map:
                    pmap_chw = pmap.permute(0, 3, 1, 2)
                    pmap_chw = self.key_transform_map[pmap_key](pmap_chw)
                    pmap = pmap_chw.permute(0, 2, 3, 1)
                
                # Calculate Relative Pointmap: P_rel = P_world - Agent_pos
                # pmap: (B, H, W, 3), agent_pos: (B, 3)
                pmap_rel = pmap # dont use relative position for now
                # - agent_pos.view(batch_size, 1, 1, 3)
                
                # Encode Pointmap to Spatial Features
                # (B, N_pmp, D_pmap) or (B, D_pmap, H, W)
                pmap_feat = self.key_model_map[pmap_key](pmap_rel, return_spatial=True)
                
                # Flatten if necessary
                if pmap_feat.dim() == 4:
                    # (B, C, H, W) -> (B, N, C)
                    pmp_tokens = pmap_feat.flatten(2).permute(0, 2, 1)
                else:
                    # (B, N, C)
                    pmp_tokens = pmap_feat
                
                # 3. Multi-Modal Transformer Fusion
                # Project to shared dim
                rgb_emb = self.rgb_proj(rgb_tokens) # (B, N_rgb, D_fusion)
                pmp_emb = self.pmp_proj(pmp_tokens) # (B, N_pmp, D_fusion)
                
                # Add Modality Embeddings
                rgb_emb = rgb_emb + self.modality_embed[:, 0:1, :]
                pmp_emb = pmp_emb + self.modality_embed[:, 1:2, :]
                
                # Concatenate Sequence
                # (B, N_rgb + N_pmp, D_fusion)
                fused_seq = torch.cat([rgb_emb, pmp_emb], dim=1)
                
                # Transformer Encoder
                fused_seq = self.fusion_transformer(fused_seq)
                
                # Global Pooling
                # (B, D_fusion)
                fused_feat = fused_seq.mean(dim=1)
                
                features.append(fused_feat)
                
        else:
            # Fallback to original logic (Concatenation)
            # process rgb input
            if self.share_rgb_model:
                imgs = list()
                for key in self.rgb_keys:
                    img = obs_dict[key]
                    if batch_size is None:
                        batch_size = img.shape[0]
                    else:
                        assert batch_size == img.shape[0]
                    img = img_Bhwc_to_Bchw(img)
                    img = self.key_transform_map[key](img)
                    imgs.append(img)
                
                if len(imgs) > 0:
                    imgs = torch.cat(imgs, dim=0)
                    feature = self.key_model_map['rgb'](imgs)
                    feature = feature.reshape(-1,batch_size,*feature.shape[1:])
                    feature = torch.moveaxis(feature,0,1)
                    feature = feature.reshape(batch_size,-1)
                    features.append(feature)
            else:
                for key in self.rgb_keys:
                    img = obs_dict[key]
                    if batch_size is None:
                        batch_size = img.shape[0]
                    else:
                        assert batch_size == img.shape[0]
                    img = img_Bhwc_to_Bchw(img)
                    img = self.key_transform_map[key](img)
                    feature = self.key_model_map[key](img)
                    features.append(feature)
            
            # process pointmap input
            for key in self.pointmap_keys:
                data = obs_dict[key]
                if batch_size is None:
                    batch_size = data.shape[0]
                else:
                    assert batch_size == data.shape[0]
                
                # Ensure float32
                if data.dtype != torch.float32:
                    data = data.float()

                # PointMapPositionEmbedding expects (B, H, W, 3)
                # Assuming input is (B, H, W, 3) or (B, 3, H, W)
                if data.shape[1] == 3 and len(data.shape) == 4:
                    data = data.permute(0, 2, 3, 1)

                # Apply transforms
                if key in self.key_transform_map:
                    data_chw = data.permute(0, 3, 1, 2)
                    data_chw = self.key_transform_map[key](data_chw)
                    data = data_chw.permute(0, 2, 3, 1)

                feature = self.key_model_map[key](data)
                features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]

            # Apply MLP if available
            if key in self.key_model_map:
                data = self.key_model_map[key](data)
                
            features.append(data)
        
        result = torch.cat(features, dim=-1)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        
        # Determine device from model parameters
        device = next(self.parameters()).device
        
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=torch.float32,
                device=device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape