import torch
import torchvision
from diffusion_policy.model.vision.models import *
from termcolor import cprint
# def some model getters, return the model instance
# the model should be a nn.Module and forward function returns feature_vector
def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    freeze_backbone = kwargs.get('freeze_backbone', True)
    resnet = ResNet18FeatureExtractor(weights=weights, freeze_backbone=freeze_backbone)
    return resnet

def get_dino_v2(name="dinov2_vits14", **kwargs):
    """
    weights: "dinov2_vitb14", "dinov2_vitl14"
    """
    dino_v2 = DINOv2Encoder(model_name=name,freeze_backbone=True)
    return dino_v2

def get_dino_v1(name="dino_vits16", **kwargs):
    """
    weights: "dino_vitb16", "dino_vits16"
    """
    dino_v1 = DINOFeatureExtractor(model_name=name, freeze_backbone=True)
    return dino_v1

def get_plain_conv(name="plain_conv", **kwargs):
    plain_conv = PlainConv(pool_feature_map=True)
    return plain_conv

def get_dino_v2_spatial(name="dinov2_vits14", **kwargs):
    dino_v2 = DINOv2EncoderSpatialAttn(model_name=name,freeze_backbone=True)
    return dino_v2

def get_rgb_model(name, **kwargs):
    if 'resnet' in name:
        cprint(f"[Model Getter] Using ResNet model: {name}", "yellow")
        return get_resnet(name, **kwargs)
    elif 'dinov2_spatial' in name:
        cprint(f"[Model Getter] Using DINOv2 Spatial Attention model: {name}", "yellow")
        return get_dino_v2_spatial('dinov2_vits14', **kwargs)
    elif 'dinov2' in name:
        cprint(f"[Model Getter] Using DINOv2 model: {name}", "yellow")
        return get_dino_v2(name, **kwargs)
    elif 'dino' in name:
        cprint(f"[Model Getter] Using DINOv1 model: {name}", "yellow")
        return get_dino_v1(name, **kwargs)
    elif 'plain_conv' in name:
        cprint(f"[Model Getter] Using Plain Conv model: {name}", "yellow")
        return get_plain_conv(name, **kwargs)
    else:
        raise ValueError(f"Unknown rgb model name: {name}")