"""
Dispersive Loss for Diffusion Policy

Implementation based on "Diffuse and Disperse: A Locality-aware Approach to 
Generative Models" (ICLR 2025). This regularization encourages intermediate 
features to be spread out, preventing mode collapse in diffusion models.

Mathematical formulation (Eq. 6 from paper):
    L_disp = log( (1/N²) * Σᵢⱼ exp(-||zᵢ - zⱼ||² / (τ * D)) )

where:
    - z ∈ R^{B×D}: flattened intermediate features
    - N = B(B-1)/2: number of unique pairs
    - τ: temperature parameter controlling repulsion strength
    - D: feature dimension for normalization

For robotics applications (Diffusion Policy):
    - Features have shape (B, T, D) where T is trajectory length
    - We apply mean pooling over T to get trajectory-level features
    - Optional action token masking for encoder-decoder architectures
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DispersiveLoss(nn.Module):
    """
    Dispersive Loss module for regularizing intermediate Transformer features.
    
    This loss encourages samples in a batch to have distinct intermediate 
    representations, which helps prevent mode collapse and improves diversity
    in generated action trajectories.
    
    Args:
        tau: Temperature parameter (default: 0.5). Lower values = stronger repulsion.
        eps: Small constant for numerical stability (default: 1e-8).
        reduction: 'mean' or 'none' (default: 'mean').
    
    Example:
        >>> disp_loss = DispersiveLoss(tau=0.5)
        >>> features = torch.randn(32, 16, 768)  # (B, T, D)
        >>> loss, stats = disp_loss(features)
    """
    
    def __init__(
        self,
        tau: float = 0.5,
        eps: float = 1e-8,
        reduction: str = 'mean',
    ) -> None:
        super().__init__()
        self.tau = tau
        self.eps = eps
        self.reduction = reduction
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute dispersive loss on intermediate features.
        
        Args:
            features: Tensor of shape (B, T, D) or (B, D) or any (B, ...).
                      Will be flattened to (B, -1) following original paper.
            mask: Optional mask (not used in current implementation, kept for API compatibility).
        
        Returns:
            loss: Scalar dispersive loss value.
            stats: Dictionary containing logging statistics:
                - 'feat_norm': Mean L2 norm of features
                - 'pairwise_dist_mean': Mean pairwise distance
                - 'pairwise_dist_std': Std of pairwise distances
        """
        # Flatten features: (B, ...) -> (B, D_flat)
        # This matches the original paper implementation
        B = features.shape[0]
        features = features.reshape(B, -1)  # (B, D_flat)
        
        D = features.shape[1]
        
        # Compute statistics for logging
        feat_norm = features.norm(dim=-1).mean()
        
        # Compute pairwise squared L2 distances using pdist
        # pdist returns a 1D tensor of size B*(B-1)/2 containing unique pairs
        pairwise_dist_sq = F.pdist(features).pow(2)  # (B*(B-1)/2,)
        
        # Normalize by dimension (as in original paper)
        pairwise_dist_sq_normalized = pairwise_dist_sq / D
        
        # Compute statistics before temperature scaling
        pairwise_dist_mean = pairwise_dist_sq_normalized.mean()
        pairwise_dist_std = pairwise_dist_sq_normalized.std() if pairwise_dist_sq_normalized.numel() > 1 else torch.tensor(0.0)
        
        # Apply temperature scaling
        # Lower tau = stronger repulsion (distances weighted more heavily)
        scaled_dist = pairwise_dist_sq_normalized / self.tau
        
        # Compute InfoNCE-style loss: -log(mean(exp(-dist)))
        # This encourages distances to be large (minimizing exp(-dist) terms)
        # 
        # Mathematical note: 
        #   When distances are large, exp(-dist) -> 0, mean(...) -> 0, -log(...) -> +inf (high loss)
        #   When distances are small, exp(-dist) -> 1, mean(...) -> 1, -log(...) -> 0 (low loss)
        #   So minimizing this loss pushes distances larger
        #
        # For numerical stability, we use logsumexp trick:
        #   -log(mean(exp(-x))) = -logsumexp(-x) + log(N)
        N = pairwise_dist_sq.numel()
        if N > 0:
            loss = -torch.logsumexp(-scaled_dist, dim=0) + torch.log(torch.tensor(N, dtype=features.dtype, device=features.device))
        else:
            # Edge case: batch size <= 1, no pairs to compute
            loss = torch.tensor(0.0, dtype=features.dtype, device=features.device)
        
        # Compile statistics for logging
        stats = {
            'feat_norm': feat_norm.detach(),
            'pairwise_dist_mean': pairwise_dist_mean.detach(),
            'pairwise_dist_std': pairwise_dist_std.detach() if isinstance(pairwise_dist_std, torch.Tensor) else torch.tensor(pairwise_dist_std),
        }
        
        return loss, stats


class TransformerHookManager:
    """
    Manages forward hooks for extracting intermediate features from Transformer layers.
    
    This class provides a clean interface for:
    1. Registering hooks on specific decoder layers
    2. Capturing intermediate activations during forward pass
    3. Cleaning up hooks after use
    
    Usage:
        >>> hook_manager = TransformerHookManager()
        >>> hook_manager.register_hooks(model.decoder.layers, target_indices=[2, 5])
        >>> output = model(input)
        >>> features = hook_manager.get_features()
        >>> hook_manager.clear_features()
    """
    
    def __init__(self) -> None:
        self._hooks = []
        self._features: Dict[int, torch.Tensor] = {}
    
    def _hook_fn(self, layer_idx: int):
        """Create a hook function for a specific layer index."""
        def hook(module, input, output):
            # TransformerDecoderLayer output is (seq_len, batch, d_model) or (batch, seq_len, d_model)
            # depending on batch_first setting
            if isinstance(output, tuple):
                output = output[0]
            self._features[layer_idx] = output.detach()
        return hook
    
    def register_hooks(
        self,
        layers: nn.ModuleList,
        target_indices: Optional[list] = None,
    ) -> None:
        """
        Register forward hooks on specified transformer layers.
        
        Args:
            layers: ModuleList of transformer layers (encoder or decoder).
            target_indices: List of layer indices to hook. If None, hooks the last layer.
        """
        self.remove_hooks()  # Clean up any existing hooks
        
        num_layers = len(layers)
        if target_indices is None:
            # Default: hook the layer at 1/4 depth (as suggested in paper)
            target_indices = [max(0, num_layers // 4 - 1)]
        
        for idx in target_indices:
            if 0 <= idx < num_layers:
                handle = layers[idx].register_forward_hook(self._hook_fn(idx))
                self._hooks.append(handle)
    
    def get_features(self, layer_idx: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Get captured features from a specific layer or all layers.
        
        Args:
            layer_idx: Specific layer index to retrieve. If None, returns first available.
        
        Returns:
            Feature tensor of shape (B, T, D) or None if no features captured.
        """
        if layer_idx is not None:
            return self._features.get(layer_idx)
        elif self._features:
            # Return first captured features
            return next(iter(self._features.values()))
        return None
    
    def get_all_features(self) -> Dict[int, torch.Tensor]:
        """Get all captured features as a dictionary."""
        return self._features.copy()
    
    def clear_features(self) -> None:
        """Clear captured features (should be called after each forward pass)."""
        self._features.clear()
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._features.clear()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
