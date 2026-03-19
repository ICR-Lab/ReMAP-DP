from diffusion_policy.workspace.train_diffusion_transformer_rgbpmp_workspace import TrainDiffusionTransformerRgbPmpWorkspace
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
from diffusion_policy.workspace.train_dp_robotwin_workspace import TrainDiffusionUnetImageWorkspace
from diffusion_policy.workspace.train_diffusion_unet_image_disp_workspace import TrainDiffusionUnetImageDispWorkspace
from diffusion_policy.workspace.train_dp_robotwin_disp_workspace import TrainDiffusionUnetImageDispWorkspace as TrainDiffusionUnetImageDispWorkspaceRoboTwin

# Expose submodules so Hydra can locate them via dotted paths
from . import train_diffusion_transformer_rgbpmp_workspace  # noqa: F401
from . import train_diffusion_unet_image_workspace  # noqa: F401
from . import train_dp_robotwin_workspace  # noqa: F401
from . import train_diffusion_unet_image_disp_workspace  # noqa: F401
from . import train_dp_robotwin_disp_workspace  # noqa: F401
