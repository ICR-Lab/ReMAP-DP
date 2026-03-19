import sys
import os
import torch
import numpy as np
import collections
import hydra
import dill
import cv2
from omegaconf import OmegaConf
from PIL import Image
sys.path.insert(0, '/home/icrlab/RoboTwin/policy/Your_Policy')

# Add current directory to path so we can import diffusion_policy
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add diffusion_policy directory to path so internal imports like 'from policy.base_policy import BasePolicy' work
diffusion_policy_dir = os.path.join(current_dir, 'diffusion_policy')
if diffusion_policy_dir not in sys.path:
    sys.path.insert(0, diffusion_policy_dir)

# Import necessary modules from diffusion_policy
# Note: Ensure diffusion_policy is in PYTHONPATH or relative import works
import sys
sys.path.insert(0, '/home/icrlab/RoboTwin/policy/Your_Policy')
try:
    from diffusion_policy.model.utils.projection import project_to_tripleplane
except ImportError:
    # Attempt to add policy/Your_Policy to path if running from root
    policy_dir = os.path.join(os.path.dirname(__file__), 'diffusion_policy') 
    if os.path.exists(policy_dir) and os.path.dirname(os.path.dirname(__file__)) not in sys.path:
         sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from diffusion_policy.model.utils.projection import project_to_tripleplane

class PolicyWrapper:
    def __init__(self, policy, cfg, device):
        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.n_obs_steps = cfg.n_obs_steps
        self.obs_cache = collections.deque(maxlen=self.n_obs_steps)
        
    def update_obs(self, obs):
        self.obs_cache.append(obs)
        
    def reset_obs_cache(self):
        self.obs_cache.clear()
        
    def get_obs_cache_len(self):
        return len(self.obs_cache)
        
    def get_action(self):
        while len(self.obs_cache) < self.n_obs_steps:
            self.obs_cache.appendleft(self.obs_cache[0])
            
        obs_seq = list(self.obs_cache)
        obs_dict = collections.defaultdict(list)
        for obs in obs_seq:
            for k, v in obs.items():
                obs_dict[k].append(v)
        
        batch_obs = {}
        for k, v in obs_dict.items():
            val = np.stack(v) # (T_obs, ...)
            val = val[np.newaxis, ...] # (1, T_obs, ...)
            
            # Type conversion and device transfer
            if isinstance(val, np.ndarray):
                tensor_val = torch.from_numpy(val).to(self.device)
                if np.issubdtype(val.dtype, np.floating):
                    tensor_val = tensor_val.float()
                # Images might be uint8, keep them as is usually (SpatialEncoder expects 0-1 float or 0-255 uint8 depending on config)
                # But SpatialEncoder expects 0-1 for float inputs usually.
                # However, convert script saves RGB as uint8.
                # Let's check if we need to normalize. DINOv2 usually expects normalized.
                # But img_Bhwc_to_Bchw in spatial_encoder checks >1.0 and divides by 255.
                # So passing uint8 is fine if encoder handles it.
            else:
                tensor_val = val.to(self.device)
            batch_obs[k] = tensor_val

        with torch.no_grad():
            action_dict = self.policy.predict_action(batch_obs)
            
        action = action_dict['action'].detach().cpu().numpy()[0]
        return action

# Helper functions for Point Cloud Processing
def _ensure_homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=matrix.dtype)
        return np.vstack([matrix, bottom])
    if matrix.shape == (4, 3):
        right = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=matrix.dtype)
        return np.hstack([matrix, right])
    if matrix.shape == (3, 3):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=matrix.dtype)
        right = np.array([[0.0], [0.0], [0.0]], dtype=matrix.dtype)
        return np.vstack([np.hstack([matrix, right]), bottom])
    raise ValueError(f"Unsupported matrix shape for homogeneous conversion: {matrix.shape}")

def _to_numpy(array):
    if array is None:
        return None
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    return np.asarray(array)

def get_point_cloud(
    sensor_data,
    sensor_param,
    sensor_uids,
    use_point_crop=True,
    target_points=1024,
):
    all_points = []
    all_colors = []
    
    # Camera Parameter Setup for Crop (Hardcoded in convert script as [-0.7, -1, 0.00] to [1, 1, 2])
    # But note: The convert script used 'use_point_crop' parameter heavily.
    # We will follow the provided script's get_point_cloud function exactly.
    
    for sensor_uid in sensor_uids:
        color = sensor_data[sensor_uid]['rgb']
        depth = sensor_data[sensor_uid]['depth']
        intrinsic_cv = sensor_param[sensor_uid]['intrinsic_cv']
        extrinsic_cv = sensor_param[sensor_uid].get('extrinsic_cv')
        cam2world_gl = sensor_param[sensor_uid].get('cam2world_gl')
        
        color = _to_numpy(color)
        depth = _to_numpy(depth)
        intrinsic_cv = _to_numpy(intrinsic_cv)
        extrinsic_cv = _to_numpy(extrinsic_cv) if extrinsic_cv is not None else None
        cam2world_gl = _to_numpy(cam2world_gl) if cam2world_gl is not None else None

        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth = np.squeeze(depth)
        
        H, W = depth.shape[:2]

        if color.ndim == 3 and color.shape[-1] == 1:
            color = color[..., 0]
        color = np.squeeze(color)
        if color.ndim == 2:
            color = np.repeat(color[..., None], repeats=3, axis=-1)
        
        color = color.reshape(-1, 3)
        
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)
        
        # robo_twin camera.py returns depth * 1000 in float64
        # convert script expects millimeters and divides by 1000.
        depth_meters = depth.astype(np.float32) / 1000.0
        valid_mask = depth_meters > 0

        flat_valid_mask = valid_mask.reshape(-1)
        colors = color[flat_valid_mask].astype(np.float32)
        if colors.max(initial=0.0) > 1.0:
            colors /= 255.0
        
        z = depth_meters[valid_mask]
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        
        fx = intrinsic_cv[0, 0]
        fy = intrinsic_cv[1, 1]
        cx = intrinsic_cv[0, 2]
        cy = intrinsic_cv[1, 2]
        
        x_cam = (u_valid - cx) * z / fx
        y_cam = (v_valid - cy) * z / fy
        z_cam = z
        
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        
        if extrinsic_cv is not None:
            world_to_cam = _ensure_homogeneous(extrinsic_cv)
            try:
                cam_to_world = np.linalg.inv(world_to_cam)
            except np.linalg.LinAlgError:
                cam_to_world = np.linalg.pinv(world_to_cam)
            points_cam_homo = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
        elif cam2world_gl is not None:
            cam_to_world = _ensure_homogeneous(cam2world_gl)
            points_cam_gl = points_cam.copy()
            points_cam_gl[:, 1] *= -1.0
            points_cam_gl[:, 2] *= -1.0
            points_cam_homo = np.concatenate([points_cam_gl, np.ones((len(points_cam_gl), 1))], axis=1)
        else:
            continue # Should not happen

        if points_cam_homo.shape[0] == 0:
            continue

        points_world_homo = (cam_to_world @ points_cam_homo.T).T
        points_world = points_world_homo[:, :3]
        w = points_world_homo[:, 3:4]
        if w.shape[0] > 0:
            points_world = points_world / np.maximum(w, 1e-8)

        # Crop logic from convert script
        if use_point_crop:
            min_bound = np.array([-0.7, -1, 0.00])
            max_bound = np.array([1, 1, 2])
            in_bound_mask = np.all((points_world >= min_bound) & (points_world <= max_bound), axis=1)
            points_world = points_world[in_bound_mask]
            colors = colors[in_bound_mask]

        if len(points_world) > 0:
            all_points.append(points_world.astype(np.float32))
            all_colors.append(colors)
    
    if all_points:
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        pointcloud = np.concatenate([all_points, all_colors], axis=1)
    else:
        pointcloud = np.zeros((0, 6), dtype=np.float32)

    return pointcloud

def encode_obs(observation):
    # Note: rely solely on observation content; no environment access needed.

    # 1) Extract sensor data + camera params directly from observation
    sensor_data = {}
    sensor_param = {}
    cameras = ['front_camera', 'head_camera', 'left_camera', 'right_camera']
    obs_data = observation.get('observation', {})

    for cam in cameras:
        if cam not in obs_data:
            continue
        cam_info = obs_data[cam]
        if 'rgb' not in cam_info or 'depth' not in cam_info:
            continue

        sensor_data[cam] = {
            'rgb': cam_info['rgb'],
            'depth': cam_info['depth']
        }

        # pull intrinsics / extrinsics if present
        intrinsic = cam_info.get('intrinsic_cv')
        extrinsic = cam_info.get('extrinsic_cv')
        cam2world_gl = cam_info.get('cam2world_gl')
        sensor_param[cam] = {
            'intrinsic_cv': intrinsic,
            'extrinsic_cv': extrinsic,
            'cam2world_gl': cam2world_gl
        }

    # 2) Main image (head camera)
    if 'head_camera' in sensor_data:
        obs_img = sensor_data['head_camera']['rgb']
        if obs_img.shape[0] != 168 or obs_img.shape[1] != 168:
            obs_img = cv2.resize(obs_img, (168, 168), interpolation=cv2.INTER_AREA)
    else:
        obs_img = np.zeros((168, 168, 3), dtype=np.uint8)

    # 3) Point cloud
    obs_point_cloud = get_point_cloud(
        sensor_data=sensor_data,
        sensor_param=sensor_param,
        sensor_uids=list(sensor_data.keys()),
        use_point_crop=False,
        target_points=0
    )

    # 4) Projection to planes
    camera_params = {
        'xy': {'pos': (-0.032, -0.6, 1.55), 'lookat': [-0.032, 0.15, 0.55], 'up': (0.0, 0.8, 0.6)},
        'xz': {'pos': (-0.05, -0.9, 0.9), 'lookat': (-0.05, 0.0, 0.9), 'up': (0.0, 0.0, 1.0)},
        'yz': {'pos': (1.0, 0.0, 0.9), 'lookat': (-0.05, 0.0, 0.9), 'up': (0.0, 0.0, 1.0)}
    }
    workspace_bounds = ((-0.6, 0.6), (-0.5, 0.38), (0.6, 1.5))

    img_xy, img_xz, img_yz, pmp_xy, pmp_xz, pmp_yz = project_to_tripleplane(
        pointcloud=obs_point_cloud[..., :3],
        rgb=obs_point_cloud[..., 3:6],
        voxel_size=0.005,
        img_size=(224, 224),
        point_size=2,
        workspace_bounds=workspace_bounds,
        camera_params=camera_params,
        projection_mode="perspective",
        return_pointmap=True,
        fov=37
    )
    
    # Convert RGB outputs to BGR for downstream consumers, only use for beat_block_hammer task
    img_xy = img_xy[..., ::-1]
    img_xz = img_xz[..., ::-1]
    img_yz = img_yz[..., ::-1]

    # 5) Robot state
    ja = observation.get('joint_action', {})
    ep = observation.get('endpose', {})

    state_parts = []
    if 'left_arm' in ja: state_parts.append(np.atleast_1d(ja['left_arm']))
    if 'left_gripper' in ja: state_parts.append(np.atleast_1d(ja['left_gripper']))
    if 'right_arm' in ja: state_parts.append(np.atleast_1d(ja['right_arm']))
    if 'right_gripper' in ja: state_parts.append(np.atleast_1d(ja['right_gripper']))
    if 'left_endpose' in ep: state_parts.append(np.atleast_1d(ep['left_endpose']))
    if 'right_endpose' in ep: state_parts.append(np.atleast_1d(ep['right_endpose']))

    obs_robot_state = np.concatenate(state_parts).astype(np.float32)

    return {
        'pmp_xy_plane': pmp_xy,
        'pmp_xz_plane': pmp_xz,
        'pmp_yz_plane': pmp_yz,
        'xy_plane': img_xy,
        'xz_plane': img_xz,
        'yz_plane': img_yz,
        'agent_pos': obs_robot_state
    }

def get_model(usr_args):
    # Standard Loading Logic
    ckpt_path = usr_args.get('ckpt_path', None)
    
    if ckpt_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        task_name = usr_args.get('task_name', 'default_task')
        policy_name = usr_args.get('policy_name', 'Your_Policy')
        ckpt_setting = usr_args.get('ckpt_setting', 'default_ckpt')
        
        # Adjust path as necessary
        ckpt_path = os.path.join(base_dir, "data", "outputs", task_name, policy_name, ckpt_setting, "checkpoints", "latest.ckpt")
        
        # Fallback search if needed, e.g. parent directories
        if not os.path.exists(ckpt_path):
             # Try assuming we are in policy/Your_Policy/
             # Data might be in ../../data/outputs
             root_outputs = os.path.abspath(os.path.join(base_dir, "../../data/outputs"))
             path_candidate = os.path.join(root_outputs, task_name, policy_name, ckpt_setting, "checkpoints", "latest.ckpt")
             if os.path.exists(path_candidate):
                 ckpt_path = path_candidate

    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found at {ckpt_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    except Exception as e:
        print(f"Failed to load with dill: {e}, trying standard torch.load")
        payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu')

    cfg = payload['cfg']
    try:
        cls = hydra.utils.get_class(cfg._target_)
    except:
        # Fallback if class not found in current path context
        # assuming standard workspace
        from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
        cls = TrainDiffusionUnetImageWorkspace

    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    policy.to(device)
    policy.eval()
    
    return PolicyWrapper(policy, cfg, device)


def eval(env, model, observation, instruction=None):
    # Instruction is optional; env.get_instruction() can be used if not provided.
    if instruction is None and hasattr(env, 'get_instruction'):
        instruction = env.get_instruction()

    obs = encode_obs(observation)

    # Handle both local execution (PolicyWrapper) and remote execution (ModelClient)
    if hasattr(model, 'call'): # Client mode
        # Check cache length on server
        if model.call('get_obs_cache_len') == 0:
            model.call('update_obs', obs=obs)
        actions = model.call('get_action')
    else: # Local mode
        if len(model.obs_cache) == 0:
            model.update_obs(obs)
        actions = model.get_action()

    for action in actions:
        # Assuming QPos control as per standard
        env.take_action(action, action_type='qpos')
        observation = env.get_obs()
        obs = encode_obs(observation)
        
        if hasattr(model, 'call'):
            model.call('update_obs', obs=obs)
        else:
            model.update_obs(obs)

def reset_model(model):  
    if hasattr(model, 'call'):
        # For remote model, we might need a reset method exposed or clearing logic
        # Assuming server reset logic isn't explicitly exposed via cmd='reset_model' in base server
        # But we can add a reset method to PolicyWrapper if needed.
        # However, PolicyWrapper creates new empty deque on init.
        # If we need to clear during run:
        model.call('reset_obs_cache')
    else:
        model.obs_cache.clear()