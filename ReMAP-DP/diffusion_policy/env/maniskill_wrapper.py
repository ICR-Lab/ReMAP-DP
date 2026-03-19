from termcolor import cprint
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import cv2
from mani_skill.utils import gym_utils
from mani_skill.utils import common
from common.utils import downsample_with_fps
import open3d as o3d
from diffusion_policy.model.utils.projection import project_to_tripleplane
from mani_skill.envs.sapien_env import BaseEnv
import torch

# the range to crop poindcloud
TASK_BOUDNS = {
    'GraspCup-v1': [-0.7, -1, 0.00, 1, 1, 2],
    'PickCube-v1': [-0.7, -1, 0.00, 1, 1, 2],
    'default': [-0.7, -1, 0.00, 1, 1, 2],
}

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


class ManiSkillEnv(gym.Wrapper):
    def __init__(self,
                 env: gym.Env,
                 task_name,
                 use_point_crop=True,
                 use_pc_fps=False,
                 num_points=1024,
                 resolution=(128, 128),
                 projection_mode="perspective",
                 ):
        super().__init__(env)

        self.width, self.height = resolution
        self.use_point_crop = use_point_crop
        self.use_pc_fps = use_pc_fps
        self.projection_mode = projection_mode
        cprint("[ManiSkillEnv] use_point_crop: {}".format(self.use_point_crop), "cyan")
        cprint("[ManiSkillEnv] use_pc_fps: {}".format(use_pc_fps), "cyan")
        cprint("[ManiSkillEnv] resolution: {}".format(resolution), "cyan")
        cprint("[ManiSkillEnv] projection_mode: {}".format(projection_mode), "cyan")
        self.num_points = num_points

        self.pc_scale = np.array([1, 1, 1])
        self.pc_offset = np.array([0, 0, 0])
        if task_name in TASK_BOUDNS:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS[task_name]
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS['default']
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]

        self.episode_length = self._max_episode_steps = 300 # lch fixed？

        # TODO: check the following action, state, observation space dimension
        self.action_space = self.base_env.action_space
        self.obs_state_dim = self.base_env.observation_space["state"].shape[1] # TODO 29 or 25
        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_state_dim,),
                dtype=np.float32
            ),
            'rgb': spaces.Box(
                low=0,
                high=255,
                shape=(self.width, self.height, 3),
                dtype=np.uint8
            ),
            'xy_plane':spaces.Box(
                low=0,
                high=255,
                shape=(self.width, self.height, 3),
                dtype=np.uint8
            ),
            'xz_plane':spaces.Box(
                low=0,
                high=255,
                shape=(self.width, self.height, 3),
                dtype=np.uint8
            ),
            'yz_plane':spaces.Box(
                low=0,
                high=255,
                shape=(self.width, self.height, 3),
                dtype=np.uint8
            ),
            'left_img':spaces.Box(
                low=0,
                high=255,
                shape=(self.width, self.height, 3),
                dtype=np.uint8
            ),
            'right_img':spaces.Box(
                low=0,
                high=255,
                shape=(self.width, self.height, 3),
                dtype=np.uint8
            ),
            'pmp_xy_plane':spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(224, 224, 3),
                dtype=np.float32
            ),
            'pmp_xz_plane':spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(224, 224, 3),
                dtype=np.float32
            ),
            'pmp_yz_plane':spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(224, 224, 3),
                dtype=np.float32
            ),
        })

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped
    
    @staticmethod
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
        self,
        sensor_data,
        sensor_param,
        sensor_uids,
        use_point_crop=True,
    target_points=1024,
    ):
        all_points = []
        all_colors = []
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
            if depth.ndim != 2:
                raise ValueError(f"Depth map for sensor {sensor_uid} must be 2D after squeeze, got shape {depth.shape}")
            
            # 获取图像尺寸
            H, W = depth.shape[:2]

            if color.ndim == 3 and color.shape[-1] == 1:
                color = color[..., 0]
            color = np.squeeze(color)
            if color.ndim == 2:
                color = np.repeat(color[..., None], repeats=3, axis=-1)
            if color.ndim != 3 or color.shape[0] != H or color.shape[1] != W:
                raise ValueError(
                    f"Color image for sensor {sensor_uid} must be HxWx3 ({H}x{W}), got shape {color.shape}"
                )
            color = color.reshape(-1, 3)
            
            # 创建像素坐标网格
            u = np.arange(W)
            v = np.arange(H)
            u, v = np.meshgrid(u, v)
            
            # 将深度转换为米，并过滤无效点
            depth_meters = depth.astype(np.float32) / 1000.0  # 毫米转米
            valid_mask = depth_meters > 0

            flat_valid_mask = valid_mask.reshape(-1)
            colors = color.astype(np.float32)[flat_valid_mask]
            if colors.max(initial=0.0) > 1.0:
                colors /= 255.0
            
            # 计算相机坐标系下的3D坐标
            z = depth_meters[valid_mask]
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]
            
            # 使用相机内参反投影到相机坐标系
            fx = intrinsic_cv[0, 0]
            fy = intrinsic_cv[1, 1]
            cx = intrinsic_cv[0, 2]
            cy = intrinsic_cv[1, 2]
            
            x_cam = (u_valid - cx) * z / fx
            y_cam = (v_valid - cy) * z / fy
            z_cam = z
            
            # 构建相机坐标系下的点云 (N, 3)
            points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
            
            # 转换到世界坐标系 (OpenCV convention)
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
                raise ValueError(f"Camera {sensor_uid} missing both extrinsic_cv and cam2world_gl transforms")

            if points_cam_homo.shape[0] == 0:
                continue

            points_world_homo = (cam_to_world @ points_cam_homo.T).T
            points_world = points_world_homo[:, :3]
            w = points_world_homo[:, 3:4]
            if w.shape[0] > 0:
                points_world = points_world / np.maximum(w, 1e-8)

            # 检查每个点是否在边界框内
            min_bound = np.array([-0.7, -1, 0.00])
            max_bound = np.array([1, 1, 2])
            in_bound_mask = np.all((points_world >= min_bound) & (points_world <= max_bound), axis=1)
            points_world = points_world[in_bound_mask]
            colors = colors[in_bound_mask]

            # 收集当前相机的点和颜色
            if len(points_world) > 0:
                all_points.append(points_world)
                all_colors.append(colors)
        
        # 合并所有相机的点云
        if all_points:
            all_points = np.concatenate(all_points, axis=0)  # [M, 3]
            all_colors = np.concatenate(all_colors, axis=0)  # [M, 3]
            
            # 组合成xyzrgb格式 [M, 6]
            pointcloud = np.concatenate([all_points, all_colors], axis=1).astype(np.float32)
        else:
            # 如果没有有效点，返回空数组
            pointcloud = np.zeros((0, 6), dtype=np.float32)

        n_points = pointcloud.shape[0]
        if target_points is None or target_points <= 0:
            return pointcloud.astype(np.float32)

        if n_points == 0:
            return np.zeros((target_points, 6), dtype=np.float32)

        fps_downsample = None
        if n_points >= target_points:
            if fps_downsample is not None and torch.cuda.is_available():
                try:
                    sampled = fps_downsample(pointcloud.astype(np.float32), num_points=target_points)
                    if isinstance(sampled, np.ndarray) and sampled.shape[0] == target_points:
                        return sampled.astype(np.float32)
                except Exception:
                    pass
            if n_points > target_points:
                idx = np.linspace(0, n_points - 1, target_points, dtype=np.int64)
                pointcloud = pointcloud[idx]
            return pointcloud.astype(np.float32)

        pad_needed = target_points - n_points
        pad_block = np.repeat(pointcloud[:1], repeats=pad_needed, axis=0)
        pointcloud = np.concatenate([pointcloud, pad_block], axis=0)
        return pointcloud.astype(np.float32)    

    def step(self, action):
        raw_obs, reward, done, truncated, env_info = self.env.step(action)
        self.cur_step += 1

        # raw_obs: dict with 3 items
        #           raw_obs["state"]: dict with state [state_shape]
        #           raw_obs["pointcloud"]: dict with pointcloud [num, 4]
        #           raw_obs["rgb"]: dict with rgb [num, 3]
        robot_state = raw_obs["state"]
        sensor_data = raw_obs["sensor_data"]
        sensor_param = raw_obs["sensor_param"]
        sensor_uids = raw_obs["sensor_data"].keys()
        
        point_cloud = self.get_point_cloud(sensor_data, sensor_param, sensor_uids,
                                           use_point_crop=self.use_point_crop, target_points=0)
        
        workspace_bounds = ((-0.7, 0.5), (-0.5, 0.5), (0, 1.5))
        img_xy, img_xz, img_yz, pmp_xy, pmp_xz, pmp_yz = project_to_tripleplane(
            pointcloud=point_cloud[..., :3],
            rgb=point_cloud[...,3:],
            voxel_size=0.005,
            img_size=(224, 224),
            point_size=2,
            workspace_bounds=workspace_bounds,
            projection_mode=self.projection_mode,
            return_pointmap=True
        )

        # only use the 'base_camera' sensor for rgbd observation
        # hand camera data is only used for point cloud generation
        rgb = sensor_data['base_camera']['rgb']
        left_rgb = cv2.resize(sensor_data['left_camera']['rgb'], (224, 224), interpolation=cv2.INTER_AREA)
        right_rgb = cv2.resize(sensor_data['right_camera']['rgb'], (224, 224), interpolation=cv2.INTER_AREA)
        resized_image = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

        obs_dict = {
            'agent_pos': robot_state,
            'rgb': resized_image,
            'right_img': right_rgb,
            'left_img': left_rgb,
            'xy_plane': img_xy,
            'xz_plane': img_xz,
            'yz_plane': img_yz,
            'pmp_xy_plane': pmp_xy,
            'pmp_xz_plane': pmp_xz,
            'pmp_yz_plane': pmp_yz,
        }

        done = done or self.cur_step >= self.episode_length

        return obs_dict, reward, done, truncated, env_info

    def reset(self, **kwargs):
        raw_obs, env_info = self.env.reset(**kwargs)
        self.cur_step = 0

        robot_state = raw_obs["state"]
        sensor_data = raw_obs["sensor_data"]
        sensor_param = raw_obs["sensor_param"]
        sensor_uids = raw_obs["sensor_data"].keys()
        
        point_cloud = self.get_point_cloud(sensor_data, sensor_param, sensor_uids,
                                           use_point_crop=self.use_point_crop, target_points=0)
        
        workspace_bounds = ((-0.7, 0.5), (-0.5, 0.5), (0, 1.5))
        img_xy, img_xz, img_yz, pmp_xy, pmp_xz, pmp_yz = project_to_tripleplane(
            pointcloud=point_cloud[..., :3],
            rgb=point_cloud[...,3:],
            voxel_size=0.005,
            img_size=(224, 224),
            point_size=2,
            workspace_bounds=workspace_bounds,
            projection_mode=self.projection_mode,
            return_pointmap=True
        )

        # only use the 'base_camera' sensor for rgbd observation
        # hand camera data is only used for point cloud generation
        rgb = sensor_data['base_camera']['rgb']
        left_rgb = cv2.resize(sensor_data['left_camera']['rgb'], (224, 224), interpolation=cv2.INTER_AREA)
        right_rgb = cv2.resize(sensor_data['right_camera']['rgb'], (224, 224), interpolation=cv2.INTER_AREA)
        resized_image = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

        obs_dict = {
            'agent_pos': robot_state,
            'rgb': resized_image,
            'right_img': right_rgb,
            'left_img': left_rgb,
            'xy_plane': img_xy,
            'xz_plane': img_xz,
            'yz_plane': img_yz,
            'pmp_xy_plane': pmp_xy,
            'pmp_xz_plane': pmp_xz,
            'pmp_yz_plane': pmp_yz,
        }
        
        return obs_dict, env_info

    def render(self):
        ret = self.env.render()
        if self.render_mode in ["rgb_array", "sensors", "all"]:
            return common.unbatch(common.to_numpy(ret))