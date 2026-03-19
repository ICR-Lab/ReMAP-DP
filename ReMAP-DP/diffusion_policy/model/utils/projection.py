import numpy as np
import torch
from typing import Optional, Tuple, List, Union
import sys
sys.path.insert(0, '/home/icrlab/imitation/improved_dp/')
fps_downsample = None

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


def _opengl_to_opencv(cam2world_gl: np.ndarray) -> np.ndarray:
    """Convert an OpenGL camera-to-world transform to OpenCV convention."""
    cam2world_gl = _ensure_homogeneous(cam2world_gl)
    flip = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=cam2world_gl.dtype)
    return flip @ cam2world_gl @ flip

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

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def voxel_downsample(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    voxel_size: float = 0.01
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    对点云进行体素下采样
    
    Args:
        xyz: (N, 3) 点云坐标
        rgb: (N, 3) 可选的颜色信息，uint8 或 float
        voxel_size: 体素大小
        
    Returns:
        xyz_ds: (M, 3) 下采样后的点云
        rgb_ds: (M, 3) 下采样后的颜色，如果提供了rgb
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    
    if HAS_OPEN3D:
        # 使用 Open3D 进行快速体素下采样
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        if rgb is not None:
            col = np.asarray(rgb, dtype=np.float64)
            if not np.issubdtype(col.dtype, np.floating):
                col = col.astype(np.float32) / 255.0
            else:
                if col.max() > 1.0:
                    col = col / 255.0
            col = np.clip(col, 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(col)
        
        pcd_ds = pcd.voxel_down_sample(voxel_size=float(voxel_size))
        xyz_ds = np.asarray(pcd_ds.points)
        rgb_ds = np.asarray(pcd_ds.colors) if len(pcd_ds.colors) > 0 else None
        
        if rgb_ds is not None and rgb is not None and not np.issubdtype(rgb.dtype, np.floating):
            rgb_ds = (rgb_ds * 255.0).round().astype(np.uint8)
    else:
        # NumPy 实现的体素下采样
        N = xyz.shape[0]
        if N == 0:
            return xyz, rgb
        
        keys = np.floor(xyz / voxel_size).astype(np.int64)
        vox_map = {}
        
        if rgb is not None:
            rgb = np.asarray(rgb, dtype=np.float64)
            for i in range(N):
                k = tuple(keys[i])
                if k not in vox_map:
                    vox_map[k] = [xyz[i].copy(), 1, rgb[i].copy()]
                else:
                    vox_map[k][0] += xyz[i]
                    vox_map[k][1] += 1
                    vox_map[k][2] += rgb[i]
            
            pts = []
            cols = []
            for s in vox_map.values():
                pts.append(s[0] / s[1])
                cols.append(s[2] / s[1])
            xyz_ds = np.asarray(pts, dtype=np.float64)
            rgb_ds = np.asarray(cols)
        else:
            for i in range(N):
                k = tuple(keys[i])
                if k not in vox_map:
                    vox_map[k] = [xyz[i].copy(), 1]
                else:
                    vox_map[k][0] += xyz[i]
                    vox_map[k][1] += 1
            
            xyz_ds = np.asarray([v[0] / v[1] for v in vox_map.values()], dtype=np.float64)
            rgb_ds = None
    
    return xyz_ds, rgb_ds

def project_to_plane(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    plane: str = "xy",
    img_size: Tuple[int, int] = (128, 128),
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    point_size: int = 1,
    projection_mode: str = "orthographic",
    camera_pos: Optional[Tuple[float, float, float]] = None,
    camera_lookat: Optional[Tuple[float, float, float]] = None,
    camera_up: Optional[Tuple[float, float, float]] = None,
    return_pointmap: bool = False,
    fov: Optional[float] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    将3D点云投影到2D平面并栅格化为图像
    
    Args:
        xyz: (N, 3) 点云坐标
        rgb: (N, 3) 可选的颜色信息
        plane: 投影平面 'xy', 'xz', 或 'yz'
        img_size: (H, W) 输出图像大小
        bounds: ((umin, umax), (vmin, vmax)) 投影范围，None则自动计算
        point_size: 点的渲染大小
        projection_mode: 'orthographic' 或 'perspective'
        return_pointmap: 是否返回对应的pointmap (H, W, 3)，存储每个像素对应的3D坐标
        
    Returns:
        img: (H, W, 3) uint8 图像
        pointmap: (H, W, 3) float32 坐标图 (仅当 return_pointmap=True 时返回)
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if rgb is not None:
        rgb = np.asarray(rgb)
    H, W = int(img_size[0]), int(img_size[1])
    
    plane = plane.lower()
    axis_map = {
        "xy": (0, 1, 2),  # depth along z
        "xz": (0, 2, 1),  # depth along y
        "yz": (1, 2, 0),  # depth along x
    }
    if plane not in axis_map:
        raise ValueError(f"Unsupported plane: {plane}")

    u_idx, v_idx, depth_idx = axis_map[plane]
    u_coord = xyz[:, u_idx]
    v_coord = xyz[:, v_idx]
    depth_coord = xyz[:, depth_idx]

    if plane == "xy":
        # Rotate 90 degrees counter-clockwise: u' = -v, v' = u
        u_temp = u_coord.copy()
        u_coord = -v_coord
        v_coord = u_temp

    projection_mode = projection_mode.lower()
    if projection_mode not in {"orthographic", "perspective"}:
        raise ValueError(f"Unsupported projection mode: {projection_mode}")

    sort_idx = None
    if projection_mode == "orthographic":
        uv = np.stack([u_coord, v_coord], axis=1)
        # Sort by depth (low to high), assuming camera at +infinity
        sort_idx = np.argsort(depth_coord)
    else:
        # Perspective projection: support explicit pinhole camera parameters when provided.
        # Default fallback (legacy, axis-aligned camera) is used when camera_pos is None.
        if camera_pos is None or camera_lookat is None:
            # Legacy axis-aligned perspective (camera placed along depth axis)
            if depth_coord.size > 0:
                depth_min = float(np.min(depth_coord))
                depth_max = float(np.max(depth_coord))
                depth_center = 0.5 * (depth_min + depth_max)
            else:
                depth_min, depth_max, depth_center = -0.5, 0.5, 0.0
            max_extent = max(
                (float(np.max(u_coord)) - float(np.min(u_coord))) * 0.5,
                (float(np.max(v_coord)) - float(np.min(v_coord))) * 0.5,
                1e-3,
            )

            # Ensure camera is placed outside the object's depth range
            depth_half_size = (depth_max - depth_min) * 0.5
            cam_dist = max(max_extent * 2.0, depth_half_size * 4.0)

            cam_pos = depth_center + cam_dist
            dz = cam_pos - depth_coord
            dz = np.maximum(dz, 1e-3)

            u_center = 0.5 * (float(np.max(u_coord)) + float(np.min(u_coord)))
            v_center = 0.5 * (float(np.max(v_coord)) + float(np.min(v_coord)))

            proj_scale = max_extent / dz
            proj_u = (u_coord - u_center) * proj_scale
            proj_v = (v_coord - v_center) * proj_scale
            uv = np.stack([proj_u, proj_v], axis=1)
            
            # Sort by depth (low to high)
            sort_idx = np.argsort(depth_coord)
        else:
            # Pinhole camera projection with arbitrary camera pose.
            # Build camera coordinate frame
            cam_pos = np.asarray(camera_pos, dtype=np.float64)
            lookat = np.asarray(camera_lookat, dtype=np.float64)
            cam_dir = lookat - cam_pos
            if np.linalg.norm(cam_dir) < 1e-9:
                raise ValueError("camera_pos and camera_lookat must be different")
            cam_dir = cam_dir / np.linalg.norm(cam_dir)

            if camera_up is None:
                # choose a world-up (prefer z-axis)
                up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                up = np.asarray(camera_up, dtype=np.float64)
            # make up orthogonal to cam_dir
            right = np.cross(cam_dir, up)
            if np.linalg.norm(right) < 1e-6:
                # camera is looking along/up axis; pick another up
                up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                right = np.cross(cam_dir, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, cam_dir)
            up = up / np.linalg.norm(up)

            # transform points into camera frame
            pts = xyz.astype(np.float64)
            rel = pts - cam_pos[None, :]
            x_cam = rel.dot(right)
            y_cam = rel.dot(up)
            z_cam = rel.dot(cam_dir)

            # Filter points behind the camera
            # Use a reasonable near plane to avoid singularity and outliers
            near_plane = 0.01 # 1cm
            valid_mask = z_cam > near_plane
            
            xyz = xyz[valid_mask]
            if rgb is not None:
                rgb = rgb[valid_mask]
            
            x_cam = x_cam[valid_mask]
            y_cam = y_cam[valid_mask]
            z_cam = z_cam[valid_mask]

            if z_cam.size == 0:
                u_img = np.zeros(0, dtype=np.float64)
                v_img = np.zeros(0, dtype=np.float64)
            else:
                # Normalize by depth to get image-plane coordinates
                u_img = x_cam / z_cam
                v_img = y_cam / z_cam

            # compute bounds in image plane if not provided
            if bounds is None:
                if fov is not None and projection_mode == "perspective":
                    # Use FOV to determine bounds
                    # fov is vertical field of view in degrees
                    half_fov_rad = np.deg2rad(fov) / 2.0
                    vmax = np.tan(half_fov_rad)
                    vmin = -vmax
                    
                    aspect_ratio = W / H
                    umax = vmax * aspect_ratio
                    umin = -umax
                elif u_img.size == 0:
                    umin, umax, vmin, vmax = -1.0, 1.0, -1.0, 1.0
                else:
                    # Use percentiles to be robust against outliers
                    umin, umax = np.percentile(u_img, [0.5, 99.5])
                    vmin, vmax = np.percentile(v_img, [0.5, 99.5])
                    
                    # If range is too small (e.g. single point), expand it
                    if umax - umin < 1e-6:
                        umin -= 0.1
                        umax += 0.1
                    if vmax - vmin < 1e-6:
                        vmin -= 0.1
                        vmax += 0.1
                        
                    # Add a small margin
                    du = 0.05 * (umax - umin)
                    dv = 0.05 * (vmax - vmin)
                    umin, umax = umin - du, umax + du
                    vmin, vmax = vmin - dv, vmax + dv
                # Persist computed bounds so later mapping uses them instead of recomputing min/max
                bounds = ((umin, umax), (vmin, vmax))
            else:
                (umin, umax), (vmin, vmax) = bounds

            # pack uv from normalized image coords
            uv = np.stack([u_img, v_img], axis=1)
            
            # Sort by z_cam (far to close)
            sort_idx = np.argsort(z_cam)[::-1]
    
    # 准备颜色
    if rgb is not None:
        rgb = np.asarray(rgb)
        if np.issubdtype(rgb.dtype, np.floating):
            if rgb.max() <= 1.0:
                rgb = (rgb * 255.0).round()
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        elif rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        
        if rgb.ndim == 1 or rgb.shape[1] != 3:
            rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)
    else:
        rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)
    
    # 计算投影范围
    if bounds is None:
        umin, umax = float(np.min(uv[:, 0])), float(np.max(uv[:, 0]))
        vmin, vmax = float(np.min(uv[:, 1])), float(np.max(uv[:, 1]))
        
        # 避免零范围
        if umax - umin < 1e-9:
            umax = umin + 1.0
        if vmax - vmin < 1e-9:
            vmax = vmin + 1.0
        
        # 添加边距
        du = 0.05 * (umax - umin)
        dv = 0.05 * (vmax - vmin)
        umin, umax = umin - du, umax + du
        vmin, vmax = vmin - dv, vmax + dv
    else:
        (umin, umax), (vmin, vmax) = bounds
    
    # 映射到像素坐标
    x = (uv[:, 0] - umin) / max(1e-12, (umax - umin)) * (W - 1)
    y = (vmax - uv[:, 1]) / max(1e-12, (vmax - vmin)) * (H - 1)
    xi = np.clip(np.round(x).astype(int), 0, W - 1)
    yi = np.clip(np.round(y).astype(int), 0, H - 1)
    
    # Apply sorting (Painter's Algorithm)
    xyz_sorted = xyz
    if sort_idx is not None:
        xi = xi[sort_idx]
        yi = yi[sort_idx]
        rgb = rgb[sort_idx]
        xyz_sorted = xyz[sort_idx]
    
    # 创建图像
    img = np.zeros((H, W, 3), dtype=np.uint8)
    pointmap = None
    if return_pointmap:
        pointmap = np.zeros((H, W, 3), dtype=np.float32)
    
    # 渲染点
    r = max(0, int(point_size) // 2)
    if r == 0:
        for i in range(uv.shape[0]):
            img[yi[i], xi[i]] = rgb[i]
            if return_pointmap:
                pointmap[yi[i], xi[i]] = xyz_sorted[i]
    else:
        for i in range(uv.shape[0]):
            cx, cy = xi[i], yi[i]
            x0, x1 = max(0, cx - r), min(W, cx + r + 1)
            y0, y1 = max(0, cy - r), min(H, cy + r + 1)
            if x0 < x1 and y0 < y1:
                img[y0:y1, x0:x1] = rgb[i]
                if return_pointmap:
                    pointmap[y0:y1, x0:x1] = xyz_sorted[i]
    
    if return_pointmap:
        return img, pointmap
    return img


def project_to_tripleplane(
    pointcloud: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    voxel_size: float = 0.01,
    img_size: Tuple[int, int] = (128, 128),
    point_size: int = 1,
    workspace_bounds: Optional[Tuple[Tuple[float, float],
                                     Tuple[float, float],
                                     Tuple[float, float]]] = None,
    projection_mode: str = "orthographic",
    camera_params: Optional[dict] = None,
    return_pointmap: bool = False,
    fov: Optional[float] = None,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    将点云在工作空间内体素化，然后投影到三个平面获得图像
    
    Args:
        pointcloud: (N, 3) 或 (N, C) 点云，C>=3时取前3列为xyz
        rgb: (N, 3) 可选的RGB颜色信息，uint8[0,255] 或 float[0,1]
        voxel_size: 体素大小，用于下采样
        img_size: (H, W) 输出图像大小
        point_size: 点的渲染大小（像素）
        workspace_bounds: ((xmin, xmax), (ymin, ymax), (zmin, zmax)) 
                         工作空间边界，None则使用点云范围
        projection_mode: 'orthographic' 或 'perspective'
        return_pointmap: 是否返回对应的pointmap
    
    Returns:
        如果 return_pointmap=False:
            img_xy, img_xz, img_yz: (H, W, 3) 投影图像
        如果 return_pointmap=True:
            img_xy, img_xz, img_yz, pmap_xy, pmap_xz, pmap_yz
            其中 pmap_* 为 (H, W, 3) float32 坐标图
    """
    pointcloud = np.asarray(pointcloud)
    if pointcloud.ndim == 3:
        Bs, N, C = pointcloud.shape
        outputs_xy: List[np.ndarray] = []
        outputs_xz: List[np.ndarray] = []
        outputs_yz: List[np.ndarray] = []
        pmaps_xy: List[np.ndarray] = []
        pmaps_xz: List[np.ndarray] = []
        pmaps_yz: List[np.ndarray] = []
        
        for b in range(Bs):
            pc_b = pointcloud[b]
            rgb_b = rgb[b] if rgb is not None else None
            res = project_to_tripleplane(
                pc_b, rgb=rgb_b, voxel_size=voxel_size,
                img_size=img_size, point_size=point_size,
                workspace_bounds=workspace_bounds,
                projection_mode=projection_mode,
                return_pointmap=return_pointmap,
                fov=fov,
            )
            if return_pointmap:
                outputs_xy.append(res[0])
                outputs_xz.append(res[1])
                outputs_yz.append(res[2])
                pmaps_xy.append(res[3])
                pmaps_xz.append(res[4])
                pmaps_yz.append(res[5])
            else:
                outputs_xy.append(res[0])
                outputs_xz.append(res[1])
                outputs_yz.append(res[2])
        
        if return_pointmap:
            return (
                np.stack(outputs_xy, axis=0),
                np.stack(outputs_xz, axis=0),
                np.stack(outputs_yz, axis=0),
                np.stack(pmaps_xy, axis=0),
                np.stack(pmaps_xz, axis=0),
                np.stack(pmaps_yz, axis=0),
            )
        else:
            return (
                np.stack(outputs_xy, axis=0),
                np.stack(outputs_xz, axis=0),
                np.stack(outputs_yz, axis=0),
            )

    pointcloud = np.asarray(pointcloud, dtype=np.float64)
    projection_mode = projection_mode.lower()
    if projection_mode not in {"orthographic", "perspective"}:
        raise ValueError(f"Unsupported projection mode: {projection_mode}")
    if pointcloud.ndim != 2 or pointcloud.shape[1] < 3:
        raise ValueError(f"点云形状应为 (N, 3) 或 (N, C) 其中 C>=3，得到: {pointcloud.shape}")

    xyz = pointcloud[:, :3]
    
    # 如果指定了工作空间边界，过滤点云
    if workspace_bounds is not None:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = workspace_bounds
        mask = (
            (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
            (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
            (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
        )
        xyz = xyz[mask]
        if rgb is not None:
            rgb = rgb[mask]
        
        if xyz.shape[0] == 0:
            # 如果过滤后没有点，返回空白图像
            empty = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
            if return_pointmap:
                empty_pmap = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
                return empty, empty, empty, empty_pmap, empty_pmap, empty_pmap
            return empty, empty, empty
    
    # 体素下采样
    xyz_ds, rgb_ds = voxel_downsample(xyz, rgb, voxel_size=voxel_size)
    
    if xyz_ds.shape[0] == 0:
        # 如果下采样后没有点，返回空白图像
        empty = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        if return_pointmap:
            empty_pmap = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
            return empty, empty, empty, empty_pmap, empty_pmap, empty_pmap
        return empty, empty, empty
    
    # 计算统一的投影范围（可选：为三个视图使用相同的范围）
    # 这里我们为每个平面独立计算范围以最大化利用图像空间
    
    # 投影到XY平面
    cam_xy = camera_params.get('xy') if camera_params is not None else None
    cam_xz = camera_params.get('xz') if camera_params is not None else None
    cam_yz = camera_params.get('yz') if camera_params is not None else None

    res_xy = project_to_plane(
        xyz_ds, rgb_ds, plane="xy",
        img_size=img_size, point_size=point_size,
        projection_mode=projection_mode,
        camera_pos=(cam_xy.get('pos') if cam_xy else None),
        camera_lookat=(cam_xy.get('lookat') if cam_xy else None),
        camera_up=(cam_xy.get('up') if cam_xy else None),
        return_pointmap=return_pointmap,
        fov=fov,
    )
    
    # 投影到XZ平面
    res_xz = project_to_plane(
        xyz_ds, rgb_ds, plane="xz",
        img_size=img_size, point_size=point_size,
        projection_mode=projection_mode,
        camera_pos=(cam_xz.get('pos') if cam_xz else None),
        camera_lookat=(cam_xz.get('lookat') if cam_xz else None),
        camera_up=(cam_xz.get('up') if cam_xz else None),
        return_pointmap=return_pointmap,
        fov=fov,
    )
    
    # 投影到YZ平面
    res_yz = project_to_plane(
        xyz_ds, rgb_ds, plane="yz",
        img_size=img_size, point_size=point_size,
        projection_mode=projection_mode,
        camera_pos=(cam_yz.get('pos') if cam_yz else None),
        camera_lookat=(cam_yz.get('lookat') if cam_yz else None),
        camera_up=(cam_yz.get('up') if cam_yz else None),
        return_pointmap=return_pointmap,
        fov=fov,
    )
    
    if return_pointmap:
        return res_xy[0], res_xz[0], res_yz[0], res_xy[1], res_xz[1], res_yz[1]
    
    return res_xy, res_xz, res_yz


if __name__ == "__main__":
    from dataset.load_trajectories import load_hdf5
    """
    测试示例：生成随机点云并投影到三平面
    """
    print("测试 project_to_tripleplane 函数...")
    
    # 生成测试数据：在球形区域内的随机点云
    np.random.seed(42)
    n_points = 5000
    
    # 生成球形点云
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = np.random.uniform(0, 0.2, n_points)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    # 从h5数据集中加载点云
    data_path = '/home/icrlab/imitation/demos/PushCube-v1/motionplanning/20251202_162142.rgbd.pd_ee_delta_pose.physx_cpu.h5'
    data = load_hdf5(data_path)
    rgb = data['traj_0']['obs']['sensor_data']['base_camera']['rgb'][0]
    depth = data['traj_0']['obs']['sensor_data']['base_camera']['depth'][0]
    intrinsics = data['traj_0']['obs']['sensor_param']['base_camera']['intrinsic_cv'][0]
    extrinsics = data['traj_0']['obs']['sensor_param']['base_camera']['extrinsic_cv'][0]
    cam2world_gl = data['traj_0']['obs']['sensor_param']['base_camera']['cam2world_gl'][0]
    sensor_params = {
        'base_camera': {
            'intrinsic_cv': intrinsics,
            'extrinsic_cv': extrinsics,
            'cam2world_gl': cam2world_gl
        }
    }
    sensor_data = {
        'base_camera': {
            'rgb': rgb,
            'depth': depth
        }
    }
    sensor_uids = ['base_camera']
    pointcloud = get_point_cloud(sensor_data, sensor_params, sensor_uids, False, 0)
    
    print(f"输入点云形状: {pointcloud.shape}")
    print(f"点云范围: X=[{x.min():.3f}, {x.max():.3f}], "
          f"Y=[{y.min():.3f}, {y.max():.3f}], "
          f"Z=[{z.min():.3f}, {z.max():.3f}]")
    
    # 测试1: 基本投影
    print("\n测试1: 基本投影（带体素化）")
    camera_params = {
        'xy': {
            'pos': (0.3, 0.0, 0.6),
            'lookat': (-0.1, 0.0, 0.1),
        }
    }
    workspace_bounds = ((-0.7, 0.5), (-0.5, 0.5), (0, 1.5))
    img_xy, img_xz, img_yz = project_to_tripleplane(
        pointcloud[..., :3], 
        rgb=pointcloud[..., 3:],
        voxel_size=0.005,
        img_size=(256, 256),
        point_size=2,
        camera_params=camera_params,
        workspace_bounds=workspace_bounds,
        projection_mode="perspective",
    )
    
    print(f"XY平面图像形状: {img_xy.shape}")
    print(f"XZ平面图像形状: {img_xz.shape}")
    print(f"YZ平面图像形状: {img_yz.shape}")
    
    # 保存结果（如果有PIL或imageio）
    try:
        from PIL import Image
        import os
        
        output_dir = "/home/icrlab/imitation/improved_dp/diffusion_policy/model/vision/"
        os.makedirs(output_dir, exist_ok=True)
        
        Image.fromarray(img_xy).save(os.path.join(output_dir, "projection_xy.png"))
        Image.fromarray(img_xz).save(os.path.join(output_dir, "projection_xz.png"))
        Image.fromarray(img_yz).save(os.path.join(output_dir, "projection_yz.png"))
        
        print(f"\n投影图像已保存到: {output_dir}")
        print("  - projection_xy.png")
        print("  - projection_xz.png")
        print("  - projection_yz.png")
    except ImportError:
        print("\n警告: 未安装PIL，跳过图像保存")
    
    # 测试2: 使用工作空间边界
    print("\n测试2: 使用工作空间边界过滤")
    workspace_bounds = ((-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15))
    img_xy2, img_xz2, img_yz2 = project_to_tripleplane(
        pointcloud[..., :3],
        rgb=pointcloud[..., 3:],
        voxel_size=0.005,
        img_size=(128, 128),
        point_size=2,
        workspace_bounds=workspace_bounds,
        projection_mode="orthographic"
    )
    print("工作空间边界投影完成")
    
    # 测试3: 无颜色信息
    print("\n测试3: 无颜色信息（白色点）")
    img_xy3, img_xz3, img_yz3 = project_to_tripleplane(
        pointcloud,
        rgb=None,
        voxel_size=0.01,
        img_size=(128, 128),
        point_size=1
    )
    print("无颜色投影完成")

    # 测试4: 透视投影
    print("\n测试4: 透视投影模式")
    img_xy4, img_xz4, img_yz4 = project_to_tripleplane(
        pointcloud,
        rgb=rgb,
        voxel_size=0.005,
        img_size=(256, 256),
        point_size=2,
        projection_mode="perspective"
    )
    print("透视投影完成")
    
    print("\n所有测试完成！")