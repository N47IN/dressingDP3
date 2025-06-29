import gym
import assistive_gym
import pybullet as p
import numpy as np
import cv2
import json
import os
import time
import sys
import pickle
from datetime import datetime
from pathlib import Path
import torch
import argparse
import matplotlib.pyplot as plt
from torch_geometric.data import Data

# Workaround to unpickle old model files (same as in enjoy.py)
import ppo.a2c_ppo_acktr
sys.modules['a2c_ppo_acktr'] = ppo.a2c_ppo_acktr
sys.path.append('a2c_ppo_acktr')

# Import the same modules as enjoy.py
from ppo.a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from ppo.a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

# Load the expert policy
EXPERT_POLICY_PATH = 'trained_models/ppo/BedBathingSawyer-v0.pt'
actor_critic, ob_rms = torch.load(EXPERT_POLICY_PATH, map_location='cuda')

def get_right_hand_pos(env):
    """Get the position of the human's right hand"""
    return np.array(p.getLinkState(env.human, 9, computeForwardKinematics=True, physicsClientId=env.id)[0])

def setup_camera_aimed_at_right_hand(env, offset=np.array([0.0, 0.4, 0.9])):
    """
    Place the camera at a fixed position above the right hand for a focused top-down view.
    This gives a consistent overhead view of the hand area.
    Offset is relative to the hand position (in meters).
    """
    hand_pos = get_right_hand_pos(env)
    camera_pos = hand_pos + offset
    camera_target = hand_pos + np.asarray([0.1, 0.3, 0.0])
    
    camera_config = {
        'position': camera_pos.tolist(),
        'target': camera_target.tolist(),
        'up': [0, 1, 0],
        'fov': 45.0,
        'near': 0.05,
        'far': 3.0,
        'width': 320,  # Reduced from 640 for better performance
        'height': 240  # Reduced from 480 for better performance
    }
    return camera_config

def get_rgb_depth_images(env, camera_config):
    """Get RGB and depth images from camera"""
    view_matrix = p.computeViewMatrix(
        camera_config['position'],
        camera_config['target'],
        camera_config['up'],
        physicsClientId=env.id
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        camera_config['fov'],
        camera_config['width'] / camera_config['height'],
        camera_config['near'],
        camera_config['far'],
        physicsClientId=env.id
    )
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        width=camera_config['width'],
        height=camera_config['height'],
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        physicsClientId=env.id
    )
    rgb_img = np.array(rgb_img).reshape(height, width, 4)[:, :, :3].astype(np.uint8)
    depth_img = np.array(depth_img).reshape(height, width)
    depth_img = camera_config['far'] * camera_config['near'] / (
        camera_config['far'] - (camera_config['far'] - camera_config['near']) * depth_img
    )
    seg_img = np.array(seg_img).reshape(height, width)
    return rgb_img, depth_img, seg_img, view_matrix, projection_matrix

def get_robot_state(env):
    """Get current robot state (joint positions)"""
    joint_positions, _, _ = env.get_motor_joint_states(env.robot)
    return np.array(joint_positions)[:7]  # 7-DOF for Sawyer

def get_end_effector_pose(env):
    """Get end-effector position and orientation"""
    # In the bed bathing environment, there's a tool attachment (sponge/cloth)
    # Get the tool base position which is the actual end-effector for the task
    # This is the position of the tool attachment that makes contact with the human
    tool_base_state = p.getBasePositionAndOrientation(env.tool, physicsClientId=env.id)
    position = np.array(tool_base_state[0])
    orientation = np.array(tool_base_state[1])
    
    return position, orientation

def calculate_eef_delta_movement(current_pose, previous_pose):
    """
    Calculate delta movement of end-effector between frames in 6D format.
    
    Args:
        current_pose: Tuple of (position, orientation) for current frame
        previous_pose: Tuple of (position, orientation) for previous frame, or None for first frame
    
    Returns:
        delta_pos: Position delta (3D vector)
        delta_ori: Orientation delta (3D axis-angle vector)
        delta_pos_magnitude: Scalar magnitude of position change
        delta_ori_magnitude: Scalar magnitude of orientation change
        delta_6d: 6D delta EEF action [pos_delta(3D), rot_delta(3D axis-angle)]
    """
    if previous_pose is None:
        # First frame - no movement
        return np.zeros(3), np.zeros(3), 0.0, 0.0, np.zeros(6)
    
    current_pos, current_ori = current_pose
    previous_pos, previous_ori = previous_pose
    
    # Calculate position delta
    delta_pos = current_pos - previous_pos
    delta_pos_magnitude = np.linalg.norm(delta_pos)
    
    # Calculate orientation delta as axis-angle using scipy (same as dressing_sim)
    try:
        from scipy.spatial.transform import Rotation as R
        
        # Create rotation objects from quaternions
        # Note: PyBullet returns quaternions as [x, y, z, w] but scipy expects [w, x, y, z]
        current_quat = [current_ori[3], current_ori[0], current_ori[1], current_ori[2]]  # Convert to [w, x, y, z]
        previous_quat = [previous_ori[3], previous_ori[0], previous_ori[1], previous_ori[2]]  # Convert to [w, x, y, z]
        
        current_R = R.from_quat(current_quat)
        previous_R = R.from_quat(previous_quat)
        
        # Compute relative rotation: current * inverse(previous)
        relative_R = current_R * previous_R.inv()
        
        # Convert to axis-angle representation
        delta_ori = relative_R.as_rotvec()  # This gives us axis-angle format
        delta_ori_magnitude = np.linalg.norm(delta_ori)
        
    except ImportError:
        # Fallback: compute axis-angle directly from quaternions
        # For small rotations, we can approximate
        dot_product = np.dot(current_ori, previous_ori)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle = 2 * np.arccos(abs(dot_product))
        delta_ori_magnitude = angle
        
        if angle > 1e-6:
            # Compute rotation axis (simplified)
            # For small rotations, the axis is approximately the cross product
            axis = np.cross(previous_ori[:3], current_ori[:3])
            if np.linalg.norm(axis) > 1e-6:
                axis = axis / np.linalg.norm(axis)
            else:
                # If no clear axis, use a default
                axis = np.array([0, 0, 1])
            
            # Sign of angle based on dot product
            if dot_product < 0:
                angle = -angle
            
            delta_ori = axis * angle
        else:
            delta_ori = np.zeros(3)
    
    # Combine into 6D delta EEF action: [pos_delta(3D), rot_delta(3D axis-angle)]
    delta_6d = np.concatenate([delta_pos, delta_ori])
    
    return delta_pos, delta_ori, delta_pos_magnitude, delta_ori_magnitude, delta_6d

def get_force_information_fast(env):
    """Get force information from the environment - optimized version that only queries relevant contacts"""
    total_force = 0.0
    tool_force = 0.0
    tool_force_on_human = 0.0
    total_force_on_human = 0.0
    force_vectors = np.zeros(3, dtype=np.float32)
    
    try:
        # Only query specific contact pairs that matter
        # Tool contacts
        tool_contacts = p.getContactPoints(bodyA=env.tool, physicsClientId=env.id)
        for c in tool_contacts:
            total_force += c[9]
            tool_force += c[9]
        
        # Robot-human contacts (only this specific pair)
        robot_human_contacts = p.getContactPoints(bodyA=env.robot, bodyB=env.human, physicsClientId=env.id)
        for c in robot_human_contacts:
            total_force_on_human += c[9]
        
        # Tool-human contacts (only this specific pair)
        tool_human_contacts = p.getContactPoints(bodyA=env.tool, bodyB=env.human, physicsClientId=env.id)
        for c in tool_human_contacts:
            total_force_on_human += c[9]
            tool_force_on_human += c[9]
        
        # Create force vector from the most significant forces
        if tool_force_on_human > 0:
            force_vectors[0] = tool_force_on_human
        if total_force_on_human > 0:
            force_vectors[1] = total_force_on_human
        if tool_force > 0:
            force_vectors[2] = tool_force
            
    except Exception as e:
        # Fallback to basic contact detection
        try:
            contact_points = p.getContactPoints(env.robot, env.human, physicsClientId=env.id)
            if contact_points:
                total_force_on_human = sum([contact[9] for contact in contact_points])
                force_vectors[0] = total_force_on_human
        except:
            pass
    
    return total_force_on_human, force_vectors

def mask_human_arm_points(rgb_img, depth_img, seg_img, env, camera_config, view_matrix, projection_matrix):
    """
    Mask pointcloud to only include right hand, forearm, and upper arm points.
    Uses the correct logic for coordinate transformation and masking.
    
    Args:
        rgb_img: RGB image (H, W, 3)
        depth_img: Depth image (H, W)
        seg_img: Segmentation mask image (H, W)
        env: Environment object
        camera_config: Camera configuration
        view_matrix: View matrix
        projection_matrix: Projection matrix
    
    Returns:
        points: Mx3 array of masked point positions in world coordinates
        depth_values: Mx1 array of masked depth values
        rgb_values: Mx3 array of masked RGB values
    """
    # Get human body ID
    human_body_id = env.human
    
    # Reshape images to 1D arrays
    rgba = np.concatenate([rgb_img, np.ones((rgb_img.shape[0], rgb_img.shape[1], 1)) * 255], axis=2)
    rgba = rgba.reshape((-1, 4))
    depth = depth_img.flatten()
    segmentation_mask = seg_img.flatten()
    
    # Create a 4x4 transform matrix that goes from pixel coordinates to world coordinates
    proj_matrix = np.asarray(projection_matrix).reshape([4, 4], order="F")
    view_matrix_np = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix_np))
    
    # Create a grid with pixel coordinates and depth values
    height, width = depth_img.shape
    y, x = np.mgrid[-1:1:2/height, -1:1:2/width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth
    h = np.ones_like(z)
    pixels = np.stack([x, y, z, h], axis=1)
    
    # Extract the objectUniqueId and linkIndex for each pixel
    # For human body (body == human_body_id), filter for right arm links (5, 7, 9) and additional links (24, 30)
    pc_mask = (
        ((segmentation_mask & ((1 << 24) - 1)) == human_body_id) &
        (
            ((segmentation_mask >> 24) - 1 == 5) |   # Right hand
            ((segmentation_mask >> 24) - 1 == 7) |   # Right forearm
            ((segmentation_mask >> 24) - 1 == 9) |   # Right upper arm
            ((segmentation_mask >> 24) - 1 == 24) |  # Additional link 24
            ((segmentation_mask >> 24) - 1 == 30)    # Additional link 30
        )
    )
    
    # Filter point cloud to only include points on the target body parts
    pixels = pixels[pc_mask]
    z = z[pc_mask]
    rgba = rgba[pc_mask]
    
    if len(pixels) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Filter out "infinite" depths
    valid_depth_mask = z < camera_config['far']
    pixels = pixels[valid_depth_mask]
    z = z[valid_depth_mask]
    rgba = rgba[valid_depth_mask]
    
    if len(pixels) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert depth to [-1, 1] range
    pixels[:, 2] = 2 * pixels[:, 2] - 1
    
    # Transform pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3:4]
    points = points[:, :3]
    
    # Return points, depth values, and RGB values (normalized to [0, 1])
    return points, z, rgba[:, :3] / 255.0

def create_human_arm_mask(env, camera_config, view_matrix, projection_matrix):
    """
    Create a binary mask for human arm points that can be reused throughout the episode.
    This mask is computed once at the beginning of each episode using the first frame's segmentation.
    
    Args:
        env: Environment object
        camera_config: Camera configuration
        view_matrix: View matrix
        projection_matrix: Projection matrix
    
    Returns:
        arm_mask: Binary mask of shape (height, width) where True indicates arm points
    """
    # Get human body ID
    human_body_id = env.human
    
    # Get the first frame's segmentation to create the mask
    rgb_img, depth_img, seg_img, _, _ = get_rgb_depth_images(env, camera_config)
    
    # Create binary mask for arm links (5, 7, 9)
    arm_mask = np.zeros((camera_config['height'], camera_config['width']), dtype=bool)
    
    # Extract the objectUniqueId and linkIndex for each pixel using vectorized operations
    seg_flat = seg_img.flatten()
    object_ids = seg_flat & ((1 << 24) - 1)
    link_indices = (seg_flat >> 24) - 1
    
    # Create mask for human arm links using vectorized operations
    target_links = [5, 7, 9]  # Right hand, forearm, upper arm
    
    # Find pixels that belong to human and are arm links
    human_pixels = (object_ids == human_body_id)
    arm_link_pixels = np.isin(link_indices, target_links)
    arm_pixels = human_pixels & arm_link_pixels
    
    # Reshape back to 2D
    arm_mask = arm_pixels.reshape((camera_config['height'], camera_config['width']))
    
    # Debug: Show which links were found (only for first episode)
    if not hasattr(env, '_debug_links_shown'):
        found_links = np.unique(link_indices[human_pixels])
        print(f"Found human links in segmentation: {sorted(found_links)}")
        print(f"Target arm links: {target_links}")
        
        # Count pixels for each target link
        for link in target_links:
            link_pixels = np.sum((object_ids == human_body_id) & (link_indices == link))
            print(f"Link {link}: {link_pixels} pixels")
        
        # Also show all human links with their pixel counts
        print("\nAll human links found:")
        for link in sorted(found_links):
            link_pixels = np.sum((object_ids == human_body_id) & (link_indices == link))
            print(f"  Link {link}: {link_pixels} pixels")
        
        env._debug_links_shown = True
    
    # If no arm pixels found, create a fallback mask based on typical arm region
    if not np.any(arm_mask):
        print("Warning: No arm pixels found in segmentation. Using fallback mask.")
        center_x, center_y = camera_config['width'] // 2, camera_config['height'] // 2
        
        # Create a region around the center where the arm typically appears
        arm_region_width = camera_config['width'] // 3
        arm_region_height = camera_config['height'] // 2
        
        arm_mask[
            center_y - arm_region_height//2:center_y + arm_region_height//2,
            center_x - arm_region_width//2:center_x + arm_region_width//2
        ] = True
    
    return arm_mask

def precompute_camera_stuff(camera_config):
    """
    Precompute camera-constant values that don't change during the episode.
    This includes projection matrix, pixel grids, and inverse transforms.
    """
    width, height = camera_config['width'], camera_config['height']
    fx = fy = width / (2 * np.tan(np.radians(camera_config['fov']) / 2))
    cx, cy = width / 2, height / 2
    
    # Create pixel grid (only need to do this once)
    y, x = np.mgrid[0:height, 0:width]
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Precompute normalized pixel coordinates
    x_norm = (x_flat - cx) / fx
    y_norm = (y_flat - cy) / fy
    
    # Create homogeneous pixel coordinates (without depth)
    pixels_h = np.stack([x_norm, y_norm, np.ones_like(x_norm), np.ones_like(x_norm)], axis=1)
    
    return {
        'width': width,
        'height': height,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'x_norm': x_norm,
        'y_norm': y_norm,
        'pixels_h': pixels_h,
        'x_flat': x_flat,
        'y_flat': y_flat
    }

def create_pointcloud_from_depth_rgb_with_fixed_mask_fast(depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env, arm_mask, voxel_size=0.025):
    """
    Fast pointcloud creation using precomputed camera cache.
    
    Args:
        depth_img: Depth image (H, W)
        rgb_img: RGB image (H, W, 3)
        seg_img: Segmentation mask image (H, W)
        camera_config: Camera configuration
        view_matrix: View matrix
        projection_matrix: Projection matrix
        env: Environment object
        arm_mask: Pre-computed binary mask for arm points (H, W)
        voxel_size: Size of each voxel cube
    
    Returns:
        pointcloud_data: torch_geometric.Data object with pointcloud
    """
    # Use precomputed camera cache
    if not hasattr(env, 'cam_cache'):
        env.cam_cache = precompute_camera_stuff(camera_config)
    
    cam_cache = env.cam_cache
    height, width = cam_cache['height'], cam_cache['width']
    
    # Apply the pre-computed arm mask to filter points
    masked_depth = depth_img * arm_mask
    masked_rgb = rgb_img * arm_mask[..., np.newaxis]
    
    # Get valid points (non-zero depth within arm mask)
    valid_mask = (masked_depth > 0) & (masked_depth < camera_config['far']) & arm_mask
    
    if not np.any(valid_mask):
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 4), dtype=torch.float32)  # RGB + depth
        )
    
    # Get valid depth values and their indices
    valid_depth_flat = masked_depth.flatten()[valid_mask.flatten()]
    valid_rgb_flat = masked_rgb.reshape(-1, 3)[valid_mask.flatten()]
    
    # Use precomputed normalized coordinates for valid pixels
    valid_pixels_h = cam_cache['pixels_h'][valid_mask.flatten()]
    
    # Scale by depth to get 3D points in camera coordinates
    points_cam = valid_pixels_h[:, :3] * valid_depth_flat[:, np.newaxis]
    
    # Transform to world coordinates
    view_matrix_np = np.array(view_matrix).reshape(4, 4)
    # The view matrix is already the transformation from world to camera
    # So we need its inverse to go from camera to world
    camera_to_world = np.linalg.inv(view_matrix_np)
    
    points_homog = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
    points_world = (camera_to_world @ points_homog.T).T[:, :3]
    
    # Apply spatial cropping (optional)
    z_crop_mask = (points_world[:, 0] >= -0.1) & (points_world[:, 0] <= 0.4)
    points_world_cropped = points_world[z_crop_mask]
    valid_rgb_cropped = valid_rgb_flat[z_crop_mask]
    valid_depth_cropped = valid_depth_flat[z_crop_mask].reshape(-1, 1)
    
    if len(points_world_cropped) == 0:
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 4), dtype=torch.float32)  # RGB + depth
        )
    
    # Voxelize the pointcloud (optimized)
    points_voxelized, features_voxelized = voxelize_pointcloud_fast(
        points_world_cropped, 
        valid_rgb_cropped, 
        valid_depth_cropped, 
        voxel_size
    )
    
    if len(points_voxelized) == 0:
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 4), dtype=torch.float32)  # RGB + depth
        )
    
    pointcloud_data = Data(
        pos=torch.from_numpy(points_voxelized).float(),
        x=torch.from_numpy(features_voxelized).float()
    )
    return pointcloud_data

def voxelize_pointcloud_fast(points, rgb_values, depth_values, voxel_size):
    """
    Fast voxelization of pointcloud using fully vectorized single pass operations.
    
    Args:
        points: Nx3 array of point positions
        rgb_values: Nx3 array of RGB values
        depth_values: Nx1 array of depth values
        voxel_size: Size of each voxel cube
    
    Returns:
        voxelized_points: Mx3 array of voxel centers
        voxelized_features: Mx4 array of averaged features (RGB + depth)
    """
    if len(points) == 0:
        return np.array([]), np.array([])
    
    # Calculate voxel indices for each point
    voxel_idx = (points / voxel_size).astype(np.int32)
    
    # Create structured array for unique operation
    voxel_keys = voxel_idx.view([('', voxel_idx.dtype)] * 3)
    keys, inv = np.unique(voxel_keys, return_inverse=True)
    
    # Get counts for each voxel
    counts = np.bincount(inv)
    
    # Sort indices for reduceat operation
    sort_indices = inv.argsort()
    
    # Calculate voxel centers using reduceat
    voxelized_points = np.add.reduceat(points[sort_indices], np.concatenate([[0], np.cumsum(counts[:-1])])) / counts[:, np.newaxis]
    
    # Calculate voxel features (RGB + depth)
    features = np.concatenate([rgb_values, depth_values], axis=1)
    voxelized_features = np.add.reduceat(features[sort_indices], np.concatenate([[0], np.cumsum(counts[:-1])])) / counts[:, np.newaxis]
    
    # Normalize RGB to [0, 1] range
    voxelized_features[:, :3] = voxelized_features[:, :3].astype(np.float32) / 255.0
    
    return voxelized_points, voxelized_features

def get_agent_pos(env):
    """
    Get agent position in the format expected by DP3.
    For Sawyer robot, this should be the end-effector position and joint positions.
    Returns array of shape (Nd) where Nd is the action dimension.
    """
    # Get end-effector position (3D)
    ee_pos, _ = get_end_effector_pose(env)
    
    # Get joint positions (7-DOF for Sawyer)
    joint_positions = get_robot_state(env)
    
    # Combine end-effector position and joint positions
    # This matches the format: 6D position of end effector + 7D joint position = 13D
    agent_pos = np.concatenate([ee_pos, joint_positions])
    
    return agent_pos

def save_episode_transitions(episode_num, transitions, dataset_path):
    episode_file = dataset_path / f"episode_{episode_num:04d}_transitions.pkl"
    episode_data = {
        'episode_num': episode_num,
        'transitions': transitions,
        'num_transitions': len(transitions),
        'timestamp': datetime.now().isoformat(),
        'task': 'bed_bathing_right_hand',
        'expert_policy': True,
        'robot_type': 'Sawyer',
        'action_dim': 7,
        'state_dim': 7,
        'gripper_dim': 3,
        'force_dim': 3
    }
    with open(episode_file, 'wb') as f:
        pickle.dump(episode_data, f)
    return episode_file

def create_cyan_arm_mask(rgb_img, depth_img, seg_img, camera_config, view_matrix, projection_matrix, cyan_threshold=0.5):
    """
    Create a mask for cyan-colored pixels and nearby colors (likely human arm) using color-based thresholding.
    
    Args:
        rgb_img: RGB image (H, W, 3)
        depth_img: Depth image (H, W)
        seg_img: Segmentation mask image (H, W)
        camera_config: Camera configuration
        view_matrix: View matrix
        projection_matrix: Projection matrix
        cyan_threshold: Threshold for cyan color similarity (0-1), higher = more inclusive
    
    Returns:
        arm_mask: Binary mask of shape (height, width) where True indicates cyan/nearby pixels
        cyan_points: 3D points in world coordinates for cyan pixels
        cyan_rgb: RGB values for cyan pixels
    """
    height, width = rgb_img.shape[:2]
    
    # Convert RGB to normalized [0, 1] range
    rgb_norm = rgb_img.astype(np.float32) / 255.0
    
    # Define multiple cyan-like colors to be more inclusive
    cyan_colors = [
        np.array([0.0, 1.0, 1.0]),    # Pure cyan
        np.array([0.0, 0.8, 1.0]),    # Light cyan
        np.array([0.0, 1.0, 0.8]),    # Cyan-green
        np.array([0.2, 0.8, 1.0]),    # Sky blue
        np.array([0.0, 0.7, 0.8]),    # Teal
        np.array([0.1, 0.9, 0.9]),    # Light teal
        np.array([0.0, 0.6, 0.7]),    # Dark teal
        np.array([0.1, 0.8, 0.8]),    # Blue-green
    ]
    
    # Calculate color distance from all cyan-like colors
    color_masks = []
    for cyan_color in cyan_colors:
        color_diff = np.linalg.norm(rgb_norm - cyan_color, axis=2)
        color_masks.append(color_diff < cyan_threshold)
    
    # Combine all cyan-like color masks
    cyan_mask = np.any(color_masks, axis=0)
    
    # Additional inclusive color criteria for nearby colors
    # High green and blue values (cyan characteristics)
    green_blue_mask = (rgb_norm[:, :, 1] > 0.5) & (rgb_norm[:, :, 2] > 0.5) & (rgb_norm[:, :, 0] < 0.5)
    
    # Blue-green range (teal, aqua, etc.)
    blue_green_mask = (rgb_norm[:, :, 1] > 0.4) & (rgb_norm[:, :, 2] > 0.6) & (rgb_norm[:, :, 0] < 0.3)
    
    # Light blue range
    light_blue_mask = (rgb_norm[:, :, 1] > 0.6) & (rgb_norm[:, :, 2] > 0.7) & (rgb_norm[:, :, 0] < 0.4)
    
    # Combine all masks
    final_cyan_mask = cyan_mask | green_blue_mask | blue_green_mask | light_blue_mask
    
    # Apply depth filtering to remove background
    valid_depth_mask = (depth_img > 0) & (depth_img < camera_config['far'])
    final_cyan_mask = final_cyan_mask & valid_depth_mask
    
    # Get cyan pixels
    cyan_pixels = final_cyan_mask.flatten()
    cyan_indices = np.where(cyan_pixels)[0]
    
    if len(cyan_indices) == 0:
        return final_cyan_mask, np.array([]), np.array([])
    
    # Get camera intrinsics
    fx = fy = width / (2 * np.tan(np.radians(camera_config['fov']) / 2))
    cx, cy = width / 2, height / 2
    
    # Convert flat indices to 2D coordinates
    y_coords = cyan_indices // width
    x_coords = cyan_indices % width
    
    # Get depth values for cyan pixels
    z_coords = depth_img[y_coords, x_coords]
    
    # Convert to 3D camera coordinates
    x_cam = (x_coords - cx) * z_coords / fx
    y_cam = (y_coords - cy) * z_coords / fy
    
    points_cam = np.stack([x_cam, y_cam, z_coords], axis=-1)
    
    # Transform to world coordinates
    view_matrix_np = np.array(view_matrix).reshape(4, 4)
    # The view matrix is already the transformation from world to camera
    # So we need its inverse to go from camera to world
    camera_to_world = np.linalg.inv(view_matrix_np)
    
    points_homog = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
    points_world = (camera_to_world @ points_homog.T).T[:, :3]
    
    # Get RGB values for cyan pixels
    cyan_rgb = rgb_img[y_coords, x_coords]
    
    return final_cyan_mask, points_world, cyan_rgb

def create_cyan_points_raw(depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env, downsample_factor=10):
    """
    Create raw cyan pointcloud with 5 features: [R,G,B,cyan_onehot,gripper_onehot].
    Args:
        depth_img: Depth image (H, W)
        rgb_img: RGB image (H, W, 3)
        seg_img: Segmentation mask image (H, W)
        camera_config: Camera configuration
        view_matrix: View matrix
        projection_matrix: Projection matrix
        env: Environment object
        downsample_factor: Factor to downsample points (1 = no downsampling, 10 = keep 1/10 points)
    Returns:
        torch_geometric.Data with 5 features per point
    """
    cyan_mask, cyan_points, cyan_rgb = create_cyan_arm_mask(rgb_img, depth_img, seg_img, camera_config, view_matrix, projection_matrix)
    if len(cyan_points) == 0:
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 5), dtype=torch.float32)
        )
    z_crop_mask = (cyan_points[:, 0] >= -0.1) & (cyan_points[:, 0] <= 0.4)
    cyan_points_cropped = cyan_points[z_crop_mask]
    cyan_rgb_cropped = cyan_rgb[z_crop_mask]
    if len(cyan_points_cropped) == 0:
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 5), dtype=torch.float32)
        )
    if downsample_factor > 1 and len(cyan_points_cropped) > downsample_factor:
        skip = downsample_factor
        cyan_points_cropped = cyan_points_cropped[::skip]
        cyan_rgb_cropped = cyan_rgb_cropped[::skip]
    cyan_rgb_normalized = cyan_rgb_cropped.astype(np.float32) / 255.0
    if cyan_rgb_normalized.shape[1] > 3:
        cyan_rgb_normalized = cyan_rgb_normalized[:, :3]
    # 5 features: [R,G,B,cyan_onehot,gripper_onehot]
    features = np.zeros((len(cyan_points_cropped), 5), dtype=np.float32)
    features[:, :3] = cyan_rgb_normalized
    features[:, 3] = 1.0  # cyan_onehot
    features[:, 4] = 0.0  # gripper_onehot
    return Data(
        pos=torch.from_numpy(cyan_points_cropped).float(),
        x=torch.from_numpy(features).float()
    )

def create_arm_points_from_segmentation(depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env, voxel_size=0.00025, downsample_factor=10):
    """
    Create arm pointcloud with 5 features: [R,G,B,cyan_onehot,gripper_onehot].
    Optimized version with reduced redundant calculations.
    """
    # Use cached camera intrinsics if available
    if hasattr(env, 'cam_cache'):
        cam_cache = env.cam_cache
        width, height = cam_cache['width'], cam_cache['height']
        fx, fy = cam_cache['fx'], cam_cache['fy']
        cx, cy = cam_cache['cx'], cam_cache['cy']
    else:
        width, height = camera_config['width'], camera_config['height']
        fx = fy = width / (2 * np.tan(np.radians(camera_config['fov']) / 2))
        cx, cy = width / 2, height / 2
    
    # Get human body ID
    human_body_id = env.human
    
    # Create binary mask for arm links (5, 7, 9) - vectorized
    seg_flat = seg_img.flatten()
    object_ids = seg_flat & ((1 << 24) - 1)
    link_indices = (seg_flat >> 24) - 1
    
    # Target arm links
    target_links = [5, 7, 9]
    
    # Find pixels that belong to human and are arm links
    human_pixels = (object_ids == human_body_id)
    arm_link_pixels = np.isin(link_indices, target_links)
    arm_pixels = human_pixels & arm_link_pixels
    
    # Reshape back to 2D
    arm_mask = arm_pixels.reshape((height, width))
    
    # Apply depth filtering
    valid_depth_mask = (depth_img > 0) & (depth_img < camera_config['far'])
    arm_mask = arm_mask & valid_depth_mask
    
    if not np.any(arm_mask):
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 5), dtype=torch.float32)
        )
    
    # Get arm pixels
    arm_indices = np.where(arm_mask.flatten())[0]
    
    # Downsample indices for performance
    if downsample_factor > 1 and len(arm_indices) > downsample_factor:
        skip = downsample_factor
        arm_indices = arm_indices[::skip]
    
    # Convert flat indices to 2D coordinates
    y_coords = arm_indices // width
    x_coords = arm_indices % width
    
    # Get depth values for arm pixels
    z_coords = depth_img[y_coords, x_coords]
    
    # Convert to 3D camera coordinates
    x_cam = (x_coords - cx) * z_coords / fx
    y_cam = (y_coords - cy) * z_coords / fy
    
    points_cam = np.stack([x_cam, y_cam, z_coords], axis=-1)
    
    # Transform to world coordinates
    view_matrix_np = np.array(view_matrix).reshape(4, 4)
    camera_to_world = np.linalg.inv(view_matrix_np)
    
    points_homog = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
    points_world = (camera_to_world @ points_homog.T).T[:, :3]
    
    # Get RGB values for arm pixels
    arm_rgb = rgb_img[y_coords, x_coords]
    
    # Apply spatial cropping
    z_crop_mask = (points_world[:, 0] >= -0.1) & (points_world[:, 0] <= 0.4)
    points_world_cropped = points_world[z_crop_mask]
    arm_rgb_cropped = arm_rgb[z_crop_mask]
    
    if len(points_world_cropped) == 0:
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 5), dtype=torch.float32)
        )
    
    # Create depth values
    arm_depth = points_world_cropped[:, 2].reshape(-1, 1)
    
    # Voxelize the arm pointcloud
    points_voxelized, features_voxelized = voxelize_pointcloud_fast(
        points_world_cropped, 
        arm_rgb_cropped, 
        arm_depth, 
        voxel_size
    )
    
    if len(points_voxelized) == 0:
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 5), dtype=torch.float32)
        )
    
    # 5 features: [R,G,B,cyan_onehot,gripper_onehot]
    features = np.zeros((len(points_voxelized), 5), dtype=np.float32)
    features[:, :3] = features_voxelized[:, :3]  # RGB
    features[:, 3] = 0.0  # cyan_onehot
    features[:, 4] = 0.0  # gripper_onehot
    
    return Data(
        pos=torch.from_numpy(points_voxelized).float(),
        x=torch.from_numpy(features).float()
    )

def create_combined_pointcloud_with_features(cyan_pcd, arm_pcd, tool_tip_pcd, gripper_point, target_total_points=150):
    """
    Create a combined pointcloud with 5 features: RGB + cyan one-hot + gripper one-hot.
    Downsample both cyan and arm points, keep only 1 EEF point, target total pointcloud size.
    
    Args:
        cyan_pcd: Cyan pointcloud (torch_geometric.Data)
        arm_pcd: Arm pointcloud (torch_geometric.Data)
        tool_tip_pcd: Tool tip pointcloud (torch_geometric.Data) - can be None, will be replaced by single gripper point
        gripper_point: EEF position (numpy array) - single point to use
        target_total_points: Target total number of points in combined pointcloud
    
    Returns:
        combined_pcd: Combined pointcloud with 5 features (torch_geometric.Data)
    """
    # Get points and features from cyan and arm pointclouds
    cyan_pos = cyan_pcd.pos
    cyan_x = cyan_pcd.x
    arm_pos = arm_pcd.pos
    arm_x = arm_pcd.x
    
    # Count points we want to keep at full resolution
    num_eef_points = 1  # Only 1 EEF point
    
    # Calculate how many points we can allocate to cyan and arm points
    available_slots = target_total_points - num_eef_points
    
    if available_slots <= 0:
        available_slots = max(2, target_total_points - 1)  # Fallback: use all but 1 for cyan+arm
    
    # Allocate slots between cyan and arm points (roughly 50/50 split)
    cyan_slots = available_slots // 2
    arm_slots = available_slots - cyan_slots  # Use remaining slots for arm
    
    # Downsample cyan points if needed
    if len(cyan_pos) > cyan_slots:
        # Use random sampling for downsampling
        indices = np.random.choice(len(cyan_pos), cyan_slots, replace=False)
        cyan_pos_downsampled = cyan_pos[indices]
        cyan_x_downsampled = cyan_x[indices]
    else:
        cyan_pos_downsampled = cyan_pos
        cyan_x_downsampled = cyan_x
    
    # Downsample arm points if needed
    if len(arm_pos) > arm_slots:
        # Use random sampling for downsampling
        indices = np.random.choice(len(arm_pos), arm_slots, replace=False)
        arm_pos_downsampled = arm_pos[indices]
        arm_x_downsampled = arm_x[indices]
    else:
        arm_pos_downsampled = arm_pos
        arm_x_downsampled = arm_x
    
    # Create feature vectors with 5 features: RGB + cyan one-hot + gripper one-hot
    cyan_features = torch.zeros(len(cyan_pos_downsampled), 5)
    arm_features = torch.zeros(len(arm_pos_downsampled), 5)
    
    # RGB features (first 3 features)
    if cyan_x_downsampled.shape[1] >= 3:
        cyan_features[:, :3] = cyan_x_downsampled[:, :3]  # RGB
    if arm_x_downsampled.shape[1] >= 3:
        arm_features[:, :3] = arm_x_downsampled[:, :3]    # RGB
    
    # Cyan one-hot encoding (4th feature)
    cyan_features[:, 3] = 1.0  # Cyan points get 1.0 for cyan feature
    arm_features[:, 3] = 0.0   # Arm points get 0.0 for cyan feature
    
    # Gripper one-hot encoding (5th feature)
    cyan_features[:, 4] = 0.0  # Cyan points get 0.0 for gripper feature
    arm_features[:, 4] = 0.0   # Arm points get 0.0 for gripper feature
    
    # Create single EEF point with features (ignore tool_tip_pcd parameter)
    eef_pos = torch.from_numpy(gripper_point).float().unsqueeze(0)  # Shape: (1, 3)
    eef_features = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)  # RGB=0, cyan=0, gripper=1
    
    # Combine all points and features
    combined_pos = torch.cat([cyan_pos_downsampled, arm_pos_downsampled, eef_pos], dim=0)
    combined_features = torch.cat([cyan_features, arm_features, eef_features], dim=0)
    
    combined_pcd = Data(
        pos=combined_pos,
        x=combined_features
    )
    
    return combined_pcd

def extract_tool_tip_from_segmentation(depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env):
    """
    Extract tool tip position from segmentation mask.
    Optimized version with reduced redundant calculations.
    """
    # Use cached camera intrinsics if available
    if hasattr(env, 'cam_cache'):
        cam_cache = env.cam_cache
        width, height = cam_cache['width'], cam_cache['height']
        fx, fy = cam_cache['fx'], cam_cache['fy']
        cx, cy = cam_cache['cx'], cam_cache['cy']
    else:
        width, height = camera_config['width'], camera_config['height']
        fx = fy = width / (2 * np.tan(np.radians(camera_config['fov']) / 2))
        cx, cy = width / 2, height / 2
    
    # Get tool body ID
    tool_body_id = env.tool
    
    # Create binary mask for tool - vectorized
    seg_flat = seg_img.flatten()
    object_ids = seg_flat & ((1 << 24) - 1)
    
    # Find pixels that belong to tool
    tool_pixels = (object_ids == tool_body_id)
    
    # Reshape back to 2D
    tool_mask = tool_pixels.reshape((height, width))
    
    # Apply depth filtering
    valid_depth_mask = (depth_img > 0) & (depth_img < camera_config['far'])
    tool_mask = tool_mask & valid_depth_mask
    
    if not np.any(tool_mask):
        return None
    
    # Get tool pixels
    tool_indices = np.where(tool_mask.flatten())[0]
    
    # Convert flat indices to 2D coordinates
    y_coords = tool_indices // width
    x_coords = tool_indices % width
    
    # Get depth values for tool pixels
    z_coords = depth_img[y_coords, x_coords]
    
    # Convert to 3D camera coordinates
    x_cam = (x_coords - cx) * z_coords / fx
    y_cam = (y_coords - cy) * z_coords / fy
    
    points_cam = np.stack([x_cam, y_cam, z_coords], axis=-1)
    
    # Transform to world coordinates
    view_matrix_np = np.array(view_matrix).reshape(4, 4)
    camera_to_world = np.linalg.inv(view_matrix_np)
    
    points_homog = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
    points_world = (camera_to_world @ points_homog.T).T[:, :3]
    
    # Find the tool tip position (farthest point from camera)
    tip_idx = np.argmax(z_coords)
    tool_tip_pos = points_world[tip_idx]
    
    return tool_tip_pos

def create_tool_tip_pointcloud(depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env):
    """
    Create tool tip pointcloud with 5 features: [R,G,B,cyan_onehot,gripper_onehot].
    Args:
        depth_img: Depth image (H, W)
        rgb_img: RGB image (H, W, 3)
        seg_img: Segmentation mask image (H, W)
        camera_config: Camera configuration
        view_matrix: View matrix
        projection_matrix: Projection matrix
        env: Environment object
    Returns:
        torch_geometric.Data with 5 features per point
    """
    # Get tool body ID
    tool_body_id = env.tool
    
    # Create binary mask for tool
    tool_mask = np.zeros((camera_config['height'], camera_config['width']), dtype=bool)
    
    # Extract the objectUniqueId and linkIndex for each pixel using vectorized operations
    seg_flat = seg_img.flatten()
    object_ids = seg_flat & ((1 << 24) - 1)
    link_indices = (seg_flat >> 24) - 1
    
    # Find pixels that belong to tool
    tool_pixels = (object_ids == tool_body_id)
    
    # Reshape back to 2D
    tool_mask = tool_pixels.reshape((camera_config['height'], camera_config['width']))
    
    # Apply depth filtering
    valid_depth_mask = (depth_img > 0) & (depth_img < camera_config['far'])
    tool_mask = tool_mask & valid_depth_mask
    
    if not np.any(tool_mask):
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 5), dtype=torch.float32)
        )
    
    # Get camera intrinsics
    height, width = camera_config['height'], camera_config['width']
    fx = fy = width / (2 * np.tan(np.radians(camera_config['fov']) / 2))
    cx, cy = width / 2, height / 2
    
    # Get tool pixels
    tool_pixels_flat = tool_mask.flatten()
    tool_indices = np.where(tool_pixels_flat)[0]
    
    # Convert flat indices to 2D coordinates
    y_coords = tool_indices // width
    x_coords = tool_indices % width
    
    # Get depth values for tool pixels
    z_coords = depth_img[y_coords, x_coords]
    
    # Convert to 3D camera coordinates
    x_cam = (x_coords - cx) * z_coords / fx
    y_cam = (y_coords - cy) * z_coords / fy
    
    points_cam = np.stack([x_cam, y_cam, z_coords], axis=-1)
    
    # Transform to world coordinates
    view_matrix_np = np.array(view_matrix).reshape(4, 4)
    # The view matrix is already the transformation from world to camera
    # So we need its inverse to go from camera to world
    camera_to_world = np.linalg.inv(view_matrix_np)
    
    points_homog = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
    points_world = (camera_to_world @ points_homog.T).T[:, :3]
    
    # Get RGB values for tool pixels
    tool_rgb = rgb_img[y_coords, x_coords]
    
    if len(points_world) == 0:
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 5), dtype=torch.float32)
        )
    tool_rgb_normalized = tool_rgb.astype(np.float32) / 255.0
    if tool_rgb_normalized.shape[1] > 3:
        tool_rgb_normalized = tool_rgb_normalized[:, :3]
    features = np.zeros((len(points_world), 5), dtype=np.float32)
    features[:, :3] = tool_rgb_normalized
    features[:, 3] = 0.0  # cyan_onehot
    features[:, 4] = 1.0  # gripper_onehot
    return Data(
        pos=torch.from_numpy(points_world).float(),
        x=torch.from_numpy(features).float()
    )

def collect_episode(vec_env, episode_num, dataset_path, episode_length=100, noise_std=0.05, visualize=False, debug_images=False, downsample_factor=10, target_total_points=150):
    """Collect a single episode of data using the expert policy with vectorized environment and optional action noise"""
    env = vec_env.venv.envs[0].env
    action_space = env.action_space
    
    obs = vec_env.reset()
    transitions = []
    
    # For real-time visualization
    if visualize:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title(f"End-Effector Trajectory (Episode {episode_num})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        traj_x, traj_y = [], []
        traj_plot, = ax.plot([], [], 'b.-', label='Trajectory')
        ax.legend()
    
    # For debug image display
    if debug_images:
        cv2.namedWindow('Camera Debug View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Debug View', 320, 240)
    
    # Initialize recurrent hidden states and masks (same as in enjoy.py)
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    
    # Cache camera setup - compute once and reuse
    camera_config = setup_camera_aimed_at_right_hand(env)
    
    # Cache frequently accessed attributes to avoid lookups in tight loops
    human_id = env.human
    robot_id = env.robot
    tool_id = env.tool
    physics_id = env.id
    
    # Pre-compute camera intrinsics for efficiency
    width, height = camera_config['width'], camera_config['height']
    fx = fy = width / (2 * np.tan(np.radians(camera_config['fov']) / 2))
    cx, cy = width / 2, height / 2
    
    # Cache view and projection matrices - compute once
    view_matrix = p.computeViewMatrix(
        camera_config['position'],
        camera_config['target'],
        camera_config['up'],
        physicsClientId=env.id
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        camera_config['fov'],
        camera_config['width'] / camera_config['height'],
        camera_config['near'],
        camera_config['far'],
        physicsClientId=env.id
    )
    
    # Track previous EEF pose for delta calculations
    previous_eef_pose = None
    episode_delta_stats = {
        'total_pos_delta': 0.0,
        'total_ori_delta': 0.0,
        'max_pos_delta': 0.0,
        'max_ori_delta': 0.0,
        'num_frames': 0
    }
    
    for frame in range(episode_length):
        current_state = get_robot_state(env)
        agent_pos = get_agent_pos(env)
        ee_pos, ee_orn = get_end_effector_pose(env)
        
        # Only collect force information every few frames to improve performance
        if frame % 10 == 0:
            total_force, force_vectors = get_force_information_fast(env)
        else:
            total_force, force_vectors = 0.0, np.zeros(3, dtype=np.float32)
        
        if visualize:
            traj_x.append(ee_pos[0])
            traj_y.append(ee_pos[1])
            traj_plot.set_data(traj_x, traj_y)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)
        
        # Get fresh images using cached camera setup
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=camera_config['width'],
            height=camera_config['height'],
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=env.id
        )
        rgb_img = np.array(rgb_img).reshape(height, width, 4)[:, :, :3].astype(np.uint8)
        depth_img = np.array(depth_img).reshape(height, width)
        depth_img = camera_config['far'] * camera_config['near'] / (
            camera_config['far'] - (camera_config['far'] - camera_config['near']) * depth_img
        )
        seg_img = np.array(seg_img).reshape(height, width)
        
        # Debug: Show RGB image in real-time
        if debug_images:
            rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            info_text = f"Frame: {frame}"
            cv2.putText(rgb_img_bgr, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            hand_pos = get_right_hand_pos(env)
            hand_text = f"Hand: ({hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f})"
            cv2.putText(rgb_img_bgr, hand_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            ee_text = f"EE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
            cv2.putText(rgb_img_bgr, ee_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow('Camera Debug View', rgb_img_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Create pointclouds using cached camera setup
        cyan_pointcloud = create_cyan_points_raw(depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env, downsample_factor=downsample_factor)
        arm_pointcloud = create_arm_points_from_segmentation(depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env, downsample_factor=downsample_factor)
        
        # Extract tool tip position from segmentation (single point only)
        extracted_eef_pos = extract_tool_tip_from_segmentation(depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env)
        
        # Use extracted EEF position if available, otherwise fall back to PyBullet tool base
        if extracted_eef_pos is not None:
            ee_pos = extracted_eef_pos
        else:
            ee_pos, ee_orn = get_end_effector_pose(env)
        
        # Calculate EEF delta movement
        current_eef_pose = (ee_pos, ee_orn)
        delta_pos, delta_ori, delta_pos_mag, delta_ori_mag, delta_6d = calculate_eef_delta_movement(current_eef_pose, previous_eef_pose)
        
        # Update episode delta statistics
        episode_delta_stats['total_pos_delta'] += delta_pos_mag
        episode_delta_stats['total_ori_delta'] += delta_ori_mag
        episode_delta_stats['max_pos_delta'] = max(episode_delta_stats['max_pos_delta'], delta_pos_mag)
        episode_delta_stats['max_ori_delta'] = max(episode_delta_stats['max_ori_delta'], delta_ori_mag)
        episode_delta_stats['num_frames'] += 1
        
        # Create combined pointcloud (no need for separate tool tip pointcloud)
        combined_pointcloud = create_combined_pointcloud_with_features(cyan_pointcloud, arm_pointcloud, None, ee_pos, target_total_points=target_total_points)
        
        # Print point counts and delta movements every 50 frames for monitoring
        if frame % 50 == 0:
            print(f"Frame {frame}: Cyan={len(cyan_pointcloud.pos)}, Arm={len(arm_pointcloud.pos)}, Combined={len(combined_pointcloud.pos)}, EEF=1")
            print(f"  Delta: pos={delta_pos_mag:.4f}m, ori={delta_ori_mag:.4f}rad")
        
        # Store full observation
        obs_np = obs.squeeze(0).cpu().numpy()
        
        # Get action from expert policy
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=True)
        
        # Convert action to numpy for environment step
        action_np = action.squeeze(0).cpu().numpy()
        
        # Inject Gaussian noise for suboptimality
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=action_np.shape)
            action_np = action_np + noise
            action_np = np.clip(action_np, action_space.low, action_space.high)
        
        # Store transition with current frame's data (no separate tool tip pointcloud)
        transition = {
            'obs': obs_np,
            'pcd_cyan': cyan_pointcloud,
            'pcd_arm': arm_pointcloud,
            'pcd_combined': combined_pointcloud,
            'gripper_point': ee_pos,
            'gripper_orientation': ee_orn,
            'delta_pos': delta_pos,
            'delta_ori': delta_ori,
            'delta_pos_magnitude': delta_pos_mag,
            'delta_ori_magnitude': delta_ori_mag,
            'delta_6d': delta_6d,  # 6D delta EEF action [pos_delta(3D), rot_delta(3D axis-angle)]
            'action': action_np,
            'reward': 0.0,
            'not_done': True,
            'total_force': total_force,
            'force_vectors': force_vectors,
            'state': current_state,
            'frame': frame,
            'success': False
        }
        transitions.append(transition)
        
        # Update previous pose for next frame
        previous_eef_pose = current_eef_pose
        
        # Step environment with noisy action
        action_tensor = torch.from_numpy(action_np).unsqueeze(0)
        obs, reward, done, infos = vec_env.step(action_tensor)
        
        # Store reward
        reward_value = reward[0] if isinstance(reward, list) else reward
        if hasattr(reward_value, 'cpu'):
            reward_value = reward_value.cpu().numpy()
        if isinstance(infos, list) and len(infos) > 0:
            info = infos[0]
            if 'success' in info and info['success']:
                transition['success'] = True
        
        # Update masks
        masks.fill_(0.0 if done else 1.0)
        
        # Update reward in transition
        transition['reward'] = reward_value
        
        if done:
            break
    
    if visualize:
        plt.ioff()
        plt.show()
    
    if debug_images:
        cv2.destroyAllWindows()
    
    # Log episode delta movement summary
    if episode_delta_stats['num_frames'] > 0:
        avg_pos_delta = episode_delta_stats['total_pos_delta'] / episode_delta_stats['num_frames']
        avg_ori_delta = episode_delta_stats['total_ori_delta'] / episode_delta_stats['num_frames']
        print(f"\nEpisode {episode_num} Delta Movement Summary:")
        print(f"  Total position delta: {episode_delta_stats['total_pos_delta']:.4f}m")
        print(f"  Total orientation delta: {episode_delta_stats['total_ori_delta']:.4f}rad")
        print(f"  Average position delta per frame: {avg_pos_delta:.4f}m")
        print(f"  Average orientation delta per frame: {avg_ori_delta:.4f}rad")
        print(f"  Max position delta in single frame: {episode_delta_stats['max_pos_delta']:.4f}m")
        print(f"  Max orientation delta in single frame: {episode_delta_stats['max_ori_delta']:.4f}rad")
    
    episode_file = save_episode_transitions(episode_num, transitions, dataset_path)
    return len(transitions)

def main():
    parser = argparse.ArgumentParser(description='DP3 Expert Data Collector with Transition Pickle Format - Optimized Version')
    parser.add_argument('--noise-std', type=float, default=0.05, help='Stddev of Gaussian noise for suboptimality (default: 0.05)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to collect')
    parser.add_argument('--episode-length', type=int, default=100, help='Length of each episode')
    parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel environments (default: 1, use 8+ for speed)')
    parser.add_argument('--visualize', action='store_true', help='Show real-time end-effector trajectory plot')
    parser.add_argument('--debug-images', action='store_true', help='Show real-time RGB camera images for debugging')
    parser.add_argument('--high-res', action='store_true', help='Use high resolution (640x480) instead of optimized (320x240)')
    parser.add_argument('--downsample-factor', type=int, default=16, help='Factor to downsample pointclouds for performance (1=no downsampling, 10=keep 1/10 points, default: 10)')
    parser.add_argument('--target-total-points', type=int, default=150, help='Target total number of points in combined pointcloud (default: 150)')
    args = parser.parse_args()

    # Override camera resolution if high-res is requested
    if args.high_res:
        # Temporarily modify the camera setup function
        original_setup = setup_camera_aimed_at_right_hand
        def high_res_setup(env, offset=np.array([0.0, 0.4, 0.9])):
            hand_pos = get_right_hand_pos(env)
            camera_pos = hand_pos + offset
            camera_target = hand_pos + np.asarray([0.1, 0.3, 0.0])
            
            camera_config = {
                'position': camera_pos.tolist(),
                'target': camera_target.tolist(),
                'up': [0, 1, 0],
                'fov': 45.0,
                'near': 0.05,
                'far': 3.0,
                'width': 640,
                'height': 480
            }
            return camera_config
        globals()['setup_camera_aimed_at_right_hand'] = high_res_setup

    dataset_path = Path("dp3_transitions_dataset")
    dataset_path.mkdir(exist_ok=True)
    dataset_metadata = {
        'name': 'DP3_BedBathing_Transitions_Expert_Optimized',
        'description': 'Transition dataset for 3D diffusion policy training on bed bathing task using expert policy - optimized version with 5-feature combined pointcloud, single EEF point, and downsampled cyan+arm points',
        'created': datetime.now().isoformat(),
        'num_episodes': 0,
        'episode_length': args.episode_length,
        'num_parallel_envs': args.num_envs,
        'format': 'pickle_transitions',
        'pointcloud_format': 'torch_geometric_Data',
        'pointcloud_types': {
            'pcd_cyan': 'Raw cyan-colored points with RGB features only (color-based detection) - downsampled',
            'pcd_arm': 'Voxelized arm points from segmentation (links 5,7,9) with RGB features only - downsampled',
            'pcd_combined': 'Combined pointcloud with 5 features: RGB + cyan one-hot + gripper one-hot + single EEF point'
        },
        'combined_pointcloud_features': {
            'feature_0': 'Red channel (0-1)',
            'feature_1': 'Green channel (0-1)', 
            'feature_2': 'Blue channel (0-1)',
            'feature_3': 'Cyan one-hot encoding (1.0 for cyan points, 0.0 for others)',
            'feature_4': 'Gripper one-hot encoding (1.0 for tool tip points and gripper point, 0.0 for others)'
        },
        'gripper_point_location': 'Single EEF point in combined pointcloud with features [0,0,0,0,1]',
        'action_space': '7-DOF joint space',
        'state_space': '7-DOF joint positions',
        'task': 'bed_bathing_right_hand',
        'expert_policy': True,
        'noise_std': args.noise_std,
        'camera_resolution': '320x240' if not args.high_res else '640x480',
        'downsample_factor': args.downsample_factor,
        'target_total_points': args.target_total_points,
        'optimizations': [
            'camera_cache',
            'vectorized_voxelization', 
            'reduced_force_frequency',
            'parallel_environments',
            'attribute_caching',
            'five_feature_combined_pointcloud',
            'pointcloud_downsampling',
            'single_eef_point',
            'cyan_arm_downsampling'
        ],
        'transition_keys': [
            'obs', 'pcd_cyan', 'pcd_arm', 'pcd_combined', 'gripper_point', 'gripper_orientation', 'delta_pos', 'delta_ori', 'delta_pos_magnitude', 'delta_ori_magnitude', 'delta_6d', 'action', 'reward', 'not_done',
            'total_force', 'force_vectors', 'state', 'frame', 'success'
        ]
    }
    
    print("Creating BedBathingSawyer-v0 vectorized environment...")
    env_name = 'BedBathingSawyer-v0'
    seed = 1
    
    # Create parallel environments for better performance
    vec_env = make_vec_envs(env_name, seed + 1000, args.num_envs, None, None, False, device='cuda', allow_early_resets=False)
    
    # Disable GUI rendering to avoid X connection issues
    render_func = None  # Disable GUI rendering
    vec_norm = get_vec_normalize(vec_env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms
    print(f"Environment created successfully with {args.num_envs} parallel environments!")
    
    total_frames = 0
    try:
        if args.num_envs == 1:
            # Single environment collection
            for episode in range(args.episodes):
                frames = collect_episode(vec_env, episode, dataset_path, episode_length=args.episode_length, noise_std=args.noise_std, visualize=args.visualize, debug_images=args.debug_images, downsample_factor=args.downsample_factor, target_total_points=args.target_total_points)
                total_frames += frames
                dataset_metadata['num_episodes'] += 1
                with open(dataset_path / "dataset_metadata.json", 'w') as f:
                    json.dump(dataset_metadata, f, indent=2)
        else:
            # Parallel environment collection
            episodes_per_env = args.episodes // args.num_envs
            remaining_episodes = args.episodes % args.num_envs
            
            for env_idx in range(args.num_envs):
                env_episodes = episodes_per_env + (1 if env_idx < remaining_episodes else 0)
                if env_episodes > 0:
                    # For parallel environments, we need to handle them differently
                    # This is a simplified version - in practice you'd want to collect from all envs simultaneously
                    for episode in range(env_episodes):
                        episode_num = env_idx * episodes_per_env + episode
                        frames = collect_episode(vec_env, episode_num, dataset_path, episode_length=args.episode_length, noise_std=args.noise_std, visualize=False, debug_images=False, downsample_factor=args.downsample_factor, target_total_points=args.target_total_points)
                        total_frames += frames
                        dataset_metadata['num_episodes'] += 1
                        with open(dataset_path / "dataset_metadata.json", 'w') as f:
                            json.dump(dataset_metadata, f, indent=2)
                        
    except KeyboardInterrupt:
        pass
    finally:
        vec_env.close()
        print(f"\nDataset collection completed!")
        print(f"Total episodes: {dataset_metadata['num_episodes']}")
        print(f"Total frames: {total_frames}")
        print(f"Dataset saved to: {dataset_path.absolute()}")
        print(f"Performance optimizations used:")
        print(f"  - Camera resolution: {dataset_metadata['camera_resolution']}")
        print(f"  - Parallel environments: {args.num_envs}")
        print(f"  - Downsample factor: {args.downsample_factor}")
        print(f"  - Target total points: {args.target_total_points}")
        print(f"  - Optimizations: {', '.join(dataset_metadata['optimizations'])}")

if __name__ == "__main__":
    main() 