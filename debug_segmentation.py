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
actor_critic, ob_rms = torch.load(EXPERT_POLICY_PATH, map_location='cpu')

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
        'width': 640,
        'height': 480
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
    link_state = p.getLinkState(env.robot, 8, computeForwardKinematics=True, physicsClientId=env.id)
    position = np.array(link_state[0])
    orientation = np.array(link_state[1])
    return position, orientation

def get_force_information(env):
    """Get force information from the environment using proper assistive gym patterns"""
    total_force = 0.0
    tool_force = 0.0
    tool_force_on_human = 0.0
    total_force_on_human = 0.0
    force_vectors = np.zeros(3, dtype=np.float32)
    
    try:
        # Get all contact points for the tool
        for c in p.getContactPoints(bodyA=env.tool, physicsClientId=env.id):
            total_force += c[9]
            tool_force += c[9]
        
        # Get all contact points for the robot (excluding tool)
        for c in p.getContactPoints(bodyA=env.robot, physicsClientId=env.id):
            bodyB = c[2]
            if bodyB != env.tool:
                total_force += c[9]
        
        # Get contact points between robot and human
        for c in p.getContactPoints(bodyA=env.robot, bodyB=env.human, physicsClientId=env.id):
            total_force_on_human += c[9]
        
        # Get contact points between tool and human
        for c in p.getContactPoints(bodyA=env.tool, bodyB=env.human, physicsClientId=env.id):
            linkA = c[3]
            total_force_on_human += c[9]
            if linkA in [1]:  # Tool link that can contact human
                tool_force_on_human += c[9]
        
        # Create force vector from the most significant forces
        if tool_force_on_human > 0:
            force_vectors[0] = tool_force_on_human
        if total_force_on_human > 0:
            force_vectors[1] = total_force_on_human
        if tool_force > 0:
            force_vectors[2] = tool_force
            
        # Debug output for first few frames
        if hasattr(env, '_debug_frame_count'):
            env._debug_frame_count += 1
        else:
            env._debug_frame_count = 0
            
        if env._debug_frame_count < 5:
            print(f"Frame {env._debug_frame_count}: Tool force: {tool_force:.3f}, Tool on human: {tool_force_on_human:.3f}, Total on human: {total_force_on_human:.3f}")
            
    except Exception as e:
        print(f"Error collecting forces: {e}")
        # Fallback to basic contact detection
        try:
            contact_points = p.getContactPoints(env.robot, env.human, physicsClientId=env.id)
            if contact_points:
                total_force_on_human = sum([contact[9] for contact in contact_points])
                force_vectors[0] = total_force_on_human
        except:
            pass
    
    return total_force_on_human, force_vectors

def mask_human_arm_points(rgb_img, depth_img, seg_img, right_arm_mask, env, camera_config, view_matrix, projection_matrix):
    """
    Mask pointcloud to only include right hand, forearm, and upper arm points.
    Uses a pre-computed mask for efficiency.
    
    Args:
        rgb_img: RGB image (H, W, 3)
        depth_img: Depth image (H, W)
        seg_img: Segmentation mask image (H, W)
        right_arm_mask: Pre-computed boolean mask for right arm pixels
        env: Environment object
        camera_config: Camera configuration
        view_matrix: View matrix
        projection_matrix: Projection matrix
    
    Returns:
        points: Mx3 array of masked point positions in world coordinates
        depth_values: Mx1 array of masked depth values
        rgb_values: Mx3 array of masked RGB values
    """
    # Debug: Check input shapes and mask
    print(f"  Input shapes: RGB={rgb_img.shape}, Depth={depth_img.shape}, Seg={seg_img.shape}")
    print(f"  Right arm mask shape: {right_arm_mask.shape}, Sum: {np.sum(right_arm_mask)}")
    
    # Reshape images to 1D arrays
    rgba = np.concatenate([rgb_img, np.ones((rgb_img.shape[0], rgb_img.shape[1], 1)) * 255], axis=2)
    rgba = rgba.reshape((-1, 4))
    depth = depth_img.flatten()
    
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
    
    # Use the pre-computed right arm mask
    pc_mask = right_arm_mask
    
    print(f"  Total pixels: {len(pixels)}, Masked pixels: {np.sum(pc_mask)}")
    
    # Filter point cloud to only include points on the target body parts
    pixels = pixels[pc_mask]
    z = z[pc_mask]
    rgba = rgba[pc_mask]
    
    print(f"  After masking: {len(pixels)} pixels")
    
    if len(pixels) == 0:
        print("  No pixels after masking!")
        return np.array([]), np.array([]), np.array([])
    
    # Filter out "infinite" depths
    valid_depth_mask = z < camera_config['far']
    pixels = pixels[valid_depth_mask]
    z = z[valid_depth_mask]
    rgba = rgba[valid_depth_mask]
    
    print(f"  After depth filtering: {len(pixels)} pixels")
    
    if len(pixels) == 0:
        print("  No pixels after depth filtering!")
        return np.array([]), np.array([]), np.array([])
    
    # Convert depth to [-1, 1] range
    pixels[:, 2] = 2 * pixels[:, 2] - 1
    
    # Transform pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3:4]
    points = points[:, :3]
    
    print(f"  After world transform: {len(points)} points")
    if len(points) > 0:
        print(f"  Point range: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # Safety check: Clip points whose world-space Z (height) is outside [-0.1 m, 0.4 m]
    height_mask = (points[:, 2] >= -0.1) & (points[:, 2] <= 0.4)
    points = points[height_mask]
    z = z[height_mask]
    rgba = rgba[height_mask]
    
    print(f"  After height filtering: {len(points)} points")
    
    # Return points, depth values, and RGB values (normalized to [0, 1])
    return points, z, rgba[:, :3] / 255.0

def create_pointcloud_from_depth_rgb(depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env, right_arm_mask=None, voxel_size=0.02):
    height, width = depth_img.shape
    fx = fy = width / (2 * np.tan(np.radians(camera_config['fov']) / 2))
    cx, cy = width / 2, height / 2
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    z = depth_img
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy
    points_cam = np.stack([x_cam, y_cam, z], axis=-1)
    valid_mask = (depth_img > 0) & (depth_img < camera_config['far'])
    valid_points_cam = points_cam[valid_mask]
    valid_rgb = rgb_img[valid_mask]
    if len(valid_points_cam) == 0:
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 4), dtype=torch.float32)  # RGB + depth
        )
    view_matrix_np = np.array(view_matrix).reshape(4, 4)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = view_matrix_np[:3, :3].T
    camera_pose[:3, 3] = -view_matrix_np[:3, :3].T @ view_matrix_np[:3, 3]
    points_homog = np.concatenate([valid_points_cam, np.ones((len(valid_points_cam), 1))], axis=1)
    points_world = (camera_pose @ points_homog.T).T[:, :3]
    z_crop_mask = (points_world[:, 0] >= -0.1) & (points_world[:, 0] <= 0.4)
    points_world_cropped = points_world[z_crop_mask]
    valid_rgb_cropped = valid_rgb[z_crop_mask]
    valid_points_cam_cropped = valid_points_cam[z_crop_mask]
    if len(points_world_cropped) == 0:
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 4), dtype=torch.float32)  # RGB + depth
        )
    
    # Apply human arm masking if mask is provided
    if right_arm_mask is not None:
        points_arm, depth_arm, rgb_arm = mask_human_arm_points(
            rgb_img, 
            depth_img, 
            seg_img, 
            right_arm_mask,  # Use pre-computed mask
            env, 
            camera_config, 
            view_matrix, 
            projection_matrix
        )
        
        if len(points_arm) == 0:
            return Data(
                pos=torch.zeros((1, 3), dtype=torch.float32),
                x=torch.zeros((1, 4), dtype=torch.float32)  # RGB + depth
            )
        
        # Voxelize the masked pointcloud
        points_voxelized, features_voxelized = voxelize_pointcloud(
            points_arm, 
            rgb_arm, 
            depth_arm.reshape(-1, 1), 
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
    else:
        # Fallback to using cropped pointcloud without arm masking
        if len(points_world_cropped) == 0:
            return Data(
                pos=torch.zeros((1, 3), dtype=torch.float32),
                x=torch.zeros((1, 4), dtype=torch.float32)  # RGB + depth
            )
        
        # Voxelize the cropped pointcloud
        points_voxelized, features_voxelized = voxelize_pointcloud(
            points_world_cropped, 
            valid_rgb_cropped, 
            valid_points_cam_cropped[:, 2:3], 
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

def voxelize_pointcloud(points, rgb_values, depth_values, voxel_size):
    """
    Voxelize pointcloud by grouping points within the same voxel and averaging their features.
    
    Args:
        points: Nx3 array of point positions
        rgb_values: Nx3 array of RGB values
        depth_values: Nx1 array of depth values
        voxel_size: Size of each voxel cube
    
    Returns:
        voxelized_points: Mx3 array of voxel centers
        voxelized_features: Mx4 array of averaged features (RGB + depth)
    """
    print(f"  Voxelization input: {len(points)} points, voxel_size={voxel_size}")
    
    if len(points) == 0:
        print("  No points to voxelize!")
        return np.array([]), np.array([])
    
    # Calculate voxel indices for each point
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Create a dictionary to group points by voxel
    voxel_dict = {}
    
    for i, voxel_idx in enumerate(voxel_indices):
        voxel_key = tuple(voxel_idx)
        
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = {
                'points': [],
                'rgb_values': [],
                'depth_values': []
            }
        
        voxel_dict[voxel_key]['points'].append(points[i])
        voxel_dict[voxel_key]['rgb_values'].append(rgb_values[i])
        voxel_dict[voxel_key]['depth_values'].append(depth_values[i])
    
    print(f"  Created {len(voxel_dict)} voxels")
    
    # Calculate voxel centers and averaged features
    voxelized_points = []
    voxelized_features = []
    
    for voxel_key, voxel_data in voxel_dict.items():
        # Calculate voxel center (average of all points in the voxel)
        voxel_center = np.mean(voxel_data['points'], axis=0)
        
        # Average RGB values
        avg_rgb = np.mean(voxel_data['rgb_values'], axis=0)
        
        # Average depth value
        avg_depth = np.mean(voxel_data['depth_values'], axis=0)
        
        # Normalize RGB to [0, 1] range
        avg_rgb_normalized = avg_rgb.astype(np.float32) / 255.0
        
        # Combine features: RGB + depth
        features = np.concatenate([avg_rgb_normalized, avg_depth])
        
        voxelized_points.append(voxel_center)
        voxelized_features.append(features)
    
    print(f"  Voxelization output: {len(voxelized_points)} voxels")
    
    return np.array(voxelized_points), np.array(voxelized_features)

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
    print(f"Saved episode {episode_num} with {len(transitions)} transitions to {episode_file}")
    return episode_file

def collect_episode(vec_env, episode_num, dataset_path, episode_length=100, noise_std=0.05, visualize=False, debug_images=False):
    """Collect a single episode of data using the expert policy with vectorized environment and optional action noise"""
    print(f"Starting episode {episode_num}... (noise_std={noise_std})")
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
        cv2.resizeWindow('Camera Debug View', 640, 480)
    
    # Initialize recurrent hidden states and masks (same as in enjoy.py)
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    
    # Compute right arm mask once at the start of episode (human body doesn't move)
    print("Computing right arm mask for this episode...")
    camera_config = setup_camera_aimed_at_right_hand(env)
    rgb_img, depth_img, seg_img, view_matrix, projection_matrix = get_rgb_depth_images(env, camera_config)
    
    # Create the right arm mask once
    human_body_id = env.human
    seg_flat = seg_img.flatten()
    
    # Debug: Print all human joint info to identify correct link indices
    print(f"Episode {episode_num}: Human body ID: {human_body_id}")
    print("Human joint information:")
    for i in range(p.getNumJoints(env.human, physicsClientId=env.id)):
        joint_info = p.getJointInfo(env.human, i, physicsClientId=env.id)
        joint_name = joint_info[12].decode() if isinstance(joint_info[12], bytes) else str(joint_info[12])
        print(f"  Link {i}: {joint_name}")
    
    # Create mask for right arm parts using correct link indices
    # Based on debug output, the right arm parts are at link indices 5, 7, 9
    # which correspond to link13, link15, link17 (right hand, forearm, upper arm)
    obj_mask = (seg_flat & ((1 << 24) - 1)) == human_body_id  # lower 24 bits = object ID
    link_idx = (seg_flat >> 24) - 1  # upper 8 bits = link index + 1, so subtract 1
    arm_mask = (link_idx == 5) | (link_idx == 7) | (link_idx == 9)  # right hand, forearm, upper arm
    right_arm_mask = obj_mask & arm_mask
    
    # Debug: Show mask statistics
    total_pixels = len(seg_flat)
    arm_pixels = np.sum(right_arm_mask)
    print(f"Episode {episode_num}: Right arm mask computed - {arm_pixels} pixels out of {total_pixels} ({100*arm_pixels/total_pixels:.1f}%)")
    
    # Show breakdown by arm part
    link_counts = {5: 0, 7: 0, 9: 0}
    for seg_val in seg_flat:
        object_id = seg_val & ((1 << 24) - 1)
        link_index = (seg_val >> 24) - 1
        if object_id == human_body_id and link_index in link_counts:
            link_counts[link_index] += 1
    print(f"Episode {episode_num}: Right arm breakdown - Hand(5): {link_counts[5]}, Forearm(7): {link_counts[7]}, Upper arm(9): {link_counts[9]}")
    
    # If no arm pixels found, this is unexpected since we know they exist
    if arm_pixels == 0:
        print("WARNING: No arm pixels found! This should not happen with correct link indices.")
        print("Check if camera position is correct or if human body is visible.")
    
    for frame in range(episode_length):
        current_state = get_robot_state(env)
        agent_pos = get_agent_pos(env)
        ee_pos, ee_orn = get_end_effector_pose(env)
        total_force, force_vectors = get_force_information(env)
        
        if visualize:
            traj_x.append(ee_pos[0])
            traj_y.append(ee_pos[1])
            traj_plot.set_data(traj_x, traj_y)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)
        
        # Update camera configuration for current frame to capture current state
        camera_config = setup_camera_aimed_at_right_hand(env)
        
        # Get fresh images and camera pose for current frame
        rgb_img, depth_img, seg_img, view_matrix, projection_matrix = get_rgb_depth_images(env, camera_config)
        
        # Debug: Show RGB image in real-time
        if debug_images:
            # Convert RGB to BGR for OpenCV display
            rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            
            # Add frame info to image
            info_text = f"Frame: {frame}, Points: {len(create_pointcloud_from_depth_rgb(depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env, right_arm_mask).pos)}"
            cv2.putText(rgb_img_bgr, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add hand position info
            hand_pos = get_right_hand_pos(env)
            hand_text = f"Hand: ({hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f})"
            cv2.putText(rgb_img_bgr, hand_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add EE position info
            ee_text = f"EE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
            cv2.putText(rgb_img_bgr, ee_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('Camera Debug View', rgb_img_bgr)
            
            # Wait for key press (1ms) - press 'q' to quit, any other key to continue
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Debug mode stopped by user")
                break
        
        # Create pointcloud with RGB (DP3 format) for current frame
        pointcloud = create_pointcloud_from_depth_rgb(depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env, right_arm_mask)
        
        # Debug: Show voxelization effect for first few frames
        if frame < 5:
            voxelized_points = len(pointcloud.pos)
            print(f"Frame {frame}: Voxelized pointcloud has {voxelized_points} points (voxel_size=0.0625)")
            
        # Store full observation
        obs_np = obs.squeeze(0).cpu().numpy()
        
        # Create camera pose matrix (4x4) from view matrix
        view_matrix_np = np.array(view_matrix).reshape(4, 4)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = view_matrix_np[:3, :3].T
        camera_pose[:3, 3] = -view_matrix_np[:3, :3].T @ view_matrix_np[:3, 3]
        
        # Get action from expert policy (same as in enjoy.py)
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=True)
        
        # Convert action to numpy for environment step
        action_np = action.squeeze(0).cpu().numpy()
        
        # Inject Gaussian noise for suboptimality
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=action_np.shape)
            action_np = action_np + noise
            # Clip to action space
            action_np = np.clip(action_np, action_space.low, action_space.high)
        
        # Store transition with current frame's data
        transition = {
            'obs': obs_np,
            'pcd': pointcloud,  # This is now unique for each frame
            'gripper_point': ee_pos,
            'action': action_np,
            'reward': 0.0,  # Reward will be updated after step
            'not_done': True,
            'total_force': total_force,
            'force_vectors': force_vectors,
            'state': current_state,
            'camera_pose': camera_pose,  # Current camera pose
            'frame': frame,
            'success': False
        }
        transitions.append(transition)
        
        # Step environment with noisy action
        action_tensor = torch.from_numpy(action_np).unsqueeze(0)  # Add batch dimension
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
        
        if frame % 10 == 0:
            print(f"  Frame {frame}/{episode_length} - Point cloud points: {len(pointcloud.pos)}")
            if total_force > 0:
                print(f"    Force detected: {total_force:.3f}, Force vectors: {force_vectors}")
        if done:
            print(f"  Episode ended early at frame {frame}")
            break
    
    if visualize:
        plt.ioff()
        plt.show()
    
    if debug_images:
        cv2.destroyAllWindows()
    
    episode_file = save_episode_transitions(episode_num, transitions, dataset_path)
    print(f"Episode {episode_num} completed: {len(transitions)} frames")
    return len(transitions)

def main():
    parser = argparse.ArgumentParser(description='DP3 Expert Data Collector with Transition Pickle Format')
    parser.add_argument('--noise-std', type=float, default=0.05, help='Stddev of Gaussian noise for suboptimality (default: 0.05)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to collect')
    parser.add_argument('--episode-length', type=int, default=100, help='Length of each episode')
    parser.add_argument('--visualize', action='store_true', help='Show real-time end-effector trajectory plot')
    parser.add_argument('--debug-images', action='store_true', help='Show real-time RGB camera images for debugging')
    args = parser.parse_args()

    dataset_path = Path("dp3_transitions_dataset")
    dataset_path.mkdir(exist_ok=True)
    dataset_metadata = {
        'name': 'DP3_BedBathing_Transitions_Expert',
        'description': 'Transition dataset for 3D diffusion policy training on bed bathing task using expert policy',
        'created': datetime.now().isoformat(),
        'num_episodes': 0,
        'episode_length': args.episode_length,
        'format': 'pickle_transitions',
        'pointcloud_format': 'torch_geometric_Data',
        'action_space': '7-DOF joint space',
        'state_space': '7-DOF joint positions',
        'task': 'bed_bathing_right_hand',
        'expert_policy': True,
        'noise_std': args.noise_std,
        'transition_keys': [
            'obs', 'pcd', 'gripper_point', 'action', 'reward', 'not_done',
            'total_force', 'force_vectors', 'state', 'camera_pose', 'frame', 'success'
        ]
    }
    
    print("Creating BedBathingSawyer-v0 vectorized environment...")
    env_name = 'BedBathingSawyer-v0'
    seed = 1
    vec_env = make_vec_envs(env_name, seed + 1000, 1, None, None, False, device='cpu', allow_early_resets=False)
    
    # Disable GUI rendering to avoid X connection issues
    # render_func = get_render_func(vec_env)
    # if render_func is not None:
    #     render_func('human')
    render_func = None  # Disable GUI rendering
    # render_func = get_render_func(vec_env)
    # render_func('human')
    vec_norm = get_vec_normalize(vec_env)
    vec_norm = get_vec_normalize(vec_env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms
    print("Environment created successfully!")
    total_frames = 0
    try:
        for episode in range(args.episodes):
            frames = collect_episode(vec_env, episode, dataset_path, episode_length=args.episode_length, noise_std=args.noise_std, visualize=args.visualize, debug_images=args.debug_images)
            total_frames += frames
            dataset_metadata['num_episodes'] += 1
            with open(dataset_path / "dataset_metadata.json", 'w') as f:
                json.dump(dataset_metadata, f, indent=2)
            print(f"Completed {episode + 1}/{args.episodes} episodes")
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    finally:
        vec_env.close()
        print(f"\nDataset collection completed!")
        print(f"Total episodes: {dataset_metadata['num_episodes']}")
        print(f"Total frames: {total_frames}")
        print(f"Dataset saved to: {dataset_path.absolute()}")

if __name__ == "__main__":
    main() 