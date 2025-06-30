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

def setup_camera_aimed_at_right_hand(env, offset=np.array([0.06, 0.4, 0.65])):
    """
    Place the camera above and slightly towards the torso to capture the whole arm length.
    This gives a view that includes the entire arm from shoulder to hand in the center of the frame.
    Offset is relative to the hand position (in meters).
    """
    hand_pos = get_right_hand_pos(env)
    camera_pos = hand_pos + offset  # Position camera above and towards torso
    camera_target = hand_pos + np.array([0.06, 0.3, -0.2])  # Look at middle of arm length
    
    # Use the same approach as dressing environment
    view_matrix = p.computeViewMatrix(
        camera_pos.tolist(),
        camera_target.tolist(),
        [0, 1, 0],  # Up vector (Y-axis up for proper orientation)
        physicsClientId=env.id
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        60.0,  # FOV
        320.0 / 240.0,  # Aspect ratio
        0.01,  # Near plane
        100.0,  # Far plane
        physicsClientId=env.id
    )
    
    camera_config = {
        'position': camera_pos.tolist(),
        'target': camera_target.tolist(),
        'up': [0, 1, 0],
        'fov': 60.0,
        'near': 0.01,
        'far': 100.0,
        'width': 320,
        'height': 240,
        'view_matrix': view_matrix,
        'projection_matrix': projection_matrix
    }
    return camera_config

def get_rgb_depth_images(env, camera_config):
    """Get RGB and depth images from camera using stored view/projection matrices"""
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        width=camera_config['width'],
        height=camera_config['height'],
        viewMatrix=camera_config['view_matrix'],
        projectionMatrix=camera_config['projection_matrix'],
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        physicsClientId=env.id
    )
    rgb_img = np.array(rgb_img).reshape(height, width, 4)[:, :, :3].astype(np.uint8)
    depth_img = np.array(depth_img).reshape(height, width)
    seg_img = np.array(seg_img).reshape(height, width)
    return rgb_img, depth_img, seg_img, camera_config['view_matrix'], camera_config['projection_matrix']

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

def create_cyan_mask(rgb_img, cyan_threshold=0.5):
    """
    Create a mask for cyan-colored pixels using color thresholding.
    Based on the approach from old_collection.py.
    
    Args:
        rgb_img: RGB image (H, W, 3)
        cyan_threshold: Threshold for cyan color similarity (0-1)
    
    Returns:
        cyan_mask: Binary mask of shape (height, width)
    """
    # Convert RGB to normalized [0, 1] range
    rgb_norm = rgb_img.astype(np.float32) / 255.0
    
    # Define multiple cyan-like colors to be more inclusive (from old_collection.py)
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
    
    return final_cyan_mask

def create_arm_mask(seg_img, env):
    """
    Create a mask for human arm points using segmentation.
    Based on the approach from old_collection.py.
    
    Args:
        seg_img: Segmentation mask image (H, W)
        env: Environment object
    
    Returns:
        arm_mask: Binary mask of shape (height, width)
    """
    # Get human body ID
    human_body_id = env.human
    
    # Extract object IDs and link indices using vectorized operations (from old_collection.py)
    seg_flat = seg_img.flatten()
    object_ids = seg_flat & ((1 << 24) - 1)
    link_indices = (seg_flat >> 24) - 1
    
    # Target arm links (right hand, forearm, upper arm)
    target_links = [5, 7, 9]
    
    # Find pixels that belong to human and are arm links
    human_pixels = (object_ids == human_body_id)
    arm_link_pixels = np.isin(link_indices, target_links)
    arm_pixels = human_pixels & arm_link_pixels
    
    # Reshape back to 2D
    arm_mask = arm_pixels.reshape(seg_img.shape)
    
    return arm_mask

def unproject_depth_to_world_simple(depth_img, rgb_img, mask, camera_config, view_matrix, projection_matrix, env):
    """
    Convert depth pixels to world coordinates using the proven approach from dressing environment.
    This is a simplified version that follows the exact same logic as the dressing environment.
    
    Args:
        depth_img: Depth image (H, W)
        rgb_img: RGB image (H, W, 3)
        mask: Binary mask (H, W) indicating which pixels to process
        camera_config: Camera configuration
        view_matrix: View matrix
        projection_matrix: Projection matrix
        env: Environment object
    
    Returns:
        points_world: Nx3 array of world coordinates
        rgb_values: Nx3 array of RGB values
    """
    height, width = depth_img.shape
    
    # Get masked pixel coordinates
    y_coords, x_coords = np.where(mask)
    
    if len(y_coords) == 0:
        return np.array([]), np.array([])
    
    # Get depth values for masked pixels
    z_coords = depth_img[y_coords, x_coords]
    
    # Filter out invalid depths
    valid_mask = (z_coords > 0) & (z_coords < camera_config['far'])
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    
    y_coords = y_coords[valid_mask]
    x_coords = x_coords[valid_mask]
    z_coords = z_coords[valid_mask]
    
    # Convert to normalized device coordinates (NDC) - same as dressing environment
    x_ndc = (2.0 * x_coords / width) - 1.0
    y_ndc = (2.0 * y_coords / height) - 1.0
    y_ndc *= -1.0  # Flip Y axis as in dressing environment
    z_ndc = 2.0 * z_coords - 1.0  # Convert depth to NDC range
    
    # Create homogeneous coordinates
    pixels = np.stack([x_ndc, y_ndc, z_ndc, np.ones_like(x_ndc)], axis=1)
    
    # Create transformation matrix - exactly as in dressing environment
    proj_matrix = np.asarray(projection_matrix).reshape([4, 4], order="F")
    view_matrix_np = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix_np))
    
    # Transform points from NDC to world coordinates - exactly as in dressing environment
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3:4]  # Divide by homogeneous coordinate
    points_world = points[:, :3]  # Extract 3D coordinates
    
    # Get RGB values for the same pixels
    rgb_values = rgb_img[y_coords, x_coords]
    
    return points_world, rgb_values

def create_simple_pointcloud(depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env, 
                           mask_type='cyan'):
    """
    Create a simple pointcloud using the proven dressing environment approach.
    No initial downsampling - we get all points and downsample later.
    Filter out points below z<0.6 to remove floor/background points.
    
    Args:
        depth_img: Depth image (H, W)
        rgb_img: RGB image (H, W, 3)
        seg_img: Segmentation mask image (H, W)
        camera_config: Camera configuration
        view_matrix: View matrix
        projection_matrix: Projection matrix
        env: Environment object
        mask_type: 'cyan' or 'arm'
    
    Returns:
        torch_geometric.Data with 5 features per point
    """
    # Create mask based on type
    if mask_type == 'cyan':
        mask = create_cyan_mask(rgb_img)
    elif mask_type == 'arm':
        mask = create_arm_mask(seg_img, env)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")
    
    # Apply depth filtering
    valid_depth_mask = (depth_img > 0) & (depth_img < camera_config['far'])
    mask = mask & valid_depth_mask
    
    if not np.any(mask):
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 5), dtype=torch.float32)
        )
    
    # Convert to world coordinates using proven dressing environment method
    points_world, rgb_values = unproject_depth_to_world_simple(
        depth_img, rgb_img, mask, camera_config, view_matrix, projection_matrix, env
    )
    
    if len(points_world) == 0:
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 5), dtype=torch.float32)
        )
    
    # Filter out points below z<0.6 (remove floor/background points)
    z_filter_mask = points_world[:, 2] >= 0.6
    points_world = points_world[z_filter_mask]
    rgb_values = rgb_values[z_filter_mask]
    
    if len(points_world) == 0:
        return Data(
            pos=torch.zeros((1, 3), dtype=torch.float32),
            x=torch.zeros((1, 5), dtype=torch.float32)
        )
    
    # Normalize RGB values
    rgb_normalized = rgb_values.astype(np.float32) / 255.0
    
    # Create features: [R, G, B, cyan_onehot, gripper_onehot]
    features = np.zeros((len(points_world), 5), dtype=np.float32)
    features[:, :3] = rgb_normalized
    features[:, 3] = 1.0 if mask_type == 'cyan' else 0.0  # cyan_onehot
    features[:, 4] = 0.0  # gripper_onehot
    
    return Data(
        pos=torch.from_numpy(points_world).float(),
        x=torch.from_numpy(features).float()
    )

def create_combined_pointcloud(cyan_pcd, arm_pcd, gripper_point, target_total_points=150):
    """
    Create a combined pointcloud with adaptive downsampling to maintain target_total_points.
    We downsample after getting the combined cloud to adaptively keep 150 points total.
    
    Args:
        cyan_pcd: Cyan pointcloud (torch_geometric.Data)
        arm_pcd: Arm pointcloud (torch_geometric.Data)
        gripper_point: EEF position (numpy array)
        target_total_points: Target total number of points (default: 150)
    
    Returns:
        combined_pcd: Combined pointcloud with 5 features (torch_geometric.Data)
    """
    # Get points and features
    cyan_pos = cyan_pcd.pos
    cyan_x = cyan_pcd.x
    arm_pos = arm_pcd.pos
    arm_x = arm_pcd.x
    
    # Reserve 1 slot for gripper point
    available_slots = target_total_points - 1
    
    if available_slots <= 0:
        available_slots = 2  # Fallback
    
    # Combine all points first (no initial downsampling)
    all_pos = torch.cat([cyan_pos, arm_pos], dim=0)
    all_x = torch.cat([cyan_x, arm_x], dim=0)
    
    # Create gripper point
    gripper_pos = torch.from_numpy(gripper_point).float().unsqueeze(0)
    gripper_x = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    
    # If we have more points than available slots, downsample adaptively
    if len(all_pos) > available_slots:
        # Random sampling to get exactly available_slots points
        indices = np.random.choice(len(all_pos), available_slots, replace=False)
        all_pos = all_pos[indices]
        all_x = all_x[indices]
    
    # Combine all points (downsampled + gripper)
    combined_pos = torch.cat([all_pos, gripper_pos], dim=0)
    combined_x = torch.cat([all_x, gripper_x], dim=0)
    
    return Data(pos=combined_pos, x=combined_x)

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

def collect_episode(vec_env, episode_num, dataset_path, episode_length=100, noise_std=0.05, visualize=False, debug_images=False, target_total_points=150):
    """Collect a single episode of data using the expert policy with simplified pointcloud creation"""
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
    
    # Initialize recurrent hidden states and masks
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    
    # Setup camera
    camera_config = setup_camera_aimed_at_right_hand(env)
    
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
        
        # Get fresh images
        rgb_img, depth_img, seg_img, view_matrix, projection_matrix = get_rgb_depth_images(env, camera_config)
        
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
        
        # Create pointclouds using simplified method (no initial downsampling)
        cyan_pointcloud = create_simple_pointcloud(
            depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env,
            mask_type='cyan'
        )
        arm_pointcloud = create_simple_pointcloud(
            depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env,
            mask_type='arm'
        )
        
        # Calculate EEF delta movement
        current_eef_pose = (ee_pos, ee_orn)
        delta_pos, delta_ori, delta_pos_mag, delta_ori_mag, delta_6d = calculate_eef_delta_movement(current_eef_pose, previous_eef_pose)
        
        # Debug: Log detailed delta information for first few frames
        if frame < 5:
           
            if previous_eef_pose is not None:
                prev_pos, prev_orn = previous_eef_pose
               
            else:
                print(f"  First frame - no previous pose")
        
        # Update episode delta statistics
        episode_delta_stats['total_pos_delta'] += delta_pos_mag
        episode_delta_stats['total_ori_delta'] += delta_ori_mag
        episode_delta_stats['max_pos_delta'] = max(episode_delta_stats['max_pos_delta'], delta_pos_mag)
        episode_delta_stats['max_ori_delta'] = max(episode_delta_stats['max_ori_delta'], delta_ori_mag)
        episode_delta_stats['num_frames'] += 1
        
        # Create combined pointcloud with adaptive downsampling
        combined_pointcloud = create_combined_pointcloud(cyan_pointcloud, arm_pointcloud, ee_pos, target_total_points=target_total_points)
        
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
        
        # Store transition
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
            'success': False,
            'eef_position': ee_pos,  # Store current EEF position for validation
            'eef_orientation': ee_orn,  # Store current EEF orientation for validation
            'previous_eef_position': previous_eef_pose[0] if previous_eef_pose is not None else None,  # Store previous for validation
            'previous_eef_orientation': previous_eef_pose[1] if previous_eef_pose is not None else None  # Store previous for validation
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
        print(f"  Total frames: {episode_delta_stats['num_frames']}")
        print(f"  Total position delta: {episode_delta_stats['total_pos_delta']:.6f}m")
        print(f"  Total orientation delta: {episode_delta_stats['total_ori_delta']:.6f}rad")
        print(f"  Average position delta per frame: {avg_pos_delta:.6f}m")
        print(f"  Average orientation delta per frame: {avg_ori_delta:.6f}rad")
        print(f"  Max position delta in single frame: {episode_delta_stats['max_pos_delta']:.6f}m")
        print(f"  Max orientation delta in single frame: {episode_delta_stats['max_ori_delta']:.6f}rad")
        
        # Validate delta calculations
        if episode_delta_stats['max_pos_delta'] > 1.0:  # More than 1 meter in one frame is suspicious
            print(f"  WARNING: Large position delta detected ({episode_delta_stats['max_pos_delta']:.6f}m)")
        if episode_delta_stats['max_ori_delta'] > 3.14:  # More than pi radians is suspicious
            print(f"  WARNING: Large orientation delta detected ({episode_delta_stats['max_ori_delta']:.6f}rad)")
        
        # Log pointcloud statistics
        if len(transitions) > 0:
            last_transition = transitions[-1]
            print(f"  Final pointcloud sizes:")
            print(f"    Cyan: {len(last_transition['pcd_cyan'].pos)} points")
            print(f"    Arm: {len(last_transition['pcd_arm'].pos)} points")
            print(f"    Combined: {len(last_transition['pcd_combined'].pos)} points")
            print(f"    Target total points: {target_total_points}")
            
            # Check if z-filtering worked
            if len(last_transition['pcd_cyan'].pos) > 1:
                cyan_z_min = torch.min(last_transition['pcd_cyan'].pos[:, 2]).item()
                print(f"    Cyan points z-range: min={cyan_z_min:.3f} (should be >= 0.6)")
            if len(last_transition['pcd_arm'].pos) > 1:
                arm_z_min = torch.min(last_transition['pcd_arm'].pos[:, 2]).item()
                print(f"    Arm points z-range: min={arm_z_min:.3f} (should be >= 0.6)")
    
    episode_file = save_episode_transitions(episode_num, transitions, dataset_path)
    return len(transitions)

def main():
    parser = argparse.ArgumentParser(description='DP3 Expert Data Collector - Simplified Version')
    parser.add_argument('--noise-std', type=float, default=0.05, help='Stddev of Gaussian noise for suboptimality (default: 0.05)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to collect')
    parser.add_argument('--episode-length', type=int, default=100, help='Length of each episode')
    parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel environments (default: 1)')
    parser.add_argument('--visualize', action='store_true', help='Show real-time end-effector trajectory plot')
    parser.add_argument('--debug-images', action='store_true', help='Show real-time RGB camera images for debugging')
    parser.add_argument('--high-res', action='store_true', help='Use high resolution (640x480) instead of optimized (320x240)')
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
        'name': 'DP3_BedBathing_Transitions_Expert_Adaptive_ZFiltered',
        'description': 'Transition dataset for 3D diffusion policy training on bed bathing task using expert policy - adaptive downsampling with z-filtering and improved delta logging',
        'created': datetime.now().isoformat(),
        'num_episodes': 0,
        'episode_length': args.episode_length,
        'num_parallel_envs': args.num_envs,
        'format': 'pickle_transitions',
        'pointcloud_format': 'torch_geometric_Data',
        'pointcloud_types': {
            'pcd_cyan': 'Cyan-colored points using color thresholding - z-filtered (z>=0.6), no initial downsampling',
            'pcd_arm': 'Arm points from segmentation (links 5,7,9) - z-filtered (z>=0.6), no initial downsampling',
            'pcd_combined': 'Combined pointcloud with 5 features: RGB + cyan one-hot + gripper one-hot, adaptively downsampled to target_total_points'
        },
        'combined_pointcloud_features': {
            'feature_0': 'Red channel (0-1)',
            'feature_1': 'Green channel (0-1)', 
            'feature_2': 'Blue channel (0-1)',
            'feature_3': 'Cyan one-hot encoding (1.0 for cyan points, 0.0 for others)',
            'feature_4': 'Gripper one-hot encoding (1.0 for gripper point, 0.0 for others)'
        },
        'z_filtering': {
            'enabled': True,
            'threshold': 0.6,
            'description': 'Remove all points below z=0.6 to eliminate floor and background points'
        },
        'downsampling_strategy': 'adaptive_combined',
        'downsampling_description': 'No initial downsampling of individual pointclouds. All points are combined first, then adaptively downsampled to maintain target_total_points including 1 gripper point.',
        'delta_eef_tracking': {
            'enabled': True,
            'validation': True,
            'debug_logging': True,
            'description': 'Track and validate EEF delta movements with detailed logging and validation checks'
        },
        'coordinate_transformation': 'Proven dressing environment approach: inv(proj * view) * NDC',
        'action_space': '7-DOF joint space',
        'state_space': '7-DOF joint positions',
        'task': 'bed_bathing_right_hand',
        'expert_policy': True,
        'noise_std': args.noise_std,
        'camera_resolution': '320x240' if not args.high_res else '640x480',
        'target_total_points': args.target_total_points,
        'simplifications': [
            'proven_dressing_environment_approach',
            'unified_coordinate_transformation',
            'adaptive_combined_downsampling',
            'no_initial_downsampling',
            'z_coordinate_filtering',
            'consistent_world_coordinates',
            'standard_matrix_transformation',
            'one_hot_feature_encoding',
            'delta_eef_validation'
        ],
        'transition_keys': [
            'obs', 'pcd_cyan', 'pcd_arm', 'pcd_combined', 'gripper_point', 'gripper_orientation', 
            'delta_pos', 'delta_ori', 'delta_pos_magnitude', 'delta_ori_magnitude', 'delta_6d', 
            'action', 'reward', 'not_done', 'total_force', 'force_vectors', 'state', 'frame', 'success',
            'eef_position', 'eef_orientation', 'previous_eef_position', 'previous_eef_orientation'
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
                frames = collect_episode(vec_env, episode, dataset_path, episode_length=args.episode_length, noise_std=args.noise_std, visualize=args.visualize, debug_images=args.debug_images, target_total_points=args.target_total_points)
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
                        frames = collect_episode(vec_env, episode_num, dataset_path, episode_length=args.episode_length, noise_std=args.noise_std, visualize=False, debug_images=False, target_total_points=args.target_total_points)
                        total_frames += frames
                        dataset_metadata['num_episodes'] += 1
                        with open(dataset_path / "dataset_metadata.json", 'w') as f:
                            json.dump(dataset_metadata, f, indent=2)
                        
    except KeyboardInterrupt:
        pass
    finally:
        vec_env.close()


if __name__ == "__main__":
    main() 