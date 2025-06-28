import gym
import assistive_gym
import pybullet as p
import numpy as np
import cv2
import json
import os
import time
import sys
from datetime import datetime
from pathlib import Path
import torch
import argparse
import matplotlib.pyplot as plt

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
    Place the camera directly above the right hand for a focused top-down view.
    This gives a close overhead view of just the hand area.
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
        physicsClientId=env.id
    )
    rgb_img = np.array(rgb_img).reshape(height, width, 4)[:, :, :3].astype(np.uint8)
    depth_img = np.array(depth_img).reshape(height, width)
    depth_img = camera_config['far'] * camera_config['near'] / (
        camera_config['far'] - (camera_config['far'] - camera_config['near']) * depth_img
    )
    return rgb_img, depth_img, view_matrix, projection_matrix

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

def create_episode_directory(dataset_path, episode_num):
    """Create directory structure for an episode"""
    episode_dir = dataset_path / f"episode_{episode_num:04d}"
    rgb_dir = episode_dir / "rgb"
    depth_dir = episode_dir / "depth"
    episode_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    return episode_dir, rgb_dir, depth_dir

def depth_to_pointcloud(depth_img, camera_config, camera_pose):
    """
    Convert depth image to pointcloud using camera intrinsics and pose.
    Returns pointcloud in world coordinates.
    """
    height, width = depth_img.shape
    fx = fy = width / (2 * np.tan(np.radians(camera_config['fov']) / 2))
    cx, cy = width / 2, height / 2
    
    # Create pixel coordinates
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Convert to camera coordinates
    z = depth_img
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy
    
    # Stack into 3D points
    points_cam = np.stack([x_cam, y_cam, z], axis=-1)
    
    # Filter out invalid depth values
    valid_mask = (depth_img > 0) & (depth_img < camera_config['far'])
    valid_points_cam = points_cam[valid_mask]
    
    if len(valid_points_cam) == 0:
        return np.zeros((0, 3))
    
    # Transform to world coordinates
    points_homog = np.concatenate([valid_points_cam, np.ones((len(valid_points_cam), 1))], axis=1)
    points_world = (camera_pose @ points_homog.T).T[:, :3]
    
    return points_world

def depth_to_pointcloud_with_rgb(depth_img, rgb_img, camera_config, camera_pose):
    """
    Convert depth image to pointcloud with RGB colors using camera intrinsics and pose.
    Returns pointcloud in world coordinates with RGB values (6D: x,y,z,r,g,b).
    This matches the official DP3 format: (T, Np, 6) where 6 denotes [x, y, z, r, g, b]
    """
    height, width = depth_img.shape
    fx = fy = width / (2 * np.tan(np.radians(camera_config['fov']) / 2))
    cx, cy = width / 2, height / 2
    
    # Create pixel coordinates
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Convert to camera coordinates
    z = depth_img
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy
    
    # Stack into 3D points
    points_cam = np.stack([x_cam, y_cam, z], axis=-1)
    
    # Filter out invalid depth values
    valid_mask = (depth_img > 0) & (depth_img < camera_config['far'])
    valid_points_cam = points_cam[valid_mask]
    valid_rgb = rgb_img[valid_mask]  # Get corresponding RGB values
    
    if len(valid_points_cam) == 0:
        return np.zeros((0, 6))
    
    # Transform to world coordinates
    points_homog = np.concatenate([valid_points_cam, np.ones((len(valid_points_cam), 1))], axis=1)
    points_world = (camera_pose @ points_homog.T).T[:, :3]
    
    # Combine world coordinates with RGB values (normalize RGB to 0-1)
    pointcloud_with_rgb = np.concatenate([points_world, valid_rgb.astype(np.float32) / 255.0], axis=1)
    
    return pointcloud_with_rgb

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

def save_episode_data(episode_dir, rgb_dir, depth_dir, episode_data):
    """Save all episode data to disk in DP3 format"""
    # Save images
    for i, (rgb_img, depth_img) in enumerate(episode_data['images']):
        cv2.imwrite(str(rgb_dir / f"{i:06d}.png"), rgb_img)
        # Save depth as numpy array to preserve float32 precision
        np.save(str(depth_dir / f"{i:06d}.npy"), depth_img.astype(np.float32))
    
    # Save numpy arrays in DP3 format
    np.save(episode_dir / "actions.npy", np.array(episode_data['actions']))
    np.save(episode_dir / "states.npy", np.array(episode_data['states']))
    np.save(episode_dir / "camera_poses.npy", np.array(episode_data['camera_poses']))
    np.save(episode_dir / "rewards.npy", np.array(episode_data['rewards']))
    np.save(episode_dir / "end_effector_poses.npy", np.array(episode_data['end_effector_poses']))
    np.save(episode_dir / "observations.npy", np.array(episode_data['observations']))
    
    # Save DP3-specific format data
    np.save(episode_dir / "agent_pos.npy", np.array(episode_data['agent_positions']))  # DP3: agent_pos
    np.save(episode_dir / "point_cloud.npy", np.array(episode_data['pointclouds'], dtype=object))    # DP3: point_cloud with RGB, variable points per frame
    
    # Convert camera_config numpy arrays to lists for JSON serialization
    camera_config_serializable = {}
    for key, value in episode_data['camera_config'].items():
        if isinstance(value, np.ndarray):
            camera_config_serializable[key] = value.tolist()
        else:
            camera_config_serializable[key] = value
    
    # Save metadata
    metadata = {
        'episode_length': len(episode_data['actions']),
        'timestamp': datetime.now().isoformat(),
        'camera_config': camera_config_serializable,
        'action_dim': 7,
        'state_dim': 7,
        'image_resolution': [640, 480],
        'success': episode_data['success'],
        'task': 'bed_bathing_right_hand',
        'expert_policy': True,
        'depth_format': 'numpy_float32',
        'pointcloud_format': 'numpy_float32_rgb',  # DP3 format: (N, 6) with RGB
        'total_reward': float(sum(episode_data['rewards'])),
        'mean_reward': float(np.mean(episode_data['rewards'])),
        'observation_dim': len(episode_data['observations'][0]) if episode_data['observations'] else 0,
        'end_effector_pose_dim': len(episode_data['end_effector_poses'][0]) if episode_data['end_effector_poses'] else 0,
        'agent_pos_dim': len(episode_data['agent_positions'][0]) if episode_data['agent_positions'] else 0,
        'dp3_format': True,  # Indicate this follows official DP3 format
        'pointcloud_shape': f"({len(episode_data['pointclouds'][0])}, 6)" if episode_data['pointclouds'] else "(0, 6)"
    }
    with open(episode_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

def collect_episode(vec_env, episode_num, dataset_path, episode_length=100, noise_std=0.05, visualize=False):
    """Collect a single episode of data using the expert policy with vectorized environment and optional action noise"""
    print(f"Starting episode {episode_num}... (noise_std={noise_std})")
    episode_dir, rgb_dir, depth_dir = create_episode_directory(dataset_path, episode_num)
    
    # Get the underlying environment for camera access
    env = vec_env.venv.envs[0].env
    action_space = env.action_space
    
    obs = vec_env.reset()
    camera_config = setup_camera_aimed_at_right_hand(env)
    episode_data = {
        'images': [],
        'actions': [],
        'states': [],
        'camera_poses': [],
        'camera_config': camera_config,
        'rewards': [],
        'end_effector_poses': [],
        'observations': [],
        'agent_positions': [],  # DP3 format: agent_pos
        'pointclouds': [],      # DP3 format: point_cloud with RGB
        'success': False
    }
    
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
    
    # Initialize recurrent hidden states and masks (same as in enjoy.py)
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    
    for frame in range(episode_length):
        current_state = get_robot_state(env)
        episode_data['states'].append(current_state)
        
        # Get agent position in DP3 format
        agent_pos = get_agent_pos(env)
        episode_data['agent_positions'].append(agent_pos)
        
        # Get end-effector pose
        ee_pos, ee_orn = get_end_effector_pose(env)
        episode_data['end_effector_poses'].append(np.concatenate([ee_pos, ee_orn]))
        
        # Real-time trajectory visualization
        if visualize:
            traj_x.append(ee_pos[0])
            traj_y.append(ee_pos[1])
            traj_plot.set_data(traj_x, traj_y)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)
        
        # Get images and camera pose
        rgb_img, depth_img, view_matrix, proj_matrix = get_rgb_depth_images(env, camera_config)
        episode_data['images'].append((rgb_img, depth_img))
        
        # Create camera pose matrix (4x4)
        view_matrix_np = np.array(view_matrix).reshape(4, 4)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = view_matrix_np[:3, :3].T
        camera_pose[:3, 3] = -view_matrix_np[:3, :3].T @ view_matrix_np[:3, 3]
        episode_data['camera_poses'].append(camera_pose)
        
        # Generate pointcloud with RGB (DP3 format)
        pointcloud_rgb = depth_to_pointcloud_with_rgb(depth_img, rgb_img, camera_config, camera_pose)
        episode_data['pointclouds'].append(pointcloud_rgb)
        
        # Store full observation
        episode_data['observations'].append(obs.squeeze(0).cpu().numpy())
        
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
        
        episode_data['actions'].append(action_np)
        
        # Step environment with noisy action
        obs, reward, done, infos = vec_env.step(torch.from_numpy(action_np).unsqueeze(0))
        
        # Store reward
        reward_value = reward[0] if isinstance(reward, list) else reward
        if hasattr(reward_value, 'cpu'):
            reward_value = reward_value.cpu().numpy()
        episode_data['rewards'].append(reward_value)
        
        # Check for success
        if isinstance(infos, list) and len(infos) > 0:
            info = infos[0]
            if 'success' in info and info['success']:
                episode_data['success'] = True
        
        # Update masks
        masks.fill_(0.0 if done else 1.0)
        
        if frame % 10 == 0:
            print(f"  Frame {frame}/{episode_length}")
        if done:
            print(f"  Episode ended early at frame {frame}")
            break
    
    if visualize:
        plt.ioff()
        plt.show()
    
    save_episode_data(episode_dir, rgb_dir, depth_dir, episode_data)
    print(f"Episode {episode_num} completed: {len(episode_data['actions'])} frames")
    return len(episode_data['actions'])

def main():
    parser = argparse.ArgumentParser(description='DP3 Expert Data Collector with Suboptimality Option')
    parser.add_argument('--noise-std', type=float, default=0.05, help='Stddev of Gaussian noise for suboptimality (default: 0.05)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to collect')
    parser.add_argument('--episode-length', type=int, default=100, help='Length of each episode')
    parser.add_argument('--visualize', action='store_true', help='Show real-time end-effector trajectory plot')
    args = parser.parse_args()

    dataset_path = Path("dp3_dataset")
    dataset_path.mkdir(exist_ok=True)
    dataset_metadata = {
        'name': 'DP3_BedBathing_RightHand_Expert',
        'description': 'RGB-D dataset for 3D diffusion policy training on bed bathing task using expert policy',
        'created': datetime.now().isoformat(),
        'num_episodes': 0,
        'episode_length': args.episode_length,
        'camera_resolution': [640, 480],
        'action_space': '7-DOF joint space',
        'state_space': '7-DOF joint positions',
        'task': 'bed_bathing_right_hand',
        'expert_policy': True,
        'noise_std': args.noise_std
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
    
    vec_norm = get_vec_normalize(vec_env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms
    print("Environment created successfully!")
    total_frames = 0
    try:
        for episode in range(args.episodes):
            frames = collect_episode(vec_env, episode, dataset_path, episode_length=args.episode_length, noise_std=args.noise_std, visualize=args.visualize)
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