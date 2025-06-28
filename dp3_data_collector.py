import gym
import assistive_gym
import pybullet as p
import numpy as np
import cv2
import json
import os
import time
from datetime import datetime
from pathlib import Path

def get_right_hand_pos(env):
    """Get the position of the human's right hand"""
    return np.array(p.getLinkState(env.human, 9, computeForwardKinematics=True, physicsClientId=env.id)[0])

def setup_camera_aimed_at_right_hand(env, offset=np.array([0.0, 0.0, 0.4])):
    """
    Place the camera directly above the right hand for a focused top-down view.
    This gives a clear overhead view of just the hand area.
    Offset is relative to the hand position (in meters).
    """
    hand_pos = get_right_hand_pos(env)
    camera_pos = hand_pos + offset
    camera_target = hand_pos
    camera_config = {
        'position': camera_pos.tolist(),
        'target': camera_target.tolist(),
        'up': [0, 1, 0],  # Proper top-down orientation
        'fov': 45.0,  # Narrower FOV for more focused view of just the hand
        'near': 0.05,  # Closer near plane for precision
        'far': 3.0,   # Closer far plane to avoid background
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

def save_episode_data(episode_dir, rgb_dir, depth_dir, episode_data):
    """Save all episode data to disk"""
    # Save images
    for i, (rgb_img, depth_img) in enumerate(episode_data['images']):
        cv2.imwrite(str(rgb_dir / f"{i:06d}.png"), rgb_img)
        cv2.imwrite(str(depth_dir / f"{i:06d}.png"), depth_img.astype(np.float32))
    
    # Save numpy arrays
    np.save(episode_dir / "actions.npy", np.array(episode_data['actions']))
    np.save(episode_dir / "states.npy", np.array(episode_data['states']))
    np.save(episode_dir / "camera_poses.npy", np.array(episode_data['camera_poses']))
    
    # Save metadata
    metadata = {
        'episode_length': len(episode_data['actions']),
        'timestamp': datetime.now().isoformat(),
        'camera_config': episode_data['camera_config'],
        'action_dim': 7,
        'state_dim': 7,
        'image_resolution': [640, 480],
        'success': episode_data.get('success', False),
        'task': 'bed_bathing_right_hand'
    }
    
    with open(episode_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

def collect_episode(env, episode_num, dataset_path, episode_length=100):
    """Collect a single episode of data"""
    print(f"Starting episode {episode_num}...")
    
    # Create episode directory
    episode_dir, rgb_dir, depth_dir = create_episode_directory(dataset_path, episode_num)
    
    # Reset environment
    observation = env.reset()
    
    # Setup camera
    camera_config = setup_camera_aimed_at_right_hand(env)
    
    # Initialize episode data
    episode_data = {
        'images': [],
        'actions': [],
        'states': [],
        'camera_poses': [],
        'camera_config': camera_config
    }
    
    # Collect episode
    for frame in range(episode_length):
        # Get current state
        current_state = get_robot_state(env)
        episode_data['states'].append(current_state)
        
        # Get images and camera pose
        rgb_img, depth_img, view_matrix, proj_matrix = get_rgb_depth_images(env, camera_config)
        episode_data['images'].append((rgb_img, depth_img))
        
        # Create camera pose matrix (4x4)
        view_matrix = np.array(view_matrix).reshape(4, 4)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = view_matrix[:3, :3].T  # Rotation
        camera_pose[:3, 3] = -view_matrix[:3, :3].T @ view_matrix[:3, 3]  # Translation
        episode_data['camera_poses'].append(camera_pose)
        
        # Generate action (random for now, can be replaced with policy)
        action = env.action_space.sample()
        episode_data['actions'].append(action)
        
        # Step environment
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, done, truncated, info = step_result
        
        # Display progress
        if frame % 10 == 0:
            print(f"  Frame {frame}/{episode_length}")
        
        if done:
            print(f"  Episode ended early at frame {frame}")
            break
    
    # Save episode data
    save_episode_data(episode_dir, rgb_dir, depth_dir, episode_data)
    print(f"Episode {episode_num} completed: {len(episode_data['actions'])} frames")
    
    return len(episode_data['actions'])

def main():
    # Setup dataset directory
    dataset_path = Path("dp3_dataset")
    dataset_path.mkdir(exist_ok=True)
    
    # Create dataset metadata
    dataset_metadata = {
        'name': 'DP3_BedBathing_RightHand',
        'description': 'RGB-D dataset for 3D diffusion policy training on bed bathing task',
        'created': datetime.now().isoformat(),
        'num_episodes': 0,
        'episode_length': 100,
        'camera_resolution': [640, 480],
        'action_space': '7-DOF joint space',
        'state_space': '7-DOF joint positions',
        'task': 'bed_bathing_right_hand'
    }
    
    print("Creating BedBathingSawyer-v0 environment...")
    env = gym.make('BedBathingSawyer-v0')
    env.render()
    print("Environment created successfully!")
    
    # Collect episodes
    num_episodes = 10  # Adjust as needed
    total_frames = 0
    
    try:
        for episode in range(num_episodes):
            frames = collect_episode(env, episode, dataset_path)
            total_frames += frames
            dataset_metadata['num_episodes'] += 1
            
            # Save updated metadata
            with open(dataset_path / "dataset_metadata.json", 'w') as f:
                json.dump(dataset_metadata, f, indent=2)
            
            print(f"Completed {episode + 1}/{num_episodes} episodes")
            
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    finally:
        env.close()
        print(f"\nDataset collection completed!")
        print(f"Total episodes: {dataset_metadata['num_episodes']}")
        print(f"Total frames: {total_frames}")
        print(f"Dataset saved to: {dataset_path.absolute()}")

if __name__ == "__main__":
    main() 