#!/usr/bin/env python3
"""
Trajectory Visualizer

This script visualizes the gripper trajectory by:
1. Extracting the initial EEF pose from frame 0 of an episode
2. Computing subsequent poses using delta EEF actions
3. Visualizing the trajectory over time

Usage:
    python trajectory_visualizer.py --payload-path converted_payload.pt --episode 0
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
from typing import Optional, Tuple, Dict, Any, List
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def load_payload(payload_path: str) -> Optional[Dict[str, Any]]:
    """Load payload file and return its components."""
    print(f"Loading payload from: {payload_path}")
    
    try:
        payload = torch.load(payload_path)
        print(f"✓ Successfully loaded payload")
        
        # Extract components
        obses = payload[0]
        next_obses = payload[1]
        actions = payload[2]
        rewards = payload[3]
        not_dones = payload[4]
        
        print(f"Payload contains:")
        print(f"  - {len(obses)} observations")
        print(f"  - {len(next_obses)} next observations")
        print(f"  - {len(actions)} actions")
        print(f"  - {len(rewards)} rewards")
        
        return {
            'obses': obses,
            'next_obses': next_obses,
            'actions': actions,
            'rewards': rewards,
            'not_dones': not_dones
        }
        
    except Exception as e:
        print(f"✗ Error loading payload: {e}")
        return None

def extract_eef_pose_from_pointcloud(pcd, frame_idx: int) -> Optional[Dict[str, Any]]:
    """Extract EEF pose from pointcloud using gripper one-hot encoding."""
    if not hasattr(pcd, 'x') or not hasattr(pcd, 'pos'):
        print(f"Invalid pointcloud structure for frame {frame_idx}")
        return None
    
    features = pcd.x
    positions = pcd.pos
    
    if len(positions) == 0:
        print(f"Empty pointcloud for frame {frame_idx}")
        return None
    
    # Determine feature format
    num_features = features.shape[1]
    
    if num_features == 2:
        # 2-feature format: [cyan_onehot, gripper_onehot]
        gripper_onehot = features[:, 1]
    elif num_features == 5:
        # 5-feature format: [R, G, B, cyan_onehot, gripper_onehot]
        gripper_onehot = features[:, 4]
    else:
        print(f"Unexpected feature dimension: {num_features}")
        return None
    
    # Find gripper points
    gripper_mask = gripper_onehot == 1.0
    gripper_positions = positions[gripper_mask]
    
    if len(gripper_positions) == 0:
        print(f"No gripper points found in frame {frame_idx}")
        return None
    
    # Use the first gripper point as EEF position
    eef_position = gripper_positions[0].cpu().numpy()
    
    # For now, assume identity orientation (you might want to extract this from the data)
    eef_orientation = np.array([0, 0, 0, 1])  # Identity quaternion [x, y, z, w]
    
    return {
        'position': eef_position,
        'orientation': eef_orientation,
        'frame_idx': frame_idx,
        'num_gripper_points': len(gripper_positions)
    }

def compute_trajectory_from_deltas(initial_pose: Dict[str, Any], actions: List[np.ndarray], 
                                 action_scale: float = 0.025) -> List[Dict[str, Any]]:
    """Compute trajectory using initial pose and delta EEF actions."""
    trajectory = [initial_pose]
    
    # Convert initial orientation to rotation matrix
    initial_quat = initial_pose['orientation']
    current_R = R.from_quat([initial_quat[0], initial_quat[1], initial_quat[2], initial_quat[3]])
    current_pos = initial_pose['position'].copy()
    
    print(f"Initial EEF pose:")
    print(f"  Position: {current_pos}")
    print(f"  Orientation: {initial_quat}")
    
    for i, action in enumerate(actions):
        # Parse delta EEF action [pos_delta(3D), rot_delta(3D axis-angle)]
        if isinstance(action, np.ndarray):
            action_np = action.reshape(-1, 6)[-1].flatten()
        else:
            action_np = action.reshape(-1, 6)[-1].flatten()
        
        # Extract position and rotation deltas
        pos_delta = action_np[:3] * action_scale
        rot_delta = action_np[3:]
        
        # Apply position delta (same coordinate transformation as training)
        # Training uses: pos += np.array([z, x, y])
        pos_delta_transformed = np.array([pos_delta[2], pos_delta[0], pos_delta[1]])
        current_pos += pos_delta_transformed
        
        # Apply rotation delta using axis-angle
        if np.linalg.norm(rot_delta) > 1e-6:
            # Convert axis-angle to rotation matrix
            rot_angle = np.linalg.norm(rot_delta)
            rot_axis = rot_delta / rot_angle
            
            # Apply rotation limits (same as training)
            max_rot_axis_ang = (5. * np.pi / 180.)
            if rot_angle > max_rot_axis_ang:
                rot_angle = max_rot_axis_ang
                rot_delta = rot_axis * rot_angle
            
            delta_R = R.from_rotvec(rot_delta)
            current_R = delta_R * current_R
        
        # Convert back to quaternion
        current_quat = current_R.as_quat()  # [x, y, z, w]
        
        # Store pose
        pose = {
            'position': current_pos.copy(),
            'orientation': current_quat,
            'frame_idx': i + 1,
            'action': action_np,
            'pos_delta': pos_delta,
            'rot_delta': rot_delta
        }
        trajectory.append(pose)
    
    return trajectory

def find_episode_boundaries(payload: Dict[str, Any]) -> Dict[int, Tuple[int, int]]:
    """Find episode boundaries in the payload."""
    print("Analyzing episode boundaries...")
    
    not_dones = payload['not_dones']
    
    episodes = {}
    current_episode = 0
    episode_start = 0
    
    for i, not_done in enumerate(not_dones):
        if not not_done:  # Episode ended
            episodes[current_episode] = (episode_start, i)
            current_episode += 1
            episode_start = i + 1
    
    # Handle last episode
    if episode_start < len(not_dones):
        episodes[current_episode] = (episode_start, len(not_dones) - 1)
    
    print(f"Found {len(episodes)} episodes:")
    for ep_num, (start, end) in episodes.items():
        print(f"  Episode {ep_num}: frames {start}-{end} ({end-start+1} frames)")
    
    return episodes

def visualize_trajectory(trajectory: List[Dict[str, Any]], episode_num: int, 
                        save_path: Optional[str] = None):
    """Visualize the computed trajectory."""
    if not trajectory:
        print("No trajectory to visualize")
        return
    
    # Extract positions and orientations
    positions = np.array([pose['position'] for pose in trajectory])
    orientations = np.array([pose['orientation'] for pose in trajectory])
    frame_indices = [pose['frame_idx'] for pose in trajectory]
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 4, (1, 5), projection='3d')
    
    # Plot trajectory line
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
             'b-', linewidth=2, alpha=0.7, label='Trajectory')
    
    # Plot start and end points
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                c='green', s=100, marker='o', label='Start', alpha=0.8)
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                c='red', s=100, marker='s', label='End', alpha=0.8)
    
    # Plot intermediate points
    if len(positions) > 2:
        ax1.scatter(positions[1:-1, 0], positions[1:-1, 1], positions[1:-1, 2], 
                   c='blue', s=20, alpha=0.6)
    
    # Add coordinate frame at start and end
    start_R = R.from_quat(orientations[0])
    end_R = R.from_quat(orientations[-1])
    
    # Draw coordinate frames
    frame_size = 0.05
    for i, (pos, R_mat) in enumerate([(positions[0], start_R), (positions[-1], end_R)]):
        color = 'green' if i == 0 else 'red'
        label = 'Start Frame' if i == 0 else 'End Frame'
        
        # Draw coordinate axes
        for j, (axis, color_axis) in enumerate([([1, 0, 0], 'red'), ([0, 1, 0], 'green'), ([0, 0, 1], 'blue')]):
            axis_world = R_mat.apply(axis) * frame_size
            ax1.quiver(pos[0], pos[1], pos[2], 
                      axis_world[0], axis_world[1], axis_world[2], 
                      color=color_axis, alpha=0.8, length=frame_size, arrow_length_ratio=0.3)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'EEF Trajectory - Episode {episode_num}')
    ax1.legend()
    
    # Position plots
    ax2 = fig.add_subplot(2, 4, 6)
    ax2.plot(frame_indices, positions[:, 0], 'r-', label='X', linewidth=2)
    ax2.plot(frame_indices, positions[:, 1], 'g-', label='Y', linewidth=2)
    ax2.plot(frame_indices, positions[:, 2], 'b-', label='Z', linewidth=2)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Position')
    ax2.set_title('Position vs Frame')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Orientation plots (quaternion components)
    ax3 = fig.add_subplot(2, 4, 7)
    ax3.plot(frame_indices, orientations[:, 0], 'r-', label='X', linewidth=2)
    ax3.plot(frame_indices, orientations[:, 1], 'g-', label='Y', linewidth=2)
    ax3.plot(frame_indices, orientations[:, 2], 'b-', label='Z', linewidth=2)
    ax3.plot(frame_indices, orientations[:, 3], 'k-', label='W', linewidth=2)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Quaternion')
    ax3.set_title('Orientation vs Frame')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Action magnitude plot
    ax4 = fig.add_subplot(2, 4, 8)
    if len(trajectory) > 1:
        pos_magnitudes = []
        rot_magnitudes = []
        for pose in trajectory[1:]:  # Skip first pose (no action)
            pos_mag = np.linalg.norm(pose['pos_delta'])
            rot_mag = np.linalg.norm(pose['rot_delta'])
            pos_magnitudes.append(pos_mag)
            rot_magnitudes.append(rot_mag)
        
        ax4.plot(frame_indices[1:], pos_magnitudes, 'b-', label='Position', linewidth=2)
        ax4.plot(frame_indices[1:], rot_magnitudes, 'r-', label='Rotation', linewidth=2)
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Action Magnitude')
        ax4.set_title('Action Magnitude vs Frame')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Summary text
    ax5 = fig.add_subplot(2, 4, 4)
    ax5.axis('off')
    
    # Calculate trajectory statistics
    total_distance = np.sum([np.linalg.norm(positions[i+1] - positions[i]) 
                           for i in range(len(positions)-1)])
    max_displacement = np.linalg.norm(positions[-1] - positions[0])
    
    summary_text = f"""
Trajectory Analysis Summary:
    
Episode: {episode_num}
Total Frames: {len(trajectory)}
Trajectory Length: {len(trajectory)-1} actions

Position Statistics:
• Start: [{positions[0, 0]:.3f}, {positions[0, 1]:.3f}, {positions[0, 2]:.3f}]
• End: [{positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f}]
• Total Distance: {total_distance:.3f} m
• Max Displacement: {max_displacement:.3f} m

Position Bounds:
• X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]
• Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]
• Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]

Action Scale: {0.025}
"""
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory visualization to: {save_path}")
    else:
        plt.show()

def analyze_trajectory_actions(trajectory: List[Dict[str, Any]]):
    """Analyze the actions used in the trajectory."""
    if len(trajectory) <= 1:
        print("No actions to analyze")
        return
    
    print(f"\nTrajectory Action Analysis:")
    print(f"Total actions: {len(trajectory) - 1}")
    
    # Extract action statistics
    pos_deltas = np.array([pose['pos_delta'] for pose in trajectory[1:]])
    rot_deltas = np.array([pose['rot_delta'] for pose in trajectory[1:]])
    
    print(f"Position deltas:")
    print(f"  Mean magnitude: {np.mean(np.linalg.norm(pos_deltas, axis=1)):.6f}")
    print(f"  Max magnitude: {np.max(np.linalg.norm(pos_deltas, axis=1)):.6f}")
    print(f"  Std magnitude: {np.std(np.linalg.norm(pos_deltas, axis=1)):.6f}")
    
    print(f"Rotation deltas:")
    print(f"  Mean magnitude: {np.mean(np.linalg.norm(rot_deltas, axis=1)):.6f}")
    print(f"  Max magnitude: {np.max(np.linalg.norm(rot_deltas, axis=1)):.6f}")
    print(f"  Std magnitude: {np.std(np.linalg.norm(rot_deltas, axis=1)):.6f}")
    
    # Show first few actions
    print(f"\nFirst 5 actions:")
    for i in range(min(5, len(trajectory) - 1)):
        pose = trajectory[i + 1]
        print(f"  Frame {pose['frame_idx']}: pos_delta={pose['pos_delta']}, rot_delta={pose['rot_delta']}")

def visualize_combined_trajectory_pointcloud(trajectory: List[Dict[str, Any]], 
                                           pointcloud_analysis: Dict[str, Any],
                                           episode_num: int, frame_idx: int,
                                           save_path: Optional[str] = None):
    """Visualize trajectory and pointcloud together in a single Open3D window."""
    if not trajectory:
        print("No trajectory to visualize")
        return
    
    if 'error' in pointcloud_analysis:
        print(f"Error in pointcloud: {pointcloud_analysis['error']}")
        return
    
    # Create Open3D pointcloud from analysis
    pcd = create_open3d_pointcloud(pointcloud_analysis)
    if pcd is None:
        print("Failed to create pointcloud")
        return
    
    # Extract trajectory positions
    positions = np.array([pose['position'] for pose in trajectory])
    
    # Create trajectory line
    trajectory_line = o3d.geometry.LineSet()
    trajectory_line.points = o3d.utility.Vector3dVector(positions)
    
    # Create line indices (connect consecutive points)
    line_indices = []
    for i in range(len(positions) - 1):
        line_indices.append([i, i + 1])
    trajectory_line.lines = o3d.utility.Vector2iVector(line_indices)
    
    # Color the trajectory line (blue)
    trajectory_line.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(line_indices))])
    
    # Create start and end point spheres
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    start_sphere.paint_uniform_color([0, 1, 0])  # Green
    start_sphere.translate(positions[0])
    
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    end_sphere.paint_uniform_color([1, 0, 0])  # Red
    end_sphere.translate(positions[-1])
    
    # Create coordinate frame at start and end
    start_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    start_frame.translate(positions[0])
    
    end_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    end_frame.translate(positions[-1])
    
    # Create intermediate trajectory points (small spheres)
    intermediate_spheres = []
    if len(positions) > 2:
        for pos in positions[1:-1]:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sphere.paint_uniform_color([0.5, 0.5, 1])  # Light blue
            sphere.translate(pos)
            intermediate_spheres.append(sphere)
    
    # Create coordinate frame for current frame position
    current_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    current_frame.paint_uniform_color([1, 1, 0])  # Yellow
    if frame_idx < len(positions):
        current_frame.translate(positions[frame_idx])
    
    # Combine all geometries
    geometries = [pcd, trajectory_line, start_sphere, end_sphere, start_frame, end_frame, current_frame]
    geometries.extend(intermediate_spheres)
    
    # Create text overlay for information
    info_text = f"Episode {episode_num} - Frame {frame_idx}\n"
    info_text += f"Trajectory: {len(trajectory)} poses\n"
    info_text += f"Pointcloud: {pointcloud_analysis['total_points']} points\n"
    info_text += f"Arm: {pointcloud_analysis['arm_points']}, Cyan: {pointcloud_analysis['cyan_points']}, Gripper: {pointcloud_analysis['gripper_points']}"
    
    print(info_text)
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Trajectory + Pointcloud - Episode {episode_num}, Frame {frame_idx}",
        width=1200,
        height=800
    )

def visualize_combined_matplotlib(trajectory: List[Dict[str, Any]], 
                                pointcloud_analysis: Dict[str, Any],
                                episode_num: int, frame_idx: int,
                                save_path: Optional[str] = None):
    """Visualize trajectory and pointcloud together using matplotlib."""
    if not trajectory or 'error' in pointcloud_analysis:
        print("Invalid data for combined visualization")
        return
    
    # Extract data
    positions = np.array([pose['position'] for pose in trajectory])
    orientations = np.array([pose['orientation'] for pose in trajectory])
    frame_indices = [pose['frame_idx'] for pose in trajectory]
    
    pcd_positions = pointcloud_analysis['positions'].cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # Main 3D plot with both trajectory and pointcloud
    ax1 = fig.add_subplot(2, 4, (1, 5), projection='3d')
    
    # Plot trajectory
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
             'b-', linewidth=3, alpha=0.8, label='Trajectory')
    
    # Plot trajectory points
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                c='green', s=100, marker='o', label='Start', alpha=0.8)
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                c='red', s=100, marker='s', label='End', alpha=0.8)
    
    if len(positions) > 2:
        ax1.scatter(positions[1:-1, 0], positions[1:-1, 1], positions[1:-1, 2], 
                   c='blue', s=30, alpha=0.6, label='Trajectory Points')
    
    # Plot current frame position
    if frame_idx < len(positions):
        ax1.scatter(positions[frame_idx, 0], positions[frame_idx, 1], positions[frame_idx, 2], 
                   c='yellow', s=150, marker='*', label=f'Frame {frame_idx}', alpha=1.0)
    
    # Plot pointcloud with segmentation
    if pointcloud_analysis['format'] == '2_features':
        cyan_onehot = pointcloud_analysis['cyan_onehot'].cpu().numpy()
        gripper_onehot = pointcloud_analysis['gripper_onehot'].cpu().numpy()
        
        # Plot different point types
        arm_mask = (cyan_onehot == 0) & (gripper_onehot == 0)
        cyan_mask = cyan_onehot == 1.0
        gripper_mask = gripper_onehot == 1.0
        
        if np.any(arm_mask):
            ax1.scatter(pcd_positions[arm_mask, 0], pcd_positions[arm_mask, 1], pcd_positions[arm_mask, 2], 
                       c='lightgreen', s=1, alpha=0.4, label=f'Arm points ({np.sum(arm_mask)})')
        
        if np.any(cyan_mask):
            ax1.scatter(pcd_positions[cyan_mask, 0], pcd_positions[cyan_mask, 1], pcd_positions[cyan_mask, 2], 
                       c='cyan', s=2, alpha=0.6, label=f'Cyan points ({np.sum(cyan_mask)})')
        
        if np.any(gripper_mask):
            ax1.scatter(pcd_positions[gripper_mask, 0], pcd_positions[gripper_mask, 1], pcd_positions[gripper_mask, 2], 
                       c='orange', s=5, alpha=0.8, label=f'Gripper points ({np.sum(gripper_mask)})')
    
    elif pointcloud_analysis['format'] == '5_features':
        rgb_features = pointcloud_analysis['rgb_features'].cpu().numpy()
        cyan_onehot = pointcloud_analysis['cyan_onehot'].cpu().numpy()
        gripper_onehot = pointcloud_analysis['gripper_onehot'].cpu().numpy()
        
        # Plot with RGB colors
        ax1.scatter(pcd_positions[:, 0], pcd_positions[:, 1], pcd_positions[:, 2], 
                   c=rgb_features, s=1, alpha=0.4, label='RGB points')
        
        # Overlay special points
        cyan_mask = cyan_onehot == 1.0
        gripper_mask = gripper_onehot == 1.0
        
        if np.any(cyan_mask):
            ax1.scatter(pcd_positions[cyan_mask, 0], pcd_positions[cyan_mask, 1], pcd_positions[cyan_mask, 2], 
                       c='cyan', s=3, alpha=0.8, label=f'Cyan points ({np.sum(cyan_mask)})')
        
        if np.any(gripper_mask):
            ax1.scatter(pcd_positions[gripper_mask, 0], pcd_positions[gripper_mask, 1], pcd_positions[gripper_mask, 2], 
                       c='orange', s=5, alpha=1.0, label=f'Gripper points ({np.sum(gripper_mask)})')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Trajectory + Pointcloud - Episode {episode_num}, Frame {frame_idx}')
    ax1.legend()
    
    # Position plots
    ax2 = fig.add_subplot(2, 4, 6)
    ax2.plot(frame_indices, positions[:, 0], 'r-', label='X', linewidth=2)
    ax2.plot(frame_indices, positions[:, 1], 'g-', label='Y', linewidth=2)
    ax2.plot(frame_indices, positions[:, 2], 'b-', label='Z', linewidth=2)
    if frame_idx < len(positions):
        ax2.axvline(x=frame_idx, color='yellow', linestyle='--', alpha=0.7, label=f'Frame {frame_idx}')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Position')
    ax2.set_title('Trajectory Position vs Frame')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Pointcloud statistics
    ax3 = fig.add_subplot(2, 4, 7)
    point_types = ['Arm', 'Cyan', 'Gripper']
    point_counts = [pointcloud_analysis['arm_points'], pointcloud_analysis['cyan_points'], pointcloud_analysis['gripper_points']]
    colors = ['lightgreen', 'cyan', 'orange']
    
    bars = ax3.bar(point_types, point_counts, color=colors, alpha=0.7)
    ax3.set_title('Pointcloud Distribution')
    ax3.set_ylabel('Number of Points')
    
    # Add value labels on bars
    for bar, count in zip(bars, point_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                str(count), ha='center', va='bottom')
    
    # Summary text
    ax4 = fig.add_subplot(2, 4, (4, 8))
    ax4.axis('off')
    
    # Calculate trajectory statistics
    total_distance = np.sum([np.linalg.norm(positions[i+1] - positions[i]) 
                           for i in range(len(positions)-1)])
    max_displacement = np.linalg.norm(positions[-1] - positions[0])
    
    summary_text = f"""
Combined Analysis Summary:
    
Episode: {episode_num}
Current Frame: {frame_idx}
Total Trajectory Frames: {len(trajectory)}

Trajectory Statistics:
• Start: [{positions[0, 0]:.3f}, {positions[0, 1]:.3f}, {positions[0, 2]:.3f}]
• End: [{positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f}]
• Total Distance: {total_distance:.3f} m
• Max Displacement: {max_displacement:.3f} m

Pointcloud Statistics:
• Total Points: {pointcloud_analysis['total_points']}
• Arm Points: {pointcloud_analysis['arm_points']} ({pointcloud_analysis['arm_points']/pointcloud_analysis['total_points']*100:.1f}%)
• Cyan Points: {pointcloud_analysis['cyan_points']} ({pointcloud_analysis['cyan_points']/pointcloud_analysis['total_points']*100:.1f}%)
• Gripper Points: {pointcloud_analysis['gripper_points']} ({pointcloud_analysis['gripper_points']/pointcloud_analysis['total_points']*100:.1f}%)

Position Bounds:
• X: [{min(positions[:, 0].min(), pcd_positions[:, 0].min()):.3f}, {max(positions[:, 0].max(), pcd_positions[:, 0].max()):.3f}]
• Y: [{min(positions[:, 1].min(), pcd_positions[:, 1].min()):.3f}, {max(positions[:, 1].max(), pcd_positions[:, 1].max()):.3f}]
• Z: [{min(positions[:, 2].min(), pcd_positions[:, 2].min()):.3f}, {max(positions[:, 2].max(), pcd_positions[:, 2].max()):.3f}]
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined visualization to: {save_path}")
    else:
        plt.show()

def interactive_browser(payload: Dict[str, Any]):
    """Interactive browser for exploring payload data."""
    episodes = find_episode_boundaries(payload)
    
    if not episodes:
        print("No episodes found. Using all frames as single episode.")
        episodes = {0: (0, len(payload['obses']) - 1)}
    
    while True:
        print("\n" + "="*50)
        print("INTERACTIVE PAYLOAD BROWSER")
        print("="*50)
        
        # Show available episodes
        print("Available episodes:")
        for ep_num in episodes.keys():
            start, end = episodes[ep_num]
            print(f"  {ep_num}: frames {start}-{end} ({end-start+1} frames)")
        
        try:
            # Get user input
            ep_input = input(f"\nEnter episode number (0-{max(episodes.keys())}) or 'q' to quit: ").strip()
            
            if ep_input.lower() == 'q':
                break
            
            episode = int(ep_input)
            if episode not in episodes:
                print(f"Invalid episode number. Please choose from 0-{max(episodes.keys())}")
                continue
            
            start_frame, end_frame = episodes[episode]
            print(f"\nEpisode {episode}: frames {start_frame}-{end_frame}")
            
            # Ask for visualization type
            viz_type = input("Visualization type (p=pointcloud, t=trajectory, c=combined, b=both): ").strip().lower()
            
            if viz_type in ['p', 'b']:
                # Pointcloud visualization
                frame_input = input(f"Enter frame number ({start_frame}-{end_frame}) or 'r' for random: ").strip()
                
                if frame_input.lower() == 'r':
                    frame_idx = np.random.randint(start_frame, end_frame + 1)
                else:
                    frame_idx = int(frame_input)
                    if frame_idx < start_frame or frame_idx > end_frame:
                        print(f"Invalid frame number. Please choose from {start_frame}-{end_frame}")
                        continue
                
                # Analyze and visualize the selected frame
                print(f"\nAnalyzing frame {frame_idx}...")
                obs = payload['obses'][frame_idx]
                analysis = analyze_pointcloud_features(obs, frame_idx)
                
                if 'error' in analysis:
                    print(f"Error: {analysis['error']}")
                    continue
                
                print(f"Frame {frame_idx} analysis:")
                print(f"  Total points: {analysis['total_points']}")
                print(f"  Arm points: {analysis['arm_points']}")
                print(f"  Cyan points: {analysis['cyan_points']}")
                print(f"  Gripper points: {analysis['gripper_points']}")
                
                # Ask for visualization method
                viz_input = input("\nVisualization method (m=matplotlib, o=open3d, s=save): ").strip().lower()
                
                if viz_input == 'm':
                    visualize_pointcloud(analysis)
                elif viz_input == 'o':
                    visualize_open3d(analysis)
                elif viz_input == 's':
                    save_path = f"frame_{frame_idx:06d}_visualization.png"
                    visualize_pointcloud(analysis, save_path)
            
            if viz_type in ['t', 'b']:
                # Trajectory visualization
                print(f"\nComputing trajectory for episode {episode}...")
                
                # Extract initial EEF pose from frame 0
                initial_obs = payload['obses'][start_frame]
                initial_pose = extract_eef_pose_from_pointcloud(initial_obs, start_frame)
                
                if initial_pose is None:
                    print("Failed to extract initial EEF pose")
                    continue
                
                # Extract actions for this episode
                episode_actions = []
                for i in range(start_frame, end_frame):
                    if i < len(payload['actions']):
                        action = payload['actions'][i]
                        if isinstance(action, np.ndarray):
                            episode_actions.append(action)
                        else:
                            episode_actions.append(action)
                
                # Compute trajectory
                trajectory = compute_trajectory_from_deltas(initial_pose, episode_actions, 0.025)
                
                # Ask for visualization method
                viz_input = input("\nTrajectory visualization method (m=matplotlib, s=save): ").strip().lower()
                
                if viz_input == 'm':
                    visualize_trajectory(trajectory, episode)
                elif viz_input == 's':
                    save_path = f"episode_{episode}_trajectory.png"
                    visualize_trajectory(trajectory, episode, save_path)
            
            if viz_type in ['c']:
                # Combined visualization
                print(f"\nComputing combined visualization for episode {episode}...")
                
                # Get frame for pointcloud
                frame_input = input(f"Enter frame number for pointcloud ({start_frame}-{end_frame}) or 'r' for random: ").strip()
                
                if frame_input.lower() == 'r':
                    frame_idx = np.random.randint(start_frame, end_frame + 1)
                else:
                    frame_idx = int(frame_input)
                    if frame_idx < start_frame or frame_idx > end_frame:
                        print(f"Invalid frame number. Please choose from {start_frame}-{end_frame}")
                        continue
                
                # Extract initial EEF pose and compute trajectory
                initial_obs = payload['obses'][start_frame]
                initial_pose = extract_eef_pose_from_pointcloud(initial_obs, start_frame)
                
                if initial_pose is None:
                    print("Failed to extract initial EEF pose")
                    continue
                
                # Extract actions for this episode
                episode_actions = []
                for i in range(start_frame, end_frame):
                    if i < len(payload['actions']):
                        action = payload['actions'][i]
                        if isinstance(action, np.ndarray):
                            episode_actions.append(action)
                        else:
                            episode_actions.append(action)
                
                # Compute trajectory
                trajectory = compute_trajectory_from_deltas(initial_pose, episode_actions, 0.025)
                
                # Analyze pointcloud for selected frame
                obs = payload['obses'][frame_idx]
                analysis = analyze_pointcloud_features(obs, frame_idx)
                
                if 'error' in analysis:
                    print(f"Error in pointcloud: {analysis['error']}")
                    continue
                
                # Ask for visualization method
                viz_input = input("\nCombined visualization method (m=matplotlib, o=open3d, s=save): ").strip().lower()
                
                if viz_input == 'm':
                    visualize_combined_matplotlib(trajectory, analysis, episode, frame_idx)
                elif viz_input == 'o':
                    visualize_combined_trajectory_pointcloud(trajectory, analysis, episode, frame_idx)
                elif viz_input == 's':
                    save_path = f"episode_{episode}_frame_{frame_idx}_combined.png"
                    visualize_combined_matplotlib(trajectory, analysis, episode, frame_idx, save_path)
            
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def analyze_pointcloud_features(obs, frame_idx: int) -> Dict[str, Any]:
    """Analyze pointcloud features and return statistics."""
    if not hasattr(obs, 'x') or not hasattr(obs, 'pos'):
        return {'error': f"Invalid pointcloud structure for frame {frame_idx}"}
    
    features = obs.x
    positions = obs.pos
    
    if len(positions) == 0:
        return {'error': f"Empty pointcloud for frame {frame_idx}"}
    
    # Determine feature format
    num_features = features.shape[1]
    
    if num_features == 2:
        # 2-feature format: [cyan_onehot, gripper_onehot]
        cyan_onehot = features[:, 0]
        gripper_onehot = features[:, 1]
        
        # Count points by type
        arm_points = torch.sum((cyan_onehot == 0) & (gripper_onehot == 0)).item()
        cyan_points = torch.sum(cyan_onehot == 1.0).item()
        gripper_points = torch.sum(gripper_onehot == 1.0).item()
        
        return {
            'format': '2_features',
            'total_points': len(positions),
            'arm_points': arm_points,
            'cyan_points': cyan_points,
            'gripper_points': gripper_points,
            'positions': positions,
            'cyan_onehot': cyan_onehot,
            'gripper_onehot': gripper_onehot,
            'frame_idx': frame_idx
        }
        
    elif num_features == 5:
        # 5-feature format: [R, G, B, cyan_onehot, gripper_onehot]
        rgb_features = features[:, :3]
        cyan_onehot = features[:, 3]
        gripper_onehot = features[:, 4]
        
        # Count points by type
        arm_points = torch.sum((cyan_onehot == 0) & (gripper_onehot == 0)).item()
        cyan_points = torch.sum(cyan_onehot == 1.0).item()
        gripper_points = torch.sum(gripper_onehot == 1.0).item()
        
        return {
            'format': '5_features',
            'total_points': len(positions),
            'arm_points': arm_points,
            'cyan_points': cyan_points,
            'gripper_points': gripper_points,
            'positions': positions,
            'rgb_features': rgb_features,
            'cyan_onehot': cyan_onehot,
            'gripper_onehot': gripper_onehot,
            'frame_idx': frame_idx
        }
    
    else:
        return {'error': f"Unexpected feature dimension: {num_features}"}

def create_open3d_pointcloud(analysis: Dict[str, Any]):
    """Create Open3D pointcloud from analysis."""
    if 'error' in analysis:
        print(f"Error in analysis: {analysis['error']}")
        return None
    
    # Create pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(analysis['positions'].cpu().numpy())
    
    # Color the points based on segmentation
    if analysis['format'] == '2_features':
        # 2-feature format: color by one-hot encoding
        colors = []
        cyan_onehot = analysis['cyan_onehot'].cpu().numpy()
        gripper_onehot = analysis['gripper_onehot'].cpu().numpy()
        
        for i in range(len(analysis['positions'])):
            if gripper_onehot[i] == 1.0:
                colors.append([1.0, 0.5, 0.0])  # Orange for gripper
            elif cyan_onehot[i] == 1.0:
                colors.append([0.0, 1.0, 1.0])  # Cyan for cyan points
            else:
                colors.append([0.5, 1.0, 0.5])  # Light green for arm
    
    elif analysis['format'] == '5_features':
        # 5-feature format: use RGB features
        rgb_features = analysis['rgb_features'].cpu().numpy()
        colors = rgb_features.tolist()
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize_pointcloud(analysis: Dict[str, Any], save_path: Optional[str] = None):
    """Visualize pointcloud using matplotlib."""
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    # Extract data
    positions = analysis['positions'].cpu().numpy()
    frame_idx = analysis['frame_idx']
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # 3D scatter plot
    ax = fig.add_subplot(111, projection='3d')
    
    if analysis['format'] == '2_features':
        # Plot different point types with different colors
        cyan_onehot = analysis['cyan_onehot'].cpu().numpy()
        gripper_onehot = analysis['gripper_onehot'].cpu().numpy()
        
        # Create masks
        arm_mask = (cyan_onehot == 0) & (gripper_onehot == 0)
        cyan_mask = cyan_onehot == 1.0
        gripper_mask = gripper_onehot == 1.0
        
        # Plot each type
        if np.any(arm_mask):
            ax.scatter(positions[arm_mask, 0], positions[arm_mask, 1], positions[arm_mask, 2], 
                      c='lightgreen', s=1, alpha=0.4, label=f'Arm points ({np.sum(arm_mask)})')
        
        if np.any(cyan_mask):
            ax.scatter(positions[cyan_mask, 0], positions[cyan_mask, 1], positions[cyan_mask, 2], 
                      c='cyan', s=2, alpha=0.6, label=f'Cyan points ({np.sum(cyan_mask)})')
        
        if np.any(gripper_mask):
            ax.scatter(positions[gripper_mask, 0], positions[gripper_mask, 1], positions[gripper_mask, 2], 
                      c='orange', s=5, alpha=0.8, label=f'Gripper points ({np.sum(gripper_mask)})')
    
    elif analysis['format'] == '5_features':
        # Plot with RGB colors
        rgb_features = analysis['rgb_features'].cpu().numpy()
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c=rgb_features, s=1, alpha=0.4, label='RGB points')
        
        # Overlay special points
        cyan_onehot = analysis['cyan_onehot'].cpu().numpy()
        gripper_onehot = analysis['gripper_onehot'].cpu().numpy()
        
        cyan_mask = cyan_onehot == 1.0
        gripper_mask = gripper_onehot == 1.0
        
        if np.any(cyan_mask):
            ax.scatter(positions[cyan_mask, 0], positions[cyan_mask, 1], positions[cyan_mask, 2], 
                      c='cyan', s=3, alpha=0.8, label=f'Cyan points ({np.sum(cyan_mask)})')
        
        if np.any(gripper_mask):
            ax.scatter(positions[gripper_mask, 0], positions[gripper_mask, 1], positions[gripper_mask, 2], 
                      c='orange', s=5, alpha=1.0, label=f'Gripper points ({np.sum(gripper_mask)})')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Pointcloud Visualization - Frame {frame_idx}')
    ax.legend()
    
    # Add statistics text
    stats_text = f"""
Pointcloud Statistics:
• Total Points: {analysis['total_points']}
• Arm Points: {analysis['arm_points']} ({analysis['arm_points']/analysis['total_points']*100:.1f}%)
• Cyan Points: {analysis['cyan_points']} ({analysis['cyan_points']/analysis['total_points']*100:.1f}%)
• Gripper Points: {analysis['gripper_points']} ({analysis['gripper_points']/analysis['total_points']*100:.1f}%)

Position Bounds:
• X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]
• Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]
• Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]
"""
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved pointcloud visualization to: {save_path}")
    else:
        plt.show()

def visualize_open3d(analysis: Dict[str, Any]):
    """Visualize pointcloud using Open3D."""
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    # Create Open3D pointcloud
    pcd = create_open3d_pointcloud(analysis)
    if pcd is None:
        return
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Visualize
    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name=f"Pointcloud - Frame {analysis['frame_idx']}",
        width=1200,
        height=800
    )

def main():
    parser = argparse.ArgumentParser(description='Visualize payload pointcloud data and trajectories')
    parser.add_argument('--payload-path', type=str, required=True, help='Path to payload file')
    parser.add_argument('--episode', type=int, default=0, help='Episode number to visualize')
    parser.add_argument('--frame', type=int, default=19300, help='Frame number within episode')
    parser.add_argument('--trajectory', action='store_true', help='Visualize trajectory instead of pointcloud')
    parser.add_argument('--combined', action='store_true', help='Visualize trajectory and pointcloud together')
    parser.add_argument('--interactive', action='store_true', help='Start interactive browser')
    parser.add_argument('--save', type=str, help='Save visualization to file')
    parser.add_argument('--method', choices=['matplotlib', 'open3d', 'both'], default='matplotlib', 
                       help='Visualization method')
    parser.add_argument('--action-scale', type=float, default=0.025, help='Action scaling factor for trajectory')
    args = parser.parse_args()
    
    # Load payload
    payload = load_payload(args.payload_path)
    if payload is None:
        return
    
    if args.interactive:
        interactive_browser(payload)
        return
    
    # Find episode boundaries
    episodes = find_episode_boundaries(payload)
    
    if args.episode not in episodes:
        print(f"Episode {args.episode} not found. Available episodes: {list(episodes.keys())}")
        return
    
    start_frame, end_frame = episodes[args.episode]
    
    if args.combined:
        # Combined visualization
        print(f"\nComputing combined visualization for episode {args.episode}...")
        
        if args.frame < start_frame or args.frame > end_frame:
            print(f"Frame {args.frame} not in episode {args.episode} (range: {start_frame}-{end_frame})")
            return
        
        # Extract initial EEF pose and compute trajectory
        initial_obs = payload['obses'][start_frame]
        initial_pose = extract_eef_pose_from_pointcloud(initial_obs, start_frame)
        
        if initial_pose is None:
            print("Failed to extract initial EEF pose")
            return
        
        # Extract actions for this episode
        episode_actions = []
        for i in range(start_frame, end_frame):
            if i < len(payload['actions']):
                action = payload['actions'][i]
                if isinstance(action, np.ndarray):
                    episode_actions.append(action)
                else:
                    episode_actions.append(action)
        
        # Compute trajectory
        trajectory = compute_trajectory_from_deltas(initial_pose, episode_actions, args.action_scale)
        
        # Analyze pointcloud for selected frame
        obs = payload['obses'][args.frame]
        analysis = analyze_pointcloud_features(obs, args.frame)
        
        if 'error' in analysis:
            print(f"Error in pointcloud: {analysis['error']}")
            return
        
        # Visualize combined
        if args.method in ['matplotlib', 'both']:
            save_path = args.save or f"episode_{args.episode}_frame_{args.frame}_combined.png"
            visualize_combined_matplotlib(trajectory, analysis, args.episode, args.frame, save_path)
        
        if args.method in ['open3d', 'both']:
            visualize_combined_trajectory_pointcloud(trajectory, analysis, args.episode, args.frame)
    
    elif args.trajectory:
        # Trajectory visualization
        print(f"\nComputing trajectory for episode {args.episode}...")
        
        # Extract initial EEF pose from frame 0
        initial_obs = payload['obses'][start_frame]
        initial_pose = extract_eef_pose_from_pointcloud(initial_obs, start_frame)
        
        if initial_pose is None:
            print("Failed to extract initial EEF pose")
            return
        
        # Extract actions for this episode
        episode_actions = []
        for i in range(start_frame, end_frame):
            if i < len(payload['actions']):
                action = payload['actions'][i]
                if isinstance(action, np.ndarray):
                    episode_actions.append(action)
                else:
                    episode_actions.append(action)
        
        # Compute trajectory
        trajectory = compute_trajectory_from_deltas(initial_pose, episode_actions, args.action_scale)
        
        # Visualize trajectory
        save_path = args.save or f"episode_{args.episode}_trajectory.png"
        visualize_trajectory(trajectory, args.episode, save_path)
        
    else:
        # Pointcloud visualization
        if args.frame < start_frame or args.frame > end_frame:
            print(f"Frame {args.frame} not in episode {args.episode} (range: {start_frame}-{end_frame})")
            return
        
        # Analyze and visualize the selected frame
        print(f"Analyzing frame {args.frame} from episode {args.episode}...")
        obs = payload['obses'][args.frame]
        analysis = analyze_pointcloud_features(obs, args.frame)
        
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        print(f"Frame {args.frame} analysis:")
        print(f"  Format: {analysis['format']}")
        print(f"  Total points: {analysis['total_points']}")
        print(f"  Arm points: {analysis['arm_points']}")
        print(f"  Cyan points: {analysis['cyan_points']}")
        print(f"  Gripper points: {analysis['gripper_points']}")
        
        # Visualize
        if args.method in ['matplotlib', 'both']:
            visualize_pointcloud(analysis, args.save)
        
        if args.method in ['open3d', 'both']:
            visualize_open3d(analysis)

if __name__ == "__main__":
    main() 