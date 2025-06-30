#!/usr/bin/env python3
"""
Enhanced Pointcloud Visualizer with Coordinate Frame Validation
Plots pointclouds with different colors based on one-hot labels and overlays trajectories
computed from delta EEF movements to validate coordinate frame consistency.
Now handles the corrected PyBullet world coordinate frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional
import json
from scipy.spatial.transform import Rotation as R

# Try to import Open3D for better 3D visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: Open3D not available. Using matplotlib for 3D visualization.")
    OPEN3D_AVAILABLE = False

def load_episode_data(episode_file: str) -> Dict[str, Any]:
    """Load episode data from pickle file"""
    with open(episode_file, 'rb') as f:
        episode_data = pickle.load(f)
    return episode_data

def extract_pointcloud_labels_from_numpy(combined_positions, combined_features) -> Dict[str, np.ndarray]:
    """
    Extract different point types based on one-hot labels from numpy arrays.
    
    Args:
        combined_positions: Nx3 array of point positions
        combined_features: Nx3 array of features [arm_onehot, cyan_onehot, gripper_onehot]
    
    Returns:
        labels_dict: Dictionary with 'arm', 'cyan', 'eef' point arrays
    """
    if len(combined_positions) == 0:
        return {'arm': np.array([]), 'cyan': np.array([]), 'eef': np.array([])}
    
    # 3-feature format: [arm_onehot, cyan_onehot, gripper_onehot]
    arm_onehot = combined_features[:, 0]
    cyan_onehot = combined_features[:, 1]
    gripper_onehot = combined_features[:, 2]
    
    # Extract different point types
    arm_mask = arm_onehot == 1.0
    cyan_mask = cyan_onehot == 1.0
    eef_mask = gripper_onehot == 1.0
    
    arm_points = combined_positions[arm_mask]
    cyan_points = combined_positions[cyan_mask]
    eef_points = combined_positions[eef_mask]
    
    return {
        'arm': arm_points,
        'cyan': cyan_points,
        'eef': eef_points
    }

def extract_pointcloud_labels(pcd_data) -> Dict[str, np.ndarray]:
    """
    Extract different point types based on one-hot labels from pointcloud.
    Handles both torch_geometric Data objects and numpy arrays.
    
    Args:
        pcd_data: torch_geometric.Data object with features OR tuple of (positions, features)
    
    Returns:
        labels_dict: Dictionary with 'arm', 'cyan', 'eef' point arrays
    """
    # Check if it's the new numpy format
    if isinstance(pcd_data, tuple) and len(pcd_data) == 2:
        combined_positions, combined_features = pcd_data
        if isinstance(combined_positions, np.ndarray) and isinstance(combined_features, np.ndarray):
            return extract_pointcloud_labels_from_numpy(combined_positions, combined_features)
    
    # Handle torch_geometric Data objects (legacy format)
    if not hasattr(pcd_data, 'x') or not hasattr(pcd_data, 'pos'):
        print("Warning: pcd_data doesn't have expected attributes 'x' and 'pos'")
        return {'arm': np.array([]), 'cyan': np.array([]), 'eef': np.array([])}
    
    features = pcd_data.x.cpu().numpy()
    positions = pcd_data.pos.cpu().numpy()
    
    if len(positions) == 0:
        return {'arm': np.array([]), 'cyan': np.array([]), 'eef': np.array([])}
    
    # Determine feature format based on number of features
    num_features = features.shape[1]
    print(f"Pointcloud has {len(positions)} points with {num_features} features")
    
    if num_features == 3:
        # 3-feature format: [arm_onehot, cyan_onehot, gripper_onehot]
        arm_onehot = features[:, 0]
        cyan_onehot = features[:, 1]
        gripper_onehot = features[:, 2]
        
        # Extract different point types
        arm_mask = arm_onehot == 1.0
        cyan_mask = cyan_onehot == 1.0
        eef_mask = gripper_onehot == 1.0
        
        cyan_points = positions[cyan_mask]
        eef_points = positions[eef_mask]
        arm_points = positions[arm_mask]
        
        print(f"3-feature format: Arm={len(arm_points)}, Cyan={len(cyan_points)}, EEF={len(eef_points)}")
        
    elif num_features == 5:
        # 5-feature format: [R, G, B, cyan_onehot, gripper_onehot] - SIMPLIFIED FORMAT
        cyan_onehot = features[:, 3]
        gripper_onehot = features[:, 4]
        
        # Extract different point types
        cyan_mask = cyan_onehot == 1.0
        eef_mask = gripper_onehot == 1.0
        arm_mask = (cyan_onehot == 0.0) & (gripper_onehot == 0.0)  # Neither cyan nor EEF
        
        cyan_points = positions[cyan_mask]
        eef_points = positions[eef_mask]
        arm_points = positions[arm_mask]
        
        print(f"5-feature format (simplified): Arm={len(arm_points)}, Cyan={len(cyan_points)}, EEF={len(eef_points)}")
        
    elif num_features == 2:
        # 2-feature format: [cyan_onehot, gripper_onehot]
        cyan_onehot = features[:, 0]
        gripper_onehot = features[:, 1]
        
        cyan_mask = cyan_onehot == 1.0
        eef_mask = gripper_onehot == 1.0
        arm_mask = (cyan_onehot == 0.0) & (gripper_onehot == 0.0)
        
        cyan_points = positions[cyan_mask]
        eef_points = positions[eef_mask]
        arm_points = positions[arm_mask]
        
        print(f"2-feature format: Arm={len(arm_points)}, Cyan={len(cyan_points)}, EEF={len(eef_points)}")
        
    else:
        # Unknown format, treat all as arm points
        arm_points = positions
        cyan_points = np.array([])
        eef_points = np.array([])
        print(f"Unknown format ({num_features} features): Treating all {len(positions)} points as arm points")
    
    return {
        'arm': arm_points,
        'cyan': cyan_points,
        'eef': eef_points
    }

def compute_trajectory_from_deltas(transitions: List[Dict[str, Any]], 
                                 initial_frame_idx: int = 0) -> List[np.ndarray]:
    """
    Compute trajectory using initial pose and delta EEF movements.
    Handles the new model_axes coordinate frame transformation.
    
    Args:
        transitions: List of transition dictionaries
        initial_frame_idx: Frame index to start trajectory from
    
    Returns:
        trajectory: List of computed EEF positions
    """
    if len(transitions) == 0:
        return []
    
    # Get initial pose from the specified frame
    initial_transition = transitions[initial_frame_idx]
    initial_pos = np.array(initial_transition['gripper_point'])
    initial_ori = np.array(initial_transition['gripper_orientation'])
    
    # Check coordinate frame
    coord_frame = initial_transition.get('coordinate_frame', 'model_axes')
    print(f"Coordinate frame: {coord_frame}")
    
    trajectory = [initial_pos.copy()]
    
    # Convert initial orientation to rotation matrix
    initial_quat = [initial_ori[3], initial_ori[0], initial_ori[1], initial_ori[2]]  # [w, x, y, z]
    current_R = R.from_quat(initial_quat)
    current_pos = initial_pos.copy()
    
    print(f"Initial EEF pose (frame {initial_frame_idx}):")
    print(f"  Position: {current_pos}")
    print(f"  Orientation: {initial_ori}")
    print(f"  Coordinate frame: {coord_frame}")
    
    # Apply delta movements from subsequent frames
    for i in range(initial_frame_idx + 1, len(transitions)):
        transition = transitions[i]
        
        if 'delta_6d' in transition:
            delta_6d = transition['delta_6d']
            
            # Extract position and rotation deltas
            pos_delta = delta_6d[:3]
            rot_delta = delta_6d[3:]
            
            # Apply position delta
            current_pos += pos_delta
            
            # Apply rotation delta using axis-angle
            if np.linalg.norm(rot_delta) > 1e-6:
                delta_R = R.from_rotvec(rot_delta)
                current_R = delta_R * current_R
            
            trajectory.append(current_pos.copy())
            
            # Debug output for first few frames
            if i < initial_frame_idx + 5:
                print(f"Frame {i}: Applied delta {pos_delta}, new pos: {current_pos}")
        else:
            # If no delta available, use the logged position
            logged_pos = np.array(transition['gripper_point'])
            trajectory.append(logged_pos)
            print(f"Frame {i}: No delta available, using logged position: {logged_pos}")
    
    print(f"Computed trajectory with {len(trajectory)} points using delta eef")
    return trajectory

def create_open3d_pointcloud(points: np.ndarray, color: List[float]) -> o3d.geometry.PointCloud:
    """Create Open3D pointcloud from numpy array"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd

def create_open3d_sphere(center: np.ndarray, radius: float, color: List[float]) -> o3d.geometry.TriangleMesh:
    """Create Open3D sphere for EEF visualization"""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere

def create_open3d_line_set(points: np.ndarray, color: List[float]) -> o3d.geometry.LineSet:
    """Create Open3D line set for trajectory visualization"""
    if len(points) < 2:
        return o3d.geometry.LineSet()
    
    lines = [[i, i+1] for i in range(len(points)-1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    return line_set

def plot_frame_with_trajectory_open3d(transitions: List[Dict[str, Any]], 
                                    frame_idx: int,
                                    trajectory: List[np.ndarray],
                                    initial_frame_idx: int = 0,
                                    show_trajectory: bool = True) -> None:
    """
    Plot a single frame's pointcloud with trajectory overlay using Open3D.
    Shows trajectory only up to the current frame and highlights EEF in pointcloud.
    
    Args:
        transitions: List of transition dictionaries
        frame_idx: Frame index to plot
        trajectory: Computed trajectory positions
        initial_frame_idx: Frame index where trajectory started
        show_trajectory: Whether to show trajectory overlay
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available. Use matplotlib version instead.")
        return
    
    if frame_idx >= len(transitions):
        print(f"Frame {frame_idx} not available (max: {len(transitions)-1})")
        return
    
    transition = transitions[frame_idx]
    
    # Create Open3D geometries list
    geometries = []
    
    # Extract pointcloud data - handle new format
    labels_dict = {'arm': np.array([]), 'cyan': np.array([]), 'eef': np.array([])}
    
    if 'pcd_combined_positions' in transition and 'pcd_combined_features' in transition:
        # New format: numpy arrays
        combined_positions = transition['pcd_combined_positions']
        combined_features = transition['pcd_combined_features']
        labels_dict = extract_pointcloud_labels_from_numpy(combined_positions, combined_features)
        print(f"Using new numpy format: {len(combined_positions)} points")
    elif 'pcd_combined' in transition:
        # Legacy format: torch_geometric Data
        pcd_data = transition['pcd_combined']
        labels_dict = extract_pointcloud_labels(pcd_data)
        print(f"Using legacy torch_geometric format")
    
    # Colors for different point types
    colors = {
        'arm': [0.0, 0.0, 1.0],    # Blue
        'cyan': [0.0, 1.0, 1.0],   # Cyan
        'eef': [1.0, 0.0, 0.0]     # Red
    }
    
    # Create pointclouds for arm and cyan points
    for label in ['arm', 'cyan']:
        points = labels_dict[label]
        if len(points) > 0:
            pcd = create_open3d_pointcloud(points, colors[label])
            geometries.append(pcd)
            print(f"Added {label} pointcloud with {len(points)} points")
    
    # Create EEF pointcloud with large spheres
    eef_points = labels_dict['eef']
    if len(eef_points) > 0:
        # Create EEF pointcloud
        eef_pcd = create_open3d_pointcloud(eef_points, [1.0, 0.0, 0.0])  # Red
        geometries.append(eef_pcd)
        
        # Add large spheres for each EEF point
        for i, eef_pos in enumerate(eef_points):
            eef_sphere = create_open3d_sphere(eef_pos, radius=0.03, color=[1.0, 0.0, 0.0])  # Large red sphere
            geometries.append(eef_sphere)
        
        print(f"Added EEF pointcloud with {len(eef_points)} points (large markers)")
        for i, eef_pos in enumerate(eef_points):
            print(f"  EEF point {i}: {eef_pos}")
    
    # Plot logged EEF position as sphere
    if 'gripper_point' in transition:
        logged_eef = np.array(transition['gripper_point'])
        eef_sphere = create_open3d_sphere(logged_eef, radius=0.02, color=[0.0, 1.0, 0.0])  # Green
        geometries.append(eef_sphere)
        print(f"Added logged EEF sphere at {logged_eef}")
    
    # Plot trajectory only up to current frame
    if show_trajectory and len(trajectory) > 0:
        # Calculate how many trajectory points to show
        trajectory_idx = frame_idx - initial_frame_idx
        if trajectory_idx >= 0 and trajectory_idx < len(trajectory):
            # Show trajectory up to current frame
            trajectory_up_to_frame = trajectory[:trajectory_idx + 1]
            trajectory_array = np.array(trajectory_up_to_frame)
            
            if len(trajectory_array) > 1:
                # Create trajectory line up to current frame
                trajectory_line = create_open3d_line_set(trajectory_array, [0.0, 1.0, 0.0])  # Green
                geometries.append(trajectory_line)
                
                # Create trajectory points as small spheres up to current frame
                for i, pos in enumerate(trajectory_array):
                    traj_sphere = create_open3d_sphere(pos, radius=0.005, color=[0.0, 0.8, 0.0])  # Light green
                    geometries.append(traj_sphere)
            
            # Highlight current position on trajectory with large sphere
            current_traj_pos = trajectory[trajectory_idx]
            current_sphere = create_open3d_sphere(current_traj_pos, radius=0.025, color=[1.0, 1.0, 0.0])  # Yellow
            geometries.append(current_sphere)
            print(f"Added current trajectory EEF sphere at {current_traj_pos}")
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(coord_frame)
    
    # Print coordinate frame information
    print(f"\n=== Frame {frame_idx} Analysis (Open3D) ===")
    print(f"Coordinate frame: {transition.get('coordinate_frame', 'model_axes')}")
    print(f"Data format: {'numpy_arrays' if 'pcd_combined_positions' in transition else 'torch_geometric'}")
    
    if 'gripper_point' in transition:
        print(f"Logged EEF position: {logged_eef}")
    
    if show_trajectory and len(trajectory) > 0 and frame_idx - initial_frame_idx < len(trajectory):
        current_traj_pos = trajectory[frame_idx - initial_frame_idx]
        print(f"Current trajectory EEF position: {current_traj_pos}")
        
        # Calculate difference
        if 'gripper_point' in transition:
            diff = np.linalg.norm(logged_eef - current_traj_pos)
            print(f"Position difference: {diff:.6f} meters")
    
    # Print pointcloud statistics
    total_points = sum(len(points) for points in labels_dict.values())
    print(f"Total pointcloud points: {total_points}")
    for label, points in labels_dict.items():
        print(f"  {label.capitalize()}: {len(points)} points")
    
    # Visualize
    print(f"\nVisualizing frame {frame_idx} with {len(geometries)} geometries...")
    o3d.visualization.draw_geometries(geometries, 
                                     window_name=f"Frame {frame_idx} - Pointcloud with Trajectory (up to current frame)",
                                     width=1200, height=800)

def plot_frame_with_trajectory(transitions: List[Dict[str, Any]], 
                             frame_idx: int,
                             trajectory: List[np.ndarray],
                             initial_frame_idx: int = 0,
                             show_trajectory: bool = True,
                             show_legend: bool = True) -> None:
    """
    Plot a single frame's pointcloud with trajectory overlay using matplotlib.
    Shows trajectory only up to the current frame and highlights EEF in pointcloud.
    
    Args:
        transitions: List of transition dictionaries
        frame_idx: Frame index to plot
        trajectory: Computed trajectory positions
        initial_frame_idx: Frame index where trajectory started
        show_trajectory: Whether to show trajectory overlay
        show_legend: Whether to show legend
    """
    if frame_idx >= len(transitions):
        print(f"Frame {frame_idx} not available (max: {len(transitions)-1})")
        return
    
    transition = transitions[frame_idx]
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract pointcloud data - handle new format
    labels_dict = {'arm': np.array([]), 'cyan': np.array([]), 'eef': np.array([])}
    data_format = "unknown"
    
    if 'pcd_combined_positions' in transition and 'pcd_combined_features' in transition:
        # New format: numpy arrays
        combined_positions = transition['pcd_combined_positions']
        combined_features = transition['pcd_combined_features']
        labels_dict = extract_pointcloud_labels_from_numpy(combined_positions, combined_features)
        data_format = "numpy_arrays"
        print(f"Using new numpy format: {len(combined_positions)} points")
    elif 'pcd_combined' in transition:
        # Legacy format: torch_geometric Data
        pcd_data = transition['pcd_combined']
        labels_dict = extract_pointcloud_labels(pcd_data)
        data_format = "torch_geometric"
        print(f"Using legacy torch_geometric format")
    
    # Plot different point types with different colors
    colors = {
        'arm': 'blue',
        'cyan': 'cyan', 
        'eef': 'red'
    }
    
    markers = {
        'arm': 'o',
        'cyan': 's',
        'eef': '*'
    }
    
    sizes = {
        'arm': 10,
        'cyan': 20,
        'eef': 100
    }
    
    # Plot arm and cyan points first
    for label in ['arm', 'cyan']:
        points = labels_dict[label]
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors[label], marker=markers[label], s=sizes[label], 
                      alpha=0.7, label=f'{label.capitalize()} ({len(points)} points)')
    
    # Plot EEF points from pointcloud with large marker
    eef_points = labels_dict['eef']
    if len(eef_points) > 0:
        ax.scatter(eef_points[:, 0], eef_points[:, 1], eef_points[:, 2], 
                  c='red', marker='*', s=200, alpha=1.0, 
                  label=f'EEF in PCD ({len(eef_points)} points)', edgecolors='black', linewidth=2)
        print(f"EEF points in pointcloud: {len(eef_points)}")
        for i, eef_pos in enumerate(eef_points):
            print(f"  EEF point {i}: {eef_pos}")
    
    # Plot logged EEF position
    if 'gripper_point' in transition:
        logged_eef = np.array(transition['gripper_point'])
        ax.scatter(logged_eef[0], logged_eef[1], logged_eef[2], 
                  c='green', marker='D', s=150, alpha=1.0, 
                  label=f'Logged EEF (frame {frame_idx})')
    
    # Plot trajectory only up to current frame
    if show_trajectory and len(trajectory) > 0:
        # Calculate how many trajectory points to show
        trajectory_idx = frame_idx - initial_frame_idx
        if trajectory_idx >= 0 and trajectory_idx < len(trajectory):
            # Show trajectory up to current frame
            trajectory_up_to_frame = trajectory[:trajectory_idx + 1]
            trajectory_array = np.array(trajectory_up_to_frame)
            
            if len(trajectory_array) > 1:
                # Plot trajectory line up to current frame
                ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 
                       'g-', linewidth=3, alpha=0.8, label=f'Computed Trajectory (up to frame {frame_idx})')
                
                # Plot trajectory points up to current frame
                ax.scatter(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 
                          c='green', marker='o', s=50, alpha=0.8)
            
            # Highlight current position on trajectory with large marker
            current_traj_pos = trajectory[trajectory_idx]
            ax.scatter(current_traj_pos[0], current_traj_pos[1], current_traj_pos[2], 
                      c='yellow', marker='D', s=300, alpha=1.0, edgecolors='black', linewidth=2,
                      label=f'Current Trajectory EEF (frame {frame_idx})')
            
            print(f"Current trajectory position: {current_traj_pos}")
    
    # Set plot properties
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f'Frame {frame_idx} - Pointcloud with Trajectory (up to current frame)\n'
                f'Format: {data_format}, Coordinate Frame: {transition.get("coordinate_frame", "model_axes")}')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    if show_legend:
        ax.legend()
    
    # Print detailed coordinate frame analysis
    print(f"\n=== Frame {frame_idx} Analysis ===")
    print(f"Data format: {data_format}")
    print(f"Coordinate frame: {transition.get('coordinate_frame', 'model_axes')}")
    print(f"Alignment approach: {transition.get('alignment_approach', 'dressing_environment_style')}")
    
    if 'gripper_point' in transition:
        print(f"Logged EEF position: {logged_eef}")
    
    if show_trajectory and len(trajectory) > 0 and frame_idx - initial_frame_idx < len(trajectory):
        current_traj_pos = trajectory[frame_idx - initial_frame_idx]
        print(f"Trajectory EEF position: {current_traj_pos}")
        
        # Calculate difference
        if 'gripper_point' in transition:
            diff = np.linalg.norm(logged_eef - current_traj_pos)
            print(f"Position difference: {diff:.6f} meters")
            
            # Detailed difference analysis
            diff_xyz = logged_eef - current_traj_pos
            print(f"  X difference: {diff_xyz[0]:.6f} meters")
            print(f"  Y difference: {diff_xyz[1]:.6f} meters") 
            print(f"  Z difference: {diff_xyz[2]:.6f} meters")
            
            if diff < 0.01:
                print("✅ EXCELLENT ALIGNMENT: Difference < 1cm")
            elif diff < 0.05:
                print("✅ GOOD ALIGNMENT: Difference < 5cm")
            elif diff < 0.1:
                print("⚠️  MODERATE ALIGNMENT: Difference < 10cm")
            else:
                print("❌ POOR ALIGNMENT: Difference > 10cm")
    
    # Print pointcloud statistics
    total_points = sum(len(points) for points in labels_dict.values())
    print(f"Total pointcloud points: {total_points}")
    for label, points in labels_dict.items():
        print(f"  {label.capitalize()}: {len(points)} points")
    
    # Print delta movement information
    if 'delta_pos_magnitude' in transition:
        print(f"Delta movement: pos={transition['delta_pos_magnitude']:.6f}m, ori={transition['delta_ori_magnitude']:.6f}rad")
    
    plt.tight_layout()
    plt.show()

def plot_trajectory_comparison(transitions: List[Dict[str, Any]], 
                             initial_frame_idx: int = 0) -> None:
    """
    Plot comparison between logged EEF positions and computed trajectory.
    
    Args:
        transitions: List of transition dictionaries
        initial_frame_idx: Frame index to start trajectory from
    """
    if len(transitions) == 0:
        return
    
    # Extract logged EEF positions
    logged_positions = []
    frame_indices = []
    
    for i, transition in enumerate(transitions):
        if 'gripper_point' in transition:
            logged_positions.append(np.array(transition['gripper_point']))
            frame_indices.append(i)
    
    logged_positions = np.array(logged_positions)
    
    # Compute trajectory
    trajectory = compute_trajectory_from_deltas(transitions, initial_frame_idx)
    trajectory = np.array(trajectory)
    
    # Create comparison plot
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 3D trajectory comparison
    ax_3d = fig.add_subplot(2, 3, 1, projection='3d')
    
    if len(logged_positions) > 0:
        ax_3d.scatter(logged_positions[:, 0], logged_positions[:, 1], logged_positions[:, 2], 
                     c='red', marker='o', s=50, alpha=0.7, label='Logged EEF')
    
    if len(trajectory) > 0:
        ax_3d.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                     c='blue', marker='s', s=50, alpha=0.7, label='Computed Trajectory')
    
    ax_3d.set_xlabel('X (meters)')
    ax_3d.set_ylabel('Y (meters)')
    ax_3d.set_zlabel('Z (meters)')
    ax_3d.set_title('3D Trajectory Comparison')
    ax_3d.legend()
    
    # Plot X, Y, Z components over time
    time_axis = np.arange(len(logged_positions))
    
    for i, coord_name in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(2, 3, i + 2)
        
        if len(logged_positions) > 0:
            ax.plot(time_axis, logged_positions[:, i], 'ro-', label='Logged EEF', alpha=0.7)
        
        if len(trajectory) > 0:
            traj_time = np.arange(len(trajectory))
            ax.plot(traj_time, trajectory[:, i], 'bs-', label='Computed Trajectory', alpha=0.7)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel(f'{coord_name} (meters)')
        ax.set_title(f'{coord_name} Component Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n=== Trajectory Comparison Statistics ===")
    print(f"Logged positions: {len(logged_positions)}")
    print(f"Computed trajectory points: {len(trajectory)}")
    
    if len(logged_positions) > 0 and len(trajectory) > 0:
        min_len = min(len(logged_positions), len(trajectory))
        differences = np.linalg.norm(logged_positions[:min_len] - trajectory[:min_len], axis=1)
        
        print(f"Mean position difference: {np.mean(differences):.6f} meters")
        print(f"Max position difference: {np.max(differences):.6f} meters")
        print(f"Std position difference: {np.std(differences):.6f} meters")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Pointcloud Visualizer with Coordinate Frame Validation')
    parser.add_argument('--episode-file', type=str, required=True, help='Path to episode pickle file')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to visualize (default: 0)')
    parser.add_argument('--initial-frame', type=int, default=0, help='Initial frame for trajectory computation (default: 0)')
    parser.add_argument('--no-trajectory', action='store_true', help='Disable trajectory overlay')
    parser.add_argument('--comparison', action='store_true', help='Show trajectory comparison plot')
    parser.add_argument('--no-legend', action='store_true', help='Hide legend')
    parser.add_argument('--open3d', action='store_true', help='Use Open3D for 3D visualization (better for pointclouds)')
    parser.add_argument('--validate-alignment', action='store_true', help='Run comprehensive alignment validation')
    args = parser.parse_args()
    
    # Load episode data
    print(f"Loading episode data from: {args.episode_file}")
    episode_data = load_episode_data(args.episode_file)
    
    transitions = episode_data['transitions']
    print(f"Loaded {len(transitions)} transitions")
    
    # Print episode metadata
    print(f"\n=== Episode Metadata ===")
    print(f"Episode number: {episode_data.get('episode_num', 'unknown')}")
    print(f"Task: {episode_data.get('task', 'unknown')}")
    print(f"Robot type: {episode_data.get('robot_type', 'unknown')}")
    
    # Check data format and coordinate frame information
    if len(transitions) > 0:
        first_transition = transitions[0]
        
        # Check data format
        if 'pcd_combined_positions' in first_transition and 'pcd_combined_features' in first_transition:
            data_format = "numpy_arrays"
            num_points = len(first_transition['pcd_combined_positions'])
            num_features = first_transition['pcd_combined_features'].shape[1]
            print(f"Data format: {data_format}")
            print(f"Pointcloud: {num_points} points with {num_features} features")
        elif 'pcd_combined' in first_transition:
            data_format = "torch_geometric"
            pcd_data = first_transition['pcd_combined']
            if hasattr(pcd_data, 'x') and hasattr(pcd_data, 'pos'):
                num_points = len(pcd_data.pos)
                num_features = pcd_data.x.shape[1] if len(pcd_data.x.shape) > 1 else 1
                print(f"Data format: {data_format}")
                print(f"Pointcloud: {num_points} points with {num_features} features")
                
                # Check if it's the simplified format
                if num_features == 5:
                    print("✅ Detected simplified format: [R, G, B, cyan_onehot, gripper_onehot]")
                elif num_features == 3:
                    print("✅ Detected legacy format: [arm_onehot, cyan_onehot, gripper_onehot]")
                else:
                    print(f"⚠️  Unknown feature format: {num_features} features")
            else:
                print(f"Data format: {data_format} (invalid torch_geometric object)")
        else:
            data_format = "unknown"
            print(f"Data format: {data_format} (unknown)")
        
        # Check coordinate frame
        coord_frame = first_transition.get('coordinate_frame', 'model_axes')
        alignment_approach = first_transition.get('alignment_approach', 'dressing_environment_style')
        print(f"Coordinate frame: {coord_frame}")
        print(f"Alignment approach: {alignment_approach}")
        
        # Check for simplified format indicators
        if 'simplifications' in episode_data:
            print(f"Simplifications applied: {episode_data['simplifications']}")
        
        # Validate coordinate frame consistency
        print(f"\n=== Coordinate Frame Validation ===")
        coord_frames = set()
        alignment_approaches = set()
        
        for i, transition in enumerate(transitions):
            coord_frames.add(transition.get('coordinate_frame', 'model_axes'))
            alignment_approaches.add(transition.get('alignment_approach', 'dressing_environment_style'))
            
            # Check for data format consistency
            if data_format == "numpy_arrays":
                if 'pcd_combined_positions' not in transition or 'pcd_combined_features' not in transition:
                    print(f"⚠️  Warning: Frame {i} missing numpy array data")
            elif data_format == "torch_geometric":
                if 'pcd_combined' not in transition:
                    print(f"⚠️  Warning: Frame {i} missing torch_geometric data")
        
        if len(coord_frames) == 1:
            print(f"✅ Coordinate frame consistency: All frames use {list(coord_frames)[0]}")
        else:
            print(f"❌ Coordinate frame inconsistency: Found {coord_frames}")
        
        if len(alignment_approaches) == 1:
            print(f"✅ Alignment approach consistency: All frames use {list(alignment_approaches)[0]}")
        else:
            print(f"❌ Alignment approach inconsistency: Found {alignment_approaches}")
    
    # Run comprehensive alignment validation if requested
    if args.validate_alignment:
        print(f"\n=== Comprehensive Alignment Validation ===")
        validate_coordinate_frame_alignment(transitions)
    
    # Compute trajectory
    print(f"\n=== Computing Trajectory ===")
    print(f"Starting from frame {args.initial_frame}")
    trajectory = compute_trajectory_from_deltas(transitions, args.initial_frame)
    
    # Plot frame with trajectory
    if args.open3d and OPEN3D_AVAILABLE:
        plot_frame_with_trajectory_open3d(
            transitions, 
            args.frame, 
            trajectory, 
            args.initial_frame,
            show_trajectory=not args.no_trajectory
        )
    else:
        plot_frame_with_trajectory(
            transitions, 
            args.frame, 
            trajectory, 
            args.initial_frame,
            show_trajectory=not args.no_trajectory,
            show_legend=not args.no_legend
        )
    
    # Show trajectory comparison if requested
    if args.comparison:
        plot_trajectory_comparison(transitions, args.initial_frame)

def validate_coordinate_frame_alignment(transitions: List[Dict[str, Any]]) -> None:
    """
    Comprehensive validation of coordinate frame alignment and consistency.
    
    Args:
        transitions: List of transition dictionaries
    """
    if len(transitions) == 0:
        return
    
    print("Running comprehensive alignment validation...")
    
    # 1. Check EEF position consistency
    print("\n1. EEF Position Consistency:")
    eef_positions = []
    for i, transition in enumerate(transitions):
        if 'gripper_point' in transition:
            eef_positions.append(np.array(transition['gripper_point']))
        else:
            print(f"  ⚠️  Frame {i}: Missing gripper_point")
    
    if len(eef_positions) > 1:
        eef_positions = np.array(eef_positions)
        eef_movement = np.linalg.norm(eef_positions[1:] - eef_positions[:-1], axis=1)
        print(f"  Total EEF movement: {np.sum(eef_movement):.4f} meters")
        print(f"  Average movement per frame: {np.mean(eef_movement):.4f} meters")
        print(f"  Max movement in single frame: {np.max(eef_movement):.4f} meters")
        
        if np.max(eef_movement) > 0.5:
            print(f"  ⚠️  Warning: Large EEF movement detected (>0.5m)")
    
    # 2. Check pointcloud data consistency
    print("\n2. Pointcloud Data Consistency:")
    point_counts = []
    feature_counts = []
    
    for i, transition in enumerate(transitions):
        if 'pcd_combined_positions' in transition and 'pcd_combined_features' in transition:
            positions = transition['pcd_combined_positions']
            features = transition['pcd_combined_features']
            point_counts.append(len(positions))
            feature_counts.append(features.shape[1])
        else:
            print(f"  ⚠️  Frame {i}: Missing pointcloud data")
    
    if point_counts:
        point_counts = np.array(point_counts)
        feature_counts = np.array(feature_counts)
        print(f"  Average points per frame: {np.mean(point_counts):.1f}")
        print(f"  Points range: {np.min(point_counts)} - {np.max(point_counts)}")
        print(f"  Features per point: {feature_counts[0]} (should be 3 for simplified format)")
        
        if len(set(feature_counts)) > 1:
            print(f"  ❌ Inconsistent feature counts: {set(feature_counts)}")
        else:
            print(f"  ✅ Consistent feature counts")
    
    # 3. Check delta movement consistency
    print("\n3. Delta Movement Consistency:")
    pos_deltas = []
    ori_deltas = []
    
    for i, transition in enumerate(transitions):
        if 'delta_6d' in transition:
            delta_6d = transition['delta_6d']
            pos_delta = delta_6d[:3]
            ori_delta = delta_6d[3:]
            pos_deltas.append(np.linalg.norm(pos_delta))
            ori_deltas.append(np.linalg.norm(ori_delta))
        else:
            print(f"  ⚠️  Frame {i}: Missing delta_6d")
    
    if pos_deltas:
        pos_deltas = np.array(pos_deltas)
        ori_deltas = np.array(ori_deltas)
        print(f"  Average position delta: {np.mean(pos_deltas):.6f} meters")
        print(f"  Average orientation delta: {np.mean(ori_deltas):.6f} radians")
        print(f"  Max position delta: {np.max(pos_deltas):.6f} meters")
        print(f"  Max orientation delta: {np.max(ori_deltas):.6f} radians")
        
        if np.max(pos_deltas) > 0.1:
            print(f"  ⚠️  Warning: Large position delta detected (>0.1m)")
    
    # 4. Check coordinate frame transformation
    print("\n4. Coordinate Frame Transformation:")
    coord_frames = set()
    alignment_approaches = set()
    
    for transition in transitions:
        coord_frames.add(transition.get('coordinate_frame', 'model_axes'))
        alignment_approaches.add(transition.get('alignment_approach', 'dressing_environment_style'))
    
    print(f"  Coordinate frames found: {coord_frames}")
    print(f"  Alignment approaches found: {alignment_approaches}")
    
    if 'model_axes' in coord_frames:
        print(f"  ✅ Using model_axes transformation (dressing environment style)")
    else:
        print(f"  ⚠️  Not using model_axes transformation")
    
    # 5. Check pointcloud centering
    print("\n5. Pointcloud Centering Validation:")
    if len(transitions) > 0 and 'pcd_combined_positions' in transitions[0]:
        # Check if pointclouds are centered on EEF
        sample_frame = transitions[0]
        positions = sample_frame['pcd_combined_positions']
        eef_pos = sample_frame['gripper_point']
        
        if len(positions) > 0:
            # Check if there's a point very close to origin (centered)
            distances_from_origin = np.linalg.norm(positions, axis=1)
            min_distance = np.min(distances_from_origin)
            
            if min_distance < 0.01:
                print(f"  ✅ Pointcloud appears to be centered (min distance from origin: {min_distance:.6f}m)")
            else:
                print(f"  ⚠️  Pointcloud may not be centered (min distance from origin: {min_distance:.6f}m)")
    
    print("\n=== Validation Complete ===")

if __name__ == "__main__":
    main() 