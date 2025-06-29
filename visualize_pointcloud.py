#!/usr/bin/env python3
"""
DP3 Pointcloud Visualizer - All Pointclouds in One Window
========================================================

Shows all pointclouds (cyan, arm, tool tip, combined) in one window with EEF trajectory.
Supports frame-by-frame navigation and coordinate frame debugging.
"""

import numpy as np
import open3d as o3d
import pickle
from pathlib import Path
import argparse
import torch

def load_episode(episode_path):
    """Load episode data from pickle file"""
    episode_path = Path(episode_path)
    
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_path}")
    
    with open(episode_path, 'rb') as f:
        episode_data = pickle.load(f)
    
    if 'transitions' not in episode_data:
        raise ValueError(f"Invalid episode file: 'transitions' key not found")
    
    transitions = episode_data['transitions']
    print(f"Loaded {len(transitions)} frames from {episode_path}")
    
    return transitions

def extract_eef_trajectory(transitions):
    """Extract end-effector trajectory from all transitions"""
    trajectory = []
    valid_frames = []
    
    for i, transition in enumerate(transitions):
        if 'gripper_point' in transition:
            gripper_point = np.array(transition['gripper_point'])
            trajectory.append(gripper_point)
            valid_frames.append(i)
    
    if len(trajectory) == 0:
        return None, None
    
    trajectory = np.array(trajectory)
    return trajectory, valid_frames

def create_trajectory_line(trajectory, color=[1.0, 0.5, 0.0], width=3, max_frame=None):
    """Create a line visualization of the EEF trajectory up to max_frame"""
    if len(trajectory) < 2:
        return None
    
    # Limit trajectory to max_frame if specified
    if max_frame is not None:
        trajectory = trajectory[:max_frame+1]
    
    if len(trajectory) < 2:
        return None
    
    # Create line segments
    lines = []
    colors = []
    
    for i in range(len(trajectory) - 1):
        lines.append([i, i + 1])
        colors.append(color)
    
    # Create line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(trajectory)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def create_trajectory_points(trajectory, radius=0.003, color=[1.0, 0.5, 0.0], max_frame=None):
    """Create spheres at each trajectory point up to max_frame"""
    if len(trajectory) == 0:
        return None
    
    # Limit trajectory to max_frame if specified
    if max_frame is not None:
        trajectory = trajectory[:max_frame+1]
    
    if len(trajectory) == 0:
        return None
    
    spheres = []
    for point in trajectory:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
    
    return spheres

def create_current_frame_marker(trajectory, current_frame_idx, valid_frames, radius=0.01, color=[0.0, 1.0, 0.0]):
    """Create a marker for the current frame position"""
    if len(valid_frames) == 0 or current_frame_idx >= len(valid_frames):
        return None
    
    # Find the trajectory index for current frame
    traj_idx = valid_frames.index(current_frame_idx) if current_frame_idx in valid_frames else 0
    if traj_idx >= len(trajectory):
        return None
    
    current_pos = trajectory[traj_idx]
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(current_pos)
    sphere.paint_uniform_color(color)
    
    return sphere

def create_eef_coordinate_frame(gripper_point, scale=0.05):
    """Create a coordinate frame for the end-effector pose"""
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    coordinate_frame.translate(gripper_point)
    return coordinate_frame

def create_eef_sphere(gripper_point, radius=0.01, color=[1.0, 0.0, 0.0]):
    """Create a sphere to mark the end-effector position"""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(gripper_point)
    sphere.paint_uniform_color(color)
    return sphere

def create_world_coordinate_frame(scale=0.1):
    """Create a world coordinate frame at origin"""
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    return coordinate_frame

def create_pointcloud_bounds(points, color=[0.5, 0.5, 0.5]):
    """Create a bounding box around the pointcloud"""
    if len(points) == 0:
        return None
    
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    bbox.color = color
    
    return bbox

def show_all_pointclouds_in_one_window(transition, frame_idx, show_coordinate_frames=True, show_bounds=True, 
                                      trajectory=None, valid_frames=None, show_trajectory=True):
    """Show all pointclouds in one window with EEF trajectory"""
    
    geometries = []
    
    # Get all pointcloud data
    pcd_data = {}
    pcd_types = ["pcd_cyan", "pcd_arm", "pcd_tool_tip", "pcd_combined"]
    
    for pcd_type in pcd_types:
        if pcd_type in transition:
            pcd_data[pcd_type] = transition[pcd_type]
        else:
            print(f"Warning: {pcd_type} not found in frame {frame_idx}")
    
    # Create Open3D pointclouds with different colors
    colors = {
        'pcd_cyan': [0.0, 0.0, 1.0],      # Blue
        'pcd_arm': [0.0, 1.0, 0.0],       # Green  
        'pcd_tool_tip': [1.0, 0.0, 0.0],  # Red
        'pcd_combined': [1.0, 1.0, 0.0]   # Yellow
    }
    
    labels = {
        'pcd_cyan': 'Cyan Points',
        'pcd_arm': 'Arm Points', 
        'pcd_tool_tip': 'Tool Tip Points',
        'pcd_combined': 'Combined Points'
    }
    
    print(f"\nFrame {frame_idx} - All Pointclouds:")
    
    for pcd_type in pcd_types:
        if pcd_type in pcd_data and hasattr(pcd_data[pcd_type], 'pos') and len(pcd_data[pcd_type].pos) > 0:
            pos = pcd_data[pcd_type].pos.cpu().numpy()
            features = pcd_data[pcd_type].x.cpu().numpy() if hasattr(pcd_data[pcd_type], 'x') else None
            
            # Create Open3D pointcloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pos)
            
            # Set colors based on pointcloud type
            if pcd_type == "pcd_combined" and features is not None and features.shape[1] >= 5:
                # For combined pointcloud, use distinct colors for different point types
                colors_combined = np.zeros((len(pos), 3))
                
                # Get masks for different point types
                cyan_mask = features[:, 3] > 0.5
                gripper_mask = features[:, 4] > 0.5
                
                # Color scheme for combined pointcloud:
                # - Cyan points: Blue [0, 0, 1]
                # - Gripper points: Red [1, 0, 0] 
                # - Regular arm points: Green [0, 1, 0]
                
                # Regular arm points (not cyan, not gripper) - Green
                regular_mask = ~cyan_mask & ~gripper_mask
                colors_combined[regular_mask] = [0.0, 1.0, 0.0]  # Green
                
                # Cyan points - Blue
                colors_combined[cyan_mask] = [0.0, 0.0, 1.0]  # Blue
                
                # Gripper points - Red
                colors_combined[gripper_mask] = [1.0, 0.0, 0.0]  # Red
                
                pcd.colors = o3d.utility.Vector3dVector(colors_combined)
                
                # Print feature statistics
                num_cyan = np.sum(cyan_mask)
                num_gripper = np.sum(gripper_mask)
                num_regular = np.sum(regular_mask)
                print(f"  {labels[pcd_type]}: {len(pos)} points (🟢{num_regular} 🟦{num_cyan} 🔴{num_gripper})")
                
            else:
                # Use uniform color for other pointclouds
                color = colors[pcd_type]
                colors_uniform = np.full((len(pos), 3), color)
                pcd.colors = o3d.utility.Vector3dVector(colors_uniform)
                print(f"  {labels[pcd_type]}: {len(pos)} points")
            
            geometries.append(pcd)
            
            # Add bounding box if requested
            if show_bounds:
                bounds = create_pointcloud_bounds(pos, color=[0.3, 0.3, 0.3])
                if bounds is not None:
                    geometries.append(bounds)
    
    # Add trajectory visualization
    if show_trajectory and trajectory is not None and len(trajectory) > 0:
        # Find the trajectory index for current frame
        current_traj_idx = None
        if valid_frames is not None and frame_idx in valid_frames:
            current_traj_idx = valid_frames.index(frame_idx)
        
        # Add trajectory line (up to current frame)
        trajectory_line = create_trajectory_line(trajectory, max_frame=current_traj_idx)
        if trajectory_line is not None:
            geometries.append(trajectory_line)
        
        # Add trajectory points (up to current frame)
        trajectory_points = create_trajectory_points(trajectory, radius=0.003, color=[1.0, 0.5, 0.0], max_frame=current_traj_idx)
        if trajectory_points is not None:
            geometries.extend(trajectory_points)
        
        # Add current frame marker
        current_marker = create_current_frame_marker(trajectory, frame_idx, valid_frames or [])
        if current_marker is not None:
            geometries.append(current_marker)
    
    # Add world coordinate frame
    if show_coordinate_frames:
        world_frame = create_world_coordinate_frame(scale=0.1)
        geometries.append(world_frame)
    
    # Add end-effector visualization if available
    if 'gripper_point' in transition:
        gripper_point = np.array(transition['gripper_point'])
        eef_sphere = create_eef_sphere(gripper_point, radius=0.015, color=[1.0, 0.0, 0.0])
        geometries.append(eef_sphere)
        
        if show_coordinate_frames:
            eef_frame = create_eef_coordinate_frame(gripper_point, scale=0.03)
            geometries.append(eef_frame)
        
        print(f"  🔴 EEF Position: {gripper_point}")
    
    # Add action and reward information
    if 'action' in transition:
        action = transition['action']
        print(f"  Action: {action}")
    
    if 'reward' in transition:
        reward = transition['reward']
        print(f"  Reward: {reward}")
    
    if 'total_force' in transition:
        total_force = transition['total_force']
        print(f"  Total Force: {total_force}")
    
    # Print coordinate frame info
    all_points = []
    for pcd_type in pcd_types:
        if pcd_type in pcd_data and hasattr(pcd_data[pcd_type], 'pos') and len(pcd_data[pcd_type].pos) > 0:
            all_points.append(pcd_data[pcd_type].pos.cpu().numpy())
    
    if all_points:
        all_points_combined = np.vstack(all_points)
        print(f"  Overall bounds:")
        print(f"    X: [{np.min(all_points_combined[:, 0]):.3f}, {np.max(all_points_combined[:, 0]):.3f}]")
        print(f"    Y: [{np.min(all_points_combined[:, 1]):.3f}, {np.max(all_points_combined[:, 1]):.3f}]")
        print(f"    Z: [{np.min(all_points_combined[:, 2]):.3f}, {np.max(all_points_combined[:, 2]):.3f}]")
    
    # Print trajectory info
    if trajectory is not None and len(trajectory) > 0:
        # Find the trajectory index for current frame
        current_traj_idx = None
        if valid_frames is not None and frame_idx in valid_frames:
            current_traj_idx = valid_frames.index(frame_idx)
        
        if current_traj_idx is not None:
            # Show trajectory up to current frame
            trajectory_up_to_current = trajectory[:current_traj_idx+1]
            print(f"  Trajectory (up to frame {frame_idx}): {len(trajectory_up_to_current)}/{len(trajectory)} points")
            print(f"    Progress: {current_traj_idx+1}/{len(trajectory)} ({100*(current_traj_idx+1)/len(trajectory):.1f}%)")
            
            if len(trajectory_up_to_current) > 1:
                # Calculate distance traveled so far
                distance_so_far = 0
                for i in range(len(trajectory_up_to_current) - 1):
                    distance_so_far += np.linalg.norm(trajectory_up_to_current[i+1] - trajectory_up_to_current[i])
                print(f"    Distance traveled: {distance_so_far:.3f} meters")
            
            print(f"    Bounds (up to current):")
            print(f"      X: [{np.min(trajectory_up_to_current[:, 0]):.3f}, {np.max(trajectory_up_to_current[:, 0]):.3f}]")
            print(f"      Y: [{np.min(trajectory_up_to_current[:, 1]):.3f}, {np.max(trajectory_up_to_current[:, 1]):.3f}]")
            print(f"      Z: [{np.min(trajectory_up_to_current[:, 2]):.3f}, {np.max(trajectory_up_to_current[:, 2]):.3f}]")
        else:
            # Show full trajectory info
            print(f"  Trajectory: {len(trajectory)} points")
            print(f"    X: [{np.min(trajectory[:, 0]):.3f}, {np.max(trajectory[:, 0]):.3f}]")
            print(f"    Y: [{np.min(trajectory[:, 1]):.3f}, {np.max(trajectory[:, 1]):.3f}]")
            print(f"    Z: [{np.min(trajectory[:, 2]):.3f}, {np.max(trajectory[:, 2]):.3f}]")
    
    # Visualize
    window_name = f"Frame {frame_idx} - All Pointclouds"
    o3d.visualization.draw_geometries(geometries, window_name=window_name)

def interactive_viewer(transitions, show_coordinate_frames=True, show_bounds=True, show_trajectory=True):
    """Interactive viewer for navigating through frames"""
    current_frame = 0
    total_frames = len(transitions)
    
    # Extract trajectory
    trajectory, valid_frames = extract_eef_trajectory(transitions)
    if trajectory is not None:
        print(f"Extracted EEF trajectory with {len(trajectory)} points")
    else:
        print("No EEF trajectory data found")
        show_trajectory = False
    
    print(f"\nInteractive Viewer - {total_frames} frames available")
    print("Commands:")
    print("  n/next - Next frame")
    print("  p/prev - Previous frame")
    print("  f <num> - Go to frame <num>")
    print("  c - Toggle coordinate frames")
    print("  b - Toggle bounding boxes")
    print("  t - Toggle trajectory")
    print("  q/quit - Exit")
    print("  h/help - Show this help")
    
    while True:
        print(f"\nCurrent frame: {current_frame}/{total_frames-1}")
        print(f"Coordinate frames: {'ON' if show_coordinate_frames else 'OFF'}")
        print(f"Bounding boxes: {'ON' if show_bounds else 'OFF'}")
        print(f"Trajectory: {'ON' if show_trajectory else 'OFF'}")
        
        # Show all pointclouds in one window
        show_all_pointclouds_in_one_window(transitions[current_frame], current_frame, 
                                         show_coordinate_frames, show_bounds, 
                                         trajectory, valid_frames, show_trajectory)
        
        # Get user input
        try:
            command = input("Enter command: ").strip().lower()
            
            if command in ['q', 'quit', 'exit']:
                print("Exiting viewer...")
                break
            elif command in ['n', 'next']:
                current_frame = min(current_frame + 1, total_frames - 1)
            elif command in ['p', 'prev', 'previous']:
                current_frame = max(current_frame - 1, 0)
            elif command.startswith('f '):
                try:
                    frame_num = int(command.split()[1])
                    if 0 <= frame_num < total_frames:
                        current_frame = frame_num
                    else:
                        print(f"Frame {frame_num} out of range [0, {total_frames-1}]")
                except (ValueError, IndexError):
                    print("Invalid frame number")
            elif command == 'c':
                show_coordinate_frames = not show_coordinate_frames
                print(f"Coordinate frames: {'ON' if show_coordinate_frames else 'OFF'}")
            elif command == 'b':
                show_bounds = not show_bounds
                print(f"Bounding boxes: {'ON' if show_bounds else 'OFF'}")
            elif command == 't':
                show_trajectory = not show_trajectory
                print(f"Trajectory: {'ON' if show_trajectory else 'OFF'}")
            elif command in ['h', 'help']:
                print("Commands:")
                print("  n/next - Next frame")
                print("  p/prev - Previous frame")
                print("  f <num> - Go to frame <num>")
                print("  c - Toggle coordinate frames")
                print("  b - Toggle bounding boxes")
                print("  t - Toggle trajectory")
                print("  q/quit - Exit")
                print("  h/help - Show this help")
            else:
                print("Unknown command. Type 'h' for help.")
                
        except KeyboardInterrupt:
            print("\nExiting viewer...")
            break
        except EOFError:
            print("\nExiting viewer...")
            break

def main():
    parser = argparse.ArgumentParser(description="DP3 pointcloud viewer - All pointclouds in one window")
    parser.add_argument("--episode", type=str, default="dp3_transitions_dataset/episode_0000_transitions.pkl", 
                       help="Path to episode file")
    parser.add_argument("--frame", type=int, default=None, help="Specific frame to show (if not specified, starts interactive mode)")
    parser.add_argument("--interactive", action="store_true", help="Start in interactive mode")
    parser.add_argument("--no-coordinate-frames", action="store_true", help="Hide coordinate frames")
    parser.add_argument("--no-bounds", action="store_true", help="Hide bounding boxes")
    parser.add_argument("--no-trajectory", action="store_true", help="Hide trajectory")
    
    args = parser.parse_args()
    
    try:
        transitions = load_episode(args.episode)
        
        show_coordinate_frames = not args.no_coordinate_frames
        show_bounds = not args.no_bounds
        show_trajectory = not args.no_trajectory
        
        if args.frame is not None:
            # Show specific frame
            if args.frame >= len(transitions):
                print(f"Frame {args.frame} not available. Max frames: {len(transitions)}")
                return
        
            # Extract trajectory for single frame view
            trajectory, valid_frames = extract_eef_trajectory(transitions)
            
            # Show all pointclouds in one window
            show_all_pointclouds_in_one_window(transitions[args.frame], args.frame, 
                                             show_coordinate_frames, show_bounds,
                                             trajectory, valid_frames, show_trajectory)
        else:
            # Interactive mode
            interactive_viewer(transitions, show_coordinate_frames, show_bounds, show_trajectory)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 