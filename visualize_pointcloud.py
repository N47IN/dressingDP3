#!/usr/bin/env python3
"""
Visualize DP3 Pointcloud Data using Open3D
==========================================

This script loads and visualizes the pointcloud data collected by the DP3 expert data collector.
It shows the 3D pointcloud with RGB colors in an interactive viewer.
"""

import numpy as np
import open3d as o3d
import json
from pathlib import Path
import argparse

def load_dp3_episode(episode_path):
    """Load DP3 episode data"""
    episode_path = Path(episode_path)
    
    # Load metadata
    with open(episode_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load pointcloud data - it's a regular numpy array, not object array
    pointcloud_data = np.load(episode_path / "point_cloud.npy", allow_pickle=True)
    
    # Load other data for reference
    actions = np.load(episode_path / "actions.npy")
    agent_pos = np.load(episode_path / "agent_pos.npy")
    rewards = np.load(episode_path / "rewards.npy")
    
    print(f"Loaded episode from: {episode_path}")
    print(f"Episode length: {metadata['episode_length']}")
    print(f"Pointcloud shape: {pointcloud_data.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Agent positions shape: {agent_pos.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Success: {metadata.get('success', False)}")
    print(f"Total reward: {metadata.get('total_reward', 'N/A')}")
    
    return pointcloud_data, metadata

def visualize_pointcloud_frame(pointcloud, frame_idx=0, title="DP3 Pointcloud"):
    """Visualize a single frame of pointcloud data"""
    if frame_idx >= len(pointcloud):
        print(f"Frame {frame_idx} not available. Max frames: {len(pointcloud)}")
        return
    
    # Get the pointcloud for this frame
    frame_points = pointcloud[frame_idx]
    
    if len(frame_points) == 0:
        print(f"No points in frame {frame_idx}")
        return
    
    print(f"Visualizing frame {frame_idx} with {len(frame_points)} points")
    print(f"Pointcloud shape: {frame_points.shape}")
    print(f"Point range: X[{frame_points[:, 0].min():.3f}, {frame_points[:, 0].max():.3f}]")
    print(f"Point range: Y[{frame_points[:, 1].min():.3f}, {frame_points[:, 1].max():.3f}]")
    print(f"Point range: Z[{frame_points[:, 2].min():.3f}, {frame_points[:, 2].max():.3f}]")
    print(f"RGB range: [{frame_points[:, 3:].min():.3f}, {frame_points[:, 3:].max():.3f}]")
    
    # Create Open3D pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame_points[:, :3])  # XYZ coordinates
    pcd.colors = o3d.utility.Vector3dVector(frame_points[:, 3:])  # RGB colors
    
    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name=f"{title} - Frame {frame_idx}")

def visualize_multiple_frames(pointcloud, frame_indices=[0, 25, 50, 75], title="DP3 Pointcloud"):
    """Visualize multiple frames side by side"""
    geometries = []
    
    for i, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(pointcloud):
            continue
            
        frame_points = pointcloud[frame_idx]
        if len(frame_points) == 0:
            continue
        
        # Create Open3D pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame_points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(frame_points[:, 3:])
        
        # Translate each frame to avoid overlap
        translation = np.array([i * 0.5, 0, 0])  # Move each frame 0.5 units in X direction
        pcd.translate(translation)
        
        geometries.append(pcd)
    
    if geometries:
        o3d.visualization.draw_geometries(geometries, window_name=f"{title} - Multiple Frames")

def main():
    parser = argparse.ArgumentParser(description="Visualize DP3 pointcloud data")
    parser.add_argument("--episode", type=str, default="dp3_dataset/episode_0000", 
                       help="Path to episode directory")
    parser.add_argument("--frame", type=int, default=0, 
                       help="Frame index to visualize (default: 0)")
    parser.add_argument("--multi", action="store_true", 
                       help="Visualize multiple frames")
    parser.add_argument("--frames", nargs='+', type=int, default=[0, 25, 50, 75],
                       help="Frame indices to visualize in multi-frame mode")
    
    args = parser.parse_args()
    
    try:
        # Load episode data
        pointcloud_data, metadata = load_dp3_episode(args.episode)
        
        if args.multi:
            visualize_multiple_frames(pointcloud_data, args.frames)
        else:
            visualize_pointcloud_frame(pointcloud_data, args.frame)
            
    except Exception as e:
        print(f"Error loading episode: {e}")
        print("Make sure you have collected data using dp3_expert_data_collector.py first")

if __name__ == "__main__":
    main() 