#!/usr/bin/env python3
"""
Debug script to investigate horizontal offset between pointcloud and EEF coordinates
"""

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

# Import functions from the data collector
from dp3_expert_data_collector import (
    setup_camera_aimed_at_right_hand,
    get_rgb_depth_images,
    get_right_hand_pos,
    create_cyan_points_raw,
    create_arm_points_from_segmentation,
    get_end_effector_pose
)

def debug_horizontal_offset():
    """Debug function to investigate horizontal offset between pointcloud and EEF"""
    
    # Create the environment
    env = gym.make('BedBathingSawyer-v0')
    env.reset()
    
    print("=== INVESTIGATING HORIZONTAL OFFSET ===")
    
    # Get camera setup
    camera_config = setup_camera_aimed_at_right_hand(env)
    print(f"Camera position: {camera_config['position']}")
    print(f"Camera target: {camera_config['target']}")
    
    # Get camera images and matrices
    rgb_img, depth_img, seg_img, view_matrix, projection_matrix = get_rgb_depth_images(env, camera_config)
    
    # Get tool position (EEF)
    tool_pos, tool_orn = get_end_effector_pose(env)
    print(f"Tool position: {tool_pos}")
    
    # Get human hand position for reference
    hand_pos = get_right_hand_pos(env)
    print(f"Human hand position: {hand_pos}")
    
    # Test different coordinate transformation approaches
    print(f"\n=== TESTING COORDINATE TRANSFORMATIONS ===")
    
    view_matrix_np = np.array(view_matrix).reshape(4, 4)
    projection_matrix_np = np.array(projection_matrix).reshape(4, 4)
    
    print(f"View matrix:\n{view_matrix_np}")
    print(f"Projection matrix:\n{projection_matrix_np}")
    
    # Test 1: Direct camera-to-world transformation
    print(f"\n--- Test 1: Direct camera-to-world ---")
    camera_to_world_direct = np.linalg.inv(view_matrix_np)
    print(f"Camera-to-world matrix:\n{camera_to_world_direct}")
    
    # Test 2: Check if we need to use projection matrix
    print(f"\n--- Test 2: Using projection matrix ---")
    # The projection matrix is used for perspective projection, not coordinate transformation
    # But let's check if there's an issue with the view matrix interpretation
    
    # Test 3: Check if the view matrix is actually world-to-camera or camera-to-world
    print(f"\n--- Test 3: View matrix interpretation ---")
    # In PyBullet, p.computeViewMatrix returns world-to-camera transformation
    # So camera_to_world should be the inverse
    
    # Test 4: Create a test point at the tool position and transform it
    print(f"\n--- Test 4: Transform tool position ---")
    tool_pos_homog = np.concatenate([tool_pos, [1]])
    tool_pos_camera = view_matrix_np @ tool_pos_homog
    tool_pos_camera = tool_pos_camera[:3] / tool_pos_camera[3]  # Perspective divide
    print(f"Tool position in camera coordinates: {tool_pos_camera}")
    
    # Transform back
    tool_pos_back = camera_to_world_direct @ np.concatenate([tool_pos_camera, [1]])
    tool_pos_back = tool_pos_back[:3] / tool_pos_back[3]
    print(f"Tool position transformed back: {tool_pos_back}")
    print(f"Transformation error: {np.linalg.norm(tool_pos - tool_pos_back)}")
    
    # Test 5: Check if the issue is in the pointcloud creation
    print(f"\n--- Test 5: Pointcloud coordinate system ---")
    
    # Create cyan pointcloud without cropping to see the full range
    cyan_pointcloud = create_cyan_points_raw(depth_img, rgb_img, seg_img, camera_config, view_matrix, projection_matrix, env, downsample_factor=1)
    
    if len(cyan_pointcloud.pos) > 0:
        cyan_pos = cyan_pointcloud.pos.cpu().numpy()
        print(f"Cyan pointcloud full range:")
        print(f"  X: [{np.min(cyan_pos[:, 0]):.3f}, {np.max(cyan_pos[:, 0]):.3f}]")
        print(f"  Y: [{np.min(cyan_pos[:, 1]):.3f}, {np.max(cyan_pos[:, 1]):.3f}]")
        print(f"  Z: [{np.min(cyan_pos[:, 2]):.3f}, {np.max(cyan_pos[:, 2]):.3f}]")
        
        # Check if tool is within the full range
        tool_in_range_x = np.min(cyan_pos[:, 0]) <= tool_pos[0] <= np.max(cyan_pos[:, 0])
        tool_in_range_y = np.min(cyan_pos[:, 1]) <= tool_pos[1] <= np.max(cyan_pos[:, 1])
        tool_in_range_z = np.min(cyan_pos[:, 2]) <= tool_pos[2] <= np.max(cyan_pos[:, 2])
        
        print(f"Tool within full cyan range: X={tool_in_range_x}, Y={tool_in_range_y}, Z={tool_in_range_z}")
        
        if not tool_in_range_x:
            print(f"  X offset: tool at {tool_pos[0]:.3f}, range is [{np.min(cyan_pos[:, 0]):.3f}, {np.max(cyan_pos[:, 0]):.3f}]")
            print(f"  X offset magnitude: {abs(tool_pos[0] - np.mean(cyan_pos[:, 0])):.3f}")
    
    # Test 6: Check if the issue is in the depth-to-3D conversion
    print(f"\n--- Test 6: Depth-to-3D conversion ---")
    
    # Get camera intrinsics
    height, width = camera_config['height'], camera_config['width']
    fx = fy = width / (2 * np.tan(np.radians(camera_config['fov']) / 2))
    cx, cy = width / 2, height / 2
    
    print(f"Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # Test a pixel at the center of the image
    center_x, center_y = width // 2, height // 2
    center_depth = depth_img[center_y, center_x]
    print(f"Center pixel ({center_x}, {center_y}) depth: {center_depth:.3f}")
    
    # Convert to 3D camera coordinates
    x_cam = (center_x - cx) * center_depth / fx
    y_cam = (center_y - cy) * center_depth / fy
    z_cam = center_depth
    
    center_point_cam = np.array([x_cam, y_cam, z_cam])
    print(f"Center point in camera coordinates: {center_point_cam}")
    
    # Transform to world coordinates
    center_point_homog = np.concatenate([center_point_cam, [1]])
    center_point_world = camera_to_world_direct @ center_point_homog
    center_point_world = center_point_world[:3] / center_point_world[3]
    print(f"Center point in world coordinates: {center_point_world}")
    
    # Test 7: Check if there's a systematic offset in the transformation
    print(f"\n--- Test 7: Systematic offset check ---")
    
    # Create a grid of test points in world coordinates
    test_points_world = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-1, 2):
                point = tool_pos + np.array([i*0.1, j*0.1, k*0.1])
                test_points_world.append(point)
    
    test_points_world = np.array(test_points_world)
    
    # Transform to camera coordinates
    test_points_homog = np.concatenate([test_points_world, np.ones((len(test_points_world), 1))], axis=1)
    test_points_cam = (view_matrix_np @ test_points_homog.T).T
    test_points_cam = test_points_cam[:, :3] / test_points_cam[:, 3:4]
    
    # Transform back to world coordinates
    test_points_back_homog = np.concatenate([test_points_cam, np.ones((len(test_points_cam), 1))], axis=1)
    test_points_back = (camera_to_world_direct @ test_points_back_homog.T).T
    test_points_back = test_points_back[:, :3] / test_points_back[:, 3:4]
    
    # Check transformation errors
    errors = np.linalg.norm(test_points_world - test_points_back, axis=1)
    print(f"Transformation errors - Mean: {np.mean(errors):.6f}, Max: {np.max(errors):.6f}")
    
    # Check if there's a systematic offset
    offsets = test_points_world - test_points_back
    mean_offset = np.mean(offsets, axis=0)
    print(f"Mean systematic offset: {mean_offset}")
    
    env.close()

if __name__ == "__main__":
    debug_horizontal_offset() 