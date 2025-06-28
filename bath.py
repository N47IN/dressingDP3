#!/usr/bin/env python3
"""
Optimized RGB-D Camera Visualization for Bathing the Right Hand
=============================================================

This script creates the environment and visualizes RGB-D images from a camera
positioned to best view the human's right hand for bathing tasks.
"""

import gym
import assistive_gym
import pybullet as p
import numpy as np
import cv2
import time

def get_right_hand_pos(env):
    """Get the world position of the human's right hand (link 9)"""
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
    return rgb_img, depth_img

def main():
    print("Creating BedBathingSawyer-v0 environment...")
    env = gym.make('BedBathingSawyer-v0')
    env.render()
    print("Environment created successfully!")
    observation = env.reset()
    print(f"Environment reset. Observation shape: {observation.shape}")
    
    # Main loop
    frame_count = 0
    try:
        while True:
            # Step the environment
            step_result = env.step(env.action_space.sample())
            
            # Handle both old and new gym versions
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                truncated = False
            else:
                obs, reward, done, truncated, info = step_result
            
            # Get RGB and depth images
            camera_config = setup_camera_aimed_at_right_hand(env)
            rgb_img, depth_img = get_rgb_depth_images(env, camera_config)
            
            # Save images
            cv2.imwrite(f'rgb_frame_{frame_count:04d}.png', rgb_img)
            cv2.imwrite(f'depth_frame_{frame_count:04d}.png', depth_img)
            
            # Display images
            cv2.imshow('RGB View', rgb_img)
            cv2.imshow('Depth View', depth_img)
            
            # Print frame info
            print(f"Saved frame {frame_count}: RGB and depth images")
            
            frame_count += 1
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping data collection...")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print(f"Collected {frame_count} frames")

if __name__ == "__main__":
    main() 