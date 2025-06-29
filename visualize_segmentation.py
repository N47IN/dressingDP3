import gym
import assistive_gym
import pybullet as p
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import random

def setup_camera_aimed_at_right_hand(env, offset=np.array([0.0, 0.4, 0.9])):
    """Place the camera at a fixed position above the right hand for a focused top-down view."""
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

def get_right_hand_pos(env):
    """Get the position of the human's right hand"""
    return np.array(p.getLinkState(env.human, 9, computeForwardKinematics=True, physicsClientId=env.id)[0])

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

def get_all_objects_and_links(env):
    """Get all objects and their link information in the environment"""
    objects_info = {}
    
    # Get all bodies in the environment
    all_bodies = []
    for i in range(p.getNumBodies(physicsClientId=env.id)):
        body_id = p.getBodyUniqueId(i, physicsClientId=env.id)
        all_bodies.append(body_id)
    
    # Get information for each body
    for body_id in all_bodies:
        try:
            body_name = p.getBodyInfo(body_id, physicsClientId=env.id)[1].decode('utf-8')
        except:
            body_name = f"Body_{body_id}"
        
        objects_info[body_id] = {
            'name': body_name,
            'links': {}
        }
        
        # Get all links for this body
        num_joints = p.getNumJoints(body_id, physicsClientId=env.id)
        for link_idx in range(-1, num_joints):  # -1 is the base link
            try:
                if link_idx == -1:
                    link_name = "base"
                else:
                    joint_info = p.getJointInfo(body_id, link_idx, physicsClientId=env.id)
                    link_name = joint_info[12].decode('utf-8') if isinstance(joint_info[12], bytes) else str(joint_info[12])
                
                objects_info[body_id]['links'][link_idx] = link_name
            except:
                objects_info[body_id]['links'][link_idx] = f"link_{link_idx}"
    
    return objects_info

def create_all_links_visualization(seg_img, objects_info):
    """Create visualization of all objects and their links in the environment"""
    seg_flat = seg_img.flatten()
    
    # Get all unique segmentation values and their corresponding objects/links
    all_links = {}
    for seg_val in np.unique(seg_flat):
        object_id = seg_val & ((1 << 24) - 1)
        link_index = (seg_val >> 24) - 1
        
        if object_id in objects_info:
            object_name = objects_info[object_id]['name']
            link_name = objects_info[object_id]['links'].get(link_index, f"link_{link_index}")
            
            pixel_count = np.sum(seg_flat == seg_val)
            if pixel_count > 0:  # Only include links that are visible
                all_links[seg_val] = {
                    'object_id': object_id,
                    'object_name': object_name,
                    'link_index': link_index,
                    'link_name': link_name,
                    'pixel_count': pixel_count,
                    'mask': seg_img == seg_val
                }
    
    # Create figure with subplots
    num_links = len(all_links)
    if num_links == 0:
        print("No visible objects found!")
        return None, {}
    
    cols = 6
    rows = (num_links + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(24, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Create random colors for each link
    colors = plt.cm.Set3(np.linspace(0, 1, num_links))
    
    # Plot each link
    for i, (seg_val, link_data) in enumerate(sorted(all_links.items())):
        row = i // cols
        col = i % cols
        
        ax = axes[row, col]
        
        # Create colored mask
        colored_mask = np.zeros((*seg_img.shape, 3))
        colored_mask[link_data['mask']] = colors[i][:3]
        
        ax.imshow(colored_mask)
        title = f"{link_data['object_name']}\nLink {link_data['link_index']}: {link_data['link_name']}\n({link_data['pixel_count']} pixels)"
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_links, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig, all_links

def create_object_summary_visualization(rgb_img, seg_img, all_links):
    """Create summary visualization grouped by objects"""
    # Group links by object
    objects_summary = {}
    for seg_val, link_data in all_links.items():
        obj_id = link_data['object_id']
        if obj_id not in objects_summary:
            objects_summary[obj_id] = {
                'name': link_data['object_name'],
                'total_pixels': 0,
                'links': [],
                'mask': np.zeros_like(seg_img, dtype=bool)
            }
        
        objects_summary[obj_id]['total_pixels'] += link_data['pixel_count']
        objects_summary[obj_id]['links'].append(link_data)
        objects_summary[obj_id]['mask'] |= link_data['mask']
    
    # Create visualization
    num_objects = len(objects_summary)
    cols = 4
    rows = (num_objects + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Create random colors for each object
    colors = plt.cm.tab10(np.linspace(0, 1, num_objects))
    
    # Plot each object
    for i, (obj_id, obj_data) in enumerate(sorted(objects_summary.items())):
        row = i // cols
        col = i % cols
        
        ax = axes[row, col]
        
        # Create colored mask for entire object
        colored_mask = np.zeros((*seg_img.shape, 3))
        colored_mask[obj_data['mask']] = colors[i][:3]
        
        ax.imshow(colored_mask)
        title = f"{obj_data['name']}\n({obj_data['total_pixels']} pixels)\n{len(obj_data['links'])} links"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_objects, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig, objects_summary

def create_right_arm_visualization(rgb_img, seg_img, human_body_id, right_arm_links=[5, 7, 9]):
    """Create visualization specifically for right arm links"""
    seg_flat = seg_img.flatten()
    
    # Create mask for right arm
    obj_mask = (seg_flat & ((1 << 24) - 1)) == human_body_id
    link_idx = (seg_flat >> 24) - 1
    arm_mask = np.zeros_like(link_idx, dtype=bool)
    for link in right_arm_links:
        arm_mask |= (link_idx == link)
    right_arm_mask = obj_mask & arm_mask
    
    # Reshape mask back to image dimensions
    right_arm_mask = right_arm_mask.reshape(seg_img.shape)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original RGB image
    axes[0].imshow(rgb_img)
    axes[0].set_title('Original RGB Image')
    axes[0].axis('off')
    
    # Right arm mask overlay
    overlay = rgb_img.copy()
    overlay[right_arm_mask] = [255, 0, 0]  # Red for right arm
    axes[1].imshow(overlay)
    axes[1].set_title(f'Right Arm Mask (Links {right_arm_links})\n({np.sum(right_arm_mask)} pixels)')
    axes[1].axis('off')
    
    # Right arm mask only
    axes[2].imshow(right_arm_mask, cmap='Reds')
    axes[2].set_title('Right Arm Mask Only')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig, right_arm_mask

def main():
    print("Creating BedBathingSawyer-v0 environment...")
    env = gym.make('BedBathingSawyer-v0')
    env.reset()
    
    print(f"Human body ID: {env.human}")
    print(f"Robot ID: {env.robot}")
    
    # Get all objects and their links
    print("Getting all objects and links...")
    objects_info = get_all_objects_and_links(env)
    
    print(f"Found {len(objects_info)} objects in the environment:")
    for obj_id, obj_data in objects_info.items():
        print(f"  {obj_id}: {obj_data['name']} ({len(obj_data['links'])} links)")
    
    # Setup camera
    camera_config = setup_camera_aimed_at_right_hand(env)
    print(f"Camera position: {camera_config['position']}")
    print(f"Camera target: {camera_config['target']}")
    
    # Get images
    rgb_img, depth_img, seg_img, view_matrix, projection_matrix = get_rgb_depth_images(env, camera_config)
    
    print(f"Image shapes: RGB={rgb_img.shape}, Depth={depth_img.shape}, Seg={seg_img.shape}")
    
    # Analyze segmentation mask
    seg_flat = seg_img.flatten()
    unique_seg_values = np.unique(seg_flat)
    print(f"Unique segmentation values: {len(unique_seg_values)}")
    
    # Check human visibility
    human_pixels = np.sum((seg_flat & ((1 << 24) - 1)) == env.human)
    print(f"Total human pixels: {human_pixels}")
    
    if human_pixels == 0:
        print("WARNING: No human pixels found!")
    
    # Create visualizations
    print("Creating all links visualizations...")
    
    # 1. All links in the environment
    fig1, all_links = create_all_links_visualization(seg_img, objects_info)
    if fig1 is not None:
        fig1.savefig('all_links_visualization.png', dpi=150, bbox_inches='tight')
        print("Saved: all_links_visualization.png")
    
    # 2. Object summary visualization
    fig2, objects_summary = create_object_summary_visualization(rgb_img, seg_img, all_links)
    fig2.savefig('objects_summary_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved: objects_summary_visualization.png")
    
    # 3. Right arm specific (if human is visible)
    if human_pixels > 0:
        fig3, right_arm_mask = create_right_arm_visualization(rgb_img, seg_img, env.human)
        fig3.savefig('right_arm_visualization.png', dpi=150, bbox_inches='tight')
        print("Saved: right_arm_visualization.png")
    
    # 4. Summary plot
    fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # RGB image
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')
    
    # Depth image
    depth_viz = axes[0, 1].imshow(depth_img, cmap='viridis')
    axes[0, 1].set_title('Depth Image')
    axes[0, 1].axis('off')
    plt.colorbar(depth_viz, ax=axes[0, 1])
    
    # All objects overlay
    all_objects_mask = np.zeros_like(seg_img, dtype=bool)
    for seg_val, link_data in all_links.items():
        all_objects_mask |= link_data['mask']
    
    axes[1, 0].imshow(all_objects_mask, cmap='Reds')
    axes[1, 0].set_title(f'All Objects\n({np.sum(all_objects_mask)} pixels)')
    axes[1, 0].axis('off')
    
    # Right arm parts (if human is visible)
    if human_pixels > 0:
        right_arm_mask = create_right_arm_visualization(rgb_img, seg_img, env.human)[1]
        axes[1, 1].imshow(right_arm_mask, cmap='Reds')
        axes[1, 1].set_title(f'Right Arm Parts (Links 5,7,9)\n({np.sum(right_arm_mask)} pixels)')
    else:
        axes[1, 1].text(0.5, 0.5, 'Human not visible', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Right Arm Parts')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    fig4.savefig('segmentation_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: segmentation_summary.png")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total image pixels: {seg_img.size}")
    print(f"Total visible objects: {len(objects_summary)}")
    print(f"Total visible links: {len(all_links)}")
    
    if human_pixels > 0:
        print(f"Human pixels: {human_pixels} ({100*human_pixels/seg_img.size:.1f}%)")
        right_arm_mask = create_right_arm_visualization(rgb_img, seg_img, env.human)[1]
        print(f"Right arm pixels: {np.sum(right_arm_mask)} ({100*np.sum(right_arm_mask)/seg_img.size:.1f}%)")
    
    # Show object breakdown
    print("\nObject breakdown:")
    for obj_id, obj_data in sorted(objects_summary.items()):
        print(f"  {obj_data['name']} (ID: {obj_id}): {obj_data['total_pixels']} pixels, {len(obj_data['links'])} links")
    
    # Show link breakdown
    print("\nLink breakdown:")
    for seg_val, link_data in sorted(all_links.items()):
        print(f"  {link_data['object_name']} - Link {link_data['link_index']} ({link_data['link_name']}): {link_data['pixel_count']} pixels")
    
    plt.show()
    env.close()

if __name__ == "__main__":
    main() 