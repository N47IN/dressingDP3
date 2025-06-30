#!/usr/bin/env python3
"""
Script to convert DP3 expert data collector dataset to payload format for pc_replay_buffer.

This script adapts the save_as_payload function from pc_replay_buffer.py to work with
the dataset format created by dp3_expert_data_collector.py.

Updated for 2-feature combined pointcloud format:
- Feature 0: Cyan one-hot encoding (1.0 for cyan points, 0.0 for others)
- Feature 1: Gripper one-hot encoding (1.0 for gripper/tool points, 0.0 for others)

Action options:
- 'joint': Use 7-DOF joint space actions (original behavior)
- 'delta_eef': Use delta end-effector movements (6D: pos_delta(3D) + rot_delta(3D rotation vector)) - DEFAULT
- 'delta_pos_only': Use only position deltas (3D)

Usage:
    python convert_dataset_to_payload.py --dataset-dir dp3_transitions_dataset --output-path converted_payload.pt
"""

import os
import pickle
import argparse
import numpy as np
import torch
import copy
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data

def validate_combined_pointcloud(pcd, expected_features=5):
    """
    Validate that a combined pointcloud has the expected 5-feature format (from collection).
    
    Args:
        pcd: torch_geometric.Data object
        expected_features: Expected number of features from collection (default: 5)
    
    Returns:
        is_valid: Boolean indicating if the pointcloud is valid
        message: Description of validation result
    """
    if not hasattr(pcd, 'x') or not hasattr(pcd, 'pos'):
        return False, "Missing 'x' or 'pos' attributes"
    
    if pcd.x.shape[1] != expected_features:
        return False, f"Expected {expected_features} features from collection, got {pcd.x.shape[1]}"
    
    if pcd.pos.shape[1] != 3:
        return False, f"Expected 3D positions, got {pcd.pos.shape[1]}D"
    
    if len(pcd.pos) == 0:
        return False, "Empty pointcloud"
    
    # Check feature ranges for 5-feature format
    rgb_features = pcd.x[:, :3]  # First 3 features should be RGB
    cyan_features = pcd.x[:, 3]  # 4th feature should be cyan one-hot
    gripper_features = pcd.x[:, 4]  # 5th feature should be gripper one-hot
    
    # RGB should be in [0, 1] range
    if torch.any(rgb_features < 0) or torch.any(rgb_features > 1):
        return False, "RGB features not in [0, 1] range"
    
    # Cyan and gripper features should be binary (0 or 1)
    if torch.any((cyan_features != 0) & (cyan_features != 1)):
        return False, "Cyan features not binary (0 or 1)"
    
    if torch.any((gripper_features != 0) & (gripper_features != 1)):
        return False, "Gripper features not binary (0 or 1)"
    
    return True, "Valid 5-feature combined pointcloud from collection"

def extract_features_for_model(pcd, feature_mode='two_features'):
    """
    Extract features from 5-feature combined pointcloud for model input.
    By default, extracts only cyan and gripper one-hot features (features 3 and 4).
    
    Args:
        pcd: torch_geometric.Data object with 5 features from collection
        feature_mode: Which features to extract ('two_features', 'cyan_only', 'gripper_only', 'all_5', 'rgb_only', 'rgb_cyan', 'rgb_gripper')
    
    Returns:
        processed_pcd: torch_geometric.Data object with extracted features
    """
    if pcd.x.shape[1] != 5:
        raise ValueError(f"Expected 5 features from collection, got {pcd.x.shape[1]}")
    
    if feature_mode == 'two_features':
        # Extract only cyan and gripper one-hot features (2 features) - DEFAULT
        # Feature 0: cyan one-hot, Feature 1: gripper one-hot
        features = pcd.x[:, 3:5]  # Take features 3 and 4 (cyan and gripper)
    elif feature_mode == 'cyan_only':
        # Extract only cyan one-hot feature (1 feature)
        features = pcd.x[:, 3:4]  # Take only feature 3 (cyan)
    elif feature_mode == 'gripper_only':
        # Extract only gripper one-hot feature (1 feature)
        features = pcd.x[:, 4:5]  # Take only feature 4 (gripper)
    elif feature_mode == 'all_5':
        # Keep all 5 features
        features = pcd.x
    elif feature_mode == 'rgb_only':
        # Extract only RGB features (first 3) - compatible with standard PointNet++
        features = pcd.x[:, :3]
    elif feature_mode == 'rgb_cyan':
        # RGB + cyan one-hot (4 features)
        features = torch.cat([pcd.x[:, :3], pcd.x[:, 3:4]], dim=1)
    elif feature_mode == 'rgb_gripper':
        # RGB + gripper one-hot (4 features)
        features = torch.cat([pcd.x[:, :3], pcd.x[:, 4:5]], dim=1)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    
    return Data(pos=pcd.pos, x=features)

def process_action(transition, action_type='delta_eef', previous_transition=None):
    """
    Process action from transition based on the specified action type.
    
    Args:
        transition: Dictionary containing transition data
        action_type: Type of action to extract ('joint', 'delta_eef', 'delta_pos_only')
        previous_transition: Previous transition for calculating deltas (not used when using pre-computed data)
    
    Returns:
        action: Processed action array
    """
    if action_type == 'joint':
        # Use joint space actions (original behavior)
        action = transition['action']
        if len(action.shape) == 1:
            # If action is 1D, reshape to 6D (assuming 7-DOF joint space)
            if len(action) == 7:
                # Take first 6 DOF for 6D action
                action = action[:6]
            elif len(action) != 6:
                # Pad or truncate to 6D
                if len(action) < 6:
                    action = np.pad(action, (0, 6 - len(action)), 'constant')
                else:
                    action = action[:6]
        return action
    
    elif action_type == 'delta_pos_only':
        # Use pre-computed position delta from data collection
        if 'delta_pos' in transition:
            delta_pos = np.array(transition['delta_pos'])
            return delta_pos
        else:
            # Fallback: calculate from stored poses
            if previous_transition is None:
                return np.zeros(3)
            current_pos = np.array(transition['gripper_point'])
            previous_pos = np.array(previous_transition['gripper_point'])
            delta_pos = current_pos - previous_pos
            return delta_pos
    
    elif action_type == 'delta_eef':
        # Use pre-computed 6D delta EEF from data collection - MUCH BETTER!
        if 'delta_6d' in transition:
            print("----------delta time------------")
            delta_6d = np.array(transition['delta_6d'])
            return delta_6d
        else:
            # Fallback: calculate from stored poses (old method)
            if previous_transition is None:
                return np.zeros(6)
            
            # Get current and previous EEF poses
            current_pos = np.array(transition['gripper_point'])
            current_ori = np.array(transition['gripper_orientation'])
            previous_pos = np.array(previous_transition['gripper_point'])
            previous_ori = np.array(previous_transition['gripper_orientation'])
            print("breh recomputing")
            # Calculate position delta
            delta_pos = current_pos - previous_pos
            
            # Calculate orientation delta as axis-angle using scipy
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
                
            except ImportError:
                # Fallback: compute axis-angle directly from quaternions
                # For small rotations, we can approximate
                dot_product = np.dot(current_ori, previous_ori)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                
                angle = 2 * np.arccos(abs(dot_product))
                
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
            
            return delta_6d
    
    else:
        raise ValueError(f"Unknown action_type: {action_type}")

def filter_pointcloud_by_z(pcd, max_z=1.2):
    """
    Filter pointcloud to remove points where z > max_z.
    
    Args:
        pcd: torch_geometric.Data object
        max_z: Maximum z-coordinate value (default: 1.2)
    
    Returns:
        filtered_pcd: Filtered pointcloud
    """
    if len(pcd.pos) == 0:
        return pcd
    
    # Create mask for points where z <= max_z
    z_mask = pcd.pos[:, 2] <= max_z
    
    if not torch.any(z_mask):
        # If no points remain, return empty pointcloud with same structure
        return Data(
            pos=torch.zeros((1, 3), dtype=pcd.pos.dtype),
            x=torch.zeros((1, pcd.x.shape[1]), dtype=pcd.x.dtype)
        )
    
    # Filter points and features
    filtered_pos = pcd.pos[z_mask]
    filtered_x = pcd.x[z_mask]
    
    return Data(pos=filtered_pos, x=filtered_x)

def convert_dataset_to_payload(dataset_dir, output_path, feature_mode='two_features', action_type='delta_eef', validate_pointclouds=True, max_z=1.2):
    """
    Convert DP3 expert data collector dataset to payload format.
    
    Args:
        dataset_dir: Directory containing episode pickle files
        output_path: Path to save the converted payload
        feature_mode: Which features to extract ('two_features', 'cyan_only', 'gripper_only', 'all_5', 'rgb_only', 'rgb_cyan', 'rgb_gripper')
        action_type: Type of action to use ('joint', 'delta_eef', 'delta_pos_only')
        validate_pointclouds: Whether to validate pointcloud format before processing
        max_z: Maximum z-coordinate for pointcloud filtering (default: 1.2)
    """
    print(f"Converting dataset from {dataset_dir} to {output_path}")
    print(f"Feature mode: {feature_mode}")
    print(f"Action type: {action_type}")
    print(f"Validation: {'enabled' if validate_pointclouds else 'disabled'}")
    print(f"Z-filtering: max_z = {max_z}")
    
    # Get all episode files
    dataset_path = Path(dataset_dir)
    episode_files = sorted([f for f in dataset_path.glob("episode_*_transitions.pkl")])
    
    if not episode_files:
        raise ValueError(f"No episode files found in {dataset_dir}")
    
    print(f"Found {len(episode_files)} episode files")
    
    # Initialize lists for payload
    obses = []
    next_obses = []
    actions = []
    rewards = []
    not_dones = []
    non_randomized_obses = []
    forces = []
    ori_obses = []
    force_vectors = []
    reward_obses = []
    next_reward_obses = []
    
    total_transitions = 0
    skipped_transitions = 0
    feature_dim_stats = []
    point_count_stats = []
    action_stats = {
        'action_dimensions': [],
        'delta_pos_magnitudes': [],
        'delta_ori_magnitudes': []
    }
    validation_stats = {
        'valid_5_feature': 0,
        'invalid_format': 0,
        'missing_combined': 0,
        'empty_pointcloud': 0,
        'missing_delta_data': 0
    }
    
    for episode_file in tqdm(episode_files, desc="Processing episodes"):
        try:
            with open(episode_file, 'rb') as f:
                episode_data = pickle.load(f)
            
            transitions = episode_data['transitions']
            episode_num = episode_data['episode_num']
            print(f"Processing episode {episode_num} with {len(transitions)} transitions")
            
            for i in range(len(transitions)):
                transition = transitions[i]
                next_transition = transitions[i+1] if i < len(transitions)-1 else transitions[i]
                
                # Use the pre-computed combined pointcloud
                if 'pcd_combined' in transition:
                    obs = copy.deepcopy(transition['pcd_combined'])
                    next_obs = copy.deepcopy(next_transition.get('pcd_combined', obs))
                else:
                    validation_stats['missing_combined'] += 1
                    print(f"  Warning: No pcd_combined found in transition {i} of episode {episode_num}")
                    skipped_transitions += 1
                    continue
                
                # Filter pointclouds to remove points where z > 1.2
                obs = filter_pointcloud_by_z(obs, max_z=max_z)
                next_obs = filter_pointcloud_by_z(next_obs, max_z=max_z)
                
                # Validate pointcloud format if enabled
                if validate_pointclouds:
                    is_valid, message = validate_combined_pointcloud(obs)
                    if not is_valid:
                        validation_stats['invalid_format'] += 1
                        print(f"  Warning: Invalid pointcloud in transition {i} of episode {episode_num}: {message}")
                        skipped_transitions += 1
                        continue
                    else:
                        validation_stats['valid_5_feature'] += 1
                
                # Check for empty pointclouds
                if len(obs.pos) == 0:
                    validation_stats['empty_pointcloud'] += 1
                    print(f"  Warning: Empty pointcloud in transition {i} of episode {episode_num}")
                    skipped_transitions += 1
                    continue
                
                # Check for required data if using delta actions
                if action_type in ['delta_eef', 'delta_pos_only']:
                    if action_type == 'delta_eef' and 'delta_6d' not in transition:
                        validation_stats['missing_delta_data'] += 1
                        print(f"  Warning: No delta_6d found in transition {i} of episode {episode_num}")
                        skipped_transitions += 1
                        continue
                    
                    if action_type == 'delta_pos_only' and 'delta_pos' not in transition:
                        validation_stats['missing_delta_data'] += 1
                        print(f"  Warning: No delta_pos found in transition {i} of episode {episode_num}")
                        skipped_transitions += 1
                        continue
                
                # Record original feature dimensions for statistics
                original_features = obs.x.shape[1]
                feature_dim_stats.append(original_features)
                point_count_stats.append(len(obs.pos))
                
                # Extract features according to the specified mode
                try:
                    obs = extract_features_for_model(obs, feature_mode)
                    next_obs = extract_features_for_model(next_obs, feature_mode)
                except Exception as e:
                    print(f"  Error extracting features in transition {i} of episode {episode_num}: {e}")
                    skipped_transitions += 1
                    continue
                
                # Validate final pointcloud structure
                if not hasattr(obs, 'x') or not hasattr(obs, 'pos'):
                    print(f"  Warning: Invalid pointcloud structure in transition {i} of episode {episode_num}")
                    skipped_transitions += 1
                    continue
                
                if obs.pos.shape[1] != 3:
                    print(f"  Error: Expected 3D positions, got {obs.pos.shape[1]}D")
                    skipped_transitions += 1
                    continue
                
                # Add observations
                obses.append(obs)
                next_obses.append(next_obs)
                
                # Process action based on action_type
                try:
                    # Get previous transition for delta calculations
                    previous_transition = transitions[i-1] if i > 0 else None
                    action = process_action(transition, action_type, previous_transition)
                    
                    # Ensure action is 6D for pc_replay_buffer compatibility
                    if len(action) == 3:  # delta_pos_only
                        # Pad with zeros for rotation components
                        action = np.concatenate([action, np.zeros(3)])
                    elif len(action) != 6:
                        # Pad or truncate to 6D
                        if len(action) < 6:
                            action = np.pad(action, (0, 6 - len(action)), 'constant')
                        else:
                            action = action[:6]
                    
                    # Ensure action is float32 for consistency
                    action = action.astype(np.float32)
                    
                    actions.append(action)
                    action_stats['action_dimensions'].append(len(action))
                    
                    # Record delta statistics if available
                    if 'delta_pos_magnitude' in transition:
                        action_stats['delta_pos_magnitudes'].append(transition['delta_pos_magnitude'])
                    if 'delta_ori_magnitude' in transition:
                        action_stats['delta_ori_magnitudes'].append(transition['delta_ori_magnitude'])
                        
                except Exception as e:
                    print(f"  Error processing action in transition {i} of episode {episode_num}: {e}")
                    skipped_transitions += 1
                    continue
                
                # Process reward and done flag
                reward = transition.get('reward', 0.0)
                rewards.append(reward)
                
                # Set not_done flag (1.0 for all except last transition)
                not_done = 1.0 if i < len(transitions) - 1 else 0.0
                not_dones.append(not_done)
                
                # Set non_randomized_obs to None (not used in this context)
                non_randomized_obses.append(None)
                
                # Process force information
                total_force = transition.get('total_force', 0.0)
                forces.append(total_force)
                
                # Process force vectors
                force_vectors_data = transition.get('force_vectors', np.zeros(3, dtype=np.float32))
                if isinstance(force_vectors_data, np.ndarray):
                    force_vectors.append(torch.from_numpy(force_vectors_data).float())
                else:
                    force_vectors.append(torch.tensor(force_vectors_data, dtype=torch.float32))
                
                # Set ori_obses to None (not used in this context)
                ori_obses.append(None)
                
                # For reward_obses and next_reward_obses, we'll use the same pointclouds
                # but with a different name to match the expected format
                reward_obses.append(copy.deepcopy(obs))
                next_reward_obses.append(copy.deepcopy(next_obs))
                
                total_transitions += 1
                
        except Exception as e:
            print(f"Error processing {episode_file}: {e}")
            continue
    
    print(f"Processed {total_transitions} total transitions")
    print(f"Skipped {skipped_transitions} transitions")
    
    # Print validation statistics
    print(f"\nValidation Statistics:")
    for key, value in validation_stats.items():
        print(f"  {key}: {value}")
    
    # Print feature dimension statistics
    if feature_dim_stats:
        unique_dims = np.unique(feature_dim_stats)
        print(f"\nOriginal feature dimensions found: {unique_dims}")
        for dim in unique_dims:
            count = feature_dim_stats.count(dim)
            print(f"  {dim}-feature data: {count} transitions")
    
    # Print point count statistics
    if point_count_stats:
        print(f"\nPoint count statistics:")
        print(f"  Average points per observation: {np.mean(point_count_stats):.1f}")
        print(f"  Min points: {min(point_count_stats)}")
        print(f"  Max points: {max(point_count_stats)}")
        print(f"  Std dev points: {np.std(point_count_stats):.1f}")
    
    # Print action statistics
    if action_stats['action_dimensions']:
        unique_dims = np.unique(action_stats['action_dimensions'])
        print(f"\nAction dimensions found: {unique_dims}")
        for dim in unique_dims:
            count = action_stats['action_dimensions'].count(dim)
            print(f"  {dim}D actions: {count} transitions")
    
    if action_stats['delta_pos_magnitudes']:
        print(f"\nDelta position magnitude statistics:")
        print(f"  Average: {np.mean(action_stats['delta_pos_magnitudes']):.6f}m")
        print(f"  Min: {min(action_stats['delta_pos_magnitudes']):.6f}m")
        print(f"  Max: {max(action_stats['delta_pos_magnitudes']):.6f}m")
        print(f"  Std dev: {np.std(action_stats['delta_pos_magnitudes']):.6f}m")
    
    if action_stats['delta_ori_magnitudes']:
        print(f"\nDelta orientation magnitude statistics:")
        print(f"  Average: {np.mean(action_stats['delta_ori_magnitudes']):.6f}rad")
        print(f"  Min: {min(action_stats['delta_ori_magnitudes']):.6f}rad")
        print(f"  Max: {max(action_stats['delta_ori_magnitudes']):.6f}rad")
        print(f"  Std dev: {np.std(action_stats['delta_ori_magnitudes']):.6f}rad")
    
    if total_transitions == 0:
        raise ValueError("No valid transitions found in dataset")
    
    # Create payload in the format expected by pc_replay_buffer
    # Format matches save_as_payload() and load2() when args.real_data=False
    payload = [
        obses,           # 0: observations (torch_geometric.Data with 2 features)
        next_obses,      # 1: next observations (torch_geometric.Data with 2 features)
        actions,         # 2: actions (6D delta EEF actions as numpy arrays)
        rewards,         # 3: rewards (float values)
        not_dones,       # 4: not_dones (1.0 for all except last transition)
        non_randomized_obses,  # 5: non_randomized_obses (None)
        forces,          # 6: forces (total_force values)
        ori_obses,       # 7: ori_obses (None)
        force_vectors,   # 8: force_vectors (torch tensors)
        reward_obses,    # 9: reward_obses (same as obses for bathing)
        next_reward_obses  # 10: next_reward_obses (same as next_obses for bathing)
    ]
    
    # Save payload
    torch.save(payload, output_path)
    print(f"Saved payload with {len(obses)} transitions to {output_path}")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"  Total transitions: {len(obses)}")
    print(f"  Average points per observation: {np.mean([len(obs.pos) for obs in obses]):.1f}")
    print(f"  Action dimension: {len(actions[0]) if actions else 0} (6D delta EEF)")
    print(f"  Pointcloud features: {obses[0].x.shape[1] if obses else 0} (2 features: cyan + gripper)")
    print(f"  Pointcloud positions: {obses[0].pos.shape[1] if obses else 0} (should be 3 for XYZ)")
    print(f"  Force vectors shape: {force_vectors[0].shape if force_vectors else 'N/A'}")
    
    # Validate final format
    if obses:
        sample_obs = obses[0]
        print(f"\nValidation:")
        print(f"  Sample obs.x shape: {sample_obs.x.shape} (features)")
        print(f"  Sample obs.pos shape: {sample_obs.pos.shape} (positions)")
        print(f"  Sample action shape: {actions[0].shape if actions else 'N/A'}")
        
        # Check that all observations have the correct format
        all_valid = True
        expected_features = obses[0].x.shape[1]  # Use first observation as reference
        
        for i, obs in enumerate(obses):
            if obs.x.shape[1] != expected_features:
                print(f"  Error: Observation {i} has {obs.x.shape[1]} features, expected {expected_features}")
                all_valid = False
            if obs.pos.shape[1] != 3:
                print(f"  Error: Observation {i} has {obs.pos.shape[1]}D positions, expected 3D")
                all_valid = False
        
        # Check that all actions are 6D
        for i, action in enumerate(actions):
            if len(action) != 6:
                print(f"  Error: Action {i} has {len(action)} dimensions, expected 6D")
                all_valid = False
        
        if all_valid:
            print(f"  ✓ All observations have correct format (2 features)")
            print(f"  ✓ All actions have correct format (6D delta EEF)")
        else:
            print(f"  ✗ Some observations or actions have incorrect format")
    
    return payload

def main():
    parser = argparse.ArgumentParser(description='Convert DP3 expert data collector dataset to payload format')
    parser.add_argument('--dataset-dir', type=str, required=True, 
                       help='Directory containing episode pickle files')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to save the converted payload (.pt file)')
    parser.add_argument('--feature-mode', type=str, default='two_features',
                       choices=['two_features', 'cyan_only', 'gripper_only', 'all_5', 'rgb_only', 'rgb_cyan', 'rgb_gripper'],
                       help='Which features to extract (default: two_features)')
    parser.add_argument('--action-type', type=str, default='delta_eef',
                       choices=['joint', 'delta_eef', 'delta_pos_only'],
                       help='Type of action to use: joint (7-DOF), delta_eef (6D: pos+ori from stored poses), delta_pos_only (3D) (default: delta_eef)')
    parser.add_argument('--no-validation', action='store_true',
                       help='Disable pointcloud validation for faster processing')
    parser.add_argument('--max-z', type=float, default=1.2,
                       help='Maximum z-coordinate for pointcloud filtering (default: 1.2)')
    
    args = parser.parse_args()
    
    # Convert dataset
    payload = convert_dataset_to_payload(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        feature_mode=args.feature_mode,
        action_type=args.action_type,
        validate_pointclouds=not args.no_validation,
        max_z=args.max_z
    )
    
    print(f"\nConversion completed successfully!")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_path}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Action type: {args.action_type}")
    print(f"Z-filtering: max_z = {args.max_z}")
    print(f"Feature dimension: {payload[0][0].x.shape[1] if payload[0] else 'N/A'}")
    print(f"Action dimension: {len(payload[2][0]) if payload[2] else 'N/A'}")
    print(f"Validation: {'enabled' if not args.no_validation else 'disabled'}")
    
    if args.action_type == 'delta_pos_only':
        print(f"\nDelta Position Actions:")
        print(f"  - Using pre-computed delta_pos from data collection")
        print(f"  - Format: [pos_delta(3D)] - simple and reliable")
        print(f"  - No orientation complexity")
        print(f"  - Perfect frame alignment with pointcloud data")
    elif args.action_type == 'delta_eef':
        print(f"\nDelta EEF Actions:")
        print(f"  - Using pre-computed delta_6d from data collection")
        print(f"  - Format: [pos_delta(3D), rot_delta(3D axis-angle)]")
        print(f"  - Already computed with proper scipy quaternion handling")
        print(f"  - Perfect frame alignment with pointcloud data")
        print(f"  - DEFAULT: Matches the format expected by pc_replay_buffer")
        print(f"  - EFFICIENT: No re-computation needed!")

if __name__ == "__main__":
    main() 