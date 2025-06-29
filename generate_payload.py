#!/usr/bin/env python3
"""
Script to convert DP3 expert data collector dataset to payload format for pc_replay_buffer.

This script adapts the save_as_payload function from pc_replay_buffer.py to work with
the dataset format created by dp3_expert_data_collector.py.

Updated for the new 5-feature combined pointcloud format:
- Feature 0-2: RGB values (0-1)
- Feature 3: Cyan one-hot encoding (1.0 for cyan points, 0.0 for others)
- Feature 4: Gripper one-hot encoding (1.0 for gripper/tool points, 0.0 for others)

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
    Validate that a combined pointcloud has the expected 5-feature format.
    
    Args:
        pcd: torch_geometric.Data object
        expected_features: Expected number of features (default: 5)
    
    Returns:
        is_valid: Boolean indicating if the pointcloud is valid
        message: Description of validation result
    """
    if not hasattr(pcd, 'x') or not hasattr(pcd, 'pos'):
        return False, "Missing 'x' or 'pos' attributes"
    
    if pcd.x.shape[1] != expected_features:
        return False, f"Expected {expected_features} features, got {pcd.x.shape[1]}"
    
    if pcd.pos.shape[1] != 3:
        return False, f"Expected 3D positions, got {pcd.pos.shape[1]}D"
    
    if len(pcd.pos) == 0:
        return False, "Empty pointcloud"
    
    # Check feature ranges
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
    
    return True, "Valid 5-feature combined pointcloud"

def extract_features_for_model(pcd, feature_mode='rgb_only'):
    """
    Extract features from 5-feature combined pointcloud for model input.
    
    Args:
        pcd: torch_geometric.Data object with 5 features
        feature_mode: Which features to extract ('rgb_only', 'all_5', 'rgb_cyan', 'rgb_gripper')
    
    Returns:
        processed_pcd: torch_geometric.Data object with extracted features
    """
    if pcd.x.shape[1] != 5:
        raise ValueError(f"Expected 5 features, got {pcd.x.shape[1]}")
    
    if feature_mode == 'rgb_only':
        # Extract only RGB features (first 3) - compatible with standard PointNet++
        features = pcd.x[:, :3]
    elif feature_mode == 'all_5':
        # Keep all 5 features
        features = pcd.x
    elif feature_mode == 'rgb_cyan':
        # RGB + cyan one-hot (4 features)
        features = torch.cat([pcd.x[:, :3], pcd.x[:, 3:4]], dim=1)
    elif feature_mode == 'rgb_gripper':
        # RGB + gripper one-hot (4 features)
        features = torch.cat([pcd.x[:, :3], pcd.x[:, 4:5]], dim=1)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    
    return Data(pos=pcd.pos, x=features)

def convert_dataset_to_payload(dataset_dir, output_path, feature_mode='rgb_only', validate_pointclouds=True):
    """
    Convert DP3 expert data collector dataset to payload format.
    
    Args:
        dataset_dir: Directory containing episode pickle files
        output_path: Path to save the converted payload
        feature_mode: Which features to extract ('rgb_only', 'all_5', 'rgb_cyan', 'rgb_gripper')
        validate_pointclouds: Whether to validate pointcloud format before processing
    """
    print(f"Converting dataset from {dataset_dir} to {output_path}")
    print(f"Feature mode: {feature_mode}")
    print(f"Validation: {'enabled' if validate_pointclouds else 'disabled'}")
    
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
    validation_stats = {
        'valid_5_feature': 0,
        'invalid_format': 0,
        'missing_combined': 0,
        'empty_pointcloud': 0
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
                
                # Process action - convert to 6D format if needed
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
                
                actions.append(action)
                
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
    
    if total_transitions == 0:
        raise ValueError("No valid transitions found in dataset")
    
    # Create payload in the format expected by pc_replay_buffer
    payload = [
        obses,           # 0: observations
        next_obses,      # 1: next observations  
        actions,         # 2: actions
        rewards,         # 3: rewards
        not_dones,       # 4: not_dones
        non_randomized_obses,  # 5: non_randomized_obses (None)
        forces,          # 6: forces
        ori_obses,       # 7: ori_obses (None)
        force_vectors,   # 8: force_vectors
        reward_obses,    # 9: reward_obses
        next_reward_obses  # 10: next_reward_obses
    ]
    
    # Save payload
    torch.save(payload, output_path)
    print(f"Saved payload with {len(obses)} transitions to {output_path}")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"  Total transitions: {len(obses)}")
    print(f"  Average points per observation: {np.mean([len(obs.pos) for obs in obses]):.1f}")
    print(f"  Action dimension: {len(actions[0]) if actions else 0}")
    print(f"  Pointcloud features: {obses[0].x.shape[1] if obses else 0}")
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
        
        if all_valid:
            print(f"  ✓ All observations have correct format")
        else:
            print(f"  ✗ Some observations have incorrect format")
    
    return payload

def main():
    parser = argparse.ArgumentParser(description='Convert DP3 expert data collector dataset to payload format')
    parser.add_argument('--dataset-dir', type=str, required=True, 
                       help='Directory containing episode pickle files')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to save the converted payload (.pt file)')
    parser.add_argument('--feature-mode', type=str, default='rgb_only',
                       choices=['rgb_only', 'all_5', 'rgb_cyan', 'rgb_gripper'],
                       help='Which features to extract (default: rgb_only)')
    parser.add_argument('--no-validation', action='store_true',
                       help='Disable pointcloud validation for faster processing')
    
    args = parser.parse_args()
    
    # Convert dataset
    payload = convert_dataset_to_payload(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        feature_mode=args.feature_mode,
        validate_pointclouds=not args.no_validation
    )
    
    print(f"\nConversion completed successfully!")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_path}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Feature dimension: {payload[0][0].x.shape[1] if payload[0] else 'N/A'}")
    print(f"Validation: {'enabled' if not args.no_validation else 'disabled'}")

if __name__ == "__main__":
    main() 