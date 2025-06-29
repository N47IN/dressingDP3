#!/usr/bin/env python3
"""
Test script to analyze payload pointcloud entries based on 5-feature one-hot encoding.

This script loads a payload file and analyzes the pointcloud data to understand:
- Distribution of cyan vs non-cyan points
- Distribution of gripper vs non-gripper points  
- RGB feature statistics
- Pointcloud sizes and characteristics

Usage:
    python test_payload_analysis.py --payload-path converted_payload.pt --sample-size 10
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import seaborn as sns
from collections import defaultdict

def analyze_pointcloud_features(pcd, transition_idx):
    """
    Analyze a single pointcloud's 5-feature encoding.
    
    Args:
        pcd: torch_geometric.Data object with 5 features
        transition_idx: Index of the transition for reporting
    
    Returns:
        analysis: Dictionary with analysis results
    """
    if pcd is None:
        return {
            'transition_idx': transition_idx,
            'total_points': 0,
            'cyan_points': 0,
            'gripper_points': 0,
            'rgb_stats': None,
            'error': 'Pointcloud is None'
        }
    
    if not hasattr(pcd, 'x') or not hasattr(pcd, 'pos'):
        return {
            'transition_idx': transition_idx,
            'total_points': 0,
            'cyan_points': 0,
            'gripper_points': 0,
            'rgb_stats': None,
            'error': 'Invalid pointcloud structure'
        }
    
    features = pcd.x
    positions = pcd.pos
    
    if features.shape[1] != 5:
        return {
            'transition_idx': transition_idx,
            'total_points': len(positions),
            'cyan_points': 0,
            'gripper_points': 0,
            'rgb_stats': None,
            'error': f'Expected 5 features, got {features.shape[1]}'
        }
    
    # Extract features
    rgb_features = features[:, :3]  # Features 0-2: RGB
    cyan_onehot = features[:, 3]    # Feature 3: Cyan one-hot
    gripper_onehot = features[:, 4] # Feature 4: Gripper one-hot
    
    # Count points by type
    total_points = len(positions)
    cyan_points = int(torch.sum(cyan_onehot).item())
    gripper_points = int(torch.sum(gripper_onehot).item())
    
    # RGB statistics
    rgb_stats = {
        'mean_r': float(torch.mean(rgb_features[:, 0]).item()),
        'mean_g': float(torch.mean(rgb_features[:, 1]).item()),
        'mean_b': float(torch.mean(rgb_features[:, 2]).item()),
        'std_r': float(torch.std(rgb_features[:, 0]).item()),
        'std_g': float(torch.std(rgb_features[:, 1]).item()),
        'std_b': float(torch.std(rgb_features[:, 2]).item()),
        'min_r': float(torch.min(rgb_features[:, 0]).item()),
        'min_g': float(torch.min(rgb_features[:, 1]).item()),
        'min_b': float(torch.min(rgb_features[:, 2]).item()),
        'max_r': float(torch.max(rgb_features[:, 0]).item()),
        'max_g': float(torch.max(rgb_features[:, 1]).item()),
        'max_b': float(torch.max(rgb_features[:, 2]).item()),
    }
    
    # Position statistics
    pos_stats = {
        'mean_x': float(torch.mean(positions[:, 0]).item()),
        'mean_y': float(torch.mean(positions[:, 1]).item()),
        'mean_z': float(torch.mean(positions[:, 2]).item()),
        'std_x': float(torch.std(positions[:, 0]).item()),
        'std_y': float(torch.std(positions[:, 1]).item()),
        'std_z': float(torch.std(positions[:, 2]).item()),
    }
    
    return {
        'transition_idx': transition_idx,
        'total_points': total_points,
        'cyan_points': cyan_points,
        'gripper_points': gripper_points,
        'rgb_stats': rgb_stats,
        'pos_stats': pos_stats,
        'error': None,
        'cyan_ratio': cyan_points / total_points if total_points > 0 else 0,
        'gripper_ratio': gripper_points / total_points if total_points > 0 else 0
    }

def print_detailed_analysis(analysis_results, sample_size=10):
    """Print detailed analysis of pointcloud features."""
    
    print("=" * 80)
    print("DETAILED POINTCLOUD ANALYSIS")
    print("=" * 80)
    
    # Filter out errors
    valid_results = [r for r in analysis_results if r['error'] is None]
    error_results = [r for r in analysis_results if r['error'] is not None]
    
    print(f"Total transitions analyzed: {len(analysis_results)}")
    print(f"Valid transitions: {len(valid_results)}")
    print(f"Error transitions: {len(error_results)}")
    
    if error_results:
        print(f"\nErrors found:")
        for result in error_results[:5]:  # Show first 5 errors
            print(f"  Transition {result['transition_idx']}: {result['error']}")
        if len(error_results) > 5:
            print(f"  ... and {len(error_results) - 5} more errors")
    
    if not valid_results:
        print("\nNo valid transitions to analyze!")
        return
    
    # Overall statistics
    total_points_all = sum(r['total_points'] for r in valid_results)
    total_cyan_all = sum(r['cyan_points'] for r in valid_results)
    total_gripper_all = sum(r['gripper_points'] for r in valid_results)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total points across all transitions: {total_points_all:,}")
    print(f"  Total cyan points: {total_cyan_all:,} ({total_cyan_all/total_points_all*100:.1f}%)")
    print(f"  Total gripper points: {total_gripper_all:,} ({total_gripper_all/total_points_all*100:.1f}%)")
    
    # Sample detailed analysis
    print(f"\nSAMPLE ANALYSIS (first {min(sample_size, len(valid_results))} transitions):")
    print("-" * 80)
    
    for i, result in enumerate(valid_results[:sample_size]):
        print(f"Transition {result['transition_idx']}:")
        print(f"  Total points: {result['total_points']}")
        print(f"  Cyan points: {result['cyan_points']} ({result['cyan_ratio']*100:.1f}%)")
        print(f"  Gripper points: {result['gripper_points']} ({result['gripper_ratio']*100:.1f}%)")
        
        if result['rgb_stats']:
            rgb = result['rgb_stats']
            print(f"  RGB - Mean: R={rgb['mean_r']:.3f}, G={rgb['mean_g']:.3f}, B={rgb['mean_b']:.3f}")
            print(f"  RGB - Std:  R={rgb['std_r']:.3f}, G={rgb['std_g']:.3f}, B={rgb['std_b']:.3f}")
        
        if result['pos_stats']:
            pos = result['pos_stats']
            print(f"  Position - Mean: X={pos['mean_x']:.3f}, Y={pos['mean_y']:.3f}, Z={pos['mean_z']:.3f}")
        
        print()

def create_visualizations(analysis_results, output_dir="analysis_plots"):
    """Create visualizations of the analysis results."""
    
    valid_results = [r for r in analysis_results if r['error'] is None]
    if not valid_results:
        print("No valid results to visualize!")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Point distribution by type
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pointcloud Feature Analysis', fontsize=16)
    
    # Total points distribution
    total_points = [r['total_points'] for r in valid_results]
    axes[0, 0].hist(total_points, bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('Distribution of Total Points per Transition')
    axes[0, 0].set_xlabel('Number of Points')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cyan points ratio
    cyan_ratios = [r['cyan_ratio'] for r in valid_results]
    axes[0, 1].hist(cyan_ratios, bins=30, alpha=0.7, color='cyan')
    axes[0, 1].set_title('Distribution of Cyan Points Ratio')
    axes[0, 1].set_xlabel('Cyan Points Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gripper points ratio
    gripper_ratios = [r['gripper_ratio'] for r in valid_results]
    axes[1, 0].hist(gripper_ratios, bins=30, alpha=0.7, color='red')
    axes[1, 0].set_title('Distribution of Gripper Points Ratio')
    axes[1, 0].set_xlabel('Gripper Points Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # RGB mean values
    rgb_means = np.array([[r['rgb_stats']['mean_r'], r['rgb_stats']['mean_g'], r['rgb_stats']['mean_b']] 
                          for r in valid_results])
    axes[1, 1].scatter(rgb_means[:, 0], rgb_means[:, 1], c=rgb_means[:, 2], 
                       cmap='viridis', alpha=0.6, s=20)
    axes[1, 1].set_title('RGB Mean Values (colored by Blue)')
    axes[1, 1].set_xlabel('Mean Red')
    axes[1, 1].set_ylabel('Mean Green')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pointcloud_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_dir}/pointcloud_analysis.png")
    
    # 2. Feature correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data for correlation
    feature_data = []
    for r in valid_results:
        feature_data.append([
            r['total_points'],
            r['cyan_points'],
            r['gripper_points'],
            r['cyan_ratio'],
            r['gripper_ratio'],
            r['rgb_stats']['mean_r'],
            r['rgb_stats']['mean_g'],
            r['rgb_stats']['mean_b']
        ])
    
    feature_data = np.array(feature_data)
    feature_names = ['Total_Points', 'Cyan_Points', 'Gripper_Points', 
                    'Cyan_Ratio', 'Gripper_Ratio', 'Mean_R', 'Mean_G', 'Mean_B']
    
    corr_matrix = np.corrcoef(feature_data.T)
    
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)
    ax.set_title('Feature Correlation Matrix')
    
    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_correlation.png", dpi=300, bbox_inches='tight')
    print(f"Saved correlation matrix to {output_dir}/feature_correlation.png")
    
    plt.show()

def analyze_payload_structure(payload):
    """Analyze the overall payload structure."""
    
    print("=" * 80)
    print("PAYLOAD STRUCTURE ANALYSIS")
    print("=" * 80)
    
    expected_fields = [
        'obses', 'next_obses', 'actions', 'rewards', 'not_dones',
        'non_randomized_obses', 'forces', 'ori_obses', 'force_vectors',
        'reward_obses', 'next_reward_obses'
    ]
    
    print(f"Payload length: {len(payload)} (expected: {len(expected_fields)})")
    
    for i, (field_name, field_data) in enumerate(zip(expected_fields, payload)):
        if field_data is None:
            print(f"  {i:2d}. {field_name}: None")
        elif isinstance(field_data, list):
            print(f"  {i:2d}. {field_name}: List with {len(field_data)} items")
            if len(field_data) > 0:
                first_item = field_data[0]
                if hasattr(first_item, 'x') and hasattr(first_item, 'pos'):
                    print(f"       First item: Pointcloud with {len(first_item.pos)} points, {first_item.x.shape[1]} features")
                elif isinstance(first_item, np.ndarray):
                    print(f"       First item: Numpy array with shape {first_item.shape}")
                elif isinstance(first_item, torch.Tensor):
                    print(f"       First item: Torch tensor with shape {first_item.shape}")
                else:
                    print(f"       First item: {type(first_item).__name__} = {first_item}")
        else:
            print(f"  {i:2d}. {field_name}: {type(field_data).__name__}")

def main():
    parser = argparse.ArgumentParser(description='Analyze payload pointcloud features')
    parser.add_argument('--payload-path', type=str, required=True,
                       help='Path to the payload file (.pt)')
    parser.add_argument('--sample-size', type=int, default=10,
                       help='Number of transitions to analyze in detail (default: 10)')
    parser.add_argument('--max-transitions', type=int, default=100,
                       help='Maximum number of transitions to analyze (default: 100)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--output-dir', type=str, default='analysis_plots',
                       help='Directory to save visualizations (default: analysis_plots)')
    
    args = parser.parse_args()
    
    # Load payload
    print(f"Loading payload from: {args.payload_path}")
    try:
        payload = torch.load(args.payload_path, map_location='cpu')
        print("✓ Payload loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading payload: {e}")
        return
    
    # Analyze payload structure
    analyze_payload_structure(payload)
    
    # Get observations
    obses = payload[0]  # observations
    if not obses:
        print("No observations found in payload!")
        return
    
    print(f"\nAnalyzing {min(args.max_transitions, len(obses))} transitions...")
    
    # Analyze pointcloud features
    analysis_results = []
    for i in range(min(args.max_transitions, len(obses))):
        analysis = analyze_pointcloud_features(obses[i], i)
        analysis_results.append(analysis)
    
    # Print detailed analysis
    print_detailed_analysis(analysis_results, args.sample_size)
    
    # Create visualizations if requested
    if args.visualize:
        print(f"\nCreating visualizations...")
        create_visualizations(analysis_results, args.output_dir)
    
    print(f"\nAnalysis completed!")

if __name__ == "__main__":
    main()