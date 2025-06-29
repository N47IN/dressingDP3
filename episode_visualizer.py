#!/usr/bin/env python3
"""
DP3 Episode Visualizer
=====================

Comprehensive visualization tool that shows all episode attributes frame-wise
and creates plots spanning the full episode for force vectors and EEF pose.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import argparse
import torch
from matplotlib.animation import FuncAnimation
import seaborn as sns

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
    
    return transitions, episode_data

def extract_episode_data(transitions):
    """Extract all relevant data from transitions for visualization"""
    frames = []
    actions = []
    rewards = []
    forces = []
    force_vectors = []
    eef_positions = []
    eef_orientations = []
    states = []
    successes = []
    pointcloud_stats = []
    
    for i, transition in enumerate(transitions):
        frames.append(i)
        
        # Actions
        if 'action' in transition:
            action = transition['action']
            # Ensure action is 1D
            if isinstance(action, np.ndarray):
                action = action.flatten()
            actions.append(action)
        
        # Rewards
        if 'reward' in transition:
            reward = transition['reward']
            # Ensure reward is scalar
            if isinstance(reward, (np.ndarray, list)):
                reward = float(reward) if len(reward) == 1 else reward[0] if len(reward) > 0 else 0.0
            rewards.append(reward)
        
        # Forces
        if 'total_force' in transition:
            force = transition['total_force']
            # Ensure force is scalar
            if isinstance(force, (np.ndarray, list)):
                force = float(force) if len(force) == 1 else force[0] if len(force) > 0 else 0.0
            forces.append(force)
        
        # Force vectors
        if 'force_vectors' in transition:
            force_vec = transition['force_vectors']
            # Ensure force vector is 1D
            if isinstance(force_vec, np.ndarray):
                force_vec = force_vec.flatten()
            force_vectors.append(force_vec)
        
        # EEF positions
        if 'gripper_point' in transition:
            eef_pos = transition['gripper_point']
            # Ensure EEF position is 1D
            if isinstance(eef_pos, np.ndarray):
                eef_pos = eef_pos.flatten()
            eef_positions.append(eef_pos)
        
        # States
        if 'state' in transition:
            state = transition['state']
            # Ensure state is 1D
            if isinstance(state, np.ndarray):
                state = state.flatten()
            states.append(state)
        
        # Success
        if 'success' in transition:
            successes.append(transition['success'])
        
        # Pointcloud statistics
        pcd_stats = {}
        for pcd_type in ['pcd_cyan', 'pcd_arm', 'pcd_tool_tip', 'pcd_combined']:
            if pcd_type in transition:
                pcd_data = transition[pcd_type]
                if hasattr(pcd_data, 'pos') and len(pcd_data.pos) > 0:
                    pcd_stats[pcd_type] = len(pcd_data.pos)
                else:
                    pcd_stats[pcd_type] = 0
            else:
                pcd_stats[pcd_type] = 0
        pointcloud_stats.append(pcd_stats)
    
    return {
        'frames': np.array(frames),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'forces': np.array(forces),
        'force_vectors': np.array(force_vectors),
        'eef_positions': np.array(eef_positions),
        'states': np.array(states),
        'successes': np.array(successes),
        'pointcloud_stats': pointcloud_stats
    }

def plot_eef_trajectory(eef_positions, title="End-Effector Trajectory"):
    """Plot EEF trajectory in 3D"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(eef_positions) > 0:
        eef_positions = np.array(eef_positions)
        x, y, z = eef_positions[:, 0], eef_positions[:, 1], eef_positions[:, 2]
        
        # Plot trajectory
        ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        
        # Plot points
        scatter = ax.scatter(x, y, z, c=range(len(x)), cmap='viridis', s=50, alpha=0.8)
        
        # Mark start and end
        ax.scatter(x[0], y[0], z[0], c='green', s=200, marker='o', label='Start', edgecolors='black')
        ax.scatter(x[-1], y[-1], z[-1], c='red', s=200, marker='s', label='End', edgecolors='black')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Frame Number')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig

def plot_force_analysis(forces, force_vectors, frames):
    """Plot force analysis over the episode"""
    # Debug: Check data shapes
    print(f"Debug - Forces shape: {forces.shape}, Frames shape: {frames.shape}")
    print(f"Debug - Force vectors shape: {force_vectors.shape}")
    
    # Ensure forces is 1D
    if forces.ndim > 1:
        forces = forces.flatten()
        print(f"Debug - Flattened forces shape: {forces.shape}")
    
    # Ensure frames is 1D
    if frames.ndim > 1:
        frames = frames.flatten()
        print(f"Debug - Flattened frames shape: {frames.shape}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total force over time
    axes[0, 0].plot(frames, forces, 'r-', linewidth=2)
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Total Force (N)')
    axes[0, 0].set_title('Total Force Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Force vector components
    if len(force_vectors) > 0:
        force_vectors = np.array(force_vectors)
        # Ensure force_vectors is 2D: (frames, components)
        if force_vectors.ndim > 2:
            force_vectors = force_vectors.reshape(force_vectors.shape[0], -1)
            print(f"Debug - Reshaped force vectors shape: {force_vectors.shape}")
        
        axes[0, 1].plot(frames, force_vectors[:, 0], 'r-', label='Tool-Human Force', linewidth=2)
        axes[0, 1].plot(frames, force_vectors[:, 1], 'g-', label='Total Human Force', linewidth=2)
        axes[0, 1].plot(frames, force_vectors[:, 2], 'b-', label='Tool Force', linewidth=2)
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Force (N)')
        axes[0, 1].set_title('Force Vector Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Force histogram
    axes[1, 0].hist(forces, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Force (N)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Force Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Force statistics
    if len(forces) > 0:
        stats_text = f"""
        Force Statistics:
        Max: {np.max(forces):.3f} N
        Min: {np.min(forces):.3f} N
        Mean: {np.mean(forces):.3f} N
        Std: {np.std(forces):.3f} N
        Total frames with force: {np.sum(forces > 0)}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig

def plot_actions(actions, frames):
    """Plot actions over the episode"""
    if len(actions) == 0:
        return None
    
    actions = np.array(actions)
    num_joints = actions.shape[1]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    joint_names = [f'Joint {i+1}' for i in range(num_joints)]
    
    for i in range(num_joints):
        if i < len(axes):
            axes[i].plot(frames, actions[:, i], linewidth=2)
            axes[i].set_xlabel('Frame')
            axes[i].set_ylabel('Joint Position')
            axes[i].set_title(f'{joint_names[i]}')
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_joints, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_rewards(rewards, frames):
    """Plot rewards over the episode"""
    # Debug: Check data shapes
    print(f"Debug - Rewards shape: {rewards.shape}, Frames shape: {frames.shape}")
    print(f"Debug - Rewards type: {type(rewards)}, Frames type: {type(frames)}")
    print(f"Debug - First few rewards: {rewards[:5]}")
    
    # Ensure rewards is 1D
    if rewards.ndim > 1:
        rewards = rewards.flatten()
        print(f"Debug - Flattened rewards shape: {rewards.shape}")
    
    # Ensure frames is 1D
    if frames.ndim > 1:
        frames = frames.flatten()
        print(f"Debug - Flattened frames shape: {frames.shape}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reward over time
    ax1.plot(frames, rewards, 'g-', linewidth=2)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative reward
    cumulative_reward = np.cumsum(rewards)
    ax2.plot(frames, cumulative_reward, 'b-', linewidth=2)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Reward')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_pointcloud_stats(pointcloud_stats, frames):
    """Plot pointcloud statistics over the episode"""
    if not pointcloud_stats:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Extract data
    pcd_types = ['pcd_cyan', 'pcd_arm', 'pcd_tool_tip', 'pcd_combined']
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, (pcd_type, color) in enumerate(zip(pcd_types, colors)):
        counts = [stats[pcd_type] for stats in pointcloud_stats]
        row, col = i // 2, i % 2
        axes[row, col].plot(frames, counts, color=color, linewidth=2, label=pcd_type.replace('pcd_', '').replace('_', ' ').title())
        axes[row, col].set_xlabel('Frame')
        axes[row, col].set_ylabel('Number of Points')
        axes[row, col].set_title(f'{pcd_type.replace("pcd_", "").replace("_", " ").title()} Points')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_frame_summary(transition, frame_idx):
    """Create a summary of a single frame"""
    print(f"\n{'='*60}")
    print(f"FRAME {frame_idx} SUMMARY")
    print(f"{'='*60}")
    
    # Basic info
    print(f"Frame: {frame_idx}")
    
    # Action
    if 'action' in transition:
        action = transition['action']
        print(f"Action: {action}")
        print(f"Action magnitude: {np.linalg.norm(action):.4f}")
    
    # Reward
    if 'reward' in transition:
        print(f"Reward: {transition['reward']:.4f}")
    
    # Force
    if 'total_force' in transition:
        print(f"Total Force: {transition['total_force']:.4f} N")
    
    # Force vectors
    if 'force_vectors' in transition:
        force_vec = transition['force_vectors']
        print(f"Force Vectors: {force_vec}")
    
    # EEF position
    if 'gripper_point' in transition:
        eef_pos = transition['gripper_point']
        print(f"EEF Position: [{eef_pos[0]:.4f}, {eef_pos[1]:.4f}, {eef_pos[2]:.4f}]")
    
    # State
    if 'state' in transition:
        state = transition['state']
        print(f"State: {state}")
    
    # Success
    if 'success' in transition:
        print(f"Success: {transition['success']}")
    
    # Pointcloud statistics
    print("\nPointcloud Statistics:")
    for pcd_type in ['pcd_cyan', 'pcd_arm', 'pcd_tool_tip', 'pcd_combined']:
        if pcd_type in transition:
            pcd_data = transition[pcd_type]
            if hasattr(pcd_data, 'pos') and len(pcd_data.pos) > 0:
                num_points = len(pcd_data.pos)
                if hasattr(pcd_data, 'x') and len(pcd_data.x) > 0:
                    num_features = pcd_data.x.shape[1]
                    print(f"  {pcd_type}: {num_points} points, {num_features} features")
                else:
                    print(f"  {pcd_type}: {num_points} points")
            else:
                print(f"  {pcd_type}: 0 points")
        else:
            print(f"  {pcd_type}: Not available")

def interactive_episode_viewer(transitions, episode_data):
    """Interactive viewer for episode analysis"""
    data = extract_episode_data(transitions)
    current_frame = 0
    total_frames = len(transitions)
    
    print(f"\nInteractive Episode Viewer - {total_frames} frames")
    print("Commands:")
    print("  n/next - Next frame")
    print("  p/prev - Previous frame")
    print("  f <num> - Go to frame <num>")
    print("  s - Show frame summary")
    print("  t - Show EEF trajectory plot")
    print("  fr - Show force analysis plot")
    print("  a - Show actions plot")
    print("  r - Show rewards plot")
    print("  pc - Show pointcloud stats plot")
    print("  all - Show all plots")
    print("  q/quit - Exit")
    print("  h/help - Show this help")
    
    while True:
        print(f"\nCurrent frame: {current_frame}/{total_frames-1}")
        
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
            elif command == 's':
                create_frame_summary(transitions[current_frame], current_frame)
            elif command == 't':
                fig = plot_eef_trajectory(data['eef_positions'])
                plt.show()
            elif command == 'fr':
                fig = plot_force_analysis(data['forces'], data['force_vectors'], data['frames'])
                plt.show()
            elif command == 'a':
                fig = plot_actions(data['actions'], data['frames'])
                if fig:
                    plt.show()
            elif command == 'r':
                fig = plot_rewards(data['rewards'], data['frames'])
                plt.show()
            elif command == 'pc':
                fig = plot_pointcloud_stats(data['pointcloud_stats'], data['frames'])
                if fig:
                    plt.show()
            elif command == 'all':
                # Show all plots
                fig1 = plot_eef_trajectory(data['eef_positions'])
                fig2 = plot_force_analysis(data['forces'], data['force_vectors'], data['frames'])
                fig3 = plot_actions(data['actions'], data['frames'])
                fig4 = plot_rewards(data['rewards'], data['frames'])
                fig5 = plot_pointcloud_stats(data['pointcloud_stats'], data['frames'])
                
                plt.show()
            elif command in ['h', 'help']:
                print("Commands:")
                print("  n/next - Next frame")
                print("  p/prev - Previous frame")
                print("  f <num> - Go to frame <num>")
                print("  s - Show frame summary")
                print("  t - Show EEF trajectory plot")
                print("  fr - Show force analysis plot")
                print("  a - Show actions plot")
                print("  r - Show rewards plot")
                print("  pc - Show pointcloud stats plot")
                print("  all - Show all plots")
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
    parser = argparse.ArgumentParser(description="DP3 Episode Visualizer - Comprehensive episode analysis")
    parser.add_argument("--episode", type=str, default="dp3_transitions_dataset/episode_0000_transitions.pkl", 
                       help="Path to episode file")
    parser.add_argument("--frame", type=int, default=None, help="Show summary for specific frame")
    parser.add_argument("--interactive", action="store_true", help="Start in interactive mode")
    parser.add_argument("--all-plots", action="store_true", help="Show all plots at once")
    parser.add_argument("--trajectory", action="store_true", help="Show EEF trajectory plot")
    parser.add_argument("--force", action="store_true", help="Show force analysis plot")
    parser.add_argument("--actions", action="store_true", help="Show actions plot")
    parser.add_argument("--rewards", action="store_true", help="Show rewards plot")
    parser.add_argument("--pointcloud", action="store_true", help="Show pointcloud stats plot")
    
    args = parser.parse_args()
    
    try:
        transitions, episode_data = load_episode(args.episode)
        data = extract_episode_data(transitions)
        
        # Print episode summary
        print(f"\nEpisode Summary:")
        print(f"  Total frames: {len(transitions)}")
        print(f"  Episode length: {episode_data.get('episode_length', 'Unknown')}")
        print(f"  Task: {episode_data.get('task', 'Unknown')}")
        print(f"  Robot type: {episode_data.get('robot_type', 'Unknown')}")
        print(f"  Expert policy: {episode_data.get('expert_policy', 'Unknown')}")
        
        if args.frame is not None:
            if args.frame >= len(transitions):
                print(f"Frame {args.frame} not available. Max frames: {len(transitions)}")
                return
            create_frame_summary(transitions[args.frame], args.frame)
        elif args.interactive:
            interactive_episode_viewer(transitions, episode_data)
        elif args.all_plots:
            # Show all plots
            fig1 = plot_eef_trajectory(data['eef_positions'])
            fig2 = plot_force_analysis(data['forces'], data['force_vectors'], data['frames'])
            fig3 = plot_actions(data['actions'], data['frames'])
            fig4 = plot_rewards(data['rewards'], data['frames'])
            fig5 = plot_pointcloud_stats(data['pointcloud_stats'], data['frames'])
            plt.show()
        else:
            # Show individual plots based on flags
            if args.trajectory:
                fig = plot_eef_trajectory(data['eef_positions'])
                plt.show()
            if args.force:
                fig = plot_force_analysis(data['forces'], data['force_vectors'], data['frames'])
                plt.show()
            if args.actions:
                fig = plot_actions(data['actions'], data['frames'])
                if fig:
                    plt.show()
            if args.rewards:
                fig = plot_rewards(data['rewards'], data['frames'])
                plt.show()
            if args.pointcloud:
                fig = plot_pointcloud_stats(data['pointcloud_stats'], data['frames'])
                if fig:
                    plt.show()
            
            # If no specific plots requested, show summary
            if not any([args.trajectory, args.force, args.actions, args.rewards, args.pointcloud]):
                print("\nNo specific plots requested. Use --interactive for interactive mode or specify plots:")
                print("  --trajectory: Show EEF trajectory")
                print("  --force: Show force analysis")
                print("  --actions: Show actions")
                print("  --rewards: Show rewards")
                print("  --pointcloud: Show pointcloud stats")
                print("  --all-plots: Show all plots")
                print("  --interactive: Interactive mode")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()