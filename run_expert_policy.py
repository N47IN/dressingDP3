#!/usr/bin/env python3
"""
Expert Policy Runner for BedBathingSawyer-v0
============================================

This script runs the pretrained expert policy for the BedBathingSawyer-v0 environment.
Based on the assistive-gym documentation for running pretrained policies.

Usage:
    python3 run_expert_policy.py --env-name "BedBathingSawyer-v0"
"""

import sys
import os
import subprocess
import argparse

def run_enjoy_command(env_name):
    """Run the enjoy command as specified in the assistive-gym documentation"""
    
    # Check if the policy file exists
    policy_file = os.path.join('trained_models', 'ppo', f'{env_name}.pt')
    if not os.path.exists(policy_file):
        print(f"Error: Policy file not found: {policy_file}")
        print("Available policies:")
        ppo_dir = os.path.join('trained_models', 'ppo')
        if os.path.exists(ppo_dir):
            for file in os.listdir(ppo_dir):
                if file.endswith('.pt'):
                    print(f"  - {file}")
        return False
    
    print(f"Running expert policy for {env_name}...")
    print(f"Policy file: {policy_file}")
    
    try:
        # Try to run the enjoy command as specified in the documentation
        # This follows the pattern: python3 -m ppo.enjoy --env-name "ScratchItchBaxter-v0"
        cmd = [sys.executable, '-m', 'ppo.enjoy', '--env-name', env_name]
        
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Expert policy execution completed successfully!")
            print("Output:")
            print(result.stdout)
        else:
            print("Error running expert policy:")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("Error: Could not find the ppo.enjoy module.")
        print("Please ensure you have installed the PPO library:")
        print("pip3 install git+https://github.com/Zackory/pytorch-a2c-ppo-acktr --no-cache-dir")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

def run_100_trials(env_name):
    """Run 100 trials evaluation as specified in the documentation"""
    
    print(f"Running 100 trials evaluation for {env_name}...")
    
    try:
        # Try to run the enjoy_100trials command
        cmd = [sys.executable, '-m', 'ppo.enjoy_100trials', '--env-name', env_name]
        
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("100 trials evaluation completed successfully!")
            print("Output:")
            print(result.stdout)
        else:
            print("Error running 100 trials evaluation:")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("Error: Could not find the ppo.enjoy_100trials module.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True

def main():
    """Main function to run the expert policy"""
    
    parser = argparse.ArgumentParser(description='Run expert policy for assistive gym environments')
    parser.add_argument('--env-name', type=str, default='BedBathingSawyer-v0',
                       help='Environment name (default: BedBathingSawyer-v0)')
    parser.add_argument('--trials', action='store_true',
                       help='Run 100 trials evaluation instead of single episode')
    
    args = parser.parse_args()
    
    print("Expert Policy Runner for Assistive Gym")
    print("=" * 50)
    print(f"Environment: {args.env_name}")
    print(f"Mode: {'100 Trials Evaluation' if args.trials else 'Single Episode'}")
    print("=" * 50)
    
    if args.trials:
        success = run_100_trials(args.env_name)
    else:
        success = run_enjoy_command(args.env_name)
    
    if success:
        print("\n✅ Expert policy execution completed successfully!")
    else:
        print("\n❌ Expert policy execution failed.")
        print("\nTroubleshooting:")
        print("1. Ensure you have installed the PPO library:")
        print("   pip3 install git+https://github.com/Zackory/pytorch-a2c-ppo-acktr --no-cache-dir")
        print("2. Ensure you have installed OpenAI Baselines:")
        print("   pip3 install git+https://github.com/openai/baselines.git")
        print("3. Check that the policy file exists in trained_models/ppo/")

if __name__ == "__main__":
    main() 