# Expert Policy Runner for Assistive Gym

This directory contains scripts to run pretrained expert policies for the Assistive Gym environments, specifically for the `BedBathingSawyer-v0` environment.

## Files

- `run_expert_policy.py` - Main script to run expert policies
- `test_environment.py` - Test script to verify environment and dependencies
- `README_expert_policy.md` - This file

## Prerequisites

Based on the [Assistive Gym documentation](https://github.com/Healthcare-Robotics/assistive-gym/wiki/4.-Running-Pretrained-Policies), you need to install the following:

### 1. Install PyTorch RL Library
```bash
pip3 install git+https://github.com/Zackory/pytorch-a2c-ppo-acktr --no-cache-dir
```

### 2. Install OpenAI Baselines
```bash
pip3 install git+https://github.com/openai/baselines.git
```

### 3. Install OpenCV (if not already installed)
**Ubuntu:**
```bash
sudo apt-get install python3-opencv
```

**Mac:**
```bash
brew install opencv
```

### 4. Verify Policy Files
Ensure you have the pretrained policy files in `trained_models/ppo/`. The script expects:
- `BedBathingSawyer-v0.pt` (already present in your setup)

## Usage

### 1. Test Environment and Dependencies
First, run the test script to verify everything is set up correctly:

```bash
python3 test_environment.py
```

This will check:
- ✅ Gym and assistive_gym imports
- ✅ BedBathingSawyer-v0 environment creation
- ✅ Policy file existence
- ✅ PPO library availability

### 2. Run Expert Policy

#### Single Episode
```bash
python3 run_expert_policy.py --env-name "BedBathingSawyer-v0"
```

#### 100 Trials Evaluation
```bash
python3 run_expert_policy.py --env-name "BedBathingSawyer-v0" --trials
```

### 3. Alternative: Direct PPO Commands

If the wrapper script doesn't work, you can try the direct commands as specified in the documentation:

#### Single Episode
```bash
python3 -m ppo.enjoy --env-name "BedBathingSawyer-v0"
```

#### 100 Trials Evaluation
```bash
python3 -m ppo.enjoy_100trials --env-name "BedBathingSawyer-v0"
```

## Expected Behavior

When running successfully, you should see:

1. **Environment Creation**: The BedBathingSawyer-v0 environment will be created and rendered
2. **Policy Loading**: The pretrained policy will be loaded from the `.pt` file
3. **Execution**: The robot will perform the bed bathing task using the expert policy
4. **Results**: Statistics about the episode(s) including rewards and success rates

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you've installed all the required packages listed above
2. **Policy File Not Found**: Verify that `trained_models/ppo/BedBathingSawyer-v0.pt` exists
3. **PPO Module Not Found**: Ensure the PyTorch RL library is installed correctly
4. **Environment Errors**: Check that assistive_gym is properly installed

### Debug Steps

1. Run `python3 test_environment.py` to identify specific issues
2. Check that all policy files are present in `trained_models/ppo/`
3. Verify Python environment has all required packages
4. Try running the direct PPO commands if the wrapper script fails

## Available Environments

Based on the policy files in your `trained_models/ppo/` directory, you can run expert policies for:

- `BedBathingSawyer-v0` (recommended for this setup)
- `BedBathingBaxter-v0`
- `BedBathingJaco-v0`
- `BedBathingPR2-v0`
- `FeedingSawyer-v0`
- `DrinkingSawyer-v0`
- `ScratchItchSawyer-v0`
- And many more...

## Notes

- The pretrained policies were trained for 10,000,000 time steps (50,000 simulation rollouts)
- These are v0.1 policies as mentioned in the documentation
- The policies use Proximal Policy Optimization (PPO) implemented in PyTorch
- For collaborative assistance (robot + human), use the `*Human-v0` variants 