# Replay Buffer Visualization Tool

Interactive visualizer for inspecting trajectory data saved by RLinf's replay buffer.

## Features

- **Lazy loading**: Uses `TrajectoryReplayBuffer` to load trajectories on-demand, avoiding loading all data into memory
- **Auto-switching**: Automatically advances to the next trajectory when reaching the last frame
- **Jump to trajectory**: Type trajectory ID in the text box to jump directly
- **Multi-camera support**: View main, wrist, or extra camera views
- **Batch navigation**: Navigate between batch indices if B > 1

## Quick Start
```bash
# Setup the python-path first
export PYTHONPATH=/path/to/RLinf:$PYTHONPATH
```

## Common Use Cases

### 1. Interactive Mode (Local Machine with Display)

```bash
python toolkits/replay_buffer/visualize.py \
    --replay_dir logs/my_run/replay_buffer/rank_0
```

Navigate with keyboard:
- `→` / `←`: Next/previous step
- `↑` / `↓`: Next/previous trajectory
- Type trajectory ID in the text box to jump directly
- Press `s` to save current view
- Press `q` to quit

### 2. SSH Without X11 (e.g. VSCode)

Use the headless interactive script:

```bash
python toolkits/replay_buffer/visualize_headless.py \
    --replay_dir logs/my_run/replay_buffer/rank_0
```

Then in VSCode:
1. Open `replay_buffer_viz.png` in the editor
2. Navigate using command-line prompts
3. The image updates automatically - VSCode will show the changes

Example session:
```
Command: info
Current position:
  Trajectory ID: 0 (#0/203)
  Step: 0/255
  Batch: 0/0

Command: n
Command: n
Command: j 10
Jumped to trajectory ID 10
Command: info
Current position:
  Trajectory ID: 10 (#10/203)
  Step: 0/255
  Batch: 0/0

Command: q
```

## Display Information
Here is the visualization result for the FrankaSim replay buffer. 
![](https://github.com/RLinf/misc/blob/main/pic/vis_buffer_example.png)


