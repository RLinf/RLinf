cd /home/shang/RLinf
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash

export RLINF_REPLAY_DATASET='/data/datasets/rlinf/20260808-19:47:34/collected_data/rank_0/id_0'
export RLINF_REPLAY_EPISODE=0
unset RLINF_REPLAY_VALIDATE_ONLY

./.venv/bin/python -m examples.embodiment.replay_lerobot_episode

# REPLAY 0