#!/bin/bash

export CURRENT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$CURRENT_PATH"))
export PYTHONPATH=$REPO_PATH:$PYTHONPATH

# Modify these environment variables as needed
export RLINF_NODE_RANK=-1 # Change this to the appropriate node rank if using multiple nodes
export RLINF_COMM_NET_DEVICES="eth0" # Change this if you use a different network interface

# In the Franka docker image, run source switch_env franky instead
source <your_venv_path>/bin/activate # Source your virtual environment here

# Legacy ROS backend only: source your own catkin workspace if franka_ros and serl_franka_controllers were not installed by the installation script
# source <your_catkin_ws>/devel/setup.bash