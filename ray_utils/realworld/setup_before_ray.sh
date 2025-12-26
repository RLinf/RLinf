#!/bin/bash

export CURRENT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$CURRENT_PATH"))
export PYTHONPATH=$REPO_PATH:$PYTHONPATH

# Modify these environment variables as needed
export RLINF_NODE_RANK=-1 # Change this to the appropriate node rank if using multiple nodes
export RLINF_COMM_NET_DEVICES="eth0" # Change this if you use a different network interface
export FRANKA_PATH="" # Path to your franka_ros and serl_franka_controllers catkin workspace
source <your_venv_path>/bin/activate # Source your virtual environment here

if [ -n "$FRANKA_CATKIN_PATH" ]; then
    source $FRANKA_CATKIN_PATH/devel/setup.bash
fi
