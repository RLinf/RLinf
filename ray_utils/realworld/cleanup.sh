#!/bin/bash
# cleanup.sh - Clean up leftover processes from RLinf real-world eval/training.
# This includes: Ray clusters, Franka ROS controllers, roslaunch/roscore/rosmaster, FrankaController.
# Required environment variables: none. The `ray` executable must be on PATH.
set -u

if ! command -v ray >/dev/null 2>&1; then
    echo "[cleanup.sh] ERROR: ray is not on PATH. Activate an RLinf environment before running this script." >&2
    exit 1
fi

echo "==== 1. Stop the Ray cluster (head + control, single-machine) ===="
ray stop 2>/dev/null || true
sleep 2

echo "==== 2. Kill leftover Ray core processes (safety net) ===="
pkill -9 -f "ray/core/src/ray/gcs/.*gcs_server" 2>/dev/null || true
pkill -9 -f "ray/core/src/ray/raylet/.*raylet" 2>/dev/null || true
pkill -9 -f "ray/_private/workers" 2>/dev/null || true
pkill -9 -f "ray/dashboard" 2>/dev/null || true

echo "==== 3. Kill Franka ROS controllers and roslaunch processes ===="
# roslaunch: FrankaController uses it to bring up franka_control/gripper/state publisher etc.
pkill -9 -f "roslaunch" 2>/dev/null || true
pkill -9 -f "rosmaster" 2>/dev/null || true
pkill -9 -f "roscore" 2>/dev/null || true

# Specific ROS nodes (past failed evals may leave multiple groups behind).
pkill -9 -f "franka_control_node" 2>/dev/null || true
pkill -9 -f "franka_gripper_node" 2>/dev/null || true
pkill -9 -f "franka_state_publisher" 2>/dev/null || true
pkill -9 -f "controller_manager/spawner" 2>/dev/null || true
pkill -9 -f "joint_state_publisher" 2>/dev/null || true
pkill -9 -f "robot_state_publisher" 2>/dev/null || true

echo "==== 4. Kill FrankaController (RLinf control worker) ===="
pkill -9 -f "FrankaController" 2>/dev/null || true
pkill -9 -f "franka_controller.py" 2>/dev/null || true

sleep 1

echo "==== 5. Verify cleanup result ===="
leftover=$(ps aux | grep -E "roslaunch|rosmaster|roscore|franka_control_node|franka_gripper_node|controller_manager/spawner|joint_state_publisher|robot_state_publisher|ray/core" | grep -v grep | wc -l)
if [ "$leftover" -eq 0 ]; then
    echo "== Cleanup complete, no leftover processes =="
else
    echo "== There are still $leftover leftover processes, please check manually: =="
    ps aux | grep -E "roslaunch|rosmaster|roscore|franka_control_node|franka_gripper_node|controller_manager/spawner|joint_state_publisher|robot_state_publisher|ray/core" | grep -v grep
fi

echo "==== 6. Clean up stale Ray temp dirs (optional, for a clean head/tail) ===="
rm -rf /tmp/rlinf_compute /tmp/rlinf_control /tmp/rlinf_collect 2>/dev/null || true

echo "==== Done. To restart: source setup_compute_node.sh start (compute) and source setup_franka_node.sh start (control) ===="
