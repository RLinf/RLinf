# RLinf Franka ROS 2 workspace

This workspace contains the RLinf-owned FR3 controller used by the real-world
GELLO collector. It builds against ROS 2 Humble and the system Franka ROS 2
installation; it does not require the GELLO repository at runtime.

```bash
source /opt/ros/humble/setup.bash
source /home/pnp/franka/franka_ros2_ws/install/setup.bash
cd /home/pnp/workspaces/RLinf/ros2_ws
colcon build --packages-select rlinf_franka_controller --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

The collector launches `single_fr3.launch.py` itself. Do not launch this
controller separately while collecting data.
