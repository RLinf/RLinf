export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=enp2s0

unset PYTHONPATH
export PYTHONPATH=/home/franka/zhoujiakai/RLinf-shuang
source ~/hongzhi/0105/RLinf/.venv/bin/activate
source /opt/ros/noetic/setup.bash  # 建议先 source 系统的 ROS
source ~/catkin_franka/devel/setup.bash
export LD_LIBRARY_PATH=/opt/ros/noetic/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export FRANKA_ROBOT_IP=10.100.25.2