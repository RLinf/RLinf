# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch one FR3 with the RLinf joint impedance controller."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    controllers_yaml = PathJoinSubstitution(
        [FindPackageShare("rlinf_franka_controller"), "config", "controllers.yaml"]
    )
    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("franka_bringup"), "launch", "franka.launch.py"]
            )
        ),
        launch_arguments={
            "arm_id": "fr3",
            "arm_prefix": LaunchConfiguration("arm_prefix"),
            "namespace": LaunchConfiguration("namespace"),
            "urdf_file": "fr3/fr3.urdf.xacro",
            "robot_ip": LaunchConfiguration("robot_ip"),
            "load_gripper": "false",
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "fake_sensor_commands": LaunchConfiguration("use_fake_hardware"),
            "joint_state_rate": "30",
            "controllers_yaml": controllers_yaml,
        }.items(),
    )
    controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_impedance_controller", "--controller-manager-timeout", "30"],
        namespace=LaunchConfiguration("namespace"),
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip"),
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument("arm_prefix", default_value=""),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            robot,
            controller,
        ]
    )
