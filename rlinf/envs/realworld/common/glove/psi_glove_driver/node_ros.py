#!/usr/bin/env python3
# Copyright (c) 2025 PSI Robot Team
# Licensed under the Apache License, Version 2.0
import logging
from pathlib import Path
from typing import Optional

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np
from collections import deque

from .controller import PSIGloveController, PSIGloveJointType
from .interface import PSIGloveStatusMessage, SerialInterface

logger = logging.getLogger(__name__)

class LowPassFilter:
    """低通滤波器类"""
    def __init__(self, delta=0.1, num_joints=6):
        """
        初始化低通滤波器
        alpha: 滤波系数 (0-1)，越小滤波越强
        num_joints: 关节数量
        """
        self.delta = delta
        self.num_joints = num_joints
        self.filtered_values = None
    
    def filter(self, values):
        """
        对输入值进行低通滤波
        values: 输入的角度数组
        """
        if self.filtered_values is None:
            self.filtered_values = np.array(values)
        else:
            _delta = np.array(values) - self.filtered_values
            # cutoff the delta that is larger than delta
            # TODO：频率低时需检查这里是否过度clip导致延迟高
            # 这个是用在手套的读数上，这个倒是没必要很低
            _delta = np.clip(_delta, -self.delta, self.delta)
            self.filtered_values = self.filtered_values + _delta
        return self.filtered_values.tolist()
    
    def reset(self):
        """重置滤波器状态"""
        self.filtered_values = None


class PSIGloveNode(Node):
    def __init__(
        self,
        left_hand: Optional[PSIGloveController] = None,
        right_hand: Optional[PSIGloveController] = None,
        frequency: Optional[int] = None,
        config: str = "default_config.yaml",
    ):
        super().__init__("psi_glove_node")
        self.left_hand = left_hand
        self.right_hand = right_hand

        self.declare_parameter("left_topic", "/ruiyan_hand/left/set_angles")
        self.declare_parameter("right_topic", "/ruiyan_hand/right/set_angles")
        self.declare_parameter("frequency", 100)
        self.declare_parameter("config_file", "default_config.yaml")
        self.declare_parameter("left_port", "/dev/ttyACM5")
        self.declare_parameter("right_port", "/dev/ttyACM4")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("auto_connect", True)

        self.left_topic = (
            self.get_parameter("left_topic").get_parameter_value().string_value
        )
        self.right_topic = (
            self.get_parameter("right_topic")
            .get_parameter_value()
            .string_value
        )
        self.frequency = (
            self.get_parameter("frequency").get_parameter_value().integer_value
        )
        config_file = (
            self.get_parameter("config_file")
            .get_parameter_value()
            .string_value
        )
        left_port = (
            self.get_parameter("left_port").get_parameter_value().string_value
        )
        right_port = (
            self.get_parameter("right_port").get_parameter_value().string_value
        )
        baudrate = (
            self.get_parameter("baudrate").get_parameter_value().integer_value
        )
        auto_connect = (
            self.get_parameter("auto_connect").get_parameter_value().bool_value
        )

        # 获取包的安装路径，然后找到share目录下的配置文件
        # 从 lib/python3.10/site-packages/psi_glove_driver/node.py 回到 install/psi_glove_driver
        package_dir = Path(__file__).parent.parent.parent.parent.parent
        config_path = package_dir / "share" / "psi_glove_driver" / "config" / config_file

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        if left_hand is None and right_hand is None:
            left_hand_interface = SerialInterface(
                port=left_port,
                baudrate=baudrate,
                mock=False,
                auto_connect=auto_connect,
            )
            self.left_hand = PSIGloveController(left_hand_interface)

            right_hand_interface = SerialInterface(
                port=right_port,
                baudrate=baudrate,
                mock=False,
                auto_connect=auto_connect,
            )
            self.right_hand = PSIGloveController(right_hand_interface)
        else:
            self.left_hand = left_hand
            self.right_hand = right_hand

        self.left_hand_joint_states_publisher = self.create_publisher(
            JointState, self.left_topic, 10
        )

        self.right_hand_joint_states_publisher = self.create_publisher(
            JointState, self.right_topic, 10
        )

        if frequency is not None:
            self.frequency = frequency
        else:
            self.frequency = self.frequency

        self.create_timer(1.0 / self.frequency, self.loop)

        self.hand_low_pass_filters = {
            "left": LowPassFilter(delta=0.1, num_joints=6),
            "right": LowPassFilter(delta=0.1, num_joints=6),
        }
        self.hand_joint_position_queues = {
            "left": deque(maxlen=10),
            "right": deque(maxlen=10),
        }

    def _psi_status_message_to_joint_state(
        self, status: PSIGloveStatusMessage, hand_type: str
    ) -> JointState:
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = f"{hand_type}_hand_base_link"

        joint_names = [
            "thumb_rotation",
            "thumb_bend",
            "index",
            "middle",
            "ring",
            "pinky"
        ]
        positions = []
        velocities = []
        efforts = []

        positions = [
            self._minmax_linear_map(
                self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["side"]["calibration"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["side"]["calibration"]["max"],
                status.thumb[PSIGloveJointType.side],
                self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["side"]["clip"]["source"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["side"]["clip"]["source"]["max"],
                self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["side"]["clip"]["target"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["side"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["back"]["calibration"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["back"]["calibration"]["max"],
                status.thumb[PSIGloveJointType.back],
                self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["back"]["clip"]["source"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["back"]["clip"]["source"]["max"],
                self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["back"]["clip"]["target"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["back"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                self.config[f"{hand_type}_glove"]["calibration"]["index"]["back"]["calibration"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["index"]["back"]["calibration"]["max"],
                status.index[PSIGloveJointType.back],
                self.config[f"{hand_type}_glove"]["calibration"]["index"]["back"]["clip"]["source"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["index"]["back"]["clip"]["source"]["max"],
                self.config[f"{hand_type}_glove"]["calibration"]["index"]["back"]["clip"]["target"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["index"]["back"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                self.config[f"{hand_type}_glove"]["calibration"]["middle"]["back"]["calibration"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["middle"]["back"]["calibration"]["max"],
                status.middle[PSIGloveJointType.back],
                self.config[f"{hand_type}_glove"]["calibration"]["middle"]["back"]["clip"]["source"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["middle"]["back"]["clip"]["source"]["max"],
                self.config[f"{hand_type}_glove"]["calibration"]["middle"]["back"]["clip"]["target"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["middle"]["back"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                self.config[f"{hand_type}_glove"]["calibration"]["ring"]["back"]["calibration"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["ring"]["back"]["calibration"]["max"],
                status.ring[PSIGloveJointType.back],
                self.config[f"{hand_type}_glove"]["calibration"]["ring"]["back"]["clip"]["source"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["ring"]["back"]["clip"]["source"]["max"],
                self.config[f"{hand_type}_glove"]["calibration"]["ring"]["back"]["clip"]["target"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["ring"]["back"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                self.config[f"{hand_type}_glove"]["calibration"]["pinky"]["back"]["calibration"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["pinky"]["back"]["calibration"]["max"],
                status.pinky[PSIGloveJointType.back],
                self.config[f"{hand_type}_glove"]["calibration"]["pinky"]["back"]["clip"]["source"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["pinky"]["back"]["clip"]["source"]["max"],
                self.config[f"{hand_type}_glove"]["calibration"]["pinky"]["back"]["clip"]["target"]["min"],
                self.config[f"{hand_type}_glove"]["calibration"]["pinky"]["back"]["clip"]["target"]["max"],
            ),
        ]

        # 应用滤波器
        positions = self.hand_low_pass_filters[hand_type].filter(positions)
        self.hand_joint_position_queues[hand_type].append(positions)
        positions = np.mean(self.hand_joint_position_queues[hand_type], axis=0).tolist()

        velocities = [3000.0] * 6
        efforts = [1000.0] * 6

        # Debug 输出：原始传感器值和映射后的值
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"\n【{hand_type} 手套原始传感器值】")
            logger.debug(f"Thumb:  [tip={status.thumb[0]:4d}, mid={status.thumb[1]:4d}, back={status.thumb[2]:4d}, side={status.thumb[3]:4d}, rotate={status.thumb[4]:4d}]")
            logger.debug(f"Index:  [tip={status.index[0]:4d}, mid={status.index[1]:4d}, back={status.index[2]:4d}, side={status.index[3]:4d}]")
            logger.debug(f"Middle: [tip={status.middle[0]:4d}, mid={status.middle[1]:4d}, back={status.middle[2]:4d}, side={status.middle[3]:4d}]")
            logger.debug(f"Ring:   [tip={status.ring[0]:4d}, mid={status.ring[1]:4d}, back={status.ring[2]:4d}, side={status.ring[3]:4d}]")
            logger.debug(f"Pinky:  [tip={status.pinky[0]:4d}, mid={status.pinky[1]:4d}, back={status.pinky[2]:4d}, side={status.pinky[3]:4d}]")
            logger.debug(f"\n【{hand_type} 手套映射后的值 (归一化)】")
            logger.debug(f"Thumb:  [side={positions[0]:.3f}, back={positions[1]:.3f}]")
            logger.debug(f"Index:  [back={positions[2]:.3f}]")
            logger.debug(f"Middle: [back={positions[3]:.3f}]")
            logger.debug(f"Ring:   [back={positions[4]:.3f}]")
            logger.debug(f"Pinky:  [back={positions[5]:.3f}]")
            logger.debug("-" * 80)

        joint_state.name = joint_names
        joint_state.position = positions
        joint_state.velocity = velocities
        joint_state.effort = efforts

        return joint_state

    def _minmax_linear_map(
        self,
        calibration_min: int,
        calibration_max: int,
        value: int,
        clip_source_min: float,
        clip_source_max: float,
        clip_target_min: float,
        clip_target_max: float,
    ) -> float:
        """
        两步映射处理:
        1. 使用 calibration 参数将原始传感器值归一化到 [0, 1]
        2. 使用 clip.source 参数将归一化值重映射到 [0, 1]
        3. 使用 clip.target 参数截断最终输出值
        """
        # 步骤 1: 工厂校准归一化到 [0, 1]
        if calibration_max == calibration_min:
            normalized_value = 0.0
        else:
            normalized_value = (value - calibration_min) / (calibration_max - calibration_min)
        
        # 步骤 2: 通过 source 范围重映射到 [0, 1]
        if clip_source_max == clip_source_min:
            remapped_value = 0.0
        else:
            remapped_value = (normalized_value - clip_source_min) / (clip_source_max - clip_source_min)
        
        # 步骤 3: 使用 target 边界截断
        final_value = np.clip(remapped_value, clip_target_min, clip_target_max)
        
        return float(final_value)

    def loop(self):
        if self.left_hand:
            try:
                left_hand_status = self.left_hand.loop()
                if left_hand_status:
                    left_joint_state = self._psi_status_message_to_joint_state(
                        left_hand_status, "left"
                    )
                    self.left_hand_joint_states_publisher.publish(
                        left_joint_state
                    )
            except Exception as e:
                logger.warning(
                    f"Error reading left glove: {type(e).__name__}: {e}"
                )
        if self.right_hand:
            try:
                right_hand_status = self.right_hand.loop()
                if right_hand_status:
                    right_joint_state = (
                        self._psi_status_message_to_joint_state(
                            right_hand_status, "right"
                        )
                    )
                    self.right_hand_joint_states_publisher.publish(
                        right_joint_state
                    )
            except Exception as e:
                logger.warning(
                    f"Error reading right glove: {type(e).__name__}: {e}"
                )


def main(args=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    rclpy.init(args=args)
    node = PSIGloveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("Node manually stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
