#!/usr/bin/env python3
# Copyright (c) 2025 PSI Robot Team
# Licensed under the Apache License, Version 2.0

import sys
import logging
import time
import yaml
from pathlib import Path
from typing import Optional
from collections import deque
import numpy as np

sys.path.insert(0, str(Path(__file__)))

from .controller import PSIGloveController, PSIGloveJointType
from .interface import PSIGloveStatusMessage, SerialInterface

logger = logging.getLogger(__name__)


class LowPassFilter:
    """低通滤波器类"""
    def __init__(self, delta=0.1, num_joints=6):
        """
        初始化低通滤波器
        delta: 滤波系数，用于限制变化幅度
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
            # 限制变化幅度，避免突变
            _delta = np.clip(_delta, -self.delta, self.delta)
            self.filtered_values = self.filtered_values + _delta
        return self.filtered_values.tolist()
    
    def reset(self):
        """重置滤波器状态"""
        self.filtered_values = None


class PSIGloveStandalone:
    """不依赖 ROS 的 PSI Glove 控制器"""
    
    def __init__(
        self,
        left_hand: Optional[PSIGloveController] = None,
        right_hand: Optional[PSIGloveController] = None,
        left_port: str = "/dev/ttyACM5",
        right_port: str = "/dev/ttyACM4",
        baudrate: int = 115200,
        frequency: int = 10,
        config_file: str = f"{Path(__file__).parent.parent}/default_config.yaml",
        auto_connect: bool = True,
    ):
        """
        初始化 PSI Glove 独立控制器
        
        Args:
            left_hand: 左手控制器（可选，如果提供则使用，否则自动创建）
            right_hand: 右手控制器（可选）
            left_port: 左手串口设备路径
            right_port: 右手串口设备路径
            baudrate: 波特率
            frequency: 读取频率(Hz)
            config_file: 配置文件名称
            auto_connect: 是否自动连接
        """
        self.left_hand = left_hand
        self.right_hand = right_hand
        self.frequency = frequency
        
        # 加载配置文件
        config_path = Path(config_file)
        
        if not config_path.exists():
            # 尝试从安装路径查找
            package_dir = Path(__file__).parent.parent.parent.parent.parent
            config_path = package_dir / "share" / "psi_glove_driver" / "config" / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        
        # 如果没有提供控制器，则创建
        if left_hand is None and right_hand is None:
            if left_port:
                left_hand_interface = SerialInterface(
                    port=left_port,
                    baudrate=baudrate,
                    mock=False,
                    auto_connect=auto_connect,
                )
                self.left_hand = PSIGloveController(left_hand_interface)
            
            if right_port:
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
        
        # 初始化滤波器
        self.hand_low_pass_filters = {
            "left": LowPassFilter(delta=0.1, num_joints=6),
            "right": LowPassFilter(delta=0.1, num_joints=6),
        }
        
        # 初始化位置队列（用于平滑）
        self.hand_joint_position_queues = {
            "left": deque(maxlen=10),
            "right": deque(maxlen=10),
        }
    
    def _minmax_linear_map(
        self,
        calibration_min: int,
        calibration_max: int,
        value: int,
        clip_source_min: float,
        clip_source_max: float,
        clip_target_min: float,
        clip_target_max: float,
        reverse: bool = True,
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
        
        # # 步骤 1.5: 反转映射（修复方向：握拳=1，张开=0）
        # if reverse:
        #     normalized_value = 1.0 - normalized_value
        # else:
        #     normalized_value = normalized_value

        # 步骤 2: 通过 source 范围重映射到 [0, 1]
        if clip_source_max == clip_source_min:
            remapped_value = 0.0
        else:
            remapped_value = (normalized_value - clip_source_min) / (clip_source_max - clip_source_min)
        
        # 步骤 3: 使用 target 边界截断
        final_value = np.clip(remapped_value, clip_target_min, clip_target_max)
        
        return float(final_value)
    
    def _process_status(
        self, status: PSIGloveStatusMessage, hand_type: str
    ) -> list:
        """
        处理状态消息，应用校准和滤波
        
        Returns:
            处理后的关节位置列表 [thumb_side, thumb_back, index_back, middle_back, ring_back, pinky_back]
        """
        # 从配置中获取校准参数
        thumb_side_calib = self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["side"]["calibration"]
        thumb_side_clip = self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["side"]["clip"]
        
        thumb_back_calib = self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["back"]["calibration"]
        thumb_back_clip = self.config[f"{hand_type}_glove"]["calibration"]["thumb"]["back"]["clip"]
        
        index_back_calib = self.config[f"{hand_type}_glove"]["calibration"]["index"]["back"]["calibration"]
        index_back_clip = self.config[f"{hand_type}_glove"]["calibration"]["index"]["back"]["clip"]
        
        middle_back_calib = self.config[f"{hand_type}_glove"]["calibration"]["middle"]["back"]["calibration"]
        middle_back_clip = self.config[f"{hand_type}_glove"]["calibration"]["middle"]["back"]["clip"]
        
        ring_back_calib = self.config[f"{hand_type}_glove"]["calibration"]["ring"]["back"]["calibration"]
        ring_back_clip = self.config[f"{hand_type}_glove"]["calibration"]["ring"]["back"]["clip"]
        
        pinky_back_calib = self.config[f"{hand_type}_glove"]["calibration"]["pinky"]["back"]["calibration"]
        pinky_back_clip = self.config[f"{hand_type}_glove"]["calibration"]["pinky"]["back"]["clip"]
        
        # 应用校准映射
        positions = [
            self._minmax_linear_map(
                thumb_side_calib["min"],
                thumb_side_calib["max"],
                status.thumb[PSIGloveJointType.side],
                thumb_side_clip["source"]["min"],
                thumb_side_clip["source"]["max"],
                thumb_side_clip["target"]["min"],
                thumb_side_clip["target"]["max"],
                reverse=False,
            ),
            self._minmax_linear_map(
                thumb_back_calib["min"],
                thumb_back_calib["max"],
                status.thumb[PSIGloveJointType.back],
                thumb_back_clip["source"]["min"],
                thumb_back_clip["source"]["max"],
                thumb_back_clip["target"]["min"],
                thumb_back_clip["target"]["max"],
            ),
            self._minmax_linear_map(
                index_back_calib["min"],
                index_back_calib["max"],
                status.index[PSIGloveJointType.back],
                index_back_clip["source"]["min"],
                index_back_clip["source"]["max"],
                index_back_clip["target"]["min"],
                index_back_clip["target"]["max"],
            ),
            self._minmax_linear_map(
                middle_back_calib["min"],
                middle_back_calib["max"],
                status.middle[PSIGloveJointType.back],
                middle_back_clip["source"]["min"],
                middle_back_clip["source"]["max"],
                middle_back_clip["target"]["min"],
                middle_back_clip["target"]["max"],
            ),
            self._minmax_linear_map(
                ring_back_calib["min"],
                ring_back_calib["max"],
                status.ring[PSIGloveJointType.back],
                ring_back_clip["source"]["min"],
                ring_back_clip["source"]["max"],
                ring_back_clip["target"]["min"],
                ring_back_clip["target"]["max"],
            ),
            self._minmax_linear_map(
                pinky_back_calib["min"],
                pinky_back_calib["max"],
                status.pinky[PSIGloveJointType.back],
                pinky_back_clip["source"]["min"],
                pinky_back_clip["source"]["max"],
                pinky_back_clip["target"]["min"],
                pinky_back_clip["target"]["max"],
            ),
        ]
        
        # 应用低通滤波器
        positions = self.hand_low_pass_filters[hand_type].filter(positions)
        
        # 添加到队列并计算平均值（进一步平滑）
        self.hand_joint_position_queues[hand_type].append(positions)
        positions = np.mean(self.hand_joint_position_queues[hand_type], axis=0).tolist()
        
        return positions
    
    def loop(self):
        """执行一次循环，读取并处理数据"""
        results = {}
        
        if self.left_hand:
            try:
                left_status = self.left_hand.loop()
                if left_status:
                    left_positions = self._process_status(left_status, "left")
                    results["left"] = {
                        "raw": left_status,
                        "processed": left_positions
                    }
            except Exception as e:
                logger.warning(f"Error reading left glove: {type(e).__name__}: {e}")
        
        if self.right_hand:
            try:
                right_status = self.right_hand.loop()
                if right_status:
                    right_positions = self._process_status(right_status, "right")
                    results["right"] = {
                        "raw": right_status,
                        "processed": right_positions
                    }
            except Exception as e:
                logger.warning(f"Error reading right glove: {type(e).__name__}: {e}")
        
        return results
    
    def get_hand_action(self):
        interval = 1.0 / self.frequency
        start_time = time.perf_counter()
        results = self.loop()
        elapsed = time.perf_counter() - start_time
        sleep_time = max(0, interval - elapsed)
        time.sleep(sleep_time)
        return results
    
    def run(self, print_output: bool = True):
        """
        运行主循环
        
        Args:
            print_output: 是否打印输出
        """
        interval = 1.0 / self.frequency
        
        try:
            while True:
                start_time = time.perf_counter()
                
                results = self.loop()
                
                if print_output:
                    self._print_results(results)
                
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            self.disconnect()
    
    def _print_results(self, results: dict):
        """打印处理结果"""
        joint_names = [
            "thumb_side",
            "thumb_back",
            "index_back",
            "middle_back",
            "ring_back",
            "pinky_back"
        ]
        
        for hand_type in ["left", "right"]:
            if hand_type in results:
                data = results[hand_type]
                status = data["raw"]
                positions = data["processed"]
                
                print(f"\n【{hand_type.upper()} 手套原始传感器值】")
                print(f"Thumb:  [tip={status.thumb[0]:4d}, mid={status.thumb[1]:4d}, back={status.thumb[2]:4d}, side={status.thumb[3]:4d}, rotate={status.thumb[4]:4d}]")
                print(f"Index:  [tip={status.index[0]:4d}, mid={status.index[1]:4d}, back={status.index[2]:4d}, side={status.index[3]:4d}]")
                print(f"Middle: [tip={status.middle[0]:4d}, mid={status.middle[1]:4d}, back={status.middle[2]:4d}, side={status.middle[3]:4d}]")
                print(f"Ring:   [tip={status.ring[0]:4d}, mid={status.ring[1]:4d}, back={status.ring[2]:4d}, side={status.ring[3]:4d}]")
                print(f"Pinky:  [tip={status.pinky[0]:4d}, mid={status.pinky[1]:4d}, back={status.pinky[2]:4d}, side={status.pinky[3]:4d}]")
                
                print(f"\n【{hand_type.upper()} 手套映射后的值 (归一化)】")
                for i, (name, pos) in enumerate(zip(joint_names, positions)):
                    print(f"  {name}: {pos:.3f}")
                print("-" * 80)
    
    def disconnect(self):
        """断开连接"""
        if self.left_hand:
            self.left_hand.disconnect()
        if self.right_hand:
            self.right_hand.disconnect()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PSI Glove Standalone Controller")
    parser.add_argument("--left-port", type=str, default="/dev/ttyACM0", help="Left hand port")
    parser.add_argument("--right-port", type=str, default="/dev/ttyACM1", help="Right hand port")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baudrate")
    parser.add_argument("--frequency", type=int, default=100, help="Reading frequency (Hz)")
    parser.add_argument("--config", type=str, default="/home/cxl/dexaoyi/robot_infra/franka_env/dexhand/psi-glove-driver/config/default_config.yaml", help="Config file")
    parser.add_argument("--no-left", action="store_true", help="Disable left hand")
    parser.add_argument("--no-right", action="store_false", help="Disable right hand")
    parser.add_argument("--debug", action="store_false", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # 配置日志
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # 创建控制器
    controller = PSIGloveStandalone(
        left_port=None if args.no_left else args.left_port,
        right_port=None if args.no_right else args.right_port,
        baudrate=args.baudrate,
        frequency=args.frequency,
        config_file=args.config,
        auto_connect=True,
    )
    
    # 检查连接
    if controller.left_hand and not controller.left_hand.is_connected():
        logger.error(f"Failed to connect to left hand: {args.left_port}")
    if controller.right_hand and not controller.right_hand.is_connected():
        logger.error(f"Failed to connect to right hand: {args.right_port}")
    
    if (controller.left_hand and not controller.left_hand.is_connected()) or \
       (controller.right_hand and not controller.right_hand.is_connected()):
        logger.error("Connection failed. Exiting.")
        return
    
    logger.info(f"Connected. Reading at {args.frequency}Hz")
    logger.info("Press Ctrl+C to stop")
    
    # 运行主循环
    controller.run(print_output=True)


if __name__ == "__main__":
    main()