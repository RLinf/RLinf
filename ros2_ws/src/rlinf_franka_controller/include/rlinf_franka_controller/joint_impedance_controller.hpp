// Copyright (c) 2025 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <franka_example_controllers/motion_generator.hpp>
#include <Eigen/Eigen>
#include <array>
#include <atomic>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <string>

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace rlinf_franka_controller {

/**
 * Controller to move the robot to a desired joint position.
 */
class JointImpedanceController
    : public controller_interface::ControllerInterface {
public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  [[nodiscard]] controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;
  controller_interface::return_type
  update(const rclcpp::Time &time, const rclcpp::Duration &period) override;
  CallbackReturn on_init() override;
  CallbackReturn
  on_configure(const rclcpp_lifecycle::State &previous_state) override;
  CallbackReturn
  on_activate(const rclcpp_lifecycle::State &previous_state) override;

private:
  enum class ControlState { HOLDING, RESETTING, TRACKING };
  struct JointTarget {
    std::array<double, 7> positions{};
    rclcpp::Time received_time{};
    bool valid{false};
  };

  std::string arm_id_;
  std::string namespace_prefix_;
  const int num_joints = 7;
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d k_gains_;
  Vector7d d_gains_;
  Vector7d motion_target_positions_;
  Vector7d filtered_target_positions_;
  double k_alpha_;
  double target_filter_cutoff_;
  double reset_speed_factor_;
  double command_timeout_;
  double max_command_error_;
  ControlState control_state_{ControlState::HOLDING};
  std::atomic<bool> reset_requested_{false};
  bool move_to_start_position_finished_{false};
  rclcpp::Time start_time_;
  rclcpp::Time command_epoch_;
  std::unique_ptr<MotionGenerator> motion_generator_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
      joint_state_subscriber_ = nullptr;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr reset_subscriber_ =
      nullptr;
  realtime_tools::RealtimeBuffer<JointTarget> joint_target_buffer_;
  realtime_tools::RealtimeBuffer<JointTarget> reset_target_buffer_;

  Vector7d calculateTauDGains_(const Vector7d &q_goal);
  bool validateGains_(const std::vector<double> &gains,
                      const std::string &gains_name);
  void initializeResetMotion_();
  void updateJointStates_();
  void jointStateCallback_(const sensor_msgs::msg::JointState msg);
  void resetTargetCallback_(const sensor_msgs::msg::JointState msg);
};

} // namespace rlinf_franka_controller
