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

#include <rlinf_franka_controller/joint_impedance_controller.hpp>

#include <Eigen/Eigen>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <string>

using std::placeholders::_1;

namespace rlinf_franka_controller {

controller_interface::InterfaceConfiguration
JointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" +
                           std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" +
                           std::to_string(i) + "/position");
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" +
                           std::to_string(i) + "/velocity");
  }
  return config;
}

controller_interface::return_type
JointImpedanceController::update(const rclcpp::Time &time,
                                 const rclcpp::Duration &period) {
  updateJointStates_();
  Vector7d q_goal;
  Vector7d tau_d_calculated;

  if (reset_requested_.exchange(false)) {
    initializeResetMotion_();
  }

  const JointTarget joint_target = *joint_target_buffer_.readFromRT();
  const bool new_target = joint_target.valid &&
                          joint_target.received_time > command_epoch_;

  if (control_state_ == ControlState::RESETTING) {
    auto trajectory_time = this->get_node()->now() - start_time_;
    auto motion_generator_output =
        motion_generator_->getDesiredJointPositions(trajectory_time);
    move_to_start_position_finished_ = motion_generator_output.second;
    q_goal = motion_generator_output.first;
    if (move_to_start_position_finished_) {
      control_state_ = ControlState::HOLDING;
      command_epoch_ = time;
      motion_target_positions_ = q_goal;
      filtered_target_positions_ = q_goal;
      RCLCPP_INFO(get_node()->get_logger(),
                  "Controlled joint motion complete; holding target.");
    }
  } else if (new_target || control_state_ == ControlState::TRACKING) {
    const double command_age = (time - joint_target.received_time).seconds();
    if (!joint_target.valid || command_age < -0.01 ||
        command_age > command_timeout_) {
      for (int i = 0; i < num_joints; ++i) {
        command_interfaces_[i].set_value(0.0);
      }
      RCLCPP_FATAL(get_node()->get_logger(),
                   "RLinf joint command timed out (age %.6f s, limit %.3f s).",
                   command_age, command_timeout_);
      rclcpp::shutdown();
      return controller_interface::return_type::ERROR;
    }
    if (new_target) {
      control_state_ = ControlState::TRACKING;
    }
    Vector7d raw_target;
    for (int i = 0; i < num_joints; ++i) {
      raw_target(i) = joint_target.positions[i];
    }
    constexpr double kTwoPi = 6.283185307179586;
    const double filter_alpha =
        1.0 - std::exp(-kTwoPi * target_filter_cutoff_ * period.seconds());
    filtered_target_positions_ +=
        filter_alpha * (raw_target - filtered_target_positions_);
    const Vector7d tracking_error = filtered_target_positions_ - q_;
    for (int i = 0; i < num_joints; ++i) {
      q_goal(i) = q_(i) + std::clamp(tracking_error(i), -max_command_error_,
                                     max_command_error_);
    }
  } else {
    q_goal = motion_target_positions_;
  }

  tau_d_calculated = calculateTauDGains_(q_goal);

  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
  }

  return controller_interface::return_type::OK;
}

void JointImpedanceController::jointStateCallback_(
    const sensor_msgs::msg::JointState msg) {
  if (msg.position.size() != static_cast<std::size_t>(num_joints)) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Received joint target size %zu; expected %d.",
                msg.position.size(), num_joints);
    return;
  }

  JointTarget target;
  std::copy(msg.position.begin(), msg.position.end(), target.positions.begin());
  const rclcpp::Time received_time = get_node()->now();
  const rclcpp::Time message_stamp(msg.header.stamp,
                                   received_time.get_clock_type());
  const double message_age = (received_time - message_stamp).seconds();
  target.valid = message_age >= -0.01 && message_age < 0.5;
  for (const double position : target.positions) {
    target.valid = target.valid && std::isfinite(position);
  }
  if (!target.valid) {
    RCLCPP_WARN(get_node()->get_logger(),
                "RLinf joint target is invalid or stale; message age: %.6f s.",
                message_age);
    return;
  }
  // Use publisher time as the epoch so a delayed pre-reset command cannot
  // become active after the controlled motion completes.
  target.received_time = message_stamp;
  joint_target_buffer_.writeFromNonRT(target);
}

void JointImpedanceController::resetTargetCallback_(
    const sensor_msgs::msg::JointState msg) {
  if (msg.position.size() != static_cast<std::size_t>(num_joints)) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Received controlled-motion target size %zu; expected %d.",
                msg.position.size(), num_joints);
    return;
  }

  JointTarget target;
  std::copy(msg.position.begin(), msg.position.end(), target.positions.begin());
  target.valid = true;
  for (const double position : target.positions) {
    target.valid = target.valid && std::isfinite(position);
  }
  if (!target.valid) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Controlled-motion target contains a non-finite position.");
    return;
  }
  reset_target_buffer_.writeFromNonRT(target);
  reset_requested_.store(true);
}

CallbackReturn JointImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<double>("k_alpha", 0.99);
    auto_declare<double>("target_filter_cutoff", 30.0);
    auto_declare<double>("reset_speed_factor", 0.012);
    auto_declare<double>("command_timeout", 0.5);
    auto_declare<double>("max_command_error", 0.05);
    auto_declare<std::string>("command_topic", "rlinf/joint_targets");
    auto_declare<std::string>("reset_topic", "rlinf/reset_joint_target");
  } catch (const std::exception &e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n",
            e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_configure(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  namespace_prefix_ = get_node()->get_namespace();
  if (namespace_prefix_ == "/" || namespace_prefix_.empty()) {
    namespace_prefix_.clear();
  } else {
    // Remove leading slash and add trailing underscore
    namespace_prefix_ = namespace_prefix_.substr(1) + "_";
  }

  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  auto k_alpha = get_node()->get_parameter("k_alpha").as_double();
  target_filter_cutoff_ =
      get_node()->get_parameter("target_filter_cutoff").as_double();
  reset_speed_factor_ =
      get_node()->get_parameter("reset_speed_factor").as_double();
  command_timeout_ = get_node()->get_parameter("command_timeout").as_double();
  max_command_error_ =
      get_node()->get_parameter("max_command_error").as_double();

  if (!validateGains_(k_gains, "k_gains") ||
      !validateGains_(d_gains, "d_gains")) {
    return CallbackReturn::FAILURE;
  }

  for (int i = 0; i < num_joints; ++i) {
    d_gains_(i) = d_gains.at(i);
    k_gains_(i) = k_gains.at(i);
  }
  if (k_alpha < 0.0 || k_alpha > 1.0) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "k_alpha should be in the range [0, 1]");
    return CallbackReturn::FAILURE;
  }
  if (reset_speed_factor_ <= 0.0 || reset_speed_factor_ > 1.0) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "reset_speed_factor must be in the range (0, 1]");
    return CallbackReturn::FAILURE;
  }
  if (target_filter_cutoff_ <= 0.0 || command_timeout_ <= 0.0 ||
      max_command_error_ <= 0.0) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "target_filter_cutoff, command_timeout, and max_command_error "
                 "must be positive");
    return CallbackReturn::FAILURE;
  }

  k_alpha_ = k_alpha;

  dq_filtered_.setZero();

  joint_state_subscriber_ =
      get_node()->create_subscription<sensor_msgs::msg::JointState>(
          get_node()->get_parameter("command_topic").as_string(), 1,
          [this](const sensor_msgs::msg::JointState &msg) {
            jointStateCallback_(msg);
          });
  reset_subscriber_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      get_node()->get_parameter("reset_topic").as_string(), 1,
      [this](const sensor_msgs::msg::JointState &msg) {
        resetTargetCallback_(msg);
      });

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_activate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  control_state_ = ControlState::HOLDING;
  reset_requested_.store(false);
  move_to_start_position_finished_ = false;
  motion_generator_.reset();
  dq_filtered_.setZero();
  start_time_ = this->get_node()->now();
  command_epoch_ = start_time_;
  JointTarget empty_target;
  empty_target.received_time = start_time_;
  joint_target_buffer_.initRT(empty_target);
  JointTarget reset_target;
  reset_target.valid = false;
  reset_target_buffer_.initRT(reset_target);
  updateJointStates_();
  motion_target_positions_ = q_;
  filtered_target_positions_ = q_;

  return CallbackReturn::SUCCESS;
}

auto JointImpedanceController::calculateTauDGains_(const Vector7d &q_goal)
    -> Vector7d {
  dq_filtered_ = (1 - k_alpha_) * dq_filtered_ + k_alpha_ * dq_;
  Vector7d tau_d_calculated;
  tau_d_calculated =
      k_gains_.cwiseProduct(q_goal - q_) + d_gains_.cwiseProduct(-dq_filtered_);

  return tau_d_calculated;
}

bool JointImpedanceController::validateGains_(const std::vector<double> &gains,
                                              const std::string &gains_name) {
  if (gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "%s parameter not set",
                 gains_name.c_str());
    return false;
  }

  if (gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(),
                 "%s should be of size %d but is of size %ld",
                 gains_name.c_str(), num_joints, gains.size());
    return false;
  }

  return true;
}

void JointImpedanceController::updateJointStates_() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto &position_interface = state_interfaces_.at(2 * i);
    const auto &velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

void JointImpedanceController::initializeResetMotion_() {
  updateJointStates_();
  const JointTarget reset_target = *reset_target_buffer_.readFromRT();
  if (!reset_target.valid) {
    return;
  }
  for (int i = 0; i < num_joints; ++i) {
    motion_target_positions_(i) = reset_target.positions[i];
  }
  start_time_ = this->get_node()->now();
  command_epoch_ = start_time_;
  motion_generator_ = std::make_unique<MotionGenerator>(
      reset_speed_factor_, q_, motion_target_positions_);
  move_to_start_position_finished_ = false;
  control_state_ = ControlState::RESETTING;
  RCLCPP_INFO(get_node()->get_logger(),
              "Starting controlled joint motion at speed factor %.4f.",
              reset_speed_factor_);
}

} // namespace rlinf_franka_controller
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(rlinf_franka_controller::JointImpedanceController,
                       controller_interface::ControllerInterface)
