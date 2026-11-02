Real-World RL with UR5
=======================

.. figure:: https://upload.wikimedia.org/wikipedia/commons/5/57/Cobot.jpg
   :align: center
   :width: 60%

   Universal Robots UR5 cobot. Image: Wikimedia Commons, GrowSkills Robotics.

Train and check a real-world Universal Robots UR5 arm through either of two
arm backends: ``ur_rtde``, which drives the controller directly over
Ethernet, and ``ur_ros2``, which commands an already-running
`Universal_Robots_ROS2_Driver <https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver>`_
instead. The shipped reach task holds a six-joint target configuration, the
smallest task that exercises the whole path from composing the robot to
scoring a result, so a UR5 can be brought up before a manipulation task is
written for it.

Overview
--------

Train an MLP SAC policy to reach a joint-space target on a UR5.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      MLP policy

   .. grid-item-card:: Algorithms
      :text-align: center

      SAC

   .. grid-item-card:: Tasks
      :text-align: center

      Joint reach

   .. grid-item-card:: Hardware
      :text-align: center

      UR5 · ``ur_rtde`` / ``ur_ros2`` · optional camera

| **You'll do:** check hardware → install → run the hardware check → launch training.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a reachable UR5 controller or URSim.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Task
     - Config / entry point
     - Description
   * - Joint reach
     - ``ur5_ursim_reach``
     - Hold a six-joint target configuration; validates the control link and
       joint-space motion end to end.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Description
   * - Observation
     - Six joint positions in radians, plus one RGB frame per configured
       wrist camera.
   * - Action
     - 6-D, one value per joint. The policy sees ``[-1, 1]`` (``normalize_actions``
       rescales it to the configured absolute-radian travel limits before the
       arm sees it); set ``normalize_actions: False`` to command absolute
       radians directly instead.
   * - Reward
     - ``1.0`` once every joint is within ``reward_threshold`` radians of the
       target (sparse); the negated joint distance when
       ``use_dense_reward`` is set.
   * - Prompt
     - ``reach a joint configuration``

Hardware Setup
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Requirement
   * - Arm
     - Universal Robots UR5, reachable over Ethernet from the controller
       node. The ``ur_rtde`` backend talks to the controller's RTDE
       interfaces directly; the ``ur_ros2`` backend instead talks to an
       already-running ``Universal_Robots_ROS2_Driver`` (or URSim standing in
       for it) over the ROS graph and never opens its own connection to the
       arm's IP.
   * - Controller node
     - Linux computer on the same network as the arm (or URSim), running the
       env worker. No GPU required.
   * - Training / rollout node
     - A computer with GPU support for the MLP policy. Can be the same
       machine as the controller node for a single-node run.
   * - Cameras (optional)
     - One or more wrist cameras (``realsense``, ``zed``, or ``lumos``); the
       reach task runs with none configured.

Mount the arm on a stable surface with clear space through its full travel,
and clear the workspace before connecting: the ``ur_ros2`` backend restarts
the driver's external-control program on connect, and either backend moves
the arm to ``reset_joint_qpos`` whenever the environment is constructed with
``reset_on_init`` left at its default of ``True``.

.. warning::

   Constructing the environment commands this initial move by default, along
   a trajectory paced by ``reset_duration`` (minimum duration, default 3 s)
   and ``reset_joint_speed`` (commanded speed limit, default 30 deg/s). Set
   ``reset_on_init: False`` in the env config to skip it, or set
   ``reset_joint_qpos`` to the arm's current pose before the first
   construction.

Task-Specific Safety Constraints
---------------------------------

``joint_limit_low`` and ``joint_limit_high`` (default ``+/- pi`` on every
joint) bound the commanded action space and are validated against the
arm driver's own travel, but they are travel limits only, not a
collision-safe workspace guarantee: a configuration inside them can still
drive the arm into itself, a camera mount, or anything else sharing the
workspace. Validate your own workspace and task bounds separately, for
example by narrowing ``joint_limit_low``/``joint_limit_high``,
``reset_joint_qpos``, and ``target_joint_qpos`` in the env config to a range
you have checked is clear. Construction rejects a ``reset_joint_qpos`` or
``target_joint_qpos`` outside the configured bounds instead of silently
clipping it.

Installation
------------

Install the robot dependencies on the controller node using the
custom-environment workflow below. Run the checks and training from the same
checkout and activated environment.

Robot Controller Node
~~~~~~~~~~~~~~~~~~~~~

Clone RLinf on the machine connected to the robot, then install its
environment.

.. include:: _setup_common.rst
   :end-before: Then set up the dependencies

Install the robot dependencies from the repository root:

.. code-block:: bash

   # Mainland China users can add --use-mirror for faster downloads.
   bash requirements/install.sh embodied --env ur5
   source .venv/bin/activate
   export REPO_PATH="$PWD"

Run subsequent commands from this checkout with the environment activated.
This installs both arm backends; ``ur_rtde`` needs nothing further. See
:doc:`Installation </rst_source/start/installation>` for platform
prerequisites.

ROS2 Backend (``ur_ros2``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ur_ros2`` backend reaches the arm through an already-running
``Universal_Robots_ROS2_Driver``; it does not launch the driver or a
simulator. Bring up URSim or the real controller, then the driver, before
the env worker connects:

.. code-block:: bash

   ros2 launch ur_robot_driver ur_control.launch.py \
     ur_type:=ur5 robot_ip:=<robot-ip> \
     launch_rviz:=false headless_mode:=true

This starts ``scaled_joint_trajectory_controller`` (the controller this
backend commands) and publishes ``/joint_states``. Match the env config's
``headless_mode`` to how you launched the driver: in headless mode (the
default), connecting restarts the driver's external-control program through
its ``resend_robot_program`` service; outside headless mode, start External
Control from the pendant first and set ``headless_mode: False``, or the
restart call reports a spurious failure.

ROS2 Jazzy's ``rclpy`` targets Python 3.12, while the UR5 install above
defaults to Python 3.11.14 and cannot import it. Give the env-worker process
a Python 3.12 environment with both ``rclpy`` and the rest of the UR5 install
instead of a bare ``venv``, which would only get ``rclpy``:

.. code-block:: bash

   python3.12 -m venv --system-site-packages .venv-ros2
   bash requirements/install.sh embodied --env ur5 \
     --python 3.12 --venv .venv-ros2

Activate that environment and source the ROS2 Jazzy setup script before
running the env worker from it instead of the default ``.venv``:

.. code-block:: bash

   source .venv-ros2/bin/activate
   source /opt/ros/jazzy/setup.bash

A controller running a UR e-Series robot restricts ``scaled_joint_trajectory_
controller`` to speeds scaled by the pendant's realtime speed slider; if you
run below 100%, set the env config's ``speed_scaling`` to that fraction
(for example ``0.1`` at 10%) so a step's completion wait is stretched to
match, instead of timing out a trajectory that is still healthy at the
reduced speed.

RTDE Backend (``ur_rtde``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ur_rtde`` backend opens ``RTDEControlInterface`` and
``RTDEReceiveInterface`` directly against the controller IP, with no
external driver to launch. Put the controller in Remote Control mode
(``Settings`` > ``System`` > ``Remote Control`` on an e-Series pendant) so an
external RTDE client can command motion instead of only the pendant.

Run It
------

With the arm connected and a backend chosen, check that it composes,
connects, reads state, and lets go, without moving the arm:

.. code-block:: bash

   python -m toolkits.realworld_check.check_robot_parts UR5 \
     --arg robot_ip=<robot-ip> --arg node_rank=0 --arg backend=ur_rtde

Swap ``--arg backend=ur_ros2`` to check the ROS2 path instead, once the
driver (or URSim) is running and the environment above is activated. The
command prints what each part observes, then disconnects and confirms
nothing still claims to be connected; it reports joint positions and, for
``ur_rtde``, the tool pose, but it never commands a reset or a target, so the
arm does not move during this check.

With the hardware check passing, launch the shipped reach task against
URSim or the real controller from the repository root:

.. code-block:: bash

   bash examples/embodiment/run_realworld.sh ur5_ursim_reach

``examples/embodiment/config/ur5_ursim_reach.yaml`` targets URSim's bridge
address by default; point ``robot_ip`` at your controller for real hardware,
matching whatever backend and ROS graph you configured above. Clear the
workspace before launching: construction moves the arm to its reset pose, as
the warning above describes.

Visualization and Results
--------------------------

Follow training in TensorBoard and the terminal log:

.. code-block:: text

   logs/<timestamp>-ur5_ursim_reach/run_embodiment.log

Watch ``env/success_once`` for reach success; see
:doc:`Training Metrics </rst_source/reference/metrics>` for the full metric
list.

New Real-World Tasks
--------------------

The reach task above is the smallest one that brings up a new UR5; follow
:doc:`New Real-World Tasks </rst_source/extending/new_task>` to define a
different task's reset procedure, observations, actions, reward, and success
condition on top of the same arm.
