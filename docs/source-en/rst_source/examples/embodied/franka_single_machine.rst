Real-World RL with Franka
=========================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%

   Run the Franka real-world RL workflow on a single machine with a GPU.

Run real-world SAC / RLPD / PPO training with the compute node (training /
rollout) and control node (Franka control) on one GPU host. This workflow uses
two independent RLinf environments and has been validated on standard Ubuntu
20.04 with Franka System Image 5.9.2 and libfranka 0.19.0. On a non-real-time
kernel, you must explicitly disable libfranka's real-time check as described
below; installing newer firmware and libfranka alone does not disable it.

.. note::

   Use this page for the current single-host workflow. If you use an older
   firmware/libfranka combination or want a dedicated real-time control host,
   follow the archived multi-machine workflow at :doc:`franka`.

.. note::

   In the single-machine setup, the compute node (actor / rollout / reward) acts
   as the Ray head (rank 0), and the control node (Franka env) joins the same
   machine's cluster as rank 1. Because the two roles use different Python
   environments, you activate and join the cluster in two separate terminals.

Overview
--------

Train a real-world manipulation policy from camera observations and robot feedback.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      CNN policy · OpenPI π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      SAC · Cross-Q · RLPD · PPO

   .. grid-item-card:: Tasks
      :text-align: center

      Peg insertion · charger · PnP

   .. grid-item-card:: Hardware
      :text-align: center

      Franka · RealSense/ZED · gripper

| **You'll do:** install controller deps → collect demos → start Ray → launch real-world training → watch ``env/reward`` and videos.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Franka firmware/libfranka match · local network · safety operator.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - Task
     - Config / entry point
     - Description
   * - Peg insertion
     - ``realworld_peginsertion_rlpd_cnn_async``
     - Insert a peg at a target end-effector pose.
   * - Charger
     - ``realworld_charger_sac_cnn_async``
     - Align and insert a charger using real-world reward feedback.
   * - PnP / eval
     - ``realworld_pnp_*``
     - Collect or deploy pick-and-place style policies.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - Field
     - Description
   * - Observation
     - RGB camera frames plus optional robot state.
   * - Action
     - 6D/7D continuous Cartesian delta action, optionally with gripper control.
   * - Reward
     - Task success, keyboard labels, or dense task-specific feedback.
   * - Prompt
     - Real-world task text in the env config when a VLA policy is used.

Hardware Setup
---------------

The single-machine setup requires the following hardware components:

- **Robotic Arm**: Franka Emika Panda
- **Cameras**: Intel RealSense cameras (default) or Stereolabs ZED cameras
- **Gripper**: Franka hand (default) or Robotiq 2F-85/2F-140
- **Host**: A single computer with GPU support that serves as both the compute node (trains the CNN policy) and the robot control node (controls the Franka arm)
- **Space Mouse (Optional)**: For teleoperation data collection or human intervention during training.
- **GELLO (Optional)**: A joint-level teleoperation device as an alternative to SpaceMouse, providing more intuitive control with native gripper support.
- **VR / PICO (Optional)**: A headset-and-controller teleoperation device for 6D end-effector control, usable as an alternative to SpaceMouse for data collection.

.. warning::

  The host must be networked in the same local network as the robot arm.

.. note::

   **Using ZED cameras or Robotiq grippers?**  See the dedicated guide
   :doc:`franka_zed_robotiq` for SDK installation, serial-device setup,
   YAML configuration fields, and data collection.

   **Using VR / PICO teleoperation?** See :doc:`franka_vr` for
   XRoboToolkit, ZeroMQ, PICO wrapper configuration, and operation steps.

Check Franka Firmware Version
-----------------------------

Go to the robot's management webpage (usually at ``http://<robot_ip>/desk``), click on the ``SETTINGS`` tab, and check the version number following ``Control`` in ``DashBoard`` as follows.
Please take a note of the firmware version for later use when setting ``LIBFRANKA_VERSION``.

.. raw:: html

  <div style="flex: 1; text-align: center;">
      <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_firmware_single_machine.png" style="width: 60%;"/>
  </div>

.. note::

   Based on your firmware version, refer to the `Franka compatibility matrix <https://frankarobotics.github.io/docs/compatibility.html>`_
   to choose a matching libfranka version, and specify it via the environment variables ``LIBFRANKA_VERSION`` and ``FRANKA_ROS_VERSION``.

Environment Installation
------------------------

The single-machine setup requires **cloning two RLinf repositories on the same
host**, one for each role:

- **`RLinf-franka`**: the control / data collection environment (Franka control dependencies: ROS Noetic, libfranka, franka_ros, serl_franka_controllers).
- **`RLinf-compute`**: the compute / rollout environment (RLinf framework and the Python dependencies for real-world RL training).

This isolates the two loosely-coupled dependency sets in separate virtual
environments to avoid them interfering with each other.

A. Clone the Repositories
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git RLinf-franka
   git clone https://github.com/RLinf/RLinf.git RLinf-compute

B. Install the Control Environment (RLinf-franka)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the Franka control dependencies in the **`RLinf-franka`** directory.
Set the libfranka and franka_ros versions according to your firmware version
(for example, firmware 5.9.2 corresponds to ``LIBFRANKA_VERSION=0.19.0``):

.. code:: bash

   cd RLinf-franka
   # Specify the libfranka / franka_ros versions based on your firmware version
   export LIBFRANKA_VERSION=0.19.0
   export FRANKA_ROS_VERSION=0.10.0

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.
   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

C. Configure Real-Time Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

libfranka enforces real-time scheduling by default. Installing System Image
5.9.0+ and libfranka 0.18.0+ does **not** automatically allow control from a
standard Linux kernel. If the host does not use a PREEMPT_RT kernel, explicitly
set the backend to ignore the real-time check.

For the ``franka_ros`` backend used by this workflow, edit the installed
configuration:

.. code-block:: yaml

   # .venv/franka_catkin_ws/src/franka_ros/franka_control/config/franka_control_node.yaml
   realtime_config: ignore  # Default: enforce

For the ``franky`` backend, pass ``RealtimeConfig.Ignore`` when constructing
the robot:

.. code-block:: python

   import franky

   robot = franky.Robot(
       "172.16.0.2",
       relative_dynamics_factor=0.2,
       realtime_config=franky.RealtimeConfig.Ignore,
   )

.. note::

   RLinf's current ``FrankyController`` constructs ``franky.Robot`` with its
   default ``RealtimeConfig.Enforce`` setting. Using that backend without a
   real-time kernel therefore requires the corresponding constructor change;
   it cannot be enabled through an RLinf YAML option yet.

.. warning::

   ``ignore`` disables libfranka's real-time startup check; it does not make a
   standard kernel real-time. Training can heavily load the host, and delayed
   OS scheduling can cause missed control deadlines or communication errors.
   Install and use a PREEMPT_RT kernel for reliable control whenever possible,
   and follow the official `real-time kernel guide
   <https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_.

D. Install the Compute Environment (RLinf-compute)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the RLinf framework and the training dependencies in the **`RLinf-compute`** directory
(corresponding to the model and simulation environment you train with):

.. code:: bash

   cd RLinf-compute
   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.
   bash requirements/install.sh --model openpi --env libero
   source .venv/bin/activate

.. note::

   The model / simulation env parameters for the compute environment should match
   your config (for example, a CNN policy uses openpi; use whatever your training
   model actually requires).

.. note::

   Both cloned repositories contain the startup scripts under ``ray_utils/realworld/``
   (``setup_compute_node.sh``, ``setup_franka_node.sh``, ``setup_franka_collect.sh``, ``cleanup.sh``).
   Each script automatically locates the repository root it lives in and uses that
   repository's virtual environment, so run the corresponding script from **its own
   repository directory** (``cd RLinf-compute`` / ``cd RLinf-franka``).

Download the Model
------------------

Before starting training, you need to download the corresponding pretrained model:

.. code:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

After downloading, make sure to correctly specify the model path in the configuration yaml file.

Run It
------

Prerequisites
~~~~~~~~~~~~~~~

**Get the Target Pose for the Task**

To acquire the target pose for the peg-insertion task, you can use the `toolkits.realworld_check.test_franka_controller` script.

First, you need to activate your Franka robot's programming mode, and manually move the robot to the desired target pose.

Then, before running, activate the **control environment (RLinf-franka)** and set the environment variable ``FRANKA_ROBOT_IP`` to your robot's IP address:

.. code-block:: bash

   cd RLinf-franka
   export FRANKA_ROBOT_IP=<your_robot_ip_address>
   source ray_utils/realworld/setup_franka_node.sh   # Only activate the control env, do not start Ray

Next, run the script:

.. code-block:: bash

   python -m toolkits.realworld_check.test_franka_controller

The script will prompt you to input command, you can enter `getpos_euler` to get the current end-effector pose in Euler angles.

Data Collection
~~~~~~~~~~~~~~~~~

For RLPD experiments, you need to first collect some initial data for training.
The data collection only needs to run on the **control node (RLinf-franka)** as a single machine, with the collection node as the only Ray head.

1. Use the data-collection startup script in the control repository. It activates the environment and starts a single-node Ray head (rank 0):

.. code-block:: bash

   cd RLinf-franka
   export FRANKA_ROBOT_IP=<your_robot_ip_address>
   export RLINF_HEAD_IP=<this_host_ip_address>
   # Optional: export RLINF_COMM_NET_DEVICES=<network_device>  # defaults to eth0
   source ray_utils/realworld/setup_franka_collect.sh start

.. note::

   The script sources the ROS and catkin setup scripts first, then activates the
   virtual environment. When using an environment installed by ``install.sh``,
   these sources are usually already handled during activation (see the script comments).

2. Modify the configuration file ``examples/embodiment/config/realworld_collect_data.yaml`` by filling your robot's IP address to the field ``robot_ip``.

.. code-block:: yaml

  cluster:
    num_nodes: 1
    component_placement:
      env:
        node_group: franka
        placement: 0
    node_groups:
      - label: franka
        node_ranks: 0
        hardware:
          type: Franka
          configs:
            - robot_ip: ROBOT_IP
              node_rank: 0

Modify the `target_ee_pose` field in the configuration file to the target pose you have acquired in the previous step.

.. code-block:: yaml

  env:
    eval:
      override_cfg:
        target_ee_pose: [0.5, 0.0, 0.1, -3.14, 0.0, 0.0]

4. Run the data collection script:

.. code-block:: bash

   bash examples/embodiment/collect_data.sh

During the data collection, you can manually intervene the robot using a space mouse to collect data.

The script will terminate after 20 episodes of data collection (can be configured with the `num_data_episodes` field in the configuration file), and the collected data will be stored in the `logs/[running-timestamp]/data.pkl` path.

5. After data collection, use the collected data path for the later training (read by the compute environment).

.. note::

   **Using ZED cameras and Robotiq grippers?**  A dedicated data collection
   script and config are available.  See the
   :ref:`Data Collection <franka-zed-robotiq-data-collection>` section in
   :doc:`franka_zed_robotiq`.

Data Collection with GELLO
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to SpaceMouse, RLinf also supports using `GELLO <https://github.com/wuphilipp/gello_software>`_ for teleoperation data collection.
GELLO is a joint-level teleoperation device that mirrors the kinematic structure of the Franka arm, providing more intuitive and precise control with full gripper support.

**Prerequisites**

- Install the ``gello`` and ``gello-teleop`` packages. See :doc:`franka_gello` for detailed installation instructions.
- A GELLO device connected to the host via USB serial.
- Identify your GELLO serial port (e.g. ``/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0``).
  You can list available serial ports with:

  .. code-block:: bash

     ls /dev/serial/by-id/

**Configuration**

Use the config file ``examples/embodiment/config/realworld_collect_data_gello.yaml``.
The key differences from the SpaceMouse config are:

.. code-block:: yaml

   env:
     eval:
       use_spacemouse: False
       use_gello: True
       gello_port: "/dev/serial/by-id/usb-FTDI_..."  # Replace with your GELLO serial port

**Running**

.. code-block:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data_gello

The workflow is the same as SpaceMouse collection: use the GELLO device to demonstrate the task, and the script will automatically save successful episodes.

Cluster Setup
~~~~~~~~~~~~~~~~~

Before starting the experiment, you will first setup the ray cluster properly.

.. warning::
  This step is essential, proceed with caution! Even the slightest misconfiguration may result in missing packages or failure to control the robot.

RLinf uses ray for managing distributed environments. So it is subject to one critical characteristic of ray: when you run `ray start` on a node, the current Python interpreter and environment variables will be recorded by ray, and all the processes started by ray on that node later will inherit the same Python interpreter and environment variables.

In the single-machine setup, the **compute node (RLinf-compute)** acts as the Ray head (rank 0), and the **control node (RLinf-franka)** joins it as rank 1. Because the two roles use different Python environments, you activate and join the cluster in two separate terminals.

The repository provides the following startup scripts (under ``ray_utils/realworld/``):

- ``setup_compute_node.sh``: the compute node (rank 0, head). ``source`` it and pass ``start`` to launch Ray.
- ``setup_franka_node.sh``: the control node (rank 1, worker). ``start`` joins the compute head.
- ``setup_franka_collect.sh``: data-collection only (single node, rank 0). See the Data Collection section.
- ``cleanup.sh``: cleans up leftover Ray / ROS / FrankaController processes (run directly with ``bash cleanup.sh``).

The first three scripts support ``source <script>`` / ``source <script> start`` / ``source <script> stop`` and verify that key dependencies can be imported before startup.

Set the required environment variables **before** sourcing each script:

- ``setup_compute_node.sh start`` requires ``RLINF_NODE_IP`` (the compute head's reachable IP).
- ``setup_franka_node.sh start`` requires ``FRANKA_ROBOT_IP`` and ``RLINF_HEAD_IP`` (the compute head's IP).
- ``setup_franka_collect.sh start`` requires ``FRANKA_ROBOT_IP`` and ``RLINF_HEAD_IP`` (this host's IP for the collection-only Ray head).
- ``cleanup.sh`` requires no environment variables, but ``ray`` must be available on ``PATH`` (activate either RLinf environment first).

The optional variables ``RLINF_VENV``, ``RLINF_COMM_NET_DEVICES``, and
``RAY_TEMP_DIR`` retain the defaults documented in the script headers.

**Terminal 1: compute environment (rank 0, head)**

.. code-block:: bash

   cd RLinf-compute
   export RLINF_NODE_IP=<this_node_reachable_ip>
   # Optional: export RLINF_COMM_NET_DEVICES=<network_device>  # defaults to eth0
   source ray_utils/realworld/setup_compute_node.sh start

**Terminal 2: control environment (rank 1, worker)**

.. code-block:: bash

   cd RLinf-franka
   export RLINF_HEAD_IP=<compute_head_ip_address>
   export FRANKA_ROBOT_IP=<your_robot_ip_address>
   source ray_utils/realworld/setup_franka_node.sh start

.. note::

   Each script activates its own virtual environment. On the control node, the
   script sources the ROS and catkin setup scripts first, then activates the
   virtual environment. When using an environment installed by ``install.sh``,
   these sources are usually already handled during activation.

.. warning::

   The two scripts use different ``--temp-dir`` by default
   (``/tmp/rlinf_compute`` and ``/tmp/rlinf_control``), which is how Ray recognizes
   them as two separate nodes; never reuse the same environment variables in one terminal.

You can run `ray status` to check if the cluster is set up correctly (it should show 2 nodes).

Configuration File
~~~~~~~~~~~~~~~~~~

Before starting the experiment, you need to modify the configuration file, ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml`` according to your setup.

Similarly, you first need to fill your robot's IP address to the field ``robot_ip`` and the target end-effector pose to the field ``target_ee_pose``.

Then, change the ``model_path`` field in both ``rollout`` and ``actor`` sections to the path where you have downloaded the pretrained model.
Change the ``data.path`` field to the path where you have uploaded the collected demo data.

Headless Keyboard Reward Wrapper (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to label rewards from a physical keyboard by human, enable the keyboard wrapper in the real-world env config.

For example, in ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``:

.. code-block:: yaml

   env:
     train:
       keyboard_reward_wrapper: single_stage  # or multi_stage

The available modes are:

- ``single_stage``: press ``a`` for failure reward, ``b`` for neutral reward, and ``c`` for success reward.
- ``multi_stage``: press ``a`` / ``b`` / ``c`` to switch among reward stages, and press ``q`` to emit a negative reward.

The keyboard listener reads Linux input devices directly, so you should export ``RLINF_KEYBOARD_DEVICE`` before starting ray on the control node.

First, list the available keyboard devices:

.. code-block:: bash

   ls -l /dev/input/by-id/*-event-kbd

This command shows the stable keyboard name and the corresponding ``eventX`` device. For example, ``usb-Logitech_USB_Keyboard-event-kbd -> ../event20`` means the keyboard device is ``/dev/input/event20``.

Before starting training, grant access to that event device:

.. code-block:: bash

   chmod 666 /dev/input/event20

Then export the event device in your setup script or shell before ``ray start``:

.. code-block:: bash

   export RLINF_KEYBOARD_DEVICE=/dev/input/event20

Testing the Setup (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide several test scripts to verify that the setup is correct before starting the experiment. This step is optional but recommended.

First, test the camera connection:

.. code-block:: bash

   python -m toolkits.realworld_check.test_franka_camera

Next, test the basic cluster setup by running a dummy setup. Refer to ``examples/embodiment/config/realworld_dummy_franka_sac_cnn.yaml`` and add `env.eval.override_cfg`.
You can set the `is_dummy` field to `True` in both `env.train.override_cfg` and `env.eval.override_cfg` sections in the configuration file to enable the dummy setup.
And fill the camera serial numbers obtained from running ``toolkits.realworld_check.test_franka_camera.py`` into the field `camera_serials` under both `env.train.override_cfg` and `env.eval.override_cfg`.

Then, run the test script in the **compute environment (head)** terminal:

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async

Run It
~~~~~~

After verifying the setup, you can start the real-world training experiment in the **compute environment (head)** terminal by running:

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async

Advance: Multi-Robot Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf supports simple management of a fleet of robots for parallel data collection and training.
To set up multiple robots, you need to modify the configuration file to include multiple robot configurations under the `node_groups` section.

An example configuration for two Franka robots is shown in ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async_2arms.yaml``, as follows:

.. code-block:: yaml

  cluster:
    num_nodes: 3 # One training/rollout node + two robot controller nodes
    component_placement:
      actor:
        node_group: "4090"
        placement: 0 # Run on the first GPU of the training/rollout node
      env:
        node_group: franka
        placement: 0-1 # Two robots assigned to two envs, rank 0 and rank 1
      rollout:
        node_group: "4090"
        placement: 0:0-1 # Two rollout processes on the first GPU of the training/rollout node
    node_groups:
      - label: "4090"
        node_ranks: 0 # Node rank 0 is the training/rollout node
      - label: franka
        node_ranks: 1-2 # Node ranks 1 and 2 are the two robot controller nodes
        hardware:
          type: Franka
          configs:
            - robot_ip: ROBOT_IP_FOR_RANK1
              node_rank: 1 # The node rank of the first robot controller node
            - robot_ip: ROBOT_IP_FOR_RANK2
              node_rank: 2 # The node rank of the second robot controller node

In the single-machine setup, each robot control role joins the same cluster as an
independent Ray node (with its own ``--temp-dir``), corresponding to different ``node_ranks`` in the config.

Naturally, the settings can be extended to more robots by following the same pattern.
For more details regarding the configuration syntax of this kind of heterogeneous hardware setup, please refer to :doc:`../../guides/hetero`.

Visualization and Results
-------------------------

**1. Tensorboard Logging**

At the ray head node, run:

.. code-block:: bash

   # Start TensorBoard
   tensorboard --logdir ./logs --port 6006

**2. Key Metrics Tracked**

- **Environment Metrics**:

  - ``env/episode_len``: Number of environment steps elapsed in the episode (unit: step).
  - ``env/return``: Episode return.
  - ``env/reward``: Step-level reward.
  - ``env/success_once``: Recommended metric to monitor training performance. It directly reflects the unnormalized episodic success rate.

- **Training Metrics**:

  - ``train/sac/critic_loss``: Loss of the Q-function.
  - ``train/critic/grad_norm``: Gradient norm of the Q-function.

  - ``train/sac/actor_loss``: Loss of the policy.
  - ``train/actor/entropy``: Entropy of the policy.
  - ``train/actor/grad_norm``: Gradient norm of the policy.

  - ``train/sac/alpha_loss``: Loss of the temperature parameter.
  - ``train/sac/alpha``: Value of the temperature parameter.
  - ``train/alpha/grad_norm``: Gradient norm of the temperature parameter.

  - ``train/replay_buffer/size``: Current size of the replay buffer.
  - ``train/replay_buffer/max_reward``: Maximum reward stored in the replay buffer.
  - ``train/replay_buffer/min_reward``: Minimum reward stored in the replay buffer.
  - ``train/replay_buffer/mean_reward``: Average reward stored in the replay buffer.
  - ``train/replay_buffer/std_reward``: Standard deviation of rewards stored in the replay buffer.
  - ``train/replay_buffer/utilization``: Utilization rate of the replay buffer.

Real World Results
~~~~~~~~~~~~~~~~~~
Here we provide demo videos and training curves for the task peg-insertion and charger task, respectively. Within 1 hour of training, the robot is able to learn a policy that can continuously successfully complete the task.

.. raw:: html

  <div style="flex: 0.8; text-align: center;">
      <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/realworld-curve.png" style="width: 100%;"/>
      <p><em>Training Curve</em></p>
    </div>

.. raw:: html

  <div style="flex: 1; text-align: center;">
    <video controls autoplay loop muted playsinline preload="metadata" width="720">
      <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/peg-insertion-compressed.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><em>Peg Insertion</em></p>
  </div>

.. raw:: html

  <div style="flex: 1; text-align: center;">
    <video controls autoplay loop muted playsinline preload="metadata" width="720">
      <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/charger-compressed.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><em>Charger</em></p>
  </div>

Troubleshooting
---------------

**Camera Disconnects Mid-Run**

If the camera disconnects while training / data collection is running, you can
reinstall opencv in the **control environment (RLinf-franka)**:

.. code-block:: bash

   cd RLinf-franka
   source ray_utils/realworld/setup_franka_node.sh   # Activate the control env
   pip uninstall -y opencv-python-headless
   pip install --force-reinstall --no-deps opencv-python

**Leftover Gripper / ROS Processes After an Abnormal Run**

If the run throws an exception, or you interrupt it manually with ``Ctrl-C``, you
may see the Franka gripper disconnect and leftover Ray / ROS / FrankaController
processes. Run the cleanup script to stop the leftover processes before restarting:

.. code-block:: bash

   bash ray_utils/realworld/cleanup.sh
