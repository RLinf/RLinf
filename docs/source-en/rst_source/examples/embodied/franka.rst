Franka Online RL
================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg
   :align: center
   :width: 80%
   :alt: Franka arm used for real-world reinforcement learning

   Franka arm used for real-world reinforcement learning.

Overview
--------

Train a CNN policy on a Franka arm with RLinf, from SpaceMouse demonstration collection to asynchronous online RLPD training. The default layout runs Franky (Python bindings to libfranka), rollout, and training on one Ubuntu 20.04 or 22.04 x86-64 host with an NVIDIA GPU, with optional multi-node separation.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Robot
      :text-align: center

      Franka Emika Panda (Franky)

   .. grid-item-card:: Sensors & Teleop
      :text-align: center

      Intel RealSense · SpaceMouse

   .. grid-item-card:: Workflow
      :text-align: center

      Demo Collection → Online RLPD (SAC)

   .. grid-item-card:: Hardware
      :text-align: center

      1× NVIDIA GPU (or Controller + GPU)

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Task
     - Config
     - Description
   * - Peg insertion
     - ``realworld_collect_data``, ``realworld_peginsertion_rlpd_cnn_async``
     - Collect SpaceMouse demonstrations, then train asynchronously on demonstrations and live robot experience with SpaceMouse intervention.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - Observation
     - RGB images from the first configured camera (``wrist_1``) and robot state.
   * - Action
     - Six Cartesian position and rotation deltas; the gripper stays closed.
   * - Reward
     - Success when the end-effector pose reaches the configured target tolerance.

Hardware and Safety
-------------------

Connect the Franka arm to an Ubuntu 20.04 or 22.04 x86-64 host over wired Ethernet, and connect the Intel RealSense camera and SpaceMouse over USB. Your Linux account needs USB read access to both devices; see the `SpaceMouse setup <https://github.com/JakubAndrysek/PySpaceMouse#installation>`_ for its udev rule.

Before connecting to the arm, unlock the joints and activate FCI in Franka Desk (``http://<robot_ip>/desk``), and close any other process holding the arm's FCI connection. On a GPU host, confirm ``nvidia-smi`` works before installing RLinf so the installer selects CUDA PyTorch.

.. list-table::
   :header-rows: 1
   :widths: 22 43 35

   * - Layout
     - Machines
     - When to use it
   * - Single-node (default)
     - One Ubuntu 20.04 or 22.04 GPU host runs Franky control, rollout, and training.
     - Most setups; one machine to install and maintain.
   * - Multi-node (optional)
     - A controller host (no GPU needed) runs Franky control; a GPU server runs actor and rollout.
     - Isolate robot control from training load, or keep the GPU server remote.

.. warning::

   Keep the emergency stop within reach and supervise every hardware run. Secure the peg and fixture, clear the workspace, and verify that reset motions are safe. Peg insertion resets approximately 10 cm above the target with randomized horizontal position and yaw; never copy a target pose from another robot.

.. dropdown:: Firmware check & libfranka compatibility

   Open Franka Desk at ``http://<robot_ip>/desk``, go to ``SETTINGS``, and record the version shown after ``Control``:

   .. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_firmware.png
      :align: center
      :width: 60%
      :alt: Franka Desk dashboard showing the Control firmware version

      Control firmware version in Franka Desk.

   Match that firmware version in the `Franka compatibility table <https://frankarobotics.github.io/docs/compatibility.html>`_ to choose a libfranka version. Prebuilt Franky wheels bundle libfranka 0.19.0 (default) or 0.15.0 on x86-64; for firmware requiring another libfranka version, use the **Legacy ROS (Ubuntu 20.04)** tab below.

   RLinf is tested with firmware 5.9.2 + libfranka 0.19.0 (standard kernel, real-time check disabled) and firmware 5.7.2–5.9.0 + libfranka 0.15.0 (real-time kernel). Do not change robot firmware just to match this example.

.. dropdown:: Real-time kernel, NVIDIA driver & host setup (first-time setup)

   **1. Install a PREEMPT_RT kernel (recommended)**

   libfranka sends a command every millisecond, and a PREEMPT_RT kernel keeps the 1 kHz loop on schedule under training load. Install it on the host (never inside Docker) following Franka's `real-time kernel guide <https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html>`_, keeping your existing kernel as a GRUB fallback. After rebooting, confirm the second command prints ``1``:

   .. code-block:: bash

      uname -r
      cat /sys/kernel/realtime

   **2. Build the NVIDIA driver for the real-time kernel**

   Install or rebuild the NVIDIA module against the matching ``linux-headers`` using ``IGNORE_PREEMPT_RT_PRESENCE=1``. For APT, configure the repository from NVIDIA's `driver installation guide <https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html>`_ and replace ``DRIVER_PACKAGE`` with your GPU package; for an existing DKMS driver, use the second command:

   .. code-block:: bash

      # APT install
      sudo env IGNORE_PREEMPT_RT_PRESENCE=1 apt-get install -y DRIVER_PACKAGE
      sudo reboot

      # Or rebuild an existing DKMS driver
      sudo env IGNORE_PREEMPT_RT_PRESENCE=1 dkms autoinstall -k "$(uname -r)"
      sudo reboot

   Verify ``nvidia-smi`` lists your GPU before installing RLinf. Repeat the environment variable whenever a driver or kernel update rebuilds the module.

   .. warning::

      ``IGNORE_PREEMPT_RT_PRESENCE=1`` only bypasses the build check; NVIDIA does not officially support PREEMPT_RT kernels. Boot the fallback kernel if the GPU is unavailable.

   **3. Grant real-time scheduling permissions**

   RLinf locks the Franky control process in memory and sets ``SCHED_FIFO`` priority 80 with CPU pinning. Add your login account to a ``realtime`` group and set limits in ``/etc/security/limits.d/99-rlinf-realtime.conf``:

   .. code-block:: bash

      getent group realtime || sudo groupadd realtime
      sudo usermod -aG realtime "$(id -un)"
      sudoedit /etc/security/limits.d/99-rlinf-realtime.conf

   Add these lines, then log out and back in so ``ulimit -r`` prints ``99`` and ``ulimit -l`` prints ``unlimited`` (helpful on standard kernels too):

   .. code-block:: text

      @realtime - rtprio 99
      @realtime - memlock unlimited

   **4. Run without a real-time kernel (optional)**

   By default, Franky uses ``realtime_config: ignore`` and logs a warning on a standard kernel. To refuse starting without PREEMPT_RT, add ``realtime_config: enforce`` beside ``robot_ip`` in the hardware config.

   .. warning::

      On a standard kernel, heavy training load can cause libfranka to miss control deadlines and trigger ``communication_constraints_violation`` reflexes. Verify control under your intended training load with an operator present.

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: Single-Node (Default)

      **1. Install**

      Clone RLinf and install the Franka environment (bundles libfranka 0.19.0 via manylinux 2.28 wheels for Ubuntu 20.04/22.04):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      Prefix ``LIBFRANKA_VERSION=0.15.0`` if your firmware requires 0.15.0, set ``FRANKY_WHEEL`` to a custom wheel URL or local path if needed, or pass ``--no-root`` when APT system packages are already installed.

      **2. Download**

      Download the pretrained ResNet-10 encoder used by both actor and rollout:

      .. code-block:: bash

         hf download RLinf/RLinf-ResNet10-pretrained --local-dir ./models/RLinf-ResNet10-pretrained

      **3. Launch**

      Check the camera stream and record its serial number:

      .. code-block:: bash

         python -m toolkits.realworld_check.test_franka_camera

      Update the ``label: franka`` entry under ``cluster.node_groups`` with your robot IP and camera serial in ``examples/embodiment/config/realworld_collect_data.yaml`` and ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``.

      In the training config, also set ``cluster.num_nodes: 1`` and leave the ``4090`` group unchanged (it names GPU node 0 regardless of GPU model):

      .. code-block:: yaml

         - label: franka
           node_ranks: 0
           hardware:
             type: Franka
             configs:
               - robot_ip: ROBOT_IP
                 node_rank: 0
                 camera_serials: ["CAMERA_SERIAL"]

      Move the peg to the target insertion pose in guiding mode, unlock the arm and activate FCI in Franka Desk, then read the pose (enter ``getpos_euler``, then ``q`` to release the arm):

      .. code-block:: bash

         export FRANKA_ROBOT_IP=192.168.1.10  # Replace with your robot IP
         python -m toolkits.realworld_check.test_franka_controller
         export FRANKA_TARGET_POSE='[x, y, z, roll, pitch, yaw]'  # Replace all six entries

      .. dropdown:: Additional controller check commands (optional)

         .. list-table::
            :header-rows: 1
            :widths: 28 72

            * - Category
              - Options
            * - Interactive commands
              - ``getpos``, ``getpos_euler``, ``getjoint``, ``getstate``, ``gethand``, ``clear``, ``home``, ``open``, ``close``, ``q``
            * - CLI flags
              - ``--robot-ip``, ``--realtime-config enforce``

      Collect 20 SpaceMouse demonstrations (stored under ``logs/franka-demo/demo_buffer``, with optional ``pickle_episode/`` and ``lerobot_dataset/`` exports; see :doc:`franka_gello` for GELLO collection), then launch online RLPD training after collection exits and releases the arm:

      .. code-block:: bash

         RLINF_LOG_DIR="$PWD/logs/franka-demo" \
           bash examples/embodiment/collect_data.sh realworld_collect_data \
           "env.eval.override_cfg.target_ee_pose=$FRANKA_TARGET_POSE"

         bash examples/embodiment/run_realworld_async.sh \
           realworld_peginsertion_rlpd_cnn_async \
           "env.train.override_cfg.target_ee_pose=$FRANKA_TARGET_POSE" \
           "algorithm.demo_buffer.load_path=$PWD/logs/franka-demo/demo_buffer" \
           "actor.model.model_path=$PWD/models/RLinf-ResNet10-pretrained"

      Use a fresh ``RLINF_LOG_DIR`` for each collection session. Supervise the arm throughout training and intervene with the SpaceMouse as needed; when done, interrupt the launcher and wait for the arm to stop.

   .. tab-item:: Multi-Node (Robot + GPU)

      **1. Install on both hosts**

      Install the same RLinf revision, Python version, and Ray version on the controller host (no GPU needed; installer selects CPU-only PyTorch) and on the GPU server:

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env franka && source .venv/bin/activate

      **2. Collect demos and download model**

      Collect demonstrations on the controller first (as in the Single-Node tab), then copy ``logs/franka-demo/demo_buffer`` to the GPU server. On the GPU server, download the pretrained ResNet-10 encoder:

      .. code-block:: bash

         hf download RLinf/RLinf-ResNet10-pretrained --local-dir ./models/RLinf-ResNet10-pretrained

      **3. Start Ray cluster and launch**

      On the GPU server, edit ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml``: set ``cluster.num_nodes: 2`` and set both ``node_ranks`` and ``node_rank`` for the Franka group to 1 (actor and rollout stay on GPU 0 of node 0).

      .. warning::

         Ray captures the active Python environment and environment variables when ``ray start`` runs. Activate ``.venv`` and export ``RLINF_NODE_RANK`` before ``ray start`` on both machines. Use each machine's IP on their shared network—not the arm's IP—and export ``RLINF_COMM_NET_DEVICES`` if a host has multiple network interfaces.

      Start Ray on both nodes:

      .. code-block:: bash

         # On the GPU server (head node)
         export RLINF_NODE_RANK=0
         export HEAD_IP=192.168.10.10
         ray start --head --port=6379 --node-ip-address="$HEAD_IP"

         # On the controller host (worker node)
         export RLINF_NODE_RANK=1
         export HEAD_IP=192.168.10.10
         export CONTROLLER_IP=192.168.10.11
         ray start --address="$HEAD_IP:6379" --node-ip-address="$CONTROLLER_IP"

      Verify both nodes appear in ``ray status`` on the GPU server, then run the training command from the Single-Node tab on the GPU server only (the controller needs neither model weights nor ``demo_buffer``). For multi-robot setups, see :doc:`/rst_source/guides/realworld_robot` and :doc:`/rst_source/guides/hetero`.

   .. tab-item:: Legacy ROS (Ubuntu 20.04)

      Use the ROS Noetic backend (``franka_ros`` and ``serl_franka_controllers``) for existing ROS deployments or firmware requiring a libfranka version other than 0.15.0 or 0.19.0. Install natively on Ubuntu 20.04 into a separate virtual environment:

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         LIBFRANKA_VERSION=0.15.0 bash requirements/install.sh embodied --env franka-ros --venv franka-ros && source franka-ros/bin/activate

      The installer builds libfranka, RLinf's ``franka_ros`` fork, and ``serl_franka_controllers`` under ``franka-ros/franka_catkin_ws`` and appends the ROS setup scripts to ``franka-ros/bin/activate``.

      Confirm the build with ``rospack find serl_franka_controllers``. Override installer defaults with these environment variables:

      .. list-table::
         :header-rows: 1
         :widths: 32 16 52

         * - Variable
           - Default
           - Purpose
         * - ``LIBFRANKA_VERSION``
           - ``0.15.0``
           - libfranka version matching your firmware.
         * - ``FRANKA_ROS_VERSION``
           - ``0.10.0``
           - ``franka_ros`` version to build.
         * - ``FRANKA_REALTIME_CONFIG``
           - ``enforce``
           - Set to ``ignore`` for a non-RT kernel and re-run the installer without rebuilding.
         * - ``SKIP_ROS``
           - ``0``
           - Set to ``1`` to use your own ROS and catkin installation.

      .. warning::

         With ``SKIP_ROS=1``, install ROS Noetic, libfranka, ``franka_ros``, and ``serl_franka_controllers`` yourself (see the `ROS Noetic <https://wiki.ros.org/noetic/Installation/Ubuntu>`_, `libfranka <https://frankarobotics.github.io/docs/libfranka/docs/installation.html>`_, and `serl_franka_controllers <https://github.com/rail-berkeley/serl_franka_controllers>`_ guides).

         Before every launch (or before ``ray start`` in multi-node setups), source ``/opt/ros/noetic/setup.bash`` and your catkin ``devel/setup.bash``, and add libfranka to ``LD_LIBRARY_PATH``.

      **2. Configure and launch**

      Add ``backend: franka_ros`` to the Franka hardware config (do not add ``realtime_config`` in YAML; the ROS backend reads ``FRANKA_REALTIME_CONFIG`` from installation instead):

      .. code-block:: yaml

         configs:
           - robot_ip: ROBOT_IP
             node_rank: 0
             camera_serials: ["CAMERA_SERIAL"]
             backend: franka_ros

      Verify the controller in the active ``franka-ros`` environment, then run the collection and training commands from the Single-Node tab (or follow the Multi-Node tab for multi-node runs):

      .. code-block:: bash

         python -m toolkits.realworld_check.test_franka_controller --backend franka_ros

      RLinf starts ``roscore``, ``roslaunch serl_franka_controllers impedance.launch``, and ``joint.launch`` (during joint resets) automatically. In multi-node setups, only the controller needs ``franka-ros``; the GPU server uses the standard ``franka`` environment.

      **Differences from Franky and ROS troubleshooting:**

      .. list-table::
         :header-rows: 1
         :widths: 30 70

         * - Topic
           - Legacy ROS behavior
         * - Control mode
           - Accepts Cartesian targets only; joint-position control requires Franky.
         * - Gripper
           - ``gripper_type: franka`` communicates over ``/franka_gripper`` and responds only while the arm is connected.
         * - Impedance gains
           - Applied via ``dynamic_reconfigure``; hardware ``compliance`` overrides in YAML are ignored.
         * - Non-RT kernel error
           - Re-run the installer with ``FRANKA_REALTIME_CONFIG=ignore`` if ``Running kernel does not have realtime capabilities`` appears.
         * - Arm not ready / FCI busy
           - Confirm FCI is active in Franka Desk and close competing controllers. After a crashed run, run ``ray stop`` and ``pkill -f roslaunch``, then check state topics with ``rostopic echo -n 1 /franka_state_controller/franka_states``.

.. _franka-motion-settings:

.. dropdown:: Configure Arm Motion (Franky Compliance Defaults)

   Collection, training, evaluation, PICO/SpaceMouse teleoperation, and Cartesian GELLO teleoperation all share the same Franky Cartesian controller defaults when ``compliance`` is omitted from the hardware entry:

   .. list-table::
      :header-rows: 1
      :widths: 30 15 55

      * - Setting
        - Default
        - Meaning
      * - ``translational_stiffness``
        - 1000 N/m
        - Initial stiffness for position errors.
      * - ``rotational_stiffness``
        - 50 Nm/rad
        - Initial stiffness for orientation errors.
      * - ``translational_clip``
        - 0.008 m
        - Initial position-error limit used to compute restoring torque.
      * - ``rotational_clip``
        - 0.04 rad
        - Initial orientation-error limit used to compute restoring torque.
      * - ``max_step``
        - 0.03 m
        - Maximum position change from the previous commanded target per call.
      * - ``max_step_rad``
        - 0.10 rad
        - Maximum orientation change from the previous commanded target per call.

   The first target is limited relative to the measured pose. Target-change limits apply per call, and error clips bound controller restoring error; the environment still applies its own action scale and workspace limits.

   At reset, tasks can request updated gains via ``compliance_param``, which Franky caps at 1200 N/m and 80 Nm/rad with error-clip floors of 5 mm and 0.02 rad (for peg insertion, reset uses 5 mm x/y, 10 mm z, and 0.02 rad clips; ``max_step`` and ``max_step_rad`` stay active after reset).

   To override specific fields, add a ``compliance`` mapping beside ``robot_ip`` in each recipe that needs it (misspelled keys fail validation before connecting):

   .. code-block:: yaml

      compliance:
        max_step: 0.02

   Joint-position commands (such as dual-arm GELLO joint teleoperation) use joint control and leave these Cartesian settings unused.

.. dropdown:: Keyboard Reward Labeling (Optional)

   For tasks without an automatic pose-based success signal, enable keyboard reward labeling in the training config:

   .. code-block:: yaml

      env:
        train:
          keyboard_reward_wrapper: single_stage  # or multi_stage

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - Mode
        - Keys
      * - ``single_stage``
        - ``a`` emits failure, ``b`` neutral, and ``c`` success.
      * - ``multi_stage``
        - ``a`` / ``b`` / ``c`` switch stages; ``q`` emits a negative reward.

   Find the keyboard event device on the robot host, grant permission, and export ``RLINF_KEYBOARD_DEVICE`` before launching (or before ``ray start`` in multi-node setups):

   .. code-block:: bash

      ls -l /dev/input/by-id/*-event-kbd
      sudo chmod 666 /dev/input/event20
      export RLINF_KEYBOARD_DEVICE=/dev/input/event20

.. dropdown:: Other Franka Workflows (VLA Models & Related Guides)

   Combine Franka dependencies (including dexterous-hand support) with a VLA policy in its own virtual environment:

   .. code-block:: bash

      # OpenPI
      bash requirements/install.sh embodied --model openpi --env franka --venv openpi && source openpi/bin/activate

      # OpenVLA
      bash requirements/install.sh embodied --model openvla --env franka --venv openvla && source openvla/bin/activate

      # OpenVLA-OFT
      bash requirements/install.sh embodied --model openvla-oft --env franka --venv openvla-oft && source openvla-oft/bin/activate

      # GR00T
      bash requirements/install.sh embodied --model gr00t --env franka --venv gr00t && source gr00t/bin/activate

   Related guides:

   - :doc:`franka_gello` and :doc:`franka_vr` for GELLO or PICO teleoperation.
   - :doc:`franka_pi0_sft_deploy` and :doc:`hg-dagger` for OpenPI policies.
   - :doc:`franka_reward_model` for learned reward models.
   - :doc:`franka_zed_robotiq` and :doc:`franka_dexhand` for other cameras and end effectors.
   - :doc:`dual_franka` for dual-arm control and :doc:`/rst_source/guides/rtc` for overlapping action execution with inference.

Results and Troubleshooting
---------------------------

Monitor ``env/success_once``, ``env/return``, and SAC actor/critic losses in TensorBoard (``tensorboard --logdir ./logs --port 6006``); see :doc:`Training metrics <../../reference/metrics>` and :doc:`/rst_source/guides/logger` for logging options. The curve and videos below show representative peg-insertion and charger runs.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/realworld-curve.png
   :align: center
   :width: 100%

   Real-world training curves.

.. raw:: html

   <video controls muted playsinline preload="metadata" width="720">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/peg-insertion-compressed.mp4" type="video/mp4">
   </video>
   <video controls muted playsinline preload="metadata" width="720">
     <source src="https://raw.githubusercontent.com/RLinf/misc/main/pic/charger-compressed.mp4" type="video/mp4">
   </video>

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Symptom
     - Fix
   * - ``torch.cuda.is_available()`` prints ``False``
     - Verify ``nvidia-smi`` works under the running kernel (rebuild the NVIDIA module with ``IGNORE_PREEMPT_RT_PRESENCE=1`` on PREEMPT_RT), then re-run ``requirements/install.sh``.
   * - ``Running kernel does not have realtime capabilities``
     - Boot the PREEMPT_RT kernel, or use Franky's default ``realtime_config: ignore``.
   * - Incompatible libfranka version error
     - Match ``LIBFRANKA_VERSION`` (0.19.0 or 0.15.0) to the Franka Desk ``Control`` firmware version using the Franka compatibility table.
   * - ``communication_constraints_violation`` reflex
     - Heavy training load on a standard kernel missed 1 kHz deadlines; switch to a PREEMPT_RT kernel or move robot control to a dedicated controller node.
   * - Franky impedance controller stops unexpectedly
     - RLinf reports the motion error and halts safely. Resolve the physical or reflex cause before restarting training (in direct Python scripts, disconnect and reconnect; clearing errors alone does not resume tracking).
