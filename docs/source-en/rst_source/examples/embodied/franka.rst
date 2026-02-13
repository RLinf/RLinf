Real-World RL with Franka
============================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

This document provides a comprehensive guide to launching and managing the 
a CNN policy training task within the RLinf framework, 
focusing on training a ResNet-based CNN policy from scratch for robotic manipulation in the real world setup. 

The primary objective is to develop a model capable of performing robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from the robot's camera.
2. **Action Generation**: Producing precise robotic actions (position, rotation), possibly with gripper control.
3. **Reinforcement Learning**: Optimizing the policy via the SAC with environment feedback.

Environment
-----------

**Real World Environment**

- **Environment**: Real world setup.

  - Franka Emika Panda robotic arm
  - Realsense cameras
  - Possibly use spacemouse for teleoperation data collection or human intervention.

- **Task**: Currently we support the peg-insertion task and the charger task. 
- **Observation**:

  - RGB images (128x128) from a wrist camera or a third-person camera.

- **Action Space**: 6 or 7-dimensional continuous actions, depending on whether gripper control is included:

  - 3D position control (x, y, z)
  - 3D rotation control (roll, pitch, yaw)
  - Gripper control (open/close)

**Data Structure**

- **Images**: RGB tensors ``[batch_size, 128, 128, 3]``
- **Actions**: Normalized continuous values ``[-1, 1]`` for each action dimension
- **Rewards**: Step-level rewards based on task completion


Algorithm
-----------------------------------------

**Core Algorithm Components**

1. **SAC (Soft Actor-Critic)**

   - Learning Q-values by Bellman backups and entropy regularization.

   - Learning policy to maximize entropy-regularized Q.

   - Learning temperature parameter for exploration-exploitation trade-off.

2. **Cross-Q**

   - A variant of SAC that removes the target Q network.

   - Concating curr-obs and next-obs in one batch, incorporating BatchNorm for stable training for Q.

3. **RLPD (Reinforcement Learning with Prior Data)**

   - A variant of SAC that incorporates prior data for improved learning efficiency.

   - High update-to-data ratio to leverage collected data effectively.

4. **CNN Policy Network**

   - ResNet-based architecture for processing visual inputs.

   - MLP layers for fusing images and states to output actions.

   - Q heads for critic functions.

Hardware Setup
----------------

The real-world setup requires the following hardware components:

- **Robotic Arm**: Franka Emika Panda
- **Cameras**: Intel RealSense cameras for capturing RGB images
- **Computing Unit**: A computer with GPU support for training the CNN policy
- **Robot Controller**: A small computer (does not require GPU) connected with the robotic arm in the same local network
- **Space Mouse (Optional)**: For teleoperation data collection or human intervention during training.

.. warning::

  Ensure all computers are networked in the same local network.
  The robot arm is only required to be in the same local network as the robot controller.

Dependency Installation
-------------------------

The controller node and the training/rollout node(s) should be set up with different software dependencies.

Robot Controller Node
~~~~~~~~~~~~~~~~~~~~~~

1. Check Franka Firmware Version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Go to the robot's management webpage (usually at ``http://<robot_ip>/desk``), click on the ``SETTINGS`` tab, and check the version number following ``Control`` in ``DashBoard`` as follows.
Please take a note of the firmware version for later use.

.. raw:: html

  <div style="flex: 1; text-align: center;">
      <img src="https://github.com/RLinf/misc/blob/main/pic/franka_firmware.png?raw=true" style="width: 60%;"/>
  </div>

.. warning::

  Make sure that the Franka firmware version is ``<5.9.0`` for compatibility with the serl_franka_controllers.

  Firmware version 5.7.2 is recommended for best compatibility.

2. Real-time Kernel Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Franka controller is recommended to run on a real-time kernel for better performance.
Follow the instructions in `Franka documentation <https://frankarobotics.github.io/docs/libfranka/docs/real_time_kernel.html>`_ to install the real-time kernel.

3. Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

a. Clone RLinf Repository
__________________________

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

b. Install Dependencies
__________________________

**Option 1: Docker Image**

Use Docker image for the experiment.

To access the robot, camera, and space mouse devices from within the docker container, it is recommended to run the container in the **privileged** mode:

.. code:: bash

   docker run -it --rm \
      --privileged \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-franka
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-franka

Currently, the docker image contains libfranka version ``0.10.0``, ``0.13.3``, ``0.14.1``, ``0.15.0``, and ``0.18.0`` with franka_ros version ``0.10.0``.

These versions are selected based on the compatibility matrix in `Franka compatibility <https://frankarobotics.github.io/docs/compatibility.html>`_.
Please check your Franka firmware version and find which libfranka version is compatible with it.

Having determined the compatible libfranka version, you can switch to the corresponding virtual environment in the docker container by running:

.. code:: bash

   source switch_env franka-<libfranka_version>
   # e.g., for libfranka version 0.15.0
   # source switch_env franka-0.15.0

**Option 2: Custom Environment**

Our installation script consists of the installation of two parts:

- Python dependencies for RLinf framework and real-world RL training.
- ROS Noetic, libfranka, franka_ros, and serl_franka_controllers for Franka control.

.. warning::

  The installation script only supports Ubuntu 20.04 due to ROS Noetic requirements.

.. warning::

  If you have already installed ROS Noetic, libfranka, franka_ros and serl_franka_controllers manually, you can skip the installation of these packages by setting the environment variable ``export SKIP_ROS=1`` before running the installation script.

  If you have skipped these installations, please make sure that you have sourced the ROS setup script (usually at ``/opt/ros/noetic/setup.bash``), as well as the franka_ros and serl_franka_controllers setup scripts (usually at ``<your_catkin_ws>/devel/setup.bash``) in your `~/.bashrc`. Also, make sure the libfranka shared library is in your ``LD_LIBRARY_PATH`` or installed in the system library path `/usr/lib`.

  This is important **every time before you start ray on the controller node** to ensure that the Franka control packages can be correctly found.

.. warning::

  Currently, the installation of ROS Noetic, libfranka, and franka_ros is only tested against Franka firmware version ``>=5.7.2`` and ``<5.9.0`` with libfranka version ``0.15``.

  For other firmware versions, please first check the compatibility matrix in `Franka compatibility <https://frankarobotics.github.io/docs/compatibility.html>`_.
  For a desired libfranka and franka_ros version, you can use `export LIBFRANKA_VERSION=<version>` and `export FRANKA_ROS_VERSION=<version>` to specify the versions before running the installation script.

.. note::

  If the script does not work for you, please refer to the official `ROS Noectic <https://wiki.ros.org/noetic/Installation/Ubuntu>`_ for ROS Noetic installation, `Franka <https://frankarobotics.github.io/docs/libfranka/docs/installation.html>`_ for libfranka and franka_ros installation, and `serl_franka_controllers <https://github.com/rail-berkeley/serl_franka_controllers>`_ for serl_franka_controllers installation.
  
Execute the following command to install the dependencies:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --env franka
   source .venv/bin/activate

Training/Rollout Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~

a. Clone RLinf Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

b. Install Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.1-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # docker.1ms.run/rlinf/rlinf:agentic-rlinf0.1-maniskill_libero

**Option 2: Custom Environment**

Install dependencies directly in your environment by running the following command:

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model openvla --env maniskill_libero
   source .venv/bin/activate
   
Model Download
---------------

Before starting training, you need to download the corresponding pretrained model:

.. code:: bash

   # Download the model (choose either method)
   # Method 1: Using git clone
   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # Method 2: Using huggingface-hub
   # For mainland China users, you can use the following for better download speed:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

After downloading, make sure to correctly specify the model path in the configuration yaml file.

Running the Experiment
-----------------------

Prerequisites
~~~~~~~~~~~~~~~

**Get the Target Pose for the Task**

To acquire the target pose for the peg-insertion task, you can use the `toolkits.realworld_check.test_controller` script.

First, you need to activate your Franka robot's programming mode, and manually move the robot to the desired target pose.

Then, before running, set the environment variable ``FRANKA_ROBOT_IP`` to your robot's IP address:

.. code-block:: bash

   export FRANKA_ROBOT_IP=<your_robot_ip_address>

Next, run the script:

.. code-block:: bash

   python -m toolkits.realworld_check.test_controller

The script will prompt you to input command, you can enter `getpos_euler` to get the current end-effector pose in Euler angles.

Data Collection
~~~~~~~~~~~~~~~~~

For RLPD experiments, you need to first collect some initial data for training.
The data collection only needs to be run on the controller node without other nodes.

1. Source the virtual python environment and franka_ros and serl_franka_controllers setup scripts:

.. code-block:: bash

   source <path_to_your_venv>/bin/activate
   source <your_catkin_ws>/devel/setup.bash

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

The script will terminate after 20 episodes of data collection (can be configured with the `num_data_episodes` field in the configuration file), and the collected data will be stored in the ``logs/[running-timestamp]/data.pkl`` folder.

5. After data collection, you can upload the collected data to the training/rollout nodes.

Cluster Setup
~~~~~~~~~~~~~~~~~

Before starting the experiment, you will first setup the ray cluster properly.

.. warning::
  This step is essential, proceed with caution! Even the slightest misconfiguration may result in missing packages or failure to control the robot.

RLinf uses ray for managing distributed environments. So it is subject to one critical characteristic of ray: when you run `ray start` on a node, the current Python interpreter and environment variables will be recorded by ray, and all the processes started by ray on that node later will inherit the same Python interpreter and environment variables.

We provide a utility script ``ray_utils/realworld/setup_before_ray.sh`` to help you set up the environment before starting ray on each node.
You can modify the script accordingly and source it before starting ray on each node.

Specifically, the script sets up the following important aspects:

1. Source the correct virtual python environment. See the section on Dependency Installation for details.
   
2. Source the franka_ros and serl_franka_controllers packages setup scripts (if on the controller node), usually at ``<your_catkin_ws>/devel/setup.bash``. **If you are using the docker image or the installation script, this is already done when you source the virtual python environment.**

3. Setup RLinf environment variables on all nodes:
   
.. code-block:: bash

   export PYTHONPATH=<path_to_your_RLinf_repo>:$PYTHONPATH
   export RLINF_NODE_RANK=<node_rank_of_this_node>
   export RLINF_COMM_NET_DEVICES=<network_device_for_communication> # Optional if you do not have multiple network devices

The ``RLINF_NODE_RANK`` is set to ``0 ~ N-1`` for each of the ``N`` nodes in the cluster, and is used by the configuration file to identify the node.

The ``RLINF_COMM_NET_DEVICES`` is optional and only needed if you have multiple network devices on your machine, e.g., ``eth0``, ``enp3s0``, which must be the network card providing the IP that can be accessed by other nodes in the cluster. 
This can be checked by running ``ifconfig`` or ``ip addr`` on your machine.

After sourcing the script, you can start ray on each node as follows:

Here `<head_node_ip_address>` is the IP address of the head node that can be accessed by other nodes in the cluster.

.. code-block:: bash

   # On the head node (node rank 0)
   ray start --head --port=6379 --node-ip-address=<head_node_ip_address>

   # On worker nodes (node rank 1 ~ N-1)
   ray start --address='<head_node_ip_address>:6379'

You can run `ray status` to check if the cluster is set up correctly.

Configuration file
~~~~~~~~~~~~~~~~~~~~~~

Before starting the experiment, you need to modify the configuration file, ``examples/embodiment/config/realworld_peginsertion_rlpd_cnn_async.yaml`` according to your setup.

Similarly, you first need to fill your robot's IP address to the field ``robot_ip`` and the target end-effector pose to the field ``target_ee_pose``.

Then, change the ``model_path`` field in both ``rollout`` and ``actor`` sections to the path where you have downloaded the pretrained model.
Change the ``data.path`` field to the path where you have uploaded the collected demo data.

Testing the Setup (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide several test scripts to verify that the setup is correct before starting the experiment. This step is optional but recommended.

First, test the camera connection by running on the controller node:

.. code-block:: bash

   python -m toolkits.realworld_check.test_camera

Next, test the basic cluster setup by running a dummy setup. Refer to ``examples/embodiment/config/real_world_dummy_sac_cnn.yaml`` and add `env.eval.override_cfg`.
You can set the `is_dummy` field to `True` in both `env.train.override_cfg` and `env.eval.override_cfg` sections in the configuration file to enable the dummy setup.
And fill the camera serial numbers obtained from ``running toolkits.realworld_check.test_camera.py`` into the field `camera_serials` under both `env.train.override_cfg` and `env.eval.override_cfg`.

Then, run the test script on the head node:

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_peginsertion_rlpd_cnn_async

Running the Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~

After verifying the setup, you can start the real-world training experiment by running the following command on the head node:

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

Naturally, the settings can be extended to more robots by following the same pattern.
For more details regarding the configuration syntax of this kind of heterogeneous hardware setup, please refer to :doc:`../../tutorials/advance/hetero`.

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
      <img src="https://github.com/RLinf/misc/raw/main/pic/realworld-curve.png" style="width: 100%;"/>
      <p><em>Training Curve</em></p>
    </div>

.. raw:: html

  <div style="flex: 1; text-align: center;">
    <video controls autoplay loop muted playsinline preload="metadata" width="720">
      <source src="https://github.com/RLinf/misc/raw/main/pic/peg-insertion-compressed.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><em>Peg Insertion</em></p>
  </div>

.. raw:: html

  <div style="flex: 1; text-align: center;">
    <video controls autoplay loop muted playsinline preload="metadata" width="720">
      <source src="https://github.com/RLinf/misc/raw/main/pic/charger-compressed.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <p><em>Charger</em></p>
  </div>