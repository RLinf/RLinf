RL with GenieSim Benchmark
==========================

.. |pi| replace:: π₀.₅

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/genie_sim_system_overview.webp
   :align: center
   :width: 100%
   :alt: Genie Sim platform and capabilities overview.

   Genie Sim system overview (image: `GenieSim <https://github.com/AgibotTech/genie_sim>`__).

This is the official `GenieSim <https://github.com/AgibotTech/genie_sim>`__
post-training project using RLinf to train and evaluate vision-language-action
(VLA) policies. Genie Sim is the simulation platform from AgiBot. It provides developers with a complete toolchain for scene reconstruction, asset generation, data collection, automated evaluation, and model post-training, accelerating the development of embodied intelligence. Its tasks span instruction following, spatial
understanding, and manipulation, with object, scene, and instruction
variations for studying generalization. RLinf supports PPO and GRPO
fine-tuning of VLA models, with single-task and multi-task training
workflows for improving dual-arm manipulation.

GenieSim simulation and RLinf training run on two GPU nodes. The simulation
node uses an NVIDIA RTX-series GPU.

Overview
--------

Improve language-guided dual-arm manipulation with PPO and GRPO.
Compared with the SFT checkpoint, the ten-task PPO recipe improves block-color
selection success from 71% to 95% and block-size selection from 60% to 92%.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      VLA models

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO, GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      Instruction · Spatial · Manipulation

   .. grid-item-card:: Hardware
      :text-align: center

      2 GPU nodes

| **You'll do:** install → start a GenieSim server → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a reachable GenieSim server · a VLA checkpoint.

Tasks
~~~~~

The 25 tasks span instruction following, spatial understanding, and manipulation
skills. Use the single-task or multi-task recipes below; see the
:ref:`task appendix <geniesim-training-tasks>` for all task identifiers.

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Category
     - Tasks
     - Focus
   * - Instruction following
     - 10
     - Select objects by color, size, shape, number, category, and language descriptions.
   * - Spatial understanding
     - 8
     - Locate, place, sort, and stack objects using spatial relationships.
   * - Manipulation skills
     - 7
     - Open doors, handle containers, organize shelves, and clean a desktop.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

The OpenPI |pi| configuration used in the G2 experiments combines three camera
views, arm and gripper state, and a natural-language task instruction.
The :ref:`task appendix <geniesim-training-tasks>` groups the 25 tasks by capability.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Specification
   * - Observation
     - Three 640×480 RGB views, resized with aspect-preserving padding to
       224×224 before model input. Proprioception uses 14 arm joint positions
       and 2 gripper opening angles.
   * - Action
     - 16-dim continuous action: 14 arm joint-position targets and 2 gripper
       targets, predicted as deltas in chunks of 10 steps.
   * - Reward
     - Sparse reward: 1 on task success and 0 otherwise.
   * - Task prompt
     - The full natural-language task instruction from ``task_descriptions``.

The model uses 16 active state and action dimensions within its 32-dim input
and output space. Additional simulator state channels are masked, and model
outputs are reduced to the 16 robot-control dimensions. State and action
normalization uses the checkpoint's ``assets/norm_stats.json``.

Installation
------------

Run the repository setup commands on the RLinf host, outside any container.

.. include:: _setup_common.rst
   :end-before: Then set up the dependencies

Choose one of the following installation methods. Both install
``rlinf-geniesim-client`` from GenieSim ``v3.5.0``, which ``install.sh``
clones automatically. To install from an existing GenieSim checkout instead,
set ``GENIESIM_PATH`` to it. See
:doc:`Installation </rst_source/start/installation>` for prerequisites and
GPU driver setup.

**Custom environment**

On the RLinf host, stay in the RLinf repository root to install OpenPI and
the GenieSim client, then activate the environment:

.. code:: bash

   # Mainland China users can add --use-mirror.
   bash requirements/install.sh embodied --model openpi --env geniesim
   source .venv/bin/activate

**Docker image**

On the RLinf host, run this build from the RLinf repository root:

.. code:: bash

   docker build -f docker/Dockerfile \
     --build-arg BUILD_TARGET=embodied-geniesim \
     --build-arg NO_MIRROR=1 \
     -t rlinf:embodied-geniesim .

The image provides an NVIDIA CUDA 12.8 training environment with OpenPI and
the GenieSim client. Connect it to your GenieSim service to run training.

Still on the RLinf host and in the RLinf repository root, start the container
with the repository mounted:

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v "$(pwd):/workspace/RLinf" \
      --workdir /workspace/RLinf \
      rlinf:embodied-geniesim

The container opens in ``/workspace/RLinf``. Inside the RLinf container,
activate OpenPI:

.. code:: bash

   source switch_env openpi

Download the Model
------------------

Download the OpenPI |pi| SFT checkpoint for GenieSim as the starting point for
PPO or GRPO training.

The checkpoint was trained with 5,000 trajectories from the
`instruction_and_robust suite of GenieSim3.0-Dataset <https://huggingface.co/datasets/agibot-world/GenieSim3.0-Dataset/tree/main/task_suite/instruction_and_robust>`__.
SFT ran for 45,000 steps with an action chunk length of 10.

**Model repository: to be added after upload.** Replace
``<GENIESIM_PI05_SFT_REPO>`` below with the published Hugging Face repository ID.

Run the download in the activated OpenPI environment: inside the RLinf
container at ``/workspace/RLinf``, or in the host's RLinf repository root for
a custom installation. If you already have the checkpoint, set
``GENIESIM_CHECKPOINT_PATH`` to its directory as seen from that environment
and continue to ``Run It``.

.. code:: bash

   pip install huggingface-hub

   # For mainland China users:
   # export HF_ENDPOINT=https://hf-mirror.com

   export GENIESIM_CHECKPOINT_PATH="$PWD/checkpoints/geniesim_sft"
   hf download "<GENIESIM_PI05_SFT_REPO>" \
      --local-dir "$GENIESIM_CHECKPOINT_PATH"

The example saves the checkpoint under the RLinf repository. Choose another
directory by changing ``GENIESIM_CHECKPOINT_PATH`` before downloading.
Training and evaluation read ``GENIESIM_CHECKPOINT_PATH`` for the model path.
You can also set the paths directly in YAML, using paths visible to the
RLinf process:

.. include:: _model_path.rst

Run It
------

On the simulation host, follow the `GenieSim post-training server guide <https://github.com/AgibotTech/genie_sim#39-post-training-server>`__
to prepare ``geniesim_assets`` and build the Isaac Sim 6.0.1-based image.
Use GenieSim ``v3.5.0``, the same revision as the installed client.

**Start the Simulation Server**

On the simulation host, outside containers and from the GenieSim repository
root, start and enter the server container:

.. code:: bash

   geniesim docker --profile post-train up --headless
   geniesim docker --profile post-train into

Inside the ``post_train_server`` container, from ``/workspace``, start the
servers and check their status. The count, starting port, and GPU below are
example values; adjust them to match your RLinf configuration:

.. code:: bash

   geniesim post-train server start --count 2 --base-port 5601 --devices 0
   geniesim post-train server status --count 2 --base-port 5601

The example starts the two servers of the single-task recipes on GPU 0.
Each recipe needs the following servers and GPUs; the simulation GPUs are
RTX 4090s. ``--devices`` places consecutive ports on the same GPU, which
matches how each recipe groups its endpoints by env process:

.. list-table::
   :header-rows: 1
   :widths: 30 14 12 12 12 20

   * - Recipe
     - Servers (ports)
     - Envs per server
     - Simulation GPUs
     - Training GPUs
     - Server start options
   * - Single-task PPO / GRPO
     - 2 (5601–5602)
     - 16
     - 1
     - 1
     - ``--count 2 --devices 0``
   * - ``geniesim_multitask4_ppo_openpi_pi05``
     - 4 (5601–5604)
     - 16
     - 2
     - 2
     - ``--count 4 --devices 0,1``
   * - ``geniesim_multitask6_ppo_openpi_pi05``
     - 6 (5601–5606)
     - 12
     - 2
     - 2
     - ``--count 6 --devices 0,1``
   * - ``geniesim_multitask10_ppo_openpi_pi05``
     - 12 (5601–5612)
     - 14
     - 4
     - 4
     - ``--count 12 --devices 0,1,2,3``

All recipes start their training servers at ``--base-port 5601``.

In another terminal on the simulation host, outside containers and from any
directory, list the host's IP addresses and listening TCP ports:

.. code:: bash

   hostname -I
   ss -ltn

Replace ``sim-host`` below with an IP reachable from the RLinf host, and use
the server's listening port. For detailed options, see the
`GenieSim server guide <https://github.com/AgibotTech/genie_sim/blob/HEAD/source/rlinf_geniesim/README.md>`__.

Run the RLinf commands in the activated OpenPI environment, from the RLinf
repository root: ``/workspace/RLinf`` inside the RLinf container, or the host
checkout for a custom installation. Use ``GENIESIM_CHECKPOINT_PATH`` set above.

.. note::

   On an RTX 4090, we recommend at most 42 parallel environments per GPU at
   the default render width of 640, or 64 with
   ``env.train.override_cfg.render_width`` set to 360. Servers on the same GPU
   share this budget.

   ``env.train.override_cfg.images_at_chunk_end_only`` is enabled by default.
   With this option, RLinf and the simulator exchange complete action chunks,
   rendering only the final frame of each chunk.

**Train**

Launch single-task PPO:

.. code:: bash

   export GENIESIM_HOST_0=sim-host
   bash examples/embodiment/run_embodiment.sh geniesim_pick_block_color_ppo_openpi_pi05

GRPO and multi-task recipes are listed in `Visualization and Results`_.
Each recipe's ``env.train.override_cfg.addresses`` lists the required server
endpoints; edit this list to change hosts or ports.

Training-time validation is disabled by default. If you enable it with a
positive ``runner.val_check_interval``, set ``GENIESIM_EVAL_HOST`` to the
validation host and start one more server for each port listed in
``env.eval.override_cfg.addresses``, as many as the training servers.

**Evaluate**

Set the simulation server's address and launch standalone evaluation.
The address below is an example; replace its host and port with your server's:

.. code:: bash

   export GENIESIM_EVAL_SERVER_ADDR=tcp://sim-host:5601
   bash evaluations/run_eval.sh geniesim_pick_block_color_eval

``GENIESIM_EVAL_SERVER_ADDR`` sets ``env.eval.override_cfg.address`` in
``evaluations/geniesim/geniesim_pick_block_color_eval.yaml``; include the
chosen port in this variable. See the :doc:`Evaluation CLI <../../evaluations/reference/cli>`
for more options.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

The key signal is ``env/success_once``. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Enable video in the env config when you want rollout videos:

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

The following results use the Agibot G2 dual-arm wheeled robot, with GenieSim
running on an RTX 4090.

Compare the SFT starting checkpoint with single-task PPO and GRPO, and
PPO policies trained on mixes of 4, 6, and 10 tasks. All checkpoints use
the same evaluation protocol.

.. list-table::
   :class: geniesim-results-table
   :header-rows: 1
   :widths: 34 11 11 11 11 11 11

   * - subtask name
     - SFT
     - Single-Task PPO
     - Single-Task GRPO
     - 4-Task PPO
     - 6-Task PPO
     - 10-Task PPO
   * - ``pick_block_color``
     - 71.00%
     - 96.00%
     - 94.00%
     - 97.00%\ :sup:`※`
     - 84.00%\ :sup:`※`
     - 95.00%\ :sup:`※`
   * - ``pick_block_size``
     - 60.00%
     - 92.00%
     - —
     - 92.00%\ :sup:`※`
     - 88.00%\ :sup:`※`
     - 92.00%\ :sup:`※`
   * - ``pick_billiards_color``
     - 66.00%
     - 93.00%
     - —
     - 91.00%\ :sup:`※`
     - 82.00%\ :sup:`※`
     - 89.00%\ :sup:`※`
   * - ``pick_block_shape``
     - 32.00%
     - 45.00%
     - —
     - 37.00%\ :sup:`※`
     - 37.00%\ :sup:`※`
     - 41.00%\ :sup:`※`
   * - ``pick_specific_object``
     - 64.00%
     - 85.00%
     - 70.00%
     - 75.00%
     - 81.00%\ :sup:`※`
     - 83.00%\ :sup:`※`
   * - ``pick_object_type``
     - 50.00%
     - 65.00%
     - —
     - 64.00%
     - 63.00%\ :sup:`※`
     - 64.00%\ :sup:`※`
   * - ``pick_follow_logic_or``
     - 56.00%
     - 80.00%
     - —
     - 76.00%
     - 71.00%
     - 78.00%\ :sup:`※`
   * - ``pick_block_number``
     - 50.47%
     - 63.55%
     - —
     - 58.88%
     - 57.01%
     - 61.68%\ :sup:`※`
   * - ``pick_common_sense``
     - 47.00%
     - 65.00%
     - —
     - 53.00%
     - 51.00%
     - 53.00%\ :sup:`※`
   * - ``straighten_object``
     - 16.00%
     - —
     - —
     - 14.00%
     - 8.00%
     - 18.00%\ :sup:`※`

:sup:`※` indicates that the task was included in the corresponding
multi-task model's training mix.

Each success rate counts every test case of the sub-task (a scene instance
paired with one instruction) exactly once. The standalone evaluation config
runs 128 trajectories, more than a sub-task has test cases, so some cases run
twice; the repeats were removed before computing the rates. The success rate
logged during the run includes the repeats and can differ slightly from the
table.

The training reference configs are in ``examples/embodiment/config/``:

- Single-task PPO: ``geniesim_pick_block_color_ppo_openpi_pi05.yaml``; change
  the task selection to train another task.
- Single-task GRPO: ``geniesim_pick_block_color_grpo_openpi_pi05.yaml``; change
  the task selection to train another task.
- 4-task PPO: ``geniesim_multitask4_ppo_openpi_pi05.yaml``.
- 6-task PPO: ``geniesim_multitask6_ppo_openpi_pi05.yaml``.
- 10-task PPO: ``geniesim_multitask10_ppo_openpi_pi05.yaml``.

Appendix
--------

.. _geniesim-parameters:

Generalization Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 18 47

   * - Field
     - Default
     - Description
   * - ``env.train.seed``
     - ``0``
     - Training seed; nonnegative values enable reset randomization.
   * - ``env.eval.seed``
     - ``-1``
     - Disable reset randomization during evaluation.
   * - ``override_cfg.randomization``
     - ``null``
     - Set under env.train or env.eval; null keeps simulator defaults. All generalization fields below belong in this mapping.
   * - ``use_task_generalization``
     - ``true``
     - Inherit task defaults; false skips this source. Explicit values take precedence.
   * - ``use_benchmark_generalization``
     - ``true``
     - Inherit benchmark settings; false skips this source. Explicit values take precedence.
   * - ``object_xy_radius``
     - ``0.03``
     - Object XY position perturbation in meters; set to 0 to disable.
   * - ``init_joint_thresh``
     - ``0.05``
     - Initial joint-position offsets within ±0.05 rad; set to 0 to disable.
   * - ``init_base_xy_thresh``
     - ``[0.05, 0.05]``
     - Base X/Y offsets within ±0.05 m; null disables them. Per-instance base poses skip the default perturbation.
   * - ``render.lights.enable``
     - ``false``
     - Lighting perturbation; true enables it.
   * - ``render.material.enable``
     - ``false``
     - Material perturbation; true enables it.
   * - ``render.background.enable``
     - ``false``
     - Background variation; true enables it.
   * - ``render.camera_pose.enable``
     - ``false``
     - Camera-pose perturbation; enable with true and set nonzero thresholds in render.camera_pose.threshold.
   * - ``joint_pd.enable``
     - ``false``
     - Joint PD-gain perturbation; enable with true and set scale ranges in joint_pd.kp_scale and joint_pd.kd_scale.

.. _geniesim-training-tasks:

Training Tasks
~~~~~~~~~~~~~~

.. rubric:: Instruction Following

.. list-table::
   :header-rows: 1
   :widths: 5 35 28 32

   * - No.
     - subtask name
     - task name
     - Description
   * - 1
     - ``pick_block_color``
     - ``table_task_1_g2_op``
     - Pick a block of the specified color.
   * - 2
     - ``pick_billiards_color``
     - ``table_task_1_g2_op``
     - Pick a billiard ball of the specified color.
   * - 3
     - ``pick_specific_object``
     - ``table_task_0_g2_op``
     - Pick the specific object named in the instruction.
   * - 4
     - ``pick_block_size``
     - ``table_task_1_g2_op``
     - Pick a block of the specified size.
   * - 5
     - ``pick_object_type``
     - ``table_task_0_g2_op``
     - Pick an object of the specified category.
   * - 6
     - ``pick_follow_logic_or``
     - ``table_task_0_g2_op``
     - Pick an object satisfying either condition in the instruction.
   * - 7
     - ``pick_block_number``
     - ``table_task_0_g2_op``
     - Pick a block bearing the specified number.
   * - 8
     - ``pick_common_sense``
     - ``table_task_0_g2_op``
     - Pick an object based on a common-sense description.
   * - 9
     - ``pick_block_shape``
     - ``table_task_0_g2_op``
     - Pick a block of the specified shape.
   * - 10
     - ``straighten_object``
     - ``table_task_0_g2_op``
     - Straighten an object on the table.

.. rubric:: Spatial Understanding

.. list-table::
   :header-rows: 1
   :widths: 5 35 28 32

   * - No.
     - subtask name
     - task name
     - Description
   * - 11
     - ``pick_object_absolute_position``
     - ``table_task_2_g2_op``
     - Pick an object at the specified absolute position.
   * - 12
     - ``pick_object_relative_position``
     - ``table_task_2_g2_op``
     - Pick an object based on its position relative to another object.
   * - 13
     - ``place_object_relative_position``
     - ``table_task_2_g2_op``
     - Place an object at the specified position relative to another object.
   * - 14
     - ``place_beverage_to_anothers_position``
     - ``table_task_2_g2_op``
     - Place a beverage at another object's position.
   * - 15
     - ``sort_cubes_by_size``
     - ``table_task_2_g2_op``
     - Arrange cubes by size.
   * - 16
     - ``sort_number``
     - ``table_task_2_g2_op``
     - Arrange numbered objects in numerical order.
   * - 17
     - ``stack_bowls``
     - ``table_task_2_g2_op``
     - Stack bowls.
   * - 18
     - ``stack_three_building_blocks``
     - ``table_task_2_g2_op``
     - Stack three building blocks.

.. rubric:: Manipulation Skills

.. list-table::
   :header-rows: 1
   :widths: 5 35 28 32

   * - No.
     - subtask name
     - task name
     - Description
   * - 19
     - ``open_door``
     - ``home_g2_op``
     - Open a door.
   * - 20
     - ``pour_workpiece``
     - ``laboratory_06_g2_op``
     - Pour out workpieces.
   * - 21
     - ``take_wrong_item_shelf``
     - ``market_01_g2_op``
     - Remove a misplaced item from a shelf.
   * - 22
     - ``hold_pot``
     - ``kitchen_02_g2_op``
     - Hold a pot with both arms.
   * - 23
     - ``place_block_into_box``
     - ``table_task_2_g2_op``
     - Place a block into a box.
   * - 24
     - ``stock_and_straighten_shelf``
     - ``market_00_g2_op``
     - Stock a shelf and straighten its items.
   * - 25
     - ``clean_the_desktop``
     - ``study_room_05_g2_op``
     - Clear the desktop.
