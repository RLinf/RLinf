RL with Bench2Dex Benchmark
===========================

.. figure:: https://raw.githubusercontent.com/Bench2Dex/Bench2Dex/main/assets/overview.png
   :align: center
   :width: 90%

   Bench2Dex (image: `Bench2Dex <https://github.com/Bench2Dex/Bench2Dex>`__).

`Bench2Dex <https://github.com/Bench2Dex/Bench2Dex>`__ is an Isaac Lab
benchmark of bimanual dexterous manipulation across twelve arm and hand
embodiments. RLinf provides CNN PPO and OpenPI pi0.5 imitation learning and
PPO recipes.

Overview
--------

Train bimanual dexterous policies on Bench2Dex tasks from camera images and joint positions.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      CNN · OpenPI pi0.5

   .. grid-item-card:: Algorithms
      :text-align: center

      IL · PPO

   .. grid-item-card:: Tasks
      :text-align: center

      26 tasks · 12 embodiments

   .. grid-item-card:: Hardware
      :text-align: center

      CNN: 1 GPU · OpenPI PPO: 8 GPUs

| **You'll do:** install → download Isaac Sim + assets + model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Isaac Sim · Bench2Dex assets · a ResNet checkpoint (steps below).

Tasks
~~~~~

Training and evaluation each run one task. Set ``init_params.task_name`` and
``max_episode_steps`` in ``env.train`` and ``env.eval``, and set
``env.eval.max_steps_per_rollout_epoch`` to the same step count.
``max_episode_steps`` is the task's official episode budget, and the
description is its official instruction:

.. list-table::
   :header-rows: 1
   :widths: 36 16 48

   * - Task
     - ``max_episode_steps``
     - Description
   * - ``03_wine_glass_plate_balance``
     - 1082
     - Carry three filled wine glasses to the plates of three diners without spilling or knocking over any objects.
   * - ``06_fruit_bowl_loading``
     - 640
     - Use the left hand to move the bowl to the table center, the right hand to place two apples into it, and the left hand to place the banana into it.
   * - ``07_citrus_plate_loading``
     - 602
     - Use the left hand to place the farther lemon onto the plate, the right hand to move the paired orange near the plate, and the left hand to place that orange onto the plate; repeat for the remaining pair.
   * - ``08_frypan_stand_pour``
     - 1144
     - Use the left hand to pick up the frypan and place it onto the display stand, then pour the soy sauce and olive oil into the frypan one by one.
   * - ``09_cleaner_moisturizer_box_loading``
     - 642
     - Use both hands to lift the cleaner and place it upright into the wooden box, then use the right hand to place the soap into the box.
   * - ``12_screwdriver_box_and_hammer``
     - 1149
     - Use the right hand to place the right screwdriver into the box and move the box to the left, use the left hand to place the left screwdriver into the box, then pick up the hammer, strike the wooden block once, and place the hammer into the box.
   * - ``21_condiment_box_loading``
     - 827
     - Use the right hand to place the monosodium glutamate into the box and move the box to the table center; then use the left hand to place the soy sauce into the box, followed by the vinegar.
   * - ``22_tool_box_loading``
     - 681
     - Use the right hand to place the drill and flat screwdriver into the wooden box, then use the left hand to place the wrench and Phillips screwdriver into the box.
   * - ``24_stationery_category_sorting``
     - 1542
     - Right hand sorts marker and battery into pen cup and plastic box, left hand sorts gray pen and glue into pen cup and plastic box.
   * - ``26_canned_food_tray_line_arrangement``
     - 871
     - Use left hand to place the master chef can and MSG onto the tray, use right hand to place the milk box and potted meat can onto the tray.
   * - ``27_ball_box_loading``
     - 1190
     - Place mini soccer ball into box, place tennis ball into box, place golf ball into box, place ping-pong ball into box.
   * - ``32_baking_tray_prep_with_tools``
     - 852
     - Use the left hand to place the brush and spatula onto the tray, use the right hand to place the small pudding box then the large gelatin box onto the tray in sequence.
   * - ``34_fridge_wine_interhand_pour``
     - 1080
     - Use the left hand to open the refrigerator and take out the wine bottle, hand it to the right hand in the air to pour a glass of wine, hand it back to the left hand to return it to the refrigerator, then close the door with the left hand.
   * - ``42_trash_disposal``
     - 668
     - Open the trash can lid, throw crumpled paper into the trash can, use left hand to throw bottle and banana peel into the trash can, close the trash can lid.
   * - ``43_fridge_fruit_shelf_sorting``
     - 1375
     - Right hand moves banana to the left side of desk, left hand opens both fridge doors, place apple on upper shelf and banana on lower shelf, then right hand closes fridge door.
   * - ``44_microwave_bowl_loading``
     - 1061
     - Open microwave door with left hand, bring bowl to microwave with right hand, place baguette into bowl with left hand, place bowl into microwave with right hand, close microwave door with left hand.
   * - ``51_toilet_lid_cleaner_pour``
     - 1023
     - Open the toilet lid with left hand, pour cleaner into toilet with right hand, close the toilet lid with left hand.
   * - ``60_breadbasket_fast_food_loading``
     - 995
     - Use the right hand to place the baguette and bread into the bread basket, move the basket to the center of the table, then use the left hand to place the bread and french fries into the basket as a fast-food setup.
   * - ``61_medicine_shoebox_pack``
     - 1034
     - Use the left hand to move the shoe box to the center of the table, then use the right hand to place the pill bottle and toothpaste into the box, and use the left hand to place the hydrating oil into the box.
   * - ``62_shoebox_accessory_pack``
     - 1077
     - Use the right hand to place the seal into the shoe box and move the box to the center of the table, then use the left hand to place the shoe and pet collar into the box.
   * - ``64_sports_ball_cup_sort``
     - 665
     - Place the tennis ball and baseball into the large cup, place the racquetball and golf ball into the small cup.
   * - ``67_faucet_cup_water_fill``
     - 964
     - Use the left hand to first place the spoon into the mug, use the right hand to place the mug under the faucet, use the left hand to open the faucet to fill the mug and then close it, and use the right hand to place the mug onto the tray.
   * - ``73_jigsaw_puzzle_assembly``
     - 1213
     - Right hand and left hand in turn assemble the green, red, blue, yellow puzzle pieces onto the fixed white puzzle piece in the center of the table, forming a rectangle.
   * - ``76_soup_serving``
     - 986
     - Hold the bowl beside the pot with the left hand, use the right hand to ladle two scoops of soup from the pot into the bowl, carry the bowl with the left hand to the front-right area, and return the ladle with the right hand.
   * - ``79_bimanual_piano_melody``
     - 681
     - Both hands play the piano key sequence C-C-G-G-A-A-G together, left hand plays the bass notes, right hand plays the treble notes.
   * - ``80_gaming_desk_setup``
     - 593
     - Straighten the monitor screen forward with both hands, press the ESC key with the left index finger, move the mouse back to the left side of the keyboard with the right hand, and click the left mouse button with the index finger.

Set ``init_params.robot_key`` to choose the embodiment and
``init_params.action_dim`` to its number of active joints:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - ``robot_key``
     - ``action_dim``
   * - ``multi_iiwa7_with_sharpa``
     - 54
   * - ``multi_jaka_zu7_dexhand021_with_flange``
     - 52
   * - ``multi_panda_with_allegro``
     - 32
   * - ``multi_panda_with_orca``
     - 40
   * - ``multi_rm_65_with_revo2``
     - 24
   * - ``multi_ur5_rh56dfx_with_flange``
     - 24
   * - ``multi_ur5_rh5dg2_with_flange``
     - 38
   * - ``multi_ur5_schunk_hand_with_flange``
     - 30
   * - ``multi_ur5_shadow_hand_with_flange``
     - 56
   * - ``multi_ur5_wuji_with_flange``
     - 52
   * - ``multi_xarm7_with_ability``
     - 26
   * - ``multi_xarm7_with_leap``
     - 40

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - RGB images from ``init_params.camera_names`` (default ``[cam_stereo_left, cam_stereo_right, cam_wrist_left, cam_wrist_right]``), rendered at 640×480 and resized to ``init_params.image_size`` (128×128 in the CNN recipe), plus the ``action_dim`` active joint positions.
   * - Action
     - ``action_dim``-dim continuous action (26 for the default ``multi_xarm7_with_ability``): absolute position targets for the active joints at 20 Hz; each action is held for three 60 Hz physics steps.
   * - Reward
     - Sparse success reward. A task succeeds once its official success condition holds for the task's dwell time.
   * - Prompt
     - The task description.

.. note::

   RLinf builds each scene with Bench2Dex scene generalization off: the task's
   YAML layout on a ground plane lit by one distant light. The official
   Bench2Dex evaluation turns generalization on, and its profiles restore the
   room background, table texture, lighting, table height, camera offsets,
   clutter, and object placements recorded with a demonstration, or resample
   some or all of them.

Installation
------------

.. include:: _setup_common.rst
   :end-before: Then set up the dependencies

Install Bench2Dex dependencies:

.. code:: bash

   # Mainland China users can add --use-mirror.
   bash requirements/install.sh embodied --env bench2dex
   source .venv/bin/activate

Download Isaac Sim
~~~~~~~~~~~~~~~~~~

Download Isaac Sim 5.1.0 and initialize its shell environment:

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
   source ./setup_conda_env.sh

.. warning::

   Run ``source ./setup_conda_env.sh`` in every new terminal before launching Bench2Dex.

Download the Assets
-------------------

Download the object and robot assets:

.. code:: bash

   # export HF_ENDPOINT=https://hf-mirror.com
   hf download Bench2Dex/Assets --repo-type dataset \
     --include 'Objects/**' 'Robots_p/**' \
     --local-dir /path/to/dex2bench_dataset
   export BENCH2DEX_ASSETS_PATH=/path/to/dex2bench_dataset

Download the Model
------------------

For the CNN + PPO recipe, download the ResNet checkpoint:

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

Then set the same checkpoint path for rollout and actor in
``examples/embodiment/config/bench2dex_ppo_cnn.yaml``:

.. code-block:: yaml

   rollout:
      model:
         model_path: /path/to/RLinf-ResNet10-pretrained
   actor:
      model:
         model_path: /path/to/RLinf-ResNet10-pretrained

Run It
------

Launch the Bench2Dex recipe from a terminal where Isaac Sim is initialized:

.. list-table::
   :header-rows: 1
   :widths: 28 46 26

   * - Recipe
     - Config
     - Command suffix
   * - CNN + PPO
     - ``examples/embodiment/config/bench2dex_ppo_cnn.yaml``
     - ``bench2dex_ppo_cnn``

.. code:: bash

   source /path/to/isaac_sim/setup_conda_env.sh
   export BENCH2DEX_ASSETS_PATH=/path/to/dex2bench_dataset

   bash examples/embodiment/run_embodiment.sh bench2dex_ppo_cnn

What this does:

1. Starts the embodied training entrypoint with the ``bench2dex_ppo_cnn`` config.
2. Creates Ray workers for the actor, rollout, and Bench2Dex env components.
3. Runs PPO on ``03_wine_glass_plate_balance`` with the ``multi_xarm7_with_ability`` embodiment.

.. note::

   The config runs on GPU ``0`` by default. Each Bench2Dex environment runs one
   scene in its own Isaac Sim process, and an env worker holds one for training
   and, when evaluation is enabled, one for evaluation. Set
   ``env.train.total_num_envs`` and ``env.eval.total_num_envs`` to the number of
   env workers. To run four env workers on GPU ``0``, set
   ``cluster.component_placement`` to ``actor,rollout: 0`` and ``env: 0:0-3``,
   and both ``total_num_envs`` to ``4``. Tune batch sizes for your hardware. See
   :doc:`Placement <../../concepts/placement>` for the syntax.

.. note::

   For CNN policies, keep ``actor.model.state_dim`` and ``actor.model.action_dim`` equal to
   ``init_params.action_dim``, and ``actor.model.image_num`` equal to the number
   of cameras in ``init_params.camera_names``, when you change the embodiment or
   the cameras.

OpenPI pi0.5
------------

The OpenPI recipes use ``03_wine_glass_plate_balance`` and
``multi_xarm7_with_ability``. They consume all four RGB views and 26 active
joint positions, and predict 20 absolute joint targets per action chunk.
The base pi0.5 model keeps its 32-dimensional action space: transforms
normalize the 26 active joints, pad to 32 for the model, then unnormalize
and select the 26 active outputs for the environment.

Install the model and simulator dependencies together:

.. code:: bash

   bash requirements/install.sh embodied --model openpi --env bench2dex
   source .venv/bin/activate

Convert the official task's HDF5 replay episodes into a LeRobot dataset:

.. code:: bash

   python -m toolkits.lerobot.convert_bench2dex_to_lerobot \
     --input-dir /path/to/dataset/03_wine_glass_plate_balance/replay-generalization \
     --output-dir /path/to/bench2dex_lerobot/03_ability \
     --active-dof-map "$BENCH2DEX_REPO_PATH/robots/active_dof_maps.yml" \
     --task 'Carry three filled wine glasses to the plates of three diners without spilling or knocking over any objects.' \
     --repo-id bench2dex/03_ability --fps 20

The converter exports RGB and active-joint trajectories, excludes invalid and
homing frames, and splits trajectories at gaps to preserve the 20 Hz timebase.
It writes ``norm_stats.json`` from every exported state and action. The output
has these fields:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Dataset field
     - Content
   * - ``observation.state``
     - 26 active joint positions in simulator joint order
   * - ``action``
     - 26 absolute active-joint position targets in the same order
   * - ``observation.images.cam_stereo_left`` / ``observation.images.cam_stereo_right``
     - Left and right stereo RGB images
   * - ``observation.images.cam_wrist_left`` / ``observation.images.cam_wrist_right``
     - Left and right wrist RGB images
   * - ``task``
     - Task instruction from the dataset metadata

Use the converter's normalization file for IL, evaluation, and PPO. For a
separately prepared LeRobot dataset, the existing statistics utility also
accepts the Bench2Dex data configuration:

.. code:: bash

   python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_bench2dex_base --repo-id /path/to/bench2dex_lerobot/03_ability

Set ``data.train_data_paths``, ``actor.model.model_path`` and
``actor.model.openpi_data.norm_stats_path`` in
``examples/sft/config/bench2dex_sft_openpi_pi05.yaml``. The model path must
contain base pi0.5 weights in RLinf's OpenPI format; see
:doc:`Checkpoint conversion <../../guides/convertor>` for conversion commands.
Set ``runner.max_steps`` to the desired number of optimizer steps, then run:

.. code:: bash

   bash examples/sft/run_vla_sft.sh bench2dex_sft_openpi_pi05

For PPO, set both model paths in
``examples/embodiment/config/bench2dex_ppo_openpi_pi05.yaml`` to the IL
``checkpoints/global_step_<step>`` directory and set its normalization path.
Initialize Isaac Sim, then keep the OpenPI environment's PyTorch ahead of the
SDK's bundled Python packages and CUDA libraries:

.. code:: bash

   source /path/to/isaac_sim/setup_conda_env.sh
   OPENPI_SITE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
   export PYTHONPATH="$OPENPI_SITE:${PYTHONPATH:-}"
   export LD_LIBRARY_PATH="$OPENPI_SITE/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
   bash examples/embodiment/run_embodiment.sh bench2dex_ppo_openpi_pi05

The PPO recipe uses eight GPUs: four actor workers, four rollout workers,
and eight environment workers sharing those GPUs. Its 1080-step horizon is
54 complete 20-action chunks. Eight environments produce 432 chunk samples;
global batch 216 and three PPO epochs give six optimizer updates per
iteration. Keep these batch divisibility constraints when changing placement.
The image order is stereo left, stereo right, wrist left, wrist right for
both IL and PPO.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

Use ``env/success_once`` for training success rate and ``eval/success_once``
for evaluation success rate. Compare IL checkpoints using the same episode
budget, reset seeds, and policy sampling settings. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Enable video in the env config when needed:

.. code:: yaml

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval
