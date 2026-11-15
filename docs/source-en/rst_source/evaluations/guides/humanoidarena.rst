HumanoidArena Evaluation
========================

Evaluate WB-WAM on the seven HumanoidArena tasks through RLinf. Each task uses
its matching checkpoint and evaluation preset. The example below runs OpenDoor
on one GPU.

Set Up the Environment
----------------------

Use a Linux host with an NVIDIA GPU. Follow the
`WB-WAM setup guide <https://github.com/WB-WaM/WB-WAM-Official/blob/6a9bf12d6c36d42ec66ee2aa7235ea2058175c40/benchmark/humanoidarena/README.md>`_
to prepare the model environment, Isaac Sim 5.1 environment, scene assets,
SONIC weights and base models. WB-WAM uses Python 3.10, and the simulator uses
Python 3.11.

From the RLinf repository root, install and activate the evaluation environment:

.. code-block:: bash

   bash requirements/install.sh --no-flash-attn embodied \
     --model wbwam --env humanoidarena
   source .venv/bin/activate

Download the Checkpoint
-----------------------

Download the OpenDoor checkpoint from the
`WB-WAM HumanoidArena release <https://huggingface.co/WB-WAM/WB-WAM-HumanoidArena>`_
using the download script in your WB-WAM checkout:

.. code-block:: bash

   python /path/to/WB-WAM-Official/scripts/download_models.py arena \
     --task open_door --output /path/to/checkpoints

The task directory is
``/path/to/checkpoints/humanoid_arena_native50/open_door``. It contains the
weights, ``config.yaml`` and ``dataset_stats.json``. To download all seven tasks,
omit ``--task open_door``.

Configure Local Paths
---------------------

Edit the paths in
``evaluations/humanoidarena/humanoidarena_wbwam_eval.yaml`` to match your setup:

.. code-block:: yaml

   env:
     eval:
       wbwam_repo: /path/to/WB-WAM-Official
       arena_root: /path/to/WB-WAM-Official/third_party/HumanoidArena
       simulation_python: /path/to/isaac/bin/python
       dependencies_root: /path/to/dependencies
       sonic_root: /path/to/sonic
       output_dir: /path/to/results/arena

   rollout:
     model:
       wbwam:
         repo_root: /path/to/WB-WAM-Official
         python: /path/to/wbwam/bin/python
         base_models_root: /path/to/base-models
         output_dir: /path/to/results/wbwam

Use the same WB-WAM checkout for both repository paths. Set ``sonic_root`` to
the directory containing ``model_encoder.onnx`` and ``model_decoder.onnx``.
The example runs the model and simulator together on GPU 0. To select another
GPU, update both ``cluster.component_placement`` and ``env.eval.simulation_gpu``.

Run Evaluation
--------------

Run one OpenDoor episode with the downloaded checkpoint:

.. code-block:: bash

   bash evaluations/run_eval.sh humanoidarena humanoidarena_wbwam_eval \
     rollout.model.model_path=/path/to/checkpoints/humanoid_arena_native50/open_door

The terminal reports ``eval/success``, ``eval/return``, ``eval/episode_len`` and
``eval/num_trajectories``. For this run, the trajectory count is 1, success and
return are 0 or 1, and episode length is measured in controller steps.

Select Another Task
~~~~~~~~~~~~~~~~~~~

Set ``task`` and the checkpoint directory together. Each preset selects the
task's instruction, weight filename, episode length and inference settings.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Task
     - Override
   * - Football
     - ``task=kick_football``
   * - DoubleDesk
     - ``task=hammer``
   * - P&PBox
     - ``task=box_shelf``
   * - OpenDoor
     - ``task=open_door``
   * - SitSofa
     - ``task=sit_sofa``
   * - Boxing
     - ``task=punch_markers``
   * - VisNavi
     - ``task=obstacle_navigation``

For example, evaluate Football with its checkpoint:

.. code-block:: bash

   bash evaluations/run_eval.sh humanoidarena humanoidarena_wbwam_eval \
     task=kick_football \
     rollout.model.model_path=/path/to/checkpoints/humanoid_arena_native50/kick_football

Run the Full Evaluation
~~~~~~~~~~~~~~~~~~~~~~~

Run 20 episodes for each of seeds 0, 1 and 2. For Football:

.. code-block:: bash

   for seed in 0 1 2; do
     bash evaluations/run_eval.sh humanoidarena humanoidarena_wbwam_eval \
       task=kick_football \
       rollout.model.model_path=/path/to/checkpoints/humanoid_arena_native50/kick_football \
       env.eval.rollout_epoch=20 \
       env.eval.seed="${seed}" \
       rollout.model.wbwam.seed="${seed}" \
       env.eval.output_dir="/path/to/results/kick_football/seed_${seed}/arena" \
       rollout.model.wbwam.output_dir="/path/to/results/kick_football/seed_${seed}/wbwam"
   done

Repeat for the other tasks with their matching checkpoints. Seven tasks,
three seeds and 20 episodes per seed give 420 episodes. Report each task's mean
and population standard deviation across its three seed success rates; compute
the overall mean and population standard deviation across all 21 task/seed rates.

View Results
------------

The evaluation log is saved to
``logs/<timestamp>-humanoidarena_wbwam_eval/eval_embodiment.log``.
See :doc:`../reference/results` for logging options.

Per-episode results are saved under
``env.eval.output_dir/env-*/episode-*/result.json``. Each result includes the
success status, termination reason and controller-step count. Episode videos are
saved in the episode's ``videos/success`` or ``videos/failure`` directory.
The model's logs and configuration are saved under
``rollout.model.wbwam.output_dir``.
