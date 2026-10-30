RL with RoboDojo Benchmark
==========================

`RoboDojo <https://robodojo-benchmark.github.io/>`__ is an Isaac Sim-based
benchmark for dual-arm manipulation. It includes tabletop rearrangement,
pick-and-place, and tool-use tasks for robots such as ARX-X5. RLinf exposes
RoboDojo as a Gym environment for VLA rollout and evaluation.

Overview
--------

Run RoboDojo on Isaac Sim and drive a dual-arm robot through RLinf's unified
environment interface.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅ (Pi_05)

   .. grid-item-card:: Algorithms
      :text-align: center

      Evaluation (π₀.₅)

   .. grid-item-card:: Tasks
      :text-align: center

      Put Bottles Into Dustbin

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 1+ GPU

| **You'll do:** install → download scene assets and the Pi_05 checkpoint →
  verify the installation → launch a single-environment task.
| **Prerequisites:** :doc:`Base installation </rst_source/start/installation>` ·
  an NVIDIA CUDA GPU · at least 120 GB of free disk space.

Tasks
~~~~~

RLinf currently provides a
``put_bottles_into_dustbin`` environment configuration as the integration and
verification example.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Task
     - Description
   * - ``put_bottles_into_dustbin``
     - Grab bottles from the table with both arms and place them in a dustbin.

The configuration is located at:

.. code-block:: text

   examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml

Other RoboDojo tasks use the same environment interface. Change
``task_config.task_name`` and the corresponding task parameters.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Head-camera RGB in ``main_images``, optional left/right wrist-camera RGB
       in ``wrist_images``, and a ``[B, 14]`` float32 proprioceptive state in
       ``states`` that holds the left/right arm joints and gripper states.
   * - Action
     - 14 continuous values in ``actions`` by default: left arm 6, right arm 6,
       left gripper 1, right gripper 1.
   * - Reward
     - Sparse task-completion reward by default.
   * - Prompt
     - Natural-language instruction generated from the current scene and exposed
       as ``task_descriptions``.

Version Lock
~~~~~~~~~~~~

RLinf locks the RoboDojo simulator, PyTorch stack, and compatibility
dependencies in ``requirements/install.sh`` and
``requirements/embodied/envs/robodojo-override.txt``. The current combination
is:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Component
     - Pinned version
   * - Python
     - 3.11
   * - RoboDojo runtime
     - 0.3.0 · commit ``6d76a0c``
   * - PyTorch
     - 2.7.0 · wheel for the detected driver (cu126, cu128 on CUDA 12.8+)
   * - TorchVision / TorchAudio
     - 0.22.0 / 2.7.0 · same wheel index as PyTorch
   * - Isaac Sim
     - 5.1.0
   * - IsaacLab
     - 0.54.3 · commit ``afca7b09``

Installation
------------

RoboDojo currently supports Linux x86_64 with an NVIDIA CUDA GPU. CPU-only,
macOS, and AMD ROCm execution are not supported. Check the driver and disk
space before installing:

.. code-block:: bash

   nvidia-smi
   uname -m
   df -h .

**Custom environment**

From the RLinf repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --env robodojo
   source .venv/bin/activate

Users in mainland China can add ``--use-mirror``. To include OpenPI:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robodojo

The installer installs PyTorch 2.7, Isaac Sim 5.1, and the pinned IsaacLab,
and applies the validated compatibility pins. Do not install
``rlinf-robodojo-runtime`` separately; a plain dependency resolve can produce
an environment that cannot start.

.. note::

   The installer sets ``OMNI_KIT_ACCEPT_EULA=YES``. Continuing with the
   installation means accepting the applicable NVIDIA Omniverse license terms.

Download Scene Assets
~~~~~~~~~~~~~~~~~~~~~

Dependency installation does not download scene assets. Export
``ROBODOJO_ASSETS_ROOT`` before running the installer so that the value is
recorded in ``.venv/bin/activate``; otherwise the recorded default is
``data/robodojo`` under the RLinf repository root. Then download the
approximately 39 GB ``Assets/`` tree:

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   source .venv/bin/activate
   uv pip install "huggingface_hub>=0.34,<2"
   hf download RoboDojo-Benchmark/RoboDojo \
     --repo-type dataset \
     --local-dir "$ROBODOJO_ASSETS_ROOT" \
     --include "Assets/**"

You can also place existing assets in this directory. The required layout is:

.. code-block:: text

   /path/to/robodojo/
   └── Assets/
       ├── Robots/
       ├── Object/
       ├── Material/
       └── Eval_Layout/

.. note::

   ``ROBODOJO_ASSETS_ROOT`` must point to the parent of ``Assets/``, not to
   ``Assets/`` itself. Set ``HF_ENDPOINT`` before the download when using a
   Hugging Face mirror.

Verify the Installation
~~~~~~~~~~~~~~~~~~~~~~~

This check validates the Python package and CUDA without starting the simulator:

.. code-block:: bash

   python - <<'PY'
   import torch
   from robodojo_runtime import assets_root, source_root
   from robodojo_runtime.bridge import VectorEnv

   print("torch:", torch.__version__)
   print("cuda:", torch.cuda.is_available())
   print("source:", source_root())
   print("assets:", assets_root())
   PY

Download the Model
------------------

The evaluation configuration uses RoboDojo's official Pi_05 checkpoint, a
π₀.₅ policy trained on the ``RoboDojo-sim-arx_x5`` simulation data. It ships
in the same ``RoboDojo-Benchmark/RoboDojo`` repository as the scene assets,
under ``ckpt/RoboDojo/Pi_05/``, with one directory per training run. Download
the run the example configuration evaluates:

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   source .venv/bin/activate
   hf download RoboDojo-Benchmark/RoboDojo \
     --repo-type dataset \
     --local-dir "$ROBODOJO_ASSETS_ROOT" \
     --include "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/**"

The download is a raw JAX/orbax checkpoint:

.. code-block:: text

   $ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/
   └── 59999/
       ├── _CHECKPOINT_METADATA
       ├── assets/arx_x5_sim/norm_stats.json
       ├── params/
       └── train_state/

RLinf's OpenPI loader reads an OpenPI-format checkpoint
(``model.safetensors``), so convert it. The converter imports ``jax`` and
``orbax-checkpoint``, which the ``--model openpi`` installation provides:

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   source .venv/bin/activate
   CKPT=$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model "$CKPT" \
     --input-norm-stats "$CKPT/assets/arx_x5_sim/norm_stats.json" \
     --output-model "$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05_robodojo_arx_x5" \
     --output-norm-stats "$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05_robodojo_arx_x5/arx_x5_sim/norm_stats.json" \
     --action-dim 32 \
     --action-horizon 50

The converted directory contains ``model.safetensors`` and
``arx_x5_sim/norm_stats.json``. Point ``rollout.model.model_path`` at it; the
norm stats path is derived from it. ``joint-1`` and ``joint-2`` are the other
two runs of the same checkpoint, and ``ckpt/RoboDojo/Pi_0/`` holds the
corresponding π₀ checkpoints.

.. note::

   Download one run. The whole ``Pi_05`` tree is about 70 GB across its three
   runs, and the evaluator loads one.

Run It
------

**Single-environment launch**

Start with one environment to verify Isaac Sim, cameras, and the downloaded
assets:

.. code-block:: bash

   python - <<'PY'
   import numpy as np
   from robodojo_runtime.bridge import VectorEnv

   env = VectorEnv(
       {
           "task_name": "put_bottles_into_dustbin",
           "env_cfg_type": "arx_x5",
           "headless": True,
           "cuda_device": 0,
           "max_episode_steps": 700,
       },
       n_envs=1,
       env_seeds=[0],
   )
   try:
       env.reset()
       obs = env.get_obs()[0]
       print(obs["full_image"].shape, obs["state"].shape)
       env.step(np.zeros((1, 1, 14), dtype=np.float32))
   finally:
       env.close()
   PY

Do not start a policy server, video recording, and multiple environments during
the first launch. Once reset, step, and close work for one environment, increase
``env.train.total_num_envs``.

**Use in a training configuration**

Reference the environment preset from your training configuration:

.. code-block:: text

   examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml

The policy's ``action_dim`` must equal the sum of ``action_dims``. The example
uses ``[6, 6, 1, 1]``, which is a 14-dimensional action.

**Evaluate a Pi_05 checkpoint**

``evaluations/robodojo/robodojo_put_bottles_into_dustbin_pi05_eval.yaml`` reuses
the environment preset above and evaluates ``put_bottles_into_dustbin`` with an
OpenPI Pi_05 checkpoint. Install the policy dependencies as well:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robodojo
   source .venv/bin/activate

Then start the evaluation:

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   bash evaluations/run_eval.sh robodojo \
     robodojo_put_bottles_into_dustbin_pi05_eval

The script logs to ``logs/<timestamp>-<config_name>/`` and writes results to the
directory given by ``runner.logger.log_path``. It runs one environment by
default; once that is stable, raise the parallelism with Hydra overrides such as
``env.eval.total_num_envs=4``, ``cluster.component_placement.env=0-3``, and
``cluster.component_placement.rollout=0-3``.

``state_action_order`` selects the layout of the 14-dimensional state and action
vectors. The official Pi_05 checkpoint was trained on the ALOHA order
``[left arm, left gripper, right arm, right gripper]``, so the evaluation
configuration sets ``state_action_order: aloha``.

``rollout.model.model_path`` in the evaluation configuration is a placeholder
for the converted checkpoint directory. The default
``rollout.model.openpi_data.norm_stats_path`` is derived from it, so overriding
``model_path`` is enough:

.. code-block:: bash

   bash evaluations/run_eval.sh robodojo \
     robodojo_put_bottles_into_dustbin_pi05_eval \
     rollout.model.model_path="$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05_robodojo_arx_x5"

Visualization and Results
-------------------------

RoboDojo is launched through the same embodied training entrypoint as other
RLinf environments. The primary metric is ``env/success_once``; see
:doc:`Training metrics <../../reference/metrics>` for the full metric list.

Enable rollout videos in the environment configuration when needed:

.. code-block:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

Common Problems
---------------

**``robodojo_runtime`` is not found**

Activate the virtual environment created by the installer. If you used
``--venv``, activate that environment.

**Dependency resolution fails**

Do not install Isaac Sim, IsaacLab, or the RoboDojo runtime manually.
Remove the failed environment and rerun the installation command so the pinned
versions and compatibility overrides are applied together.

**``pip check`` reports Isaac Sim/IsaacLab version differences**

These packages use overlapping exact version pins. The installer selects the
runtime-tested combination. Do not manually upgrade ``starlette``,
``uvicorn``, or ``typing-extensions``.

**``torch.cuda.is_available()`` returns ``False``**

Run ``nvidia-smi`` to check the driver and GPU visibility. RoboDojo does not
currently support CPU-only, macOS, or AMD ROCm execution.

**``Assets``, ``cam_head``, or task layouts are missing**

Verify that ``ROBODOJO_ASSETS_ROOT`` points to the parent of ``Assets/`` and
that all four required subdirectories exist. Remove incomplete downloads and
download them again.

**GPU memory is insufficient**

Set the environment count to one and stop other training or inference
processes. RoboDojo starts a separate Isaac Sim process per parallel
environment, so multi-environment settings depend on the measured per-scene
memory use.

**GPU memory remains allocated after exit**

Call ``env.close()`` from a ``try/finally`` block. The RoboDojo adapter supports
a complete close and recreate, not suspend and resume.
