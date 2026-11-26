StarVLA RL
==========

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/starvla.png
   :align: center
   :width: 70%

   StarVLA: a modular VLM backbone + action head.

Run RL fine-tuning for **StarVLA** in RLinf. StarVLA is an open-source
Vision-Language-Action toolkit that composes a VLM backbone with an action head
in a modular way; this example uses the **QwenOFT** setup and trains it on
**LIBERO** with GRPO.

Overview
--------

Fine-tune StarVLA (QwenOFT) on LIBERO Spatial with GRPO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      LIBERO

   .. grid-item-card:: Algorithms
      :text-align: center

      GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      LIBERO Spatial

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · NVIDIA CUDA · :ref:`Huawei Ascend CANN <starvla-hardware>` (QwenOFT + LIBERO)

| **You'll do:** install → download the StarVLA checkpoint + base VLM → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a StarVLA LIBERO checkpoint and the Qwen2.5-VL base (steps below).

Tasks
~~~~~

Select the model page by matching the environment, task family, and config or checkpoint artifact.

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - LIBERO
     - LIBERO-Spatial
     - ``libero_spatial_grpo_starvla``
     - GRPO fine-tuning for StarVLA in LIBERO.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - LIBERO image observations and robot state formatted for StarVLA.
   * - Action
     - Continuous robot control commands generated through the StarVLA policy API.
   * - Reward
     - LIBERO task success or shaped reward used by GRPO.
   * - Prompt
     - Natural-language LIBERO task instruction.

Interface Conventions
~~~~~~~~~~~~~~~~~~~~~~

In the RLinf StarVLA wrapper, ``env_obs`` is a batch-first dict (dimension 0 is batch size ``B``).

Required fields:

* ``main_images``: main-view RGB, ``torch.uint8``, shape ``[B, H, W, 3]``.
* ``states``: proprio/state tensor, ``torch.float32``, shape ``[B, D_state]``.
* ``task_descriptions``: natural-language descriptions, ``list[str]`` with length ``B``.

Optional fields:

* ``wrist_images``: wrist-view RGB, ``torch.uint8``, shape ``[B, H, W, 3]``.
* ``extra_view_images``: additional RGB views, recommended shape ``[B, V, H, W, 3]`` where ``V`` is the number of extra views. A single extra view may also be provided as ``[B, H, W, 3]`` and is treated as ``V=1``.

In default LIBERO usage, ``states`` is commonly end-effector position ``(x, y, z)`` (3-D),
end-effector axis-angle ``(rx, ry, rz)`` (3-D), and gripper state (originally 2-D), so
``D_state`` is often ``3 + 3 + 2 = 8``. If a checkpoint expects 7-D state, the wrapper
compresses the 2-D gripper state into ``[x, y, z, rx, ry, rz, g_mean]`` where
``g_mean = 0.5 * (g0 + g1)``.

StarVLA inference outputs chunked actions ``[B, T, D_action]`` with
``T = actor.model.num_action_chunks`` (planning horizon) and
``D_action = actor.model.action_dim`` (commonly 7 on LIBERO). Rollout follows a
receding-horizon strategy: each forward pass predicts ``T`` actions, the environment
executes the first ``N`` steps (``1 <= N <= T``), then replans.

.. _starvla-hardware:

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: NVIDIA CUDA
      :sync: cuda

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model starvla --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      Download the StarVLA checkpoint and the Qwen2.5-VL base model:

      .. code-block:: bash

         hf download StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1 --local-dir ./Qwen2.5-VL-OFT-LIBERO-4in1
         hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

      Set ``framework.qwenvl.base_vlm`` in ``Qwen2.5-VL-OFT-LIBERO-4in1/config.yaml`` to your local ``Qwen2.5-VL-3B-Instruct`` directory.

      .. include:: _model_path.rst

      **3. Launch**

      StarVLA + GRPO + LIBERO Spatial uses ``examples/embodiment/config/libero_spatial_grpo_starvla.yaml``. Launch GRPO training:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_starvla

      For standalone evaluation, see the :doc:`LIBERO evaluation guide <../../evaluations/guides/libero>`.

   .. tab-item:: Huawei Ascend CANN
      :sync: ascend

      Huawei Ascend CANN supports StarVLA (QwenOFT) training on LIBERO with GRPO using the same checkpoint and task configuration, with OSMesa for CPU rendering.

      **1. Install**

      The RLinf installer pins starVLA to ``starVLA-v1.6`` and builds ``decord`` from source on aarch64. If you reuse an upstream checkout through ``STARVLA_PATH``, keep it at ``starVLA-v1.6``.

      Start with the Ascend LIBERO container or a host with CANN and the NPU driver installed:

      .. include:: _ascend_libero.rst

      Create a StarVLA environment inside the container or directly on the Ascend host (installs matching ``torch-npu`` and skips CUDA flash-attention):

      .. code-block:: bash

         bash requirements/install.sh --platform ascend embodied --model starvla --env libero && source .venv/bin/activate

      **2. Download**

      Download the StarVLA checkpoint and Qwen2.5-VL base model:

      .. code-block:: bash

         hf download StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1 --local-dir ./Qwen2.5-VL-OFT-LIBERO-4in1
         hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

      Set ``framework.qwenvl.base_vlm`` in ``Qwen2.5-VL-OFT-LIBERO-4in1/config.yaml`` to your local ``Qwen2.5-VL-3B-Instruct`` directory.

      .. include:: _model_path.rst

      **3. Launch**

      Enable software rendering in the active environment:

      .. include:: _libero_osmesa.rst

      The default recipe uses one node and places actor, rollout, and environment workers on all available devices. Adjust ``cluster.component_placement``, ``actor.micro_batch_size``, ``actor.global_batch_size``, and the training and evaluation ``total_num_envs`` in ``examples/embodiment/config/libero_spatial_grpo_starvla.yaml`` for your NPU count and memory, then launch LIBERO Spatial training:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_starvla

      For Qwen2.5-VL checkpoints, you can reduce attention overhead by adding this option to the experiment configuration:

      .. code-block:: yaml

         actor:
           model:
             starvla:
               enable_npu_attention: True

      This option applies to both actor and rollout models, is disabled by default, and takes effect only on Ascend. The StarVLA installer provides the required Transformers version, ``4.57.6``.

Visualization and Results
-------------------------

Launch TensorBoard with ``tensorboard --logdir ./logs --port 6006`` and watch ``env/success_once``. Pass ``env.eval.video_cfg.save_video=True`` to record evaluation videos; see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.

Reference curves (using the model from
`LIBERO_BASELIEN_FORJINHUI_10K_QWENOFT <https://huggingface.co/JasonYang66/LIBERO_BASELIEN_FORJINHUI_10K_QWENOFT>`_):

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/libero_goal_starvla_baseline.png
   :alt: LIBERO Goal StarVLA baseline result curve
   :width: 95%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/libero_object_starvla_baseline.png
   :alt: LIBERO Object StarVLA baseline result curve
   :width: 95%
   :align: center

.. image:: https://raw.githubusercontent.com/RLinf/misc/main/pic/libero_spatial_starvla_baseline.png
   :alt: LIBERO Spatial StarVLA baseline result curve
   :width: 95%
   :align: center
