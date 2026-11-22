π₀ / π₀.₅ RL
==============

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 35%

   The π₀ / π₀.₅ flow-based VLA models.

Fine-tune the **π₀** and **π₀.₅** flow-based VLA models with
reinforcement learning (PPO / GRPO) across several simulators using RLinf. For the full
method, see the paper
`πRL: Online RL Fine-Tuning for Flow-Based Vision-Language-Action Models <https://arxiv.org/abs/2510.25889>`__.

Use :doc:`Visual Short Memory <pi0/short_memory>` to condition OpenPI on past camera
frames during SFT and evaluation.

.. toctree::
   :hidden:

   Visual Short Memory <pi0/short_memory>

Overview
--------

RL-fine-tune π₀ / π₀.₅ on LIBERO, ManiSkill, MetaWorld, and CALVIN with PPO or GRPO.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      LIBERO · ManiSkill · MetaWorld · CALVIN

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO · GRPO

   .. grid-item-card:: Tasks
      :text-align: center

      Spatial · Object · Goal · Long

   .. grid-item-card:: Hardware
      :text-align: center

      NVIDIA CUDA · :ref:`AMD ROCm · Huawei Ascend CANN · Moore Threads MUSA <pi0-hardware>` (π₀ / π₀.₅, LIBERO · ManiSkill)

| **You'll do:** install → download an SFT checkpoint → pick a config → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a π₀ / π₀.₅ SFT checkpoint (steps below).

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
     - Spatial · Object · Goal · Long
     - ``libero_spatial_ppo_openpi_pi05`` / ``libero_10_grpo_openpi_pi05``
     - Fine-tune π0 / π0.5 on LIBERO manipulation suites.
   * - ManiSkill3
     - PickCube and related tasks
     - ``maniskill_ppo_openpi_pi05``
     - Fine-tune π0.5 on ManiSkill3 robot-control tasks.
   * - MetaWorld
     - MT50
     - ``metaworld_50_ppo_openpi_pi05``
     - Evaluate generalization across MetaWorld manipulation tasks.
   * - CALVIN
     - ABC-D
     - ``calvin_abc_d_ppo_openpi_pi05``
     - Train on long-horizon language-conditioned manipulation.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - Field
     - Description
   * - Observation
     - Main-view and wrist-view RGB plus robot state from LIBERO, ManiSkill3, MetaWorld, or CALVIN.
   * - Action
     - 7-D continuous control for end-effector position, rotation, and gripper state.
   * - Reward
     - Environment success or shaped reward used by PPO / GRPO.
   * - Prompt
     - Environment-provided natural-language task description consumed by the VLA processor.

π₀ / π₀.₅ train with PPO (actor-critic; GAE, ratio clipping, value clipping, entropy regularization) or GRPO (group-relative advantages over *G* sampled actions).

.. _pi0-hardware:

Setup and Run
-------------

.. include:: _setup_common.rst

NVIDIA CUDA supports all four environments listed above (select **π₀** or **π₀.₅** below). AMD ROCm, Huawei Ascend CANN, and Moore Threads MUSA support OpenPI π₀ / π₀.₅ on LIBERO and ManiSkill through the shared platform installer and scheduler device API; ManiSkill uses CPU simulation on non-CUDA backends, and MUSA also requires vendor simulator packages.

.. tab-set::

   .. tab-item:: π₀
      :sync: pi0

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      Download the LIBERO Spatial, Object, and Goal SFT checkpoint (also available on `ModelScope <https://www.modelscope.cn/models/RLinf/RLinf-Pi0-SFT-Spatial-Object-Goal>`__), or pick another environment from the table below:

      .. code-block:: bash

         hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT --local-dir ./RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

      .. list-table:: **π₀ Model List**
         :header-rows: 1
         :widths: 15 25 15 12 12

         * - Environment
           - Task Description
           - SFT Model
           - Flow-SDE
           - Flow-Noise

         * - LIBERO
           - Spatial, Object, Goal
           - |huggingface| `SFT Model <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT>`__
           - -
           - -

         * - LIBERO
           - Long
           - |huggingface| `SFT Model <https://huggingface.co/RLinf/RLinf-Pi0-LIBERO-Long-SFT>`__
           - -
           - -

         * - ManiSkill3
           - Multi-task
           - |huggingface| `38.4% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-SFT>`__
           - |huggingface| `78.8% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-RL-FlowSDE>`__
           - |huggingface| `77.8% <https://huggingface.co/RLinf/RLinf-Pi0-ManiSkill-25Main-RL-FlowNoise>`__

         * - MetaWorld
           - MT50
           - |huggingface| `50.8% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-SFT>`__
           - |huggingface| `78.1% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowSDE>`__
           - |huggingface| `85.8% <https://huggingface.co/RLinf/RLinf-Pi0-MetaWorld-RL-FlowNoise>`__

         * - CALVIN
           - ABC-D
           - |huggingface| `57.5% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-SFT>`__
           - |huggingface| `61.7% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-RL-FlowSDE>`__
           - |huggingface| `59.9% <https://huggingface.co/RLinf/RLinf-Pi0-CALVIN-ABC-D-RL-FlowNoise>`__

      .. include:: _model_path.rst

      **3. Launch**

      Shipped π₀ LIBERO configs under ``examples/embodiment/config/`` (see :doc:`Placement <../../concepts/placement>` to customize GPU layout):

      .. list-table::
         :header-rows: 1
         :widths: 30 70

         * - Recipe
           - Config
         * - π₀ + PPO (Spatial / Quickstart)
           - ``libero_spatial_ppo_openpi.yaml`` · ``libero_spatial_ppo_openpi_quickstart.yaml``
         * - π₀ + PPO / GRPO (LIBERO-10)
           - ``libero_10_ppo_openpi.yaml`` · ``libero_10_grpo_openpi.yaml``

      Launch LIBERO Spatial PPO training:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_openpi

      Pass ``libero_spatial_ppo_openpi_quickstart`` on smaller GPU setups. For standalone evaluation, see :doc:`evaluation <../../evaluations/index>`.

   .. tab-item:: π₀.₅
      :sync: pi05

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. Download**

      Download the LIBERO SFT checkpoint (covers Spatial, Object, Goal, and Long), or pick another environment from the table below:

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      .. list-table:: **π₀.₅ Model List**
         :header-rows: 1
         :widths: 15 25 15 12 12

         * - Environment
           - Task Description
           - SFT Model
           - Flow-SDE
           - Flow-Noise

         * - LIBERO
           - Spatial, Object, Goal, Long
           - |huggingface| `SFT Model <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT>`__
           - -
           - -

         * - ManiSkill3
           - Multi-task
           - |huggingface| `40.1% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT>`__
           - |huggingface| `90.9% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE>`__
           - |huggingface| `89.7% <https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise>`__

         * - MetaWorld
           - MT50
           - |huggingface| `43.8% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-SFT>`__
           - |huggingface| `70.7% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowSDE>`__
           - |huggingface| `66.1% <https://huggingface.co/RLinf/RLinf-Pi05-MetaWorld-RL-FlowNoise>`__

         * - CALVIN
           - ABC-D
           - |huggingface| `61.3% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-SFT>`__
           - |huggingface| `87.0% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowSDE>`__
           - |huggingface| `84.5% <https://huggingface.co/RLinf/RLinf-Pi05-CALVIN-ABC-D-RL-FlowNoise>`__

      .. include:: _model_path.rst

      **3. Launch**

      Shipped π₀.₅ LIBERO configs under ``examples/embodiment/config/`` (see :doc:`Placement <../../concepts/placement>` to customize GPU layout):

      .. list-table::
         :header-rows: 1
         :widths: 30 70

         * - Recipe
           - Config
         * - π₀.₅ + PPO (Spatial)
           - ``libero_spatial_ppo_openpi_pi05.yaml``
         * - π₀.₅ + PPO / GRPO (LIBERO-10)
           - ``libero_10_ppo_openpi_pi05.yaml`` · ``libero_10_grpo_openpi_pi05.yaml``

      Launch LIBERO-10 PPO training:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05

      For standalone evaluation, see :doc:`evaluation <../../evaluations/index>` (or :doc:`RTC <../../guides/rtc>` for π₀.₅ inference latency hiding).

   .. tab-item:: AMD ROCm
      :sync: rocm

      ROCm uses PyTorch's CUDA-compatible API, so OpenPI follows the shared AMD accelerator and installation path.

      **1. Install (LIBERO)**

      .. include:: _amd_libero.rst

      Inside the container, or directly on a host with ROCm installed, create the OpenPI LIBERO environment (omit ``--rocm`` to detect the installed version):

      .. code-block:: bash

         bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openpi --env libero && source .venv/bin/activate

      **2. Download**

      Download the π₀.₅ LIBERO SFT checkpoint (or pick another checkpoint from the model list in the **π₀** or **π₀.₅** tab):

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      .. include:: _model_path.rst

      **3. Launch LIBERO**

      Set the model path in ``examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml`` and enable software rendering (or ``egl`` on supported Radeon GPUs) in the active environment:

      .. include:: _libero_osmesa.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05

      **ManiSkill on AMD ROCm**

      Install the combined ManiSkill and LIBERO environment for OpenPI:

      .. code-block:: bash

         bash requirements/install.sh --platform amd --rocm 6.4 embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      Configure CPU simulation:

      .. include:: _maniskill_non_cuda.rst

      For π₀.₅, download ``RLinf/RLinf-Pi05-ManiSkill-25Main-SFT`` and set the model path in ``examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml`` (π₀ uses ``maniskill_ppo_openpi.yaml``). Adjust placement and batch sizes for your devices, then launch:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

   .. tab-item:: Huawei Ascend CANN
      :sync: ascend

      Use the Ascend LIBERO container or a host with CANN and the NPU driver installed.

      **1. Install (LIBERO)**

      .. include:: _ascend_libero.rst

      The published LIBERO image does not contain an OpenPI environment. Create one inside that container, or run the same command directly on an Ascend host (adds the matching ``torch-npu`` package and skips CUDA flash-attention):

      .. code-block:: bash

         bash requirements/install.sh --platform ascend embodied --model openpi --env libero && source .venv/bin/activate

      .. include:: _ascend_torch.rst

      **2. Download**

      Download the π₀.₅ LIBERO SFT checkpoint (or pick another checkpoint from the model list in the **π₀** or **π₀.₅** tab):

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      .. include:: _model_path.rst

      **3. Launch LIBERO**

      Set the model path in ``examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml`` and enable software rendering in the active environment:

      .. include:: _libero_osmesa.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05

      **ManiSkill on Huawei Ascend CANN**

      Install the combined ManiSkill and LIBERO environment for OpenPI:

      .. code-block:: bash

         bash requirements/install.sh --platform ascend embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      Configure CPU simulation:

      .. include:: _maniskill_non_cuda.rst

      For π₀.₅, download ``RLinf/RLinf-Pi05-ManiSkill-25Main-SFT`` and set the model path in ``examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml`` (π₀ uses ``maniskill_ppo_openpi.yaml``). Adjust placement and batch sizes for your devices, then launch:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

   .. tab-item:: Moore Threads MUSA
      :sync: musa

      MUSA reuses the image's Python, PyTorch, and ``torch_musa`` through a virtual environment with system site-packages enabled. The installer preserves those vendor packages and skips CUDA-only dependencies.

      **1. Install (LIBERO)**

      .. include:: _musa_libero.rst

      Inside the container, create the OpenPI LIBERO environment:

      .. code-block:: bash

         bash requirements/install.sh --platform musa embodied --model openpi --env libero && source .venv/bin/activate

      If a Transformers model path requests ``attn_implementation: flash_attention_2`` and Transformers cannot detect the vendor package, select ``sdpa`` instead.

      **2. Download**

      Download the π₀.₅ LIBERO SFT checkpoint (or pick another checkpoint from the model list in the **π₀** or **π₀.₅** tab):

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      .. include:: _model_path.rst

      **3. Launch LIBERO**

      Set the model path in ``examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml`` and enable software rendering in the active environment:

      .. include:: _libero_osmesa.rst

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05

      **ManiSkill on Moore Threads MUSA**

      Use the vendor simulator image and its existing OpenPI environment:

      .. include:: _musa_maniskill.rst

      .. code-block:: bash

         source switch_env openpi

      Configure CPU simulation:

      .. include:: _maniskill_non_cuda.rst

      For π₀.₅, download ``RLinf/RLinf-Pi05-ManiSkill-25Main-SFT`` and set the model path in ``examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml`` (π₀ uses ``maniskill_ppo_openpi.yaml``). Adjust placement and batch sizes for your devices, then launch:

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

.. dropdown:: Advanced configuration (optional)

   **OpenPI model and flow-RL parameters**

   - **Action chunk vs. network horizon**: ``num_action_chunks`` / ``openpi.action_chunk`` sets the steps executed per rollout step, while ``openpi.action_horizon`` defaults to the official ``TrainConfig`` horizon for ``openpi.config_name`` (``50`` for ``pi0_libero``, ``10`` for ``pi05_libero``). Override ``openpi.action_horizon`` only when a checkpoint uses a different horizon.
   - **π₀.₅ selector and critic placement**: Select π₀.₅ with ``openpi.config_name`` (such as ``pi05_libero``) and ``actor.model.pi05: True``. Set ``openpi.value_after_vlm: True`` for π₀.₅ PPO to attach the critic after the VLM output instead of the action expert.
   - **Flow-RL noise method and steps**: Control denoising steps with ``num_steps`` and choose ``openpi.noise_method`` between `flow_sde <https://arxiv.org/abs/2505.05470>`__ (``noise_level: 0.5``, ``joint_logprob: False``, ``algorithm.entropy_bonus: 0.0``; see ``libero_spatial_ppo_openpi.yaml``) and `flow_noise <https://arxiv.org/abs/2505.22094>`__ (``noise_logvar_range: [0.08, 0.16]``, ``joint_logprob: True``, ``algorithm.entropy_bonus: 0.005``; see ``maniskill_ppo_openpi.yaml``).
   - **Kernel compilation**: OpenPI compiles activation, rotary embedding, and gated residual kernels with ``torch.compile`` when ``actor.model.openpi.torch_compile`` is ``True`` and ``triton`` is installed (π₀.₅ enables it by default; backends without Triton stay eager). Disable it when Triton is installed but Inductor cannot run (such as with Isaac Sim):

     .. code-block:: yaml

        actor:
          model:
            openpi:
              torch_compile: False

   **LoRA and FSDP sharding**

   - Set ``actor.model.is_lora: True`` and ``actor.model.lora_rank: 8`` to fine-tune the VLM with LoRA. Keep ``gradient_checkpointing: False``, as gradient checkpointing is not supported for OpenPI.
   - OpenPI supports FSDP1 ``no_shard``, ``shard_grad_op``, and ``full_shard``. Set ``actor.fsdp_config.sharding_strategy: full_shard`` and ``actor.fsdp_config.use_orig_params: true`` to shard training state across actor ranks while keeping the recipe's mixed-precision settings.

   **Minimum test case and OOM tuning**

   ``libero_spatial_ppo_openpi_quickstart.yaml`` reduces ``env.train.rollout_epoch`` (8 → 2), ``env.train.total_num_envs`` (64 → 32), ``actor.micro_batch_size`` (128 → 64), ``actor.global_batch_size`` (2048 → 256), and ``actor.optim.lr`` (5e-6 → 1e-6), and enables offload on actor and rollout. On 4 H100 GPUs it matches the standard config at the same wall-clock time:

   .. image:: https://github.com/user-attachments/assets/80d098f6-5286-4ff4-89be-547f43a4dc86
      :alt: Minimum test case comparison
      :width: 95%
      :align: center

   If OOM persists:

   - **Rollout OOM**: Switch rendering from ``egl`` to ``osmesa``, halve ``env.train.total_num_envs`` (32 → 16) while doubling ``env.train.rollout_epoch`` (2 → 4), and set ``actor.enable_offload: True``.
   - **Actor OOM**: Lower ``actor.micro_batch_size`` (64 → 32) while keeping ``actor.global_batch_size`` divisible by ``micro_batch_size`` × GPU count, and set ``rollout.enable_offload: True``.

   **Alternative evaluation script**

   For per-subtask LIBERO success rates matching upstream OpenPI, or for MetaWorld (which does not support ``env.eval.auto_reset=True``), use the standalone script in `toolkits/standalone_eval_scripts/openpi/README.md <https://github.com/RLinf/RLinf/blob/main/toolkits/standalone_eval_scripts/openpi/README.md>`__.

Visualization and Results
-------------------------

Launch TensorBoard with ``tensorboard --logdir ./logs --port 6006`` and watch ``env/success_once``. Pass ``env.eval.video_cfg.save_video=True`` to record evaluation videos; see :doc:`Training metrics <../../reference/metrics>` for all logged metrics and logger backends.

LIBERO Results
~~~~~~~~~~~~~~

We trained π₀ and π₀.₅ with PPO and GRPO in the LIBERO environment.
The results achieved through RL training are shown below:

.. list-table:: **π₀ model results on LIBERO**
   :header-rows: 1

   * - Model
     - Spatial
     - Object
     - Goal
     - Long
     - Average
     - Δ Avg.

   * - π₀ (few-shot)
     - 65.3%
     - 64.4%
     - 49.8%
     - 51.2%
     - 57.6%
     - ---

   * - +GRPO
     - 97.8%
     - 97.8%
     - 83.2%
     - 81.4%
     - 90.0%
     - +32.4

   * - +PPO
     - **98.4%**
     - **99.4%**
     - **96.2%**
     - **90.2%**
     - **96.0%**
     - **+38.4**

.. list-table:: **π₀.₅ model results on LIBERO**
   :header-rows: 1

   * - Model
     - Spatial
     - Object
     - Goal
     - Long
     - Average
     - Δ Avg.

   * - π₀.₅ (few-shot)
     - 84.6%
     - 95.4%
     - 84.6%
     - 43.9%
     - 77.1%
     - ---

   * - +GRPO
     - 97.4%
     - 99.8%
     - 91.2%
     - 77.6%
     - 91.5%
     - +14.4

   * - +PPO
     - **99.6%**
     - **100%**
     - **98.8%**
     - **93.0%**
     - **97.9%**
     - **+20.8**

MetaWorld Results
~~~~~~~~~~~~~~~~~
For MetaWorld results, please check :doc:`MetaWorld Page <metaworld>`.

CALVIN Results
~~~~~~~~~~~~~~~~~
For CALVIN results, please check :doc:`CALVIN Page <calvin>`.
