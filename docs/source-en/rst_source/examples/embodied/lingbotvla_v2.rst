SFT and GRPO on LingBot-VLA 2.0
================================

RLinf provides LingBot-VLA 2.0 integration for SFT with LeRobot-format data
and GRPO on RoboTwin. Use the ``lingbotvla_v2`` model type and V2 configs;
the :doc:`Lingbot-VLA 1.0 <lingbotvla>` checkpoints and configs are not
interchangeable with these examples.

Installation
------------

Run the following commands from the RLinf repository root. The installer
requires Python 3.12, Git, an NVIDIA CUDA toolkit, and network access to
download dependencies. It builds CUDA extensions, including FlashAttention.

.. code-block:: bash

    PYTHON=python3.12 bash requirements/install.sh embodied \
      --model lingbotvla_v2 --env robotwin

The destination must not already exist. Use a new environment directory
when retrying a failed installation; existing environments are not modified.
Packages are installed with ``uv``, which the installer sets up when it is
missing. uv keeps the wheels built from pinned sources in its cache, so a
second installation on the same machine skips the CUDA extension builds.
This model uses pinned dependencies; version overrides, ``--use-mirror``,
and ``--no-flash-attn`` are not supported, and the ``[tool.uv]`` settings
of RLinf's ``pyproject.toml`` do not apply.
CUDA extensions are compiled for Ampere and Hopper GPUs by default
(``TORCH_CUDA_ARCH_LIST=8.0;9.0``, ``FLASH_ATTN_CUDA_ARCHS=80;90``); set both
variables before installing for other architectures.

The installer automatically clones pinned LingBot-VLA V2, RoboTwin, and
LeRobot sources inside the virtual environment and installs RLinf from
the current checkout. A separate sibling LingBot checkout is not required.

To use Docker instead, build the ``embodied-robotwin-lingbotvla-v2`` target.
The image runs the same installer into ``/opt/venv/lingbotvla-v2`` and
activates it in the shell, so skip ``source .venv/bin/activate`` below:

.. code-block:: bash

    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
      --build-arg BUILD_TARGET=embodied-robotwin-lingbotvla-v2 \
      -t rlinf:embodied-robotwin-lingbotvla-v2 .

V2 does not ship dedicated launch scripts; it reuses the generic
``run_vla_sft.sh`` and ``run_embodiment.sh`` launchers. Activate the
installed ``.venv`` and export the paths below before launching. The
defaults assume ``resource_root`` is two directories above the repository;
override the variables for custom locations.

.. code-block:: bash

    source .venv/bin/activate
    resource_root="${resource_root:-$(realpath ../..)}"
    export LINGBOT_VLA_V2_PATH="${VIRTUAL_ENV}/lingbot-vla-v2"
    export ROBOTWIN_PATH="${VIRTUAL_ENV}/RoboTwin"
    export ROBOTWIN_ASSETS_PATH="${ROBOTWIN_PATH}"
    export LINGBOT_VLA_V2_CHECKPOINT="${resource_root}/weights/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"
    export QWEN3VL_PATH="${resource_root}/weights/Qwen3-VL-4B-Instruct"
    export LINGBOT_VLA_SFT_DATASET="${resource_root}/datasets/robotwin2_lerobot/click_bell_clean_50"
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    export TMPDIR="${PWD}/logs/lingbotvla_v2/tmp"
    export XDG_CACHE_HOME="${PWD}/logs/lingbotvla_v2/cache"
    export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
    export RAY_TMPDIR="/dev/shm/rlinf-lbv2-${UID}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false

The installer applies the compatibility patches required by LingBot-VLA V2 and RoboTwin.

.. warning::

   Git certificate verification is enabled by default. If a proxy breaks
   HTTPS validation, configure a trusted CA. Setting
   ``LINGBOT_VLA_V2_GIT_SSL_VERIFY=false`` disables verification for the
   LingBot-VLA V2, RoboTwin, and LeRobot checkouts only; use it only on a
   trusted network.

Prepare Models and Data
-----------------------

Installation does not download model weights, a training dataset, or
RoboTwin simulation assets. Prepare compatible resources once, then export
the paths from the block above. By default, ``resource_root`` is two
directories above the repository, with these paths:

.. code-block:: text

    weights/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt
    weights/Qwen3-VL-4B-Instruct
    datasets/robotwin2_lerobot/click_bell_clean_50

The V2 RoboTwin checkpoint and the Qwen3-VL backbone come from Hugging Face.
The checkpoint repository already uses the ``checkpoints/global_step_50000/hf_ckpt``
layout above:

.. code-block:: bash

    hf download robbyant/lingbot-vla-v2-6b-robotwin \
      --local-dir "${resource_root}/weights/lingbot-vla-v2-6b-robotwin"
    hf download Qwen/Qwen3-VL-4B-Instruct \
      --local-dir "${resource_root}/weights/Qwen3-VL-4B-Instruct"

Shard sizes do not reveal a truncated or zero-filled download, and such a
checkpoint still loads. Compare the SHA-256 of each ``*.safetensors`` file with
the value listed on its Hugging Face file page before training.

The SFT dataset is the official RoboTwin 2.0 Click Bell demonstration set
``click_bell/aloha-agilex_clean_50`` (50 expert episodes in the clean scene),
converted to LeRobot format. Download and extract the release from the
``TianxingChen/RoboTwin2.0`` Hugging Face dataset, then convert it with
``toolkits/lerobot/convert_robotwin_to_lerobot.py``:

.. code-block:: bash

    raw_root="${resource_root}/datasets/robotwin2_official"
    hf download TianxingChen/RoboTwin2.0 dataset/click_bell/aloha-agilex_clean_50.zip \
      --repo-type dataset --local-dir "${raw_root}"
    unzip -q "${raw_root}/dataset/click_bell/aloha-agilex_clean_50.zip" \
      -d "${raw_root}/click_bell"
    python toolkits/lerobot/convert_robotwin_to_lerobot.py \
      --raw-root "${raw_root}/click_bell/aloha-agilex_clean_50" \
      --output-root "${LINGBOT_VLA_SFT_DATASET}"

The converter pairs the joint state and the three camera images at each
frame with the joint state of the next frame as the action, which is the
RoboTwin control contract used by the evaluation environment. The result has
50 episodes and 3,855 frames at 50 FPS. Pass several ``--raw-root``
directories to merge releases, for example ``aloha-agilex_randomized_500``.

RoboTwin assets default to ``${ROBOTWIN_PATH}/assets`` inside the installed
environment; ``ROBOTWIN_ASSETS_PATH`` denotes their parent directory, not
the ``assets`` directory itself. For another layout, override
``LINGBOT_VLA_V2_CHECKPOINT``, ``QWEN3VL_PATH``,
``LINGBOT_VLA_SFT_DATASET``, and ``ROBOTWIN_ASSETS_PATH``.
Verify the resources exist before launching; they are never downloaded
implicitly.

``LINGBOT_VLA_V2_CHECKPOINT`` must contain V2 model weights;
``QWEN3VL_PATH`` must provide the matching backbone configuration and
tokenizer. SFT accepts one LeRobot dataset path or an upstream multi-dataset
manifest. GRPO requires RoboTwin assets; see the
:doc:`RoboTwin guide <robotwin>` for asset preparation.

The model configs resolve the robot configuration and normalization
statistics under ``LINGBOT_VLA_V2_PATH``:

* ``configs/robot_configs/robotwin.yaml``
* ``assets/norm_stats/robotwin.json``

Use statistics and robot settings appropriate for your data. Override
``actor.model.lingbotvla_v2.stats_path`` and the corresponding robot
configuration fields when using a different dataset or robot.

Run SFT
-------

The default SFT config uses one node with eight GPUs, BF16, FSDP full
sharding, a micro batch size of 1, and a global batch size of 8. RLinf
launches the distributed workers through Ray; do not wrap the launcher
in ``torchrun``. After activating the environment and exporting the
paths from the installation section, launch with the generic SFT script:

.. code-block:: bash

    bash examples/sft/run_vla_sft.sh robotwin_sft_lingbotvla_v2

The config is
``examples/sft/config/robotwin_sft_lingbotvla_v2.yaml``. Like 1.0, it defaults
to 30,000 optimizer steps, a 30,000-step learning-rate schedule, 1,000 warmup
steps, and checkpoint saving every 1,000 steps. The peak learning rate is
``1.0e-5`` with cosine decay to ``1.0e-6``. The launcher writes under a
timestamped ``logs/<timestamp>-robotwin_sft_lingbotvla_v2`` directory and does
not forward extra Hydra overrides. For a short execution check, call the entry
script directly with the variables the launcher sets:

.. code-block:: bash

    REPO_PATH="${PWD}" EMBODIED_PATH="${PWD}/examples/sft" \
      python examples/sft/train_vla_sft.py \
      --config-path "${PWD}/examples/sft/config" \
      --config-name robotwin_sft_lingbotvla_v2 \
      runner.logger.log_path=logs/lingbotvla_v2/sft_check \
      runner.max_steps=3 runner.save_interval=3

Run GRPO
--------

When no Vulkan ICD override is set and ``libGLX_nvidia.so.0`` cannot load,
export the installed Mesa/llvmpipe ICD for real CPU raster rendering with
``camera_shader: default``. The launcher does not set this for you, so add it
to the exports above when the NVIDIA Vulkan ICD is unavailable:

.. code-block:: bash

    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
    export LP_NUM_THREADS=2

Explicit ``VK_ICD_FILENAMES`` / ``VK_DRIVER_FILES`` settings take precedence.
The fallback renders real observations on the CPU; it does not generate dummy
observations.

The Click Bell config uses eight GPUs on one node: GPU 0 hosts the actor,
and GPUs 1–7 host rollout and environment workers. Unlike SFT, the actor
uses ``no_shard`` and the default trainable scope is ``action_expert``.
The robot embodiment is selected in the RoboTwin YAML. RoboTwin uses
SAPIEN/Vulkan, not MuJoCo, so ``MUJOCO_GL`` and ``PYOPENGL_PLATFORM`` are
not required. Pass ``ALOHA`` as the second argument to the launcher so
``ROBOT_PLATFORM`` is set correctly:

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh robotwin_click_bell_grpo_lingbotvla_v2 ALOHA

GRPO starts from ``LINGBOT_VLA_V2_CHECKPOINT``, which defaults to the official
RoboTwin checkpoint. To continue from an SFT run instead, point it at a saved
step directory; the rollout model reads the same path as the actor, and the
loader picks up ``actor/model_state_dict/full_weights.pt`` inside it:

.. code-block:: bash

    export LINGBOT_VLA_V2_CHECKPOINT=logs/<timestamp>-robotwin_sft_lingbotvla_v2/checkpoints/global_step_1000

The config is
``examples/embodiment/config/robotwin_click_bell_grpo_lingbotvla_v2.yaml``.
Like 1.0, the defaults are 1,000 epochs with no additional step cap,
evaluation and checkpoint saving every 20 iterations, and 400 steps per
episode and rollout epoch for both training and evaluation. V2 retains
14 training environments, group size 2, and its verified GPU placement.
The launcher writes under a timestamped
``logs/<timestamp>-robotwin_click_bell_grpo_lingbotvla_v2`` directory. As with
SFT, a bounded execution check (not the full default training run) calls the
entry script directly:

.. code-block:: bash

    ROBOT_PLATFORM=ALOHA REPO_PATH="${PWD}" EMBODIED_PATH="${PWD}/examples/embodiment" \
      python examples/embodiment/train_embodied_agent.py \
      --config-path "${PWD}/examples/embodiment/config" \
      --config-name robotwin_click_bell_grpo_lingbotvla_v2 \
      runner.logger.log_path=logs/lingbotvla_v2/grpo_check \
      runner.max_steps=2 runner.save_interval=1 runner.val_check_interval=1 \
      env.train.max_episode_steps=50 env.train.max_steps_per_rollout_epoch=50 \
      env.eval.max_episode_steps=50 env.eval.max_steps_per_rollout_epoch=50

The exports above already point temporary files and compilation
caches at ``logs/lingbotvla_v2`` and use a short Ray temporary path under
``/dev/shm``. ``CUDA_HOME`` defaults to ``/usr/local/cuda``; adjust it for
another CUDA toolkit location. System GPU/Vulkan drivers are still
prerequisites, not Python packages the installer provides.

Model/data paths and ``EMBODIED_PATH`` are required by the example configs;
GRPO also needs the asset path and ``REPO_PATH`` for its seed files.
Cache, temporary-directory and CPU-thread settings are operational safeguards,
not model requirements. Inductor uses ``TMPDIR`` by default, and Matplotlib's
cache uses ``XDG_CACHE_HOME``, so separate ``TORCHINDUCTOR_CACHE_DIR`` and
``MPLCONFIGDIR`` exports are unnecessary. Triton does not follow either
variable, so ``TRITON_CACHE_DIR`` keeps its cache out of the home directory.

Validation Scope
----------------

A successful short run checks execution, not convergence. Inspect rewards,
advantages, gradients, and parameter updates before treating a GRPO run as
effective learning. All-zero rewards within groups can yield zero
advantages and no useful policy update.

For reference, the following Click Bell results were measured with this
integration:

* The official RoboTwin checkpoint, evaluated without training, succeeds in
  90.6% of 64 episodes.
* SFT from that checkpoint on the ``clean_50`` demonstrations reaches 76.6%
  of 64 episodes after 1,000 steps. This run used a peak learning rate of
  ``1.0e-4``; the default ``1.0e-5`` has not been evaluated.
* A five-step GRPO run from the official checkpoint keeps the rollout/actor
  probability ratio near 1 and the KL below ``1.2e-3``, and its step-5
  evaluation succeeds in 7 of 8 episodes. This confirms that the loop runs
  correctly; it does not measure convergence.

Complete simulation assets, including
``assets/objects/objaverse/list.json``, are required even when clutter is
disabled.

The upstream packages are installed with ``--no-deps`` under RLinf's
runtime constraints. Their original dependency declarations still conflict
with this stack (including older torch pins in LingBot, LeRobot, and MDM),
so ``pip check`` is not clean. Successful installation and runtime smoke
tests do not imply that all upstream optional features are compatible.
