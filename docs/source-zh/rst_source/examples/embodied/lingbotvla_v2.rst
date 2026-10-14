LingBot-VLA 2.0 的 SFT 与 GRPO 训练
============================================

RLinf 提供 LingBot-VLA 2.0 的 LeRobot 格式数据 SFT 和 RoboTwin GRPO 训练集成。请使用 ``lingbotvla_v2`` 模型类型和 V2 配置；:doc:`Lingbot-VLA 1.0 <lingbotvla>` 的 checkpoint 和配置不能与本示例混用。

安装环境
----------------

在 RLinf 仓库根目录执行以下命令。安装脚本需要 Python 3.12、Git、NVIDIA CUDA toolkit，以及可下载依赖的网络。脚本会编译 FlashAttention 等 CUDA 扩展。

.. code-block:: bash

    PYTHON=python3.12 bash requirements/install.sh embodied \
      --model lingbotvla_v2 --env robotwin

目标目录不能已存在。安装失败后重试时，请使用新的环境目录；不会修改已有环境。依赖包通过 ``uv`` 安装，缺少 ``uv`` 时脚本会先安装它。uv 会在缓存中保留由固定版本源码构建的 wheel，因此同一台机器上的第二次安装不再编译 CUDA 扩展。该模型使用固定依赖版本，不支持版本覆盖参数、``--use-mirror`` 和 ``--no-flash-attn``，RLinf ``pyproject.toml`` 中的 ``[tool.uv]`` 设置也不生效。CUDA 扩展默认面向 Ampere 和 Hopper GPU 编译（``TORCH_CUDA_ARCH_LIST=8.0;9.0``、``FLASH_ATTN_CUDA_ARCHS=80;90``），其他架构请在安装前同时设置这两个变量。

脚本会自动将固定版本的 LingBot-VLA V2、RoboTwin 和 LeRobot 源码下载到虚拟环境中，并从当前仓库安装 RLinf，无需手动准备同级 LingBot 仓库。

如需使用 Docker，请构建 ``embodied-robotwin-lingbotvla-v2`` 目标。镜像使用同一安装脚本将环境安装到 ``/opt/venv/lingbotvla-v2``，并在 shell 中自动激活，因此可跳过下方的 ``source .venv/bin/activate``：

.. code-block:: bash

    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
      --build-arg BUILD_TARGET=embodied-robotwin-lingbotvla-v2 \
      -t rlinf:embodied-robotwin-lingbotvla-v2 .

V2 不提供独立启动脚本，复用通用的 ``run_vla_sft.sh`` 和 ``run_embodiment.sh``。启动前请激活安装好的 ``.venv`` 并 export 以下路径。默认 ``resource_root`` 为仓库的上两级目录；自定义位置可通过同名环境变量覆盖。

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

安装脚本会自动应用 LingBot-VLA V2 和 RoboTwin 所需的兼容性修复。

.. warning::

   Git 证书验证默认开启。代理导致 HTTPS 验证失败时，应配置受信任的 CA。设置 ``LINGBOT_VLA_V2_GIT_SSL_VERIFY=false`` 只会关闭 LingBot-VLA V2、RoboTwin 和 LeRobot 源码下载时的验证，仅可在受信任的网络中使用。

准备模型和数据
----------------------------

安装脚本不会下载模型权重、训练数据集或 RoboTwin 仿真资产。兼容资源只需准备一次，准备完成后按上方代码块 export 对应路径。``resource_root`` 默认是仓库的上两级目录，其下的资源布局为：

.. code-block:: text

    weights/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt
    weights/Qwen3-VL-4B-Instruct
    datasets/robotwin2_lerobot/click_bell_clean_50

V2 RoboTwin checkpoint 和 Qwen3-VL 骨干模型均从 Hugging Face 下载，checkpoint 仓库本身已是上方的 ``checkpoints/global_step_50000/hf_ckpt`` 布局：

.. code-block:: bash

    hf download robbyant/lingbot-vla-v2-6b-robotwin \
      --local-dir "${resource_root}/weights/lingbot-vla-v2-6b-robotwin"
    hf download Qwen/Qwen3-VL-4B-Instruct \
      --local-dir "${resource_root}/weights/Qwen3-VL-4B-Instruct"

下载不完整或含零填充块的分片大小与正常文件相同，并且仍能正常加载。训练前请将每个 ``*.safetensors`` 文件的 SHA-256 与 Hugging Face 文件页面上列出的值逐一比对。

SFT 数据集来自 RoboTwin 2.0 官方 Click Bell 示教数据 ``click_bell/aloha-agilex_clean_50``（clean 场景下的 50 条 expert episode），需转换为 LeRobot 格式。先从 Hugging Face 数据集 ``TianxingChen/RoboTwin2.0`` 下载并解压，再用 ``toolkits/lerobot/convert_robotwin_to_lerobot.py`` 转换：

.. code-block:: bash

    raw_root="${resource_root}/datasets/robotwin2_official"
    hf download TianxingChen/RoboTwin2.0 dataset/click_bell/aloha-agilex_clean_50.zip \
      --repo-type dataset --local-dir "${raw_root}"
    unzip -q "${raw_root}/dataset/click_bell/aloha-agilex_clean_50.zip" \
      -d "${raw_root}/click_bell"
    python toolkits/lerobot/convert_robotwin_to_lerobot.py \
      --raw-root "${raw_root}/click_bell/aloha-agilex_clean_50" \
      --output-root "${LINGBOT_VLA_SFT_DATASET}"

转换脚本把每一帧的关节状态和三路相机图像作为输入，以下一帧的关节状态作为 action，与评估环境使用的 RoboTwin 控制约定一致。转换结果为 50 个 episode、3,855 帧，帧率 50 FPS。传入多个 ``--raw-root`` 可合并多份数据，例如再加入 ``aloha-agilex_randomized_500``。

RoboTwin 资产默认放在安装环境的 ``${ROBOTWIN_PATH}/assets`` 中。``ROBOTWIN_ASSETS_PATH`` 指向 ``assets`` 的父目录，而不是 ``assets`` 本身。其他目录布局可通过 ``LINGBOT_VLA_V2_CHECKPOINT``、``QWEN3VL_PATH``、``LINGBOT_VLA_SFT_DATASET`` 和 ``ROBOTWIN_ASSETS_PATH`` 环境变量覆盖。启动前请确认资源存在，脚本不会在启动训练时隐式下载。

``LINGBOT_VLA_V2_CHECKPOINT`` 应包含 V2 模型权重，``QWEN3VL_PATH`` 应提供匹配的骨干模型配置与 tokenizer。SFT 接受单个 LeRobot 数据集路径或上游多数据集 manifest。GRPO 需要 RoboTwin 资产，下载方法见 :doc:`RoboTwin 教程 <robotwin>`。

默认模型配置使用 ``LINGBOT_VLA_V2_PATH`` 下的以下文件：

* ``configs/robot_configs/robotwin.yaml``
* ``assets/norm_stats/robotwin.json``

请确保机器人配置和归一化统计与数据一致。更换数据或机器人时，覆盖 ``actor.model.lingbotvla_v2.stats_path`` 及对应机器人配置字段。

运行 SFT
----------------

默认 SFT 配置为单节点 8 GPU、BF16、FSDP full sharding，micro batch size 为 1，global batch size 为 8。RLinf 通过 Ray 启动分布式 worker，不要再用 ``torchrun`` 包装启动脚本。激活环境并按安装小节 export 所需路径后，使用通用 SFT 脚本启动：

.. code-block:: bash

    bash examples/sft/run_vla_sft.sh robotwin_sft_lingbotvla_v2

配置文件为 ``examples/sft/config/robotwin_sft_lingbotvla_v2.yaml``，默认进行 30,000 次优化器更新，学习率调度总步数为 30,000，warmup 为 1,000 步，每 1,000 步保存 checkpoint，与 1.0 对齐。峰值学习率为 ``1.0e-5``，按 cosine 衰减至 ``1.0e-6``。启动脚本将输出写入带时间戳的 ``logs/<timestamp>-robotwin_sft_lingbotvla_v2`` 目录，且不会转发额外的 Hydra 参数。需要短步检查时，请直接调用入口脚本，并设置启动脚本原本提供的变量：

.. code-block:: bash

    REPO_PATH="${PWD}" EMBODIED_PATH="${PWD}/examples/sft" \
      python examples/sft/train_vla_sft.py \
      --config-path "${PWD}/examples/sft/config" \
      --config-name robotwin_sft_lingbotvla_v2 \
      runner.logger.log_path=logs/lingbotvla_v2/sft_check \
      runner.max_steps=3 runner.save_interval=3

运行 GRPO
------------------

当没有显式设置 Vulkan ICD 且无法加载 ``libGLX_nvidia.so.0`` 时，请手动 export 已安装的 Mesa/llvmpipe ICD，配合 ``camera_shader: default`` 进行真实 CPU raster 渲染，不使用假图像。脚本不会自动设置，NVIDIA Vulkan ICD 不可用时请在上方 export 中补充：

.. code-block:: bash

    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
    export LP_NUM_THREADS=2

用户显式设置的 ``VK_ICD_FILENAMES`` / ``VK_DRIVER_FILES`` 优先级更高。该回退方式在 CPU 上渲染真实观测，不会生成假观测。

Click Bell 配置使用单节点 8 GPU：GPU 0 运行 actor，GPU 1–7 运行 rollout 和 environment worker。与 SFT 不同，actor 使用 ``no_shard``，默认可训练范围为 ``action_expert``。机器人类型由 RoboTwin YAML 指定。RoboTwin 使用 SAPIEN/Vulkan，不使用 MuJoCo，因此无需设置 ``MUJOCO_GL`` 或 ``PYOPENGL_PLATFORM``。启动时将 ``ALOHA`` 作为第二个参数传入，以便正确设置 ``ROBOT_PLATFORM``：

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh robotwin_click_bell_grpo_lingbotvla_v2 ALOHA

GRPO 从 ``LINGBOT_VLA_V2_CHECKPOINT`` 初始化，默认是官方 RoboTwin checkpoint。如需从 SFT 结果继续训练，请将其指向某个已保存的 step 目录；rollout 模型与 actor 读取同一路径，加载器会读取其中的 ``actor/model_state_dict/full_weights.pt``：

.. code-block:: bash

    export LINGBOT_VLA_V2_CHECKPOINT=logs/<timestamp>-robotwin_sft_lingbotvla_v2/checkpoints/global_step_1000

配置文件为 ``examples/embodiment/config/robotwin_click_bell_grpo_lingbotvla_v2.yaml``。默认训练 1,000 个 epoch，不额外限制总步数，每 20 轮进行评估和 checkpoint 保存；训练与评估的 episode 和 rollout epoch 均为 400 步，与 1.0 对齐。保留 V2 的 14 个训练环境、group size 2 及已验证的 GPU 卡位。启动脚本将输出写入带时间戳的 ``logs/<timestamp>-robotwin_click_bell_grpo_lingbotvla_v2`` 目录。与 SFT 相同，短步执行检查（不等同于完整默认训练）需直接调用入口脚本：

.. code-block:: bash

    ROBOT_PLATFORM=ALOHA REPO_PATH="${PWD}" EMBODIED_PATH="${PWD}/examples/embodiment" \
      python examples/embodiment/train_embodied_agent.py \
      --config-path "${PWD}/examples/embodiment/config" \
      --config-name robotwin_click_bell_grpo_lingbotvla_v2 \
      runner.logger.log_path=logs/lingbotvla_v2/grpo_check \
      runner.max_steps=2 runner.save_interval=1 runner.val_check_interval=1 \
      env.train.max_episode_steps=50 env.train.max_steps_per_rollout_epoch=50 \
      env.eval.max_episode_steps=50 env.eval.max_steps_per_rollout_epoch=50

上方 export 已将临时文件和编译缓存指向 ``logs/lingbotvla_v2``，Ray 临时目录使用 ``/dev/shm`` 下的短路径。``CUDA_HOME`` 默认是 ``/usr/local/cuda``，CUDA toolkit 位于其他位置时请调整。系统 GPU/Vulkan 驱动仍是运行前提，不属于安装脚本提供的 Python 依赖。

模型、数据路径和 ``EMBODIED_PATH`` 是当前示例配置所需的变量；GRPO 还需要资产路径及用于种子文件的 ``REPO_PATH``。缓存、临时目录和 CPU 线程限制属于运行保障，不是模型硬性依赖。Inductor 默认跟随 ``TMPDIR``，Matplotlib 缓存跟随 ``XDG_CACHE_HOME``，因此无需单独设置 ``TORCHINDUCTOR_CACHE_DIR`` 和 ``MPLCONFIGDIR``。Triton 不跟随上述两个变量，因此通过 ``TRITON_CACHE_DIR`` 让其缓存不写入 home 目录。

验证范围
----------------

短步运行成功只代表执行检查，不代表收敛。判断 GRPO 是否有效学习时，还需要检查 reward、advantage、梯度及参数更新；组内 reward 全零可能导致 advantage 为零，无法产生有效的 policy 更新。

以下 Click Bell 结果使用本集成测得，可作为参考：

* 官方 RoboTwin checkpoint 不经训练直接评估，64 个 episode 成功率为 90.6%。
* 以该 checkpoint 为起点，在 ``clean_50`` 示教数据上 SFT 1,000 步后，64 个 episode 成功率为 76.6%。该次运行的峰值学习率为 ``1.0e-4``，默认的 ``1.0e-5`` 尚未评估。
* 从官方 checkpoint 出发运行 5 步 GRPO，rollout 与 actor 的概率比保持在 1 附近，KL 低于 ``1.2e-3``，第 5 步评估 8 个 episode 中成功 7 个。该结果只说明训练循环运行正确，不衡量收敛。

即使禁用 clutter，也需要完整仿真资产，包括 ``assets/objects/objaverse/list.json``。

上游包通过 ``--no-deps`` 安装，并使用 RLinf 的运行时依赖约束。原始依赖声明仍与当前环境冲突，包括 LingBot、LeRobot 和 MDM 中的旧版 torch 约束，因此 ``pip check`` 未通过。安装成功和运行冒烟测试不代表所有上游可选功能兼容。
