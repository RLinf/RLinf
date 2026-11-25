LIBERO 评测
===========

概览
----

LIBERO 是基于 robosuite（MuJoCo）的机器人操作仿真基准，涵盖 Spatial、Object、Goal 与 Long（``libero_10``）套件以及 LIBERO-PRO、LIBERO-PLUS 扰动变体。RLinf 支持在 LIBERO 上并行评测 VLA policy，并输出 ``eval/success_once`` 等任务级指标。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 评测基准
      :text-align: center

      LIBERO（Spatial · Object · Goal · Long · Mem · Pro · Plus）

   .. grid-item-card:: 支持模型
      :text-align: center

      OpenPI · OpenVLA-OFT · FastWAM · MolmoAct2 · StarVLA · Evo-1 · π₀-FAST · DreamZero · Cosmos3

   .. grid-item-card:: 配置目录
      :text-align: center

      ``evaluations/libero/``

   .. grid-item-card:: 关联训练
      :text-align: center

      :doc:`LIBERO <../../examples/embodied/libero>` · :doc:`FastWAM <../../examples/embodied/sft_fastwam>` · :doc:`DreamZero <../../examples/embodied/sft_dreamzero>`

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: OpenPI π₀.₅

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate

      **2. 下载模型**

      .. code-block:: bash

         hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

      **3. 启动评测（LIBERO-Spatial）**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
           rollout.model.model_path=./RLinf-Pi05-LIBERO-SFT

      评测其他套件时，可将配置名替换为 ``libero_object_openpi_pi05_eval``、``libero_goal_openpi_eval``\ （π₀）或 ``libero_10_openpi_pi05_eval``。

   .. tab-item:: OpenVLA-OFT

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env libero && source .venv/bin/activate

      **2. 下载模型**

      .. code-block:: bash

         hf download RLinf/Openvla-oft-SFT-libero10-trajall --local-dir ./Openvla-oft-SFT-libero10-trajall

      **3. 启动评测（LIBERO-10）**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval \
           rollout.model.model_path=./Openvla-oft-SFT-libero10-trajall

      评测其他套件时，可将配置名替换为 ``libero_object_openvlaoft_eval`` 或 ``libero_goal_openvlaoft_eval``。

   .. tab-item:: FastWAM

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model fastwam --env libero && source .venv/bin/activate

      **2. 下载模型与归一化统计**

      .. code-block:: bash

         hf download yuanty/fastwam libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json --local-dir ./fastwam

      **3. 启动评测（LIBERO-Spatial）**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval \
           rollout.model.model_path=./fastwam/libero_uncond_2cam224.pt \
           rollout.model.dataset_stats_path=./fastwam/libero_uncond_2cam224_dataset_stats.json

      默认使用 4 卡（20 槽位 × 25 回合 = 500 个初始状态）；可替换为 ``libero_object_fastwam_eval``、``libero_goal_fastwam_eval`` 或 ``libero_10_fastwam_eval``。

   .. tab-item:: MolmoAct2

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model molmoact2 --env libero && source .venv/bin/activate

      **2. 下载模型**

      .. code-block:: bash

         hf download allenai/MolmoAct2-LIBERO --local-dir ./MolmoAct2-LIBERO

      **3. 启动评测（LIBERO-10）**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_10_molmoact2_eval \
           rollout.model.model_path=./MolmoAct2-LIBERO

      评测其他套件时，可将配置名替换为 ``libero_spatial_molmoact2_eval``、``libero_object_molmoact2_eval`` 或 ``libero_goal_molmoact2_eval``\ （详见 :doc:`../../examples/embodied/molmoact2`）。

   .. tab-item:: StarVLA

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model starvla --env libero && source .venv/bin/activate

      **2. 下载模型与基座 VLM**

      .. code-block:: bash

         hf download StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1 --local-dir ./Qwen2.5-VL-OFT-LIBERO-4in1
         hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct

      将 ``Qwen2.5-VL-OFT-LIBERO-4in1/config.yaml`` 中的 ``framework.qwenvl.base_vlm`` 改为 ``./Qwen2.5-VL-3B-Instruct``\ （详见 :doc:`../../examples/embodied/starvla`）。

      **3. 启动评测（LIBERO-Spatial）**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_spatial_starvla_eval \
           rollout.model.model_path=./Qwen2.5-VL-OFT-LIBERO-4in1

   .. tab-item:: π₀-FAST

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         UV_TORCH_BACKEND=cu128 bash requirements/install.sh embodied \
           --model pi0_fast --env libero \
           --python 3.12.12 --torch 2.11.0 --no-flash-attn && source .venv/bin/activate

      **2. 下载模型与 tokenizer**

      .. code-block:: bash

         hf download lerobot/pi0fast-libero \
           --revision 840f4b503f4c09110421c33c810a85b6684fd658 \
           --local-dir ./pi0fast-libero
         hf download google/paligemma-3b-pt-224 \
           --revision 35e4f46485b4d07967e7e9935bc3786aad50687c \
           --local-dir ./paligemma-3b-pt-224
         hf download jadechoghari/tokenizer-lib-mean \
           --revision 79ae83e3cbd8786dcb84b628569f8d076ca8151e \
           --local-dir ./tokenizer-lib-mean

      **3. 启动评测（LIBERO-10）**

      .. code-block:: bash

         bash evaluations/run_eval.sh libero libero_10_pi0_fast_eval \
           rollout.model.model_path=./pi0fast-libero \
           rollout.model.pi0_fast.text_tokenizer_name=./paligemma-3b-pt-224 \
           rollout.model.pi0_fast.action_tokenizer_name=./tokenizer-lib-mean

      下载前须在 Hugging Face 同意 PaliGemma 使用协议并执行 ``hf auth login``。默认配置使用 2 节点 10 卡（``cluster.num_nodes=2``），详见 :doc:`../../examples/embodied/pi0_fast`。

   .. tab-item:: DreamZero

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model dreamzero --env libero && source .venv/bin/activate
         git clone https://github.com/RLinf/dreamzero.git ../dreamzero

      **2. 下载权重、WAN2.2 组件并生成 LIBERO metadata**

      .. code-block:: bash

         hf download RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000 --local-dir ./RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000
         hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
         hf download Wan-AI/Wan2.1-I2V-14B-480P models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --local-dir ./Wan2.1-CLIP
         hf download google/umt5-xxl --local-dir ./umt5-xxl
         hf download physical-intelligence/libero --repo-type dataset --local-dir ./libero
         python toolkits/lerobot/generate_dreamzero_metadata.py \
           --preset libero_sim \
           --dataset-root ./libero \
           --output-metadata ./libero_metadata.json

      在 ``evaluations/libero/libero_spatial_dreamzero_eval.yaml`` 的 ``rollout.model`` 下填入上述本地目录路径（``tokenizer_path``、``diffusion_model_pretrained_path``、``image_encoder_pretrained_path``、``text_encoder_pretrained_path`` 与 ``vae_pretrained_path``）。

      **3. 启动评测（LIBERO-Spatial）**

      .. code-block:: bash

         export DREAMZERO_PATH="$(realpath ../dreamzero)"
         bash evaluations/run_eval.sh libero libero_spatial_dreamzero_eval \
           runner.ckpt_path=./RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step18000/model.pt \
           rollout.model.metadata_json_path=./libero_metadata.json

      将 ``runner.ckpt_path`` 指向实际的 ``.pt`` 权重文件（如 SFT 导出的 ``full_weights.pt``）。如需以 SGLang 后端评测 Diffusers 格式权重，见 :doc:`dreamzero_sglang`。

   .. tab-item:: ApxInf

      **1. 安装 RLinf 并编译 APXinf-robo CUDA 绑定**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero && source .venv/bin/activate
         git clone --recursive https://github.com/RLinf/APXinf-robo.git ../APXinf-robo
         (cd ../APXinf-robo && pip install maturin && \
           CARGO_TARGET_DIR=target/wheel maturin build --release --features cuda \
             --auditwheel skip -m apxinf/crates/apxinf-py/Cargo.toml && \
           pip install --force-reinstall target/wheel/wheels/apxinf_py-*.whl && \
           pip install -e ./apxinf/python/apxinf --config-settings editable_mode=strict && \
           pip install -e ".[rlinf]")

      **2. 下载 π₀.₅ 权重与归一化统计**

      .. code-block:: bash

         hf download lerobot/pi05_libero_base --local-dir ./pi05_libero_base
         curl -fL https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
           -o ./pi05_libero_base/norm_stats.json

      **3. 启动评测（LIBERO-10）**

      .. code-block:: bash

         export APXINF_PI05_MODEL_DIR="$(pwd)/pi05_libero_base"
         bash evaluations/run_eval.sh libero libero_10_apxinf_pi05_eval

      需要 Linux + NVIDIA GPU、CUDA 工具链与 Rust 工具链，详见 :ref:`ApxInf 推理后端 <apxinf-backend>` 与 :doc:`../../guides/apxinf`。

.. note::

   **主机内存与完整套件覆盖：** 单个 LIBERO 环境约占 2 GiB 主机内存，默认 ``total_num_envs=500`` 约需 1 TiB 内存。常规机器上请同步调小 ``env.eval.total_num_envs`` 并调大 ``env.eval.max_steps_per_rollout_epoch``\ （例如 ``libero_spatial`` 设 ``env.eval.total_num_envs=128 env.eval.max_steps_per_rollout_epoch=960``），通过 ``auto_reset`` 完整覆盖 500 个初始状态（详见 :ref:`评测配置详解 <libero-eval-config>`）。

使用 ``--env libero`` 时，``install.sh`` 会将 LIBERO 克隆到 ``.venv/libero``\ （若已设置 ``LIBERO_PATH`` 则复用已有目录）并写入 ``.venv/bin/activate`` 的 ``PYTHONPATH``。SGLang 后端见 :doc:`dreamzero_sglang` 与 :doc:`cosmos3_sglang`；如需与动作块执行重叠以隐藏推理延迟，见 :doc:`RTC <../../guides/rtc>`。

.. dropdown:: 示例配置

   ``evaluations/libero/`` 已有示例配置：

   .. list-table::
      :header-rows: 1
      :widths: 38 28 34

      * - 配置文件
        - 任务套件
        - 模型
      * - ``libero_spatial_openpi_pi05_eval.yaml``
        - Spatial
        - π₀.₅
      * - ``libero_spatial_eval_pi05_RTC.yaml``
        - Spatial
        - π₀.₅（RTC）
      * - ``libero_spatial_evo1_eval.yaml``
        - Spatial
        - Evo-1
      * - ``libero_spatial_starvla_eval.yaml``
        - Spatial
        - StarVLA
      * - ``libero_spatial_dreamzero_eval.yaml``
        - Spatial
        - DreamZero
      * - ``libero_spatial_dreamzero_eval_sglang.yaml``
        - Spatial
        - DreamZero（SGLang 后端）
      * - ``libero_spatial_cosmos3_eval_sglang.yaml``
        - Spatial
        - Cosmos3（SGLang 后端）
      * - ``libero_spatial_fastwam_eval.yaml``
        - Spatial
        - FastWAM
      * - ``libero_spatial_molmoact2_eval.yaml``
        - Spatial
        - MolmoAct2
      * - ``libero_object_openpi_pi05_eval.yaml``
        - Object
        - π₀.₅
      * - ``libero_object_openvlaoft_eval.yaml``
        - Object
        - OpenVLA-OFT
      * - ``libero_object_fastwam_eval.yaml``
        - Object
        - FastWAM
      * - ``libero_object_molmoact2_eval.yaml``
        - Object
        - MolmoAct2
      * - ``libero_goal_openpi_eval.yaml``
        - Goal
        - π₀
      * - ``libero_goal_openvlaoft_eval.yaml``
        - Goal
        - OpenVLA-OFT
      * - ``libero_goal_fastwam_eval.yaml``
        - Goal
        - FastWAM
      * - ``libero_goal_molmoact2_eval.yaml``
        - Goal
        - MolmoAct2
      * - ``libero_10_openpi_pi05_eval.yaml``
        - Long（``libero_10``）
        - π₀.₅
      * - ``libero_10_apxinf_pi05_eval.yaml``
        - Long（``libero_10``）
        - π₀.₅（ApxInf 后端）
      * - ``libero_10_openvlaoft_eval.yaml``
        - Long（``libero_10``）
        - OpenVLA-OFT
      * - ``libero_10_fastwam_eval.yaml``
        - Long（``libero_10``）
        - FastWAM
      * - ``libero_10_molmoact2_eval.yaml``
        - Long（``libero_10``）
        - MolmoAct2
      * - ``libero_10_pi0_fast_eval.yaml``
        - Long（``libero_10``）
        - π₀-FAST
      * - ``libero_mem_openpi_pi05_eval.yaml``
        - LIBERO-Mem（``libero_mem``）
        - π₀.₅（Short Memory）

.. _apxinf-backend:

.. dropdown:: ApxInf 推理后端

   ApxInf 路径仅用于评测。RLinf 通过 `APXinf-robo <https://github.com/RLinf/APXinf-robo>`_ 调用 `ApxInf <https://github.com/infinigence/ApxInf>`_ 的 L1 接口。将 ``APXINF_PI05_MODEL_DIR`` 指向含 ``config.json``、``model.safetensors``、``tokenizer.model`` 与 ``norm_stats.json`` 的目录：

   .. code-block:: bash

      export APXINF_PI05_MODEL_DIR="$(pwd)/pi05_libero_base"
      bash evaluations/run_eval.sh libero libero_10_apxinf_pi05_eval

   该配置保持原生 OpenPI 语义：两路方向已处理的 LIBERO 图像、8 维 state、π₀.₅ 无 state prompt、5 个 flow step、预测 10 步并执行前 5 步。做数值对齐测试时可设 ``rollout.model.apxinf.noise_source=observation`` 并传入显式 ``[B,10,32]`` 噪声张量。默认 10 个环境 × 10 轮评测 LIBERO-10 每任务 10 个初始状态（共 100 条轨迹）。完整编译与后端说明见 :doc:`../../guides/apxinf`。

.. _libero-eval-config:

.. dropdown:: 评测配置详解

   LIBERO 评测对每个 ``(task_id, trial_id)`` 组合运行一条轨迹，并汇总 ``eval/success_once``。`官方仓库 <https://github.com/Lifelong-Robot-Learning/LIBERO>`_ 的四个标准套件（``libero_spatial``、``libero_object``、``libero_goal``、``libero_10``）各含 10 个任务、每任务约 50 个初始状态（共约 500 条轨迹）。

   **评测协议**

   各任务的 trial 依次拼接为全局 ``reset_state_id``，评测模式（``is_eval=True``）下按交错顺序遍历——``(task0, trial0), (task1, trial0), …, (task0, trial1), …``——使并行环境在各任务间均匀推进。在一轮 ``rollout_epoch`` 内覆盖全部组合可用大并行（``total_num_envs >= S``）或自动重置（``auto_reset=True`` 配合 ``max_steps_per_rollout_epoch = N * max_episode_steps``）。

   **各套件步数上限参考**

   各套件最长训练 demo 决定 ``max_episode_steps`` 参考下限，评测配置须不低于下限且与训练一致：

   .. list-table::
      :header-rows: 1
      :widths: 22 20 24 34

      * - 套件
        - 参考下限
        - RLinf 示例值
        - 示例配置
      * - ``libero_spatial``
        - 220
        - 240 / 480
        - ``libero_spatial_openpi_pi05_eval`` / ``libero_spatial_dreamzero_eval``
      * - ``libero_object``
        - 280
        - 280 / 512
        - ``libero_object_openpi_pi05_eval`` / ``libero_object_openvlaoft_eval``
      * - ``libero_goal``
        - 300
        - 320 / 512
        - ``libero_goal_openpi_eval`` / ``libero_goal_openvlaoft_eval``
      * - ``libero_10``
        - 520
        - 520
        - ``libero_10_openpi_pi05_eval`` / ``libero_10_openvlaoft_eval``

   通用 ``env.eval`` 字段说明见 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`。

   **覆盖完整测试集**

   设套件总初始状态数为 ``S``，并行环境数为 ``E``，单轨迹步数上限为 ``T``\ （``max_episode_steps``）。

   *方式一：大并行*

   .. code-block:: yaml

      env:
        eval:
          total_num_envs: 500                # S，适用于 Spatial / Object / Goal / Long
          max_episode_steps: 240
          max_steps_per_rollout_epoch: 240   # 等于 max_episode_steps
          auto_reset: True                   # E >= S 时可选
          rollout_epoch: 1

   每个 LIBERO 环境运行在独立进程中，reset 后约占 2 GiB 非共享主机内存（``total_num_envs: 500`` 合计约需 1 TiB）。

   .. warning::

      环境进程是 env worker 的子进程，不属于 Ray worker。主机内存耗尽时，Ray 内存监控可能无法选中可终止的 worker，导致节点失去响应。

   *方式二：自动重置（除非每个环境能分配超过 2 GiB 主机内存并留出 rollout 显存余量，否则推荐此方式）*

   .. code-block:: yaml

      env:
        eval:
          total_num_envs: 128
          max_episode_steps: 480
          # N = ceil(S / E)；Spatial: ceil(500/128) = 4
          max_steps_per_rollout_epoch: 1920  # N * max_episode_steps = 4 * 480
          auto_reset: True
          ignore_terminations: True
          use_fixed_reset_state_ids: True
          use_ordered_reset_state_ids: True
          rollout_epoch: 1

   设置 ``env.eval.rollout_epoch: 2`` 可多轮取平均。``max_steps_per_rollout_epoch`` 须能被 ``rollout.model.num_action_chunks`` 整除。多个 env worker 的种子带偏移 ``seed + rank × stage_num + stage_id``，保证分配到不同初始状态子集；开启 ``auto_reset`` 时 ``eval/success_once`` 仅在回合完成时计入一次。

.. dropdown:: 进阶用法（LIBERO-PRO、LIBERO-PLUS 与并行规模）

   LIBERO-PRO 与 LIBERO-PLUS 环境安装见 :ref:`LIBERO-Pro 与 LIBERO-Plus <zh-liberopro-plus-benchmark>`。

   **LIBERO-PRO**

   .. code-block:: bash

      export LIBERO_TYPE=pro
      export LIBERO_PERTURBATION=all
      bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval \
        rollout.model.model_path=./Openvla-oft-SFT-libero10-trajall

   **LIBERO-PLUS**

   .. code-block:: bash

      export LIBERO_TYPE=plus
      export LIBERO_SUFFIX=all
      bash evaluations/run_eval.sh libero libero_10_openvlaoft_eval \
        rollout.model.model_path=./Openvla-oft-SFT-libero10-trajall

   **调整并行规模**

   只调小 ``total_num_envs`` 会缩小覆盖范围（64 个环境配合 240 步预算仅覆盖 500 个初始状态中的 64 个），须同步调大 ``max_steps_per_rollout_epoch``。``total_num_envs`` 须能被 env worker 数整除（``128`` 兼容 1、2、4、8 卡）：

   .. code-block:: bash

      # max_steps_per_rollout_epoch = ceil(500 / 128) * 240 = 960
      bash evaluations/run_eval.sh libero libero_spatial_openpi_pi05_eval \
        env.eval.total_num_envs=128 \
        env.eval.max_steps_per_rollout_epoch=960 \
        rollout.model.model_path=./RLinf-Pi05-LIBERO-SFT

结果与常见问题
--------------

终端输出 ``eval/success_once``，日志与视频保存在 ``logs/<timestamp>-<config>/``\ （详见 :doc:`../reference/results` 与 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`）。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理
   * - 无头服务器 EGL 渲染报错
     - ``run_eval.sh`` 默认使用 EGL 渲染；机器不支持 EGL 时执行 ``export MUJOCO_GL=osmesa`` 与 ``export PYOPENGL_PLATFORM=osmesa``。
   * - 测试集覆盖不完整
     - 参考 :ref:`评测配置详解 <libero-eval-config>`，协调设置 ``total_num_envs``、``auto_reset`` 与 ``max_steps_per_rollout_epoch``。
   * - 500 并行环境导致主机内存耗尽
     - 减小 ``total_num_envs``\ （如 ``128``）并启用 ``auto_reset: True`` 与更大的 ``max_steps_per_rollout_epoch``；单个 LIBERO 环境约占 2 GiB 主机内存。
   * - 步数校验失败
     - 将 ``max_steps_per_rollout_epoch`` 设为 ``rollout.model.num_action_chunks`` 的整数倍。
