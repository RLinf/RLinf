OpenPI 监督微调
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 40%

   OpenPI π₀ / π₀.₅ 视觉-语言-动作模型。

在 RLinf 中对 π₀ / π₀.₅ 做 **全量 SFT** 或 **LoRA**。policy 统一使用 ``model_type: openpi``，即与 JAX 参考实现对齐的 PyTorch 版本。SFT 通常作为强化学习的冷启动：先模仿示范数据，再在更好的先验上继续做 RL。

在 LIBERO-Mem 上使用视觉历史时，参阅 :doc:`OpenPI Short Memory 评测与训练 </rst_source/examples/embodied/pi0/short_memory>`；双 Franka 真机的 SFT 与部署见 :doc:`dual_franka_openpi_pytorch`。

概览
----------------------------------------

在 LIBERO、真机 Franka、BEHAVIOR-1K、RoboTwin 及自定义 LeRobot 数据集上对 OpenPI π₀ / π₀.₅ 做全量或 LoRA SFT。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀ · π₀.₅

   .. grid-item-card:: 方法
      :text-align: center

      Full SFT · LoRA

   .. grid-item-card:: 数据
      :text-align: center

      LeRobot · BEHAVIOR

   .. grid-item-card:: 硬件
      :text-align: center

      1+ 节点 · GPU

| **你将完成：** 安装依赖 → 准备数据与归一化统计 → ``bash examples/sft/run_vla_sft.sh <启动名>`` → 观察训练损失。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 对应格式的数据集与基座权重。

配置一览
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

实验配置通过 Hydra ``defaults`` 引入不含路径的模型模板，并设置任务选项。启动时直接通过 Hydra 命令行覆盖本地数据集、基座权重和归一化统计路径。

.. list-table::
   :header-rows: 1
   :widths: 18 38 22 22

   * - 任务
     - 实验配置（``examples/sft/config/``）
     - 模型模板
     - 启动名
   * - LIBERO π₀
     - ``libero_sft_openpi.yaml``
     - ``model/pi0``
     - ``libero_sft_openpi``
   * - 真机 Franka bin-relocation
     - ``realworld_bin_relocation_sft_openpi.yaml``
     - ``model/pi0``
     - ``realworld_bin_relocation_sft_openpi``
   * - 自定义 LeRobot
     - ``custom_sft_openpi.yaml``
     - ``model/pi0``
     - ``custom_sft_openpi``
   * - BEHAVIOR-1K π₀.₅
     - ``behavior_sft_openpi_pi05.yaml``
     - ``model/pi0_5``
     - ``behavior_sft_openpi_pi05``
   * - RoboTwin adjust_bottle π₀
     - ``robotwin_adjust_bottle_sft_openpi.yaml``
     - ``model/pi0``
     - ``robotwin_adjust_bottle_sft_openpi``

安装与运行
----------

.. include:: _setup_common.rst

``--model openpi`` 会安装官方 OpenPI 运行环境（transforms 与数据管线），训练网络使用 RLinf 的 ``openpi`` 实现。

.. tab-set::

   .. tab-item:: LIBERO 与真机 Franka
      :sync: libero-franka

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. 准备数据**

      新采集的真机 Franka 数据集需先计算 ``norm_stats.json``\ （见下文“新 LeRobot 数据集的归一化统计”）。

      **3. 启动训练**

      在仓库根目录通过 Hydra CLI 覆盖数据集与基座权重路径启动训练（日志和 checkpoint 写入 ``runner.logger.log_path``，每 ``runner.save_interval`` 步保存到 ``.../checkpoints/global_step_<N>/``）：

      .. code-block:: bash

         # LIBERO
         bash examples/sft/run_vla_sft.sh libero_sft_openpi \
           data.train_data_paths=./libero-data \
           actor.model.model_path=./pi0_base_openpi

         # 真机 Franka
         bash examples/sft/run_vla_sft.sh realworld_bin_relocation_sft_openpi \
           data.train_data_paths=./realworld-franka-bin-relocation \
           actor.model.model_path=./pi0_base_openpi \
           +actor.model.openpi_data.norm_stats_path=./assets/pi0_realworld/realworld_franka_bin_relocation/norm_stats.json

      .. dropdown:: 进阶配置（可选）

         可选的 placement、LoRA 和 OpenPI 覆写（各实验 YAML 已有默认值）：

         .. code-block:: yaml

            cluster:
              num_nodes: 1
              component_placement:
                actor,env,rollout: 0-0

            actor:
              model:
                num_action_chunks: 4          # 环境执行长度，非网络 horizon
                is_lora: False                # LoRA 时改为 True，并设 lora_rank
                openpi:
                  task: sft
                  embed_dtype: bf16
                  config_name: "pi0_libero"   # 真机改为 pi0_realworld

   .. tab-item:: BEHAVIOR-1K
      :sync: behavior

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env behavior && source .venv/bin/activate

      **2. 下载并转换基座权重**

      从 ``gs://openpi-assets/checkpoints/pi05_base`` 下载 π₀.₅ JAX 基座，并转换为 RLinf OpenPI 格式（以 BEHAVIOR 的 32-step horizon 为例；需提前准备好任务的 ``norm_stats.json``，转换器默认处理 π₀.₅，不要加 ``--no-pi05``）：

      .. code-block:: bash

         mkdir -p ./checkpoints/jax
         gcloud storage cp --recursive \
           gs://openpi-assets/checkpoints/pi05_base ./checkpoints/jax/

         python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
           --input-model ./checkpoints/jax/pi05_base \
           --input-norm-stats ./assets/behavior-1k/2025-challenge-demos/norm_stats.json \
           --output-model ./pi05_base_openpi \
           --output-norm-stats ./pi05_base_openpi/task/norm_stats.json \
           --action-dim 32 \
           --action-horizon 32 \
           --max-token-len 200

      确认 ``./pi05_base_openpi`` 包含 ``model.safetensors`` 和 ``config.json``。

      其他 π₀.₅ SFT 配置按各自 horizon 调整 ``--action-horizon``\ （ManiSkill RLT 为 10，Franka RLT 与 dual-Franka 为 20）。官方 OpenPI PyTorch 格式（键名为 ``paligemma_with_expert.*``）即使包含 ``model.safetensors``，仍需使用 ``--mode openpi_pytorch_to_openpi`` 转换。

      **3. 启动训练**

      通过 Hydra CLI 覆盖数据集、转换后的基座权重与归一化统计路径：

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh behavior_sft_openpi_pi05 \
           data.train_data_paths=./2025-challenge-demos \
           data.behavior_dataset_root=./2025-challenge-demos \
           actor.model.model_path=./pi05_base_openpi \
           actor.model.openpi_data.norm_stats_path=./assets/behavior-1k/2025-challenge-demos/norm_stats.json

      .. dropdown:: 进阶配置（可选）

         仓库自带配置已引入 ``model/pi0_5`` 与 ``hybrid_engines/openpi_fsdp``，并设置好 BEHAVIOR 流式加载项：

         .. code-block:: yaml

            data:
              repo_id: "behavior-1k/2025-challenge-demos"
              modalities: ["rgb"]
              num_workers: 8
              tasks: ["turning_on_radio"]
              use_skill: false
              task_subtasks:
                turning_on_radio:
                  - "move to radio"
                  - "pick up radio from coffee table"
                  - "press radio"
                  - "place radio on coffee table"

            actor:
              model:
                openpi:
                  task: sft
                  embed_dtype: bf16

         - ``use_skill: false`` 用主任务文本；``true`` 按窗口从 ``task_subtasks`` 取逐帧技能文本（数据集里折叠后的 orchestrator 不能作为标签）。

         - ``fine_grained_level`` 与 ``tolerance_s`` 控制流式时间对齐。

   .. tab-item:: RoboTwin
      :sync: robotwin

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env robotwin && source .venv/bin/activate

      **2. 下载数据与参考权重**

      下载处理好的 `adjust_bottle <https://huggingface.co/datasets/RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data>`_ 数据集（以及可选的参考 SFT 权重 `RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle <https://huggingface.co/RLinf/RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle>`_）：

      .. code-block:: bash

         hf download RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data --repo-type dataset --local-dir ./robotwin-adjust-bottle-data
         hf download RLinf/RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle --local-dir ./RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle

      **3. 启动训练**

      通过 Hydra CLI 覆盖数据集与基座权重路径（``norm_stats_path`` 自动按 ``actor.model.model_path`` 解析）：

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh robotwin_adjust_bottle_sft_openpi \
           data.train_data_paths=./robotwin-adjust-bottle-data \
           actor.model.model_path=./pi0_base_pytorch_new

      .. dropdown:: 进阶配置（可选）

         使用 14 维 ALOHA 动作与 3 路图像（进模型前按 OpenPI 规则补零至 32 维）。切换任务时修改 ``norm_stats_path`` 中的任务名：

         .. code-block:: yaml

            data:
              num_workers: 4

            actor:
              model:
                num_action_chunks: 50
                action_dim: 14
                openpi:
                  task: sft
                  embed_dtype: bf16
                  config_name: "pi0_aloha_robotwin"
                  num_images_in_input: 3
                openpi_data:
                  norm_stats_path: ${actor.model.model_path}/physical-intelligence/robotwin/adjust_bottle/norm_stats.json

   .. tab-item:: 自定义 LeRobot 数据集
      :sync: custom-lerobot

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env maniskill_libero && source .venv/bin/activate

      **2. 准备数据集与 dataconfig**

      将 LeRobot 数据集目录（直接包含 ``meta/info.json`` 和 ``data/``）放到训练机，数据需包含相机图像、状态、动作与任务文本（``task`` 或 ``prompt``）。

      ``examples/sft/config/custom_sft_openpi.yaml`` 默认使用 ``actor.model.openpi.config_name: "pi0_custom"``。按你的相机键名、状态与动作维度以及增量或绝对动作格式调整该 dataconfig，或在 OpenPI dataconfig 注册表中新增条目。

      用与训练一致的 ``--config-name`` 计算 ``norm_stats.json``：

      .. code-block:: bash

         python toolkits/lerobot/calculate_norm_stats.py \
           --config-name pi0_custom \
           --repo-id ./custom-lerobot-data

      该命令会在 ``./assets/pi0_custom/custom-lerobot-data/norm_stats.json`` 生成统计文件。

      **3. 启动训练**

      通过 Hydra CLI 覆盖数据集、基座权重和归一化统计路径：

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh custom_sft_openpi \
           data.train_data_paths=./custom-lerobot-data \
           actor.model.model_path=./pi0_base_openpi \
           +actor.model.openpi_data.norm_stats_path=./assets/pi0_custom/custom-lerobot-data/norm_stats.json

      .. dropdown:: 进阶配置（可选）

         ``examples/sft/config/custom_sft_openpi.yaml`` 中的核心路径配置：

         .. code-block:: yaml

            data:
              train_data_paths: ./custom-lerobot-data

            actor:
              model:
                model_path: ./pi0_base_openpi
                num_action_chunks: 4
                openpi:
                  task: sft
                  config_name: "pi0_custom"
                openpi_data:
                  norm_stats_path: ./assets/pi0_custom/custom-lerobot-data/norm_stats.json

.. dropdown:: 通用约定（精度、FSDP 与 action horizon）

   .. list-table::
      :header-rows: 1
      :widths: 26 74

      * - 项目
        - 约定
      * - 任务模式
        - 训练时设 ``openpi.task: sft``，评测配置设 ``eval``。
      * - 动作 chunk 与网络 horizon
        - ``num_action_chunks`` 控制环境单次执行步数；``openpi.action_horizon`` 控制网络预测长度（``pi05_behavior`` 为 **32**，``pi0_aloha_robotwin`` 与 ``pi0_libero`` 为 **50**，``pi0_realworld`` 为 **10**）。
      * - 数据窗口长度
        - 标准 LeRobot SFT 按 dataconfig 的 horizon 切分数据窗口；BEHAVIOR 与双臂 Franka 流式加载器按 ``num_action_chunks`` 切分，这两份配置需将其设为与网络 horizon 一致。
      * - 主权重与激活精度
        - 设 ``actor.model.precision: fp32`` 保持主权重与优化器更新为 FP32，设 ``actor.model.openpi.embed_dtype: bf16`` 让主干激活使用 BF16。省略 ``embed_dtype`` 时跟随 ``model.precision``\ （``precision: null`` 对应 BF16）。
      * - 混合精度（FSDP1）
        - 使用 ``hybrid_engines/openpi_fsdp`` 可在主干 BF16 计算时，保留 projection、RMSNorm、视觉 embedding 与 RLT Stage 1 权重为 FP32。
      * - 分片策略
        - OpenPI 支持 ``no_shard``、``shard_grad_op`` 和 ``full_shard``，且需开启 ``use_orig_params: true``。
      * - 激活显存与学习率
        - 开启 ``gradient_checkpointing: true`` 与 ``gradient_checkpointing_use_reentrant: false`` 可降低激活显存。BEHAVIOR 与 RoboTwin 使用 ``openpi_cosine`` 学习率调度。
      * - Tokenizer
        - 自动从基座模型加载。

   .. code-block:: yaml

      defaults:
        - model/pi0_5@actor.model
        - hybrid_engines/openpi_fsdp@actor.fsdp_config
        - _self_

      actor:
        model:
          precision: fp32
          openpi:
            task: sft
            embed_dtype: bf16
        fsdp_config:
          sharding_strategy: no_shard
          use_orig_params: true
          mixed_precision:
            param_dtype: bf16
            reduce_dtype: fp32
            buffer_dtype: fp32

.. dropdown:: 支持的数据集与自定义 LeRobot 配置

   用 ``actor.model.openpi.config_name`` 选择数据格式：

   .. list-table::
      :header-rows: 1
      :widths: 44 56

      * - ``config_name``
        - 数据集 / 环境
      * - ``pi0_libero`` · ``pi05_libero``
        - LIBERO
      * - ``pi0_realworld``
        - 真机 Franka
      * - ``pi05_yam_joint``
        - 真机双臂 YAM
      * - ``pi0_aloha_robotwin``
        - RoboTwin（ALOHA）
      * - ``pi05_behavior``
        - BEHAVIOR-1K（流式加载）
      * - ``pi0_maniskill`` · ``pi05_maniskill``
        - ManiSkill
      * - ``pi05_metaworld``
        - MetaWorld
      * - ``pi05_calvin``
        - CALVIN
      * - ``pi05_dexjoco``
        - DexJoCo WaterPlant
      * - ``pi0_custom``
        - 自定义 LeRobot

   **自定义 LeRobot 数据集：** 在 ``examples/sft/config/custom_sft_openpi.yaml`` 中设置 ``actor.model.openpi.config_name: "pi0_custom"``，在 OpenPI dataconfig 注册表中注册 ``pi0_custom``，并按实际观测字段和动作空间调整自定义数据配置类。

.. dropdown:: 新 LeRobot 数据集的归一化统计

   新采集的 LeRobot 数据需在 SFT 前计算 ``state`` 和 ``actions`` 的 ``norm_stats``，真机数据尤其需要：

   .. code-block:: bash

      # 本地数据集目录（含 meta/info.json）：
      python toolkits/lerobot/calculate_norm_stats.py \
          --config-name pi0_realworld \
          --repo-id ./realworld_franka_bin_relocation

      # 或 Hugging Face repo id（默认缓存在 ~/.cache/huggingface/lerobot）：
      python toolkits/lerobot/calculate_norm_stats.py \
          --config-name pi0_realworld \
          --repo-id realworld_franka_bin_relocation

   - **输入与缓存**：``--repo-id`` 支持本地路径或 LeRobot HF repo id，可用 ``HF_LEROBOT_HOME`` 修改缓存目录。

   - **配置一致**：``--config-name`` 必须与训练使用的 dataconfig 一致。

   - **输出位置**：脚本默认写出到 ``./assets/<config_name>/<repo_id>/norm_stats.json``。

   - **显式传入统计路径**：训练和评测不会自动从 SFT 的 ``full_weights.pt`` 目录寻找统计文件，需显式传入 ``openpi_data.norm_stats_path``\ （省略时会回退到 ``{assets_dir}/{asset_id}/norm_stats.json`` 并打印警告）。

   - **RoboTwin 与 BEHAVIOR**：RoboTwin 将统计固定在基础权重旁的 ``physical-intelligence/robotwin/<task>/norm_stats.json``；BEHAVIOR 不使用该脚本，直接用 ``actor.model.openpi_data.norm_stats_path`` 指向已有统计文件。

   - **训练稳定性**：若标准差过小或 q99–q01 范围过窄，适当放宽通常更稳，尤其是 SFT 后继续做在线 RL 时。

.. dropdown:: 评测 SFT 检查点

   ``openpi`` 评测可直接加载 ``.../checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``\ （或 ``global_step_<N>`` 目录），无需先转成 ``model.safetensors``。

   - 评测配置保持 ``openpi.task: eval``，并让 ``config_name`` 与 ``num_action_chunks`` 同训练一致。

   - 显式将 ``openpi_data.norm_stats_path`` 指回训练统计文件：

   .. code-block:: bash

      bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
        rollout.model.model_path=./logs/checkpoints/global_step_30000 \
        rollout.model.openpi_data.norm_stats_path=./pi0_base_pytorch_new/physical-intelligence/robotwin/adjust_bottle/norm_stats.json

   详细评测步骤见 :doc:`BEHAVIOR-1K 评测 <../../evaluations/guides/behavior>` 和 :doc:`RoboTwin 评测 <../../evaluations/guides/robotwin>`。如需导出独立目录（``model.safetensors``、``config.json`` 与 ``norm_stats.json``），使用 OpenPI 检查点转换器的 ``sft_to_openpi`` 模式（见 :doc:`检查点转换 <../../guides/convertor>`）。

可视化与结果
----------------------------------------

运行 ``tensorboard --logdir ./logs --port 6006`` 观察 **训练损失** 变化。完整指标定义与日志后端见 :doc:`训练指标 <../../reference/metrics>`。
