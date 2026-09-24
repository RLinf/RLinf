OpenPI 监督微调
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 40%

   OpenPI π₀ / π₀.₅ 视觉-语言-动作模型。

在 RLinf 里对 π₀ / π₀.₅ 做 **全量 SFT** 或 **LoRA**。策略统一为 ``model_type: openpi``：这是与 JAX 参考对齐的 PyTorch 实现（官方 openpi 仓库自带的 PyTorch 代码并未对齐）。SFT 通常是强化学习前的冷启动：先模仿示范，再在较好先验上做 RL。

配方一览
----------------------------------------

实验配置通过 Hydra ``defaults`` 引入模型模板，再覆盖路径和任务字段。启动前把 ``/path/to/...`` 换成本地路径。

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

双 Franka 真机的 SFT 与部署见 :doc:`dual_franka_openpi_pytorch`。

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

| **你将完成：** 安装依赖 → 准备数据与归一化统计 → ``bash examples/sft/run_vla_sft.sh <启动名>`` → 看训练损失。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 对应格式的数据集。

通用约定
----------------------------------------

训练时 ``openpi.task`` 必须为 ``sft``；评测 YAML 用 ``eval``，不要把训练的 ``sft`` 抄过去。

``num_action_chunks`` 和环境实际执行的 chunk 是一回事，**不是**\ 网络 horizon：

- ``num_action_chunks`` / ``openpi.action_chunk``：环境一次执行多长。
- 网络 ``action_horizon``：优先用 YAML 的 ``openpi.action_horizon``；否则用 ``openpi.config_name`` 对应官方 ``TrainConfig.model.action_horizon``。默认模板只把 ``num_action_chunks`` 插值到 ``action_chunk``，**不会**\ 把它写成网络 horizon。

官方 LeRobot SFT 的数据窗口也来自 ``TrainConfig.model.action_horizon``，不读 ``num_action_chunks``。常见官方 horizon：BEHAVIOR ``pi05_behavior`` **32**\ ，RoboTwin ``pi0_aloha_robotwin`` / LIBERO ``pi0_libero`` **50**\ ，真机 ``pi0_realworld`` **10**。只有 checkpoint 的 horizon 和该 ``TrainConfig`` 不一致时，才覆写 ``openpi.action_horizon``。

BEHAVIOR / dual-Franka 的流式 SFT 加载器目前仍用 ``num_action_chunks`` 当数据窗口，这两份配方里才需要把它设成和官方 horizon 一样。

精度与 FSDP：

- ``actor.model.precision: null``\ （OpenPI 默认：Gemma / SigLIP 为 bf16，action head 为 fp32）。
- ``actor.fsdp_config.sharding_strategy`` **必须** 为 ``no_shard``\ （``full_shard`` / ``shard_grad_op`` 会在启动时失败）。每个 rank 持有完整参数、梯度和 optimizer state。
- FSDP dtype 绑到 ``actor.model.precision``。BEHAVIOR / RoboTwin 配方还会打开 ``gradient_checkpointing: True``，并用 ``actor.optim.lr_scheduler: openpi_cosine``\ （warmup 从 ``peak / (warmup + 1)`` 开始，再余弦衰减到 ``min_lr``）。

.. code:: yaml

   actor:
     model:
       precision: null
       openpi:
         task: sft
     fsdp_config:
       sharding_strategy: no_shard
       mixed_precision:
         param_dtype: ${actor.model.precision}
         reduce_dtype: ${actor.model.precision}
         buffer_dtype: ${actor.model.precision}

Tokenizer 由 OpenPI 的 ``ModelTransformFactory`` 按基础模型加载，YAML 里不用单独写 SentencePiece 路径。

数据集
----------------------------------------

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
   * - ``pi0_custom``
     - 自定义 LeRobot

自定义 LeRobot 数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 在 ``examples/sft/config/custom_sft_openpi.yaml`` 里指定 ``config_name``：

   .. code:: yaml

      actor:
        model:
          openpi:
            task: sft
            config_name: "pi0_custom"

2. 在 :file:`rlinf/models/embodiment/openpi/dataconfig/__init__.py` 注册 ``pi0_custom``\ （仓库里已有一份模板）：

   .. code:: python

      TrainConfig(
          name="pi0_custom",
          model=pi0_config.Pi0Config(),
          data=CustomDataConfig(
              repo_id="physical-intelligence/custom_dataset",
              base_config=DataConfig(prompt_from_task=True),
              assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets"),
              extra_delta_transform=False,
              action_train_with_rotation_6d=False,
          ),
          pytorch_weight_path="checkpoints/torch/pi0_base",
      ),

3. 按自己的 key / 动作空间改 ``CustomDataConfig``，定义在 ``rlinf/models/embodiment/openpi/dataconfig/franka_dataconfig.py``。

归一化统计
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

新采集的 LeRobot 数据要在 SFT 前算 ``state`` / ``actions`` 的 ``norm_stats``，真机数据尤其需要：

.. code:: bash

   # 本地数据集目录（含 meta/info.json）：
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id /path/to/realworld_franka_bin_relocation

   # 或 Hugging Face repo id（默认缓存在 ~/.cache/huggingface/lerobot）：
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi0_realworld \
       --repo-id realworld_franka_bin_relocation

- ``--repo-id`` 可以是本地路径或 LeRobot HF repo id；可用 ``HF_LEROBOT_HOME`` 改缓存父目录。
- ``--config-name`` 必须和训练用的 dataconfig 一致。
- ``calculate_norm_stats.py`` 默认写到 ``./assets/<config_name>/<repo_id>/norm_stats.json``\ （``TrainConfig.assets_dirs / repo_id``）。训练和评测\ **不会**\ 自动从 SFT 的 ``full_weights.pt`` 目录找这份文件：要用 ``openpi_data.norm_stats_path`` 显式指过去。省略该字段时，OpenPI 按 TrainConfig 默认读取 ``{assets_dir}/{asset_id}/norm_stats.json``\ （会在 ``model_path`` 覆盖之后），并打一条 warning。RoboTwin 配方把统计钉在基础权重旁的 ``physical-intelligence/robotwin/<task>/norm_stats.json``。

若标准差过小或 q99–q01 过窄，适当放大通常更稳，尤其是 SFT 之后还要做在线训练时。

BEHAVIOR 不跑上面的脚本，而是用 ``actor.model.openpi_data.norm_stats_path`` 指向已有的 ``{assets}/{asset_id}/norm_stats.json``。

安装
----------------------------------------

.. include:: _setup_common.rst

**方式一：Docker** —— 镜像 ``agentic-rlinf0.4-maniskill_libero``：

.. code:: bash

    docker run -it --rm --gpus all \
        --shm-size 20g \
        --network host \
        --name rlinf \
        -v .:/workspace/RLinf \
        rlinf/rlinf:agentic-rlinf0.4-maniskill_libero
        # 国内镜像：infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

    source switch_env openpi

**方式二：自建环境**：

.. code:: bash

    # 国内可加 --use-mirror
    bash requirements/install.sh embodied --model openpi --env maniskill_libero
    source .venv/bin/activate

``--model openpi`` 装的是官方 OpenPI **运行环境**\ （transforms、数据管线）；训练用的网络仍是 ``openpi``。

LIBERO 与真机 Franka
----------------------------------------

先改实验 YAML 里的数据路径和 ``actor.model.model_path``。常用覆写：

.. code:: yaml

   data:
     train_data_paths: /path/to/libero-data

   cluster:
     num_nodes: 1
     component_placement:
       actor,env,rollout: 0-0

   actor:
     model:
       model_path: /path/to/pi0-model
       num_action_chunks: 4          # 环境执行长度，不是网络 horizon
       is_lora: False                # LoRA 时改为 True，并设 lora_rank
       openpi:
         task: sft
         config_name: "pi0_libero"   # 真机改为 pi0_realworld

启动：

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openpi
   bash examples/sft/run_vla_sft.sh realworld_bin_relocation_sft_openpi

日志和 checkpoint 写在 ``runner.logger.log_path`` 下，每 ``runner.save_interval`` 步保存到 ``.../checkpoints/global_step_<N>/``。

Pi0.5 + BEHAVIOR-1K
----------------------------------------

路径写在实验配置里，模型形状来自 ``model/pi0_5``：

.. code:: yaml

   defaults:
     - model/pi0_5@actor.model
     - hybrid_engines/fsdp@actor.fsdp_config

流式加载器只读 ``data:``，没有隐藏默认值。至少改这些路径：

.. code:: yaml

   data:
     train_data_paths: /path/to/2025-challenge-demos
     behavior_dataset_root: /path/to/2025-challenge-demos
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
       model_path: /path/to/pi05_base_pytorch_new   # 新格式 fp32 基础权重
       openpi:
         task: sft
       openpi_data:
         norm_stats_path: /path/to/assets/behavior-1k/2025-challenge-demos/norm_stats.json

- ``use_skill: false`` 用主任务文本；``true`` 时按窗口从 ``task_subtasks`` 取逐帧技能文本（数据集里折叠后的 orchestrator 不能当标签）。
- ``fine_grained_level`` / ``tolerance_s`` 控制流式时间对齐。

.. code:: bash

   bash examples/sft/run_vla_sft.sh behavior_sft_openpi_pi05

Pi0 + RoboTwin
----------------------------------------

用官方 OpenPI / LeRobot map-style 加载器，``config_name: pi0_aloha_robotwin``。示例数据：`adjust_bottle <https://huggingface.co/datasets/RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data>`_；对应 HF SFT 权重：`RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle <https://huggingface.co/RLinf/RLinf-Pi0-NEW-RoboTwin-SFT-adjust_bottle>`_。

.. code:: yaml

   data:
     train_data_paths: /path/to/robotwin-data
     num_workers: 4

   actor:
     model:
       model_path: /path/to/pi0_base_pytorch_new
       num_action_chunks: 50
       action_dim: 14
       openpi:
         task: sft
         config_name: "pi0_aloha_robotwin"
         num_images_in_input: 3
       openpi_data:
         norm_stats_path: ${actor.model.model_path}/physical-intelligence/robotwin/adjust_bottle/norm_stats.json

14 维 ALOHA 动作、3 路图像；进模型前按 OpenPI 规则 pad 到 32 维。``openpi_data.norm_stats_path`` 把训练和评估钉到同一份 ``norm_stats.json``。换任务时改路径里的任务名。

.. code:: bash

   bash examples/sft/run_vla_sft.sh robotwin_adjust_bottle_sft_openpi

评测
----------------------------------------

SFT 会把 ``full_weights.pt`` 写到 ``.../checkpoints/global_step_<N>/actor/model_state_dict/``。``openpi`` 评测可以直接加载这个 ``.pt``\ （或 ``global_step_<N>`` 目录）：加载时会剥掉 FSDP / wrapper 前缀，模型形状来自评测 YAML，不必先转成 ``model.safetensors``。

不要把训练的 ``openpi.task: sft`` 带到评测里——评测配置必须是 ``task: eval``\ （才会走 ``Pi0Eval`` 和观测 transform）。``config_name``、``num_action_chunks`` 与训练对齐。

归一化统计也\ **不会**\ 跟着 ``full_weights.pt`` 走。现成评测 YAML 常把 ``openpi_data.norm_stats_path`` 插值成 ``${rollout.model.model_path}/.../norm_stats.json``，那是给「权重和统计在同一目录」的转换产物用的。只改 ``model_path`` 指到 SFT checkpoint 时，必须\ **另外**\ 把统计指回训练用的那份（RoboTwin：基础权重旁的 ``physical-intelligence/robotwin/<task>/norm_stats.json``；BEHAVIOR：训练时的 ``openpi_data.norm_stats_path``）。

.. code:: bash

   # 示例：直接评测 SFT 产出的 .pt（统计仍用训练时的文件，不要跟 model_path 走）
   bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
     rollout.model.model_path=/path/to/logs/.../checkpoints/global_step_30000 \
     rollout.model.openpi_data.norm_stats_path=/path/to/pi0_base_pytorch_new/physical-intelligence/robotwin/adjust_bottle/norm_stats.json

评测步骤见 :doc:`BEHAVIOR-1K <../../evaluations/guides/behavior>` 和 :doc:`RoboTwin <../../evaluations/guides/robotwin>`。如果要导出可分发的裸 ``Pi0`` 目录（``model.safetensors`` + ``config.json`` + 拷进去的 ``norm_stats.json``），用 ``sft_to_openpi``，命令见 ``rlinf/utils/ckpt_convertor/openpi/README.md``。

可视化
----------------------------------------

看 **训练损失** 是否在下降。指标含义见 :doc:`训练指标 <../../reference/metrics>`。

.. code:: bash

   tensorboard --logdir ./logs
