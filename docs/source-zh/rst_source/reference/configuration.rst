训练配置
========

RLinf 配方采用可组合的 Hydra YAML 预设。本页是一页式配置地图，涵盖带注释的最小 YAML、单卡最常调参数与预设结构；具体示例页面直接链接到此页，不再重复罗列长字段表。

带注释的最小 YAML
-----------------

核心具身配方通过 ``defaults`` 组合 ``base/embodied_rl``、环境预设、模型预设与算法预设，实验文件只需写出当前任务的差异项：

.. code-block:: yaml

   defaults:
     - base/embodied_rl                              # 默认全卡 cluster、runner、FSDP 与权重同步器
     - env/manikill_put_carrot_on_plate_in_scene@env.train
     - env/manikill_put_carrot_on_plate_in_scene@env.eval
     - model/openvla@actor.model                     # 模型结构与默认精度
     - algorithm/ppo@algorithm                       # PPO / GAE 超参数
     - _self_

   runner:
     logger:
       experiment_name: "maniskill_ppo_openvla_quickstart"

   env:
     train:
       total_num_envs: 32                            # 并行仿真环境数
       rollout_epoch: 1

   rollout:
     pipeline_stage_num: 2                           # env 与 rollout 重叠执行的流水线阶段数

   actor:
     micro_batch_size: 20                            # 单卡前向/反向微批次
     global_batch_size: 160                          # 须能被 micro_batch_size * actor 卡数整除
     model:
       model_path: "gen-robot/openvla-7b-rlvla-warmup"  # 支持 HF Repo ID 或本地目录；rollout 自动继承
       add_value_head: True
       is_lora: True

任意配置项均可在命令行直接覆盖，无需修改仓库内的 YAML 文件：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart \
     actor.model.model_path=/path/to/local-checkpoint \
     env.train.total_num_envs=8 \
     actor.micro_batch_size=4 \
     actor.global_batch_size=16 \
     runner.max_epochs=10

单卡运行最常调的 5 个字段
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 配置项
     - 作用与建议
   * - ``actor.model.model_path``
     - 支持 Hugging Face ``owner/repo``\ （自动缓存下载）或本地权重目录；训练时 ``rollout.model.model_path`` 自动继承该值。
   * - ``cluster.component_placement``
     - 通过 ``cluster/collocated`` 默认使用 ``actor,env,rollout: all``。单卡限定可改为 ``actor,env,rollout: 0``，多卡分离部署见 :doc:`../concepts/overview`。
   * - ``env.train.total_num_envs``
     - 并行环境数量。仿真渲染或内存紧张时优先调小该项。
   * - ``actor.micro_batch_size`` / ``global_batch_size``
     - 训练显存不足（OOM）时调小 ``micro_batch_size``，同时保证 ``global_batch_size`` 能被 ``micro_batch_size * actor 卡数`` 整除。
   * - ``actor.enable_offload`` / ``rollout.enable_offload``
     - ``base/embodied_rl`` 默认开启 ``True`` 以适配单卡/共置显存；多卡分离部署时可设为 ``False`` 提速。

Hydra 预设与配置位置
--------------------

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - 预设 / 工作负载
     - 路径
     - 职责 / 启动入口
   * - 具身基础与集群预设
     - ``examples/embodiment/config/{base,cluster}/``
     - 通用 ``base/embodied_rl`` 骨架与 ``cluster/collocated`` 放置（自定义放置可加 ``override /cluster@cluster: null``）。
   * - 算法预设
     - ``examples/embodiment/config/algorithm/``
     - 挂载在 ``@algorithm`` 的 ``ppo`` 与 ``grpo`` 公共默认参数。
   * - 环境、模型与后端预设
     - ``examples/embodiment/config/{env,model,hybrid_engines}/``
     - 仿真器/真机预设、模型模板与 FSDP 默认配置。
   * - 具身 RL 配方
     - ``examples/embodiment/config/*.yaml``
     - ``bash examples/embodiment/run_embodiment.sh <config_name>``
   * - 推理与智能体 RL
     - ``examples/{reasoning,agent}/**/config/*.yaml``
     - ``run_main_grpo_{math,vqa}.sh`` 或 ``examples/agent/<recipe>/`` 下对应启动脚本
   * - SFT
     - ``examples/sft/config/*.yaml``
     - ``bash examples/sft/run_vla_sft.sh <config_name>``\ （或 ``run_vlm_sft.sh``）
   * - 独立评测
     - ``evaluations/<benchmark>/*.yaml``
     - ``bash evaluations/run_eval.sh <benchmark> <config_name>``

顶层配置段
----------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 配置段
     - 职责
   * - ``cluster``
     - 节点数、硬件节点组，以及 actor、rollout、env、reward 或 agent worker 的 ``component_placement``。
   * - ``runner``
     - 任务类型、``logger.log_path``\ （TensorBoard、视频与 checkpoint 根目录）、训练轮数、验证/保存间隔与续训路径。
   * - ``algorithm``
     - 优势估计器、策略损失、裁剪阈值、KL 系数及离策略 / 演示缓冲区设置。
   * - ``env`` / ``data``
     - 仿真与真机环境设置（``env``），或数据集路径与 prompt / answer 字段映射（``data``）。
   * - ``rollout``
     - 推理引擎（HuggingFace、SGLang、vLLM）、采样参数及可选模型覆盖项。
   * - ``actor``
     - 训练后端（FSDP、Megatron）、模型路径、LoRA、优化器、batch size 与并行切分策略。

完整字段参考
------------

- :doc:`架构总览 <../concepts/overview>`：三张图速览组件分工、部署模式与数据格式。
- :doc:`基础配置 <../guides/basic_config>`：通用 ``hydra``、``cluster``、``runner``、``algorithm`` 与 ``actor`` 逐字段说明。
- :doc:`具身智能配置 <../guides/embodiment_config>`：完整具身 ``env``、``rollout`` 与 ``model`` 参考。
- :doc:`智能体配置 <../guides/agentic_config>`：LLM / VLM 推理与智能体循环配置。
- :doc:`训练指标 <metrics>` 与 :doc:`评测配置 <../evaluations/reference/configuration>`。
