ManiSkill OOD 评测
==================

ManiSkill OOD 评测用于检验 VLA 策略在 ManiSkill 分布外（Out-of-Distribution）场景上的泛化能力。评测基于 **Put-on-Plate** 任务族（将胡萝卜放到盘子上），沿用 `rl4vla <https://arxiv.org/abs/2505.19789>`_ 论文中的 OOD 测试协议，将场景划分为 **Vision**、**Semantic**、**Execution** 三类。

相关训练文档：:doc:`../../examples/embodied/maniskill`

环境准备
--------

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate

当前 ``evaluations/maniskill/`` 仅提供 OpenVLA-OFT 示例配置；OpenVLA、OpenPI 等模型在 ManiSkill 上有训练配置，但尚无独立评测 YAML，可从训练配置派生（见下文 :ref:`maniskill-derive-from-train`）。

下载 ManiSkill 资源（若尚未下载）：

.. code-block:: bash

   cd rlinf/envs/maniskill
   # 为提升国内下载速度，可以设置：
   # export HF_ENDPOINT=https://hf-mirror.com
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets

模型与权重
----------

OpenVLA-OFT 评测通常需要两类权重：

1. **基座模型** ``rollout.model.model_path``：如 `RLinf/Openvla-oft-SFT-libero10-trajall <https://huggingface.co/RLinf/Openvla-oft-SFT-libero10-trajall>`_
2. **ManiSkill LoRA** ``rollout.model.lora_path``：如 `RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora>`_

若评测 RL 训练产出的策略，通过 ``runner.ckpt_path`` 或 ``CKPT_PATH`` 传入 ``.pt`` 权重即可，会覆盖模型初始化参数。

示例配置
--------

``evaluations/maniskill/`` 目录下已有以下示例：

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - 配置文件
     - 说明
     - 模型
   * - ``maniskill_ood_openvlaoft_eval.yaml``
     - OOD 泛化评测模板（默认训练场景）
     - OpenVLA-OFT

完整评测流程
------------

**Step 1：激活环境**

.. code-block:: bash

   source .venv/bin/activate

**Step 2：编辑配置**

复制或编辑目标 YAML，至少修改 ``rollout.model.model_path`` 与 ``rollout.model.lora_path``。``env.eval`` 通用字段见 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`；ManiSkill 场景选择与协议见下文 :ref:`maniskill-eval-config`。

**Step 3：启动评测**

.. code-block:: bash

   bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
     rollout.model.model_path=/path/to/model \
     rollout.model.lora_path=/path/to/lora

**Step 4：查看结果**

终端输出 ``eval/success_once``；日志与视频见 :doc:`../reference/results`。

.. _maniskill-eval-config:

评测配置详解
------------

ManiSkill 评测在 ``env.eval.init_params`` 下通过 ``id`` （环境 ID）与 ``obj_set`` （物体集合）选择具体场景，并汇总 ``eval/success_once`` （轨迹中至少成功一次的比例）。

评测协议概述
~~~~~~~~~~~~

RLinf 的 ManiSkill OOD 协议与 `rl4vla <https://arxiv.org/abs/2505.19789>`_ 保持一致，便于与已发表结果对比。

- **训练分布内（In-Distribution）**：``PutOnPlateInScene25Main-v3`` + ``obj_set=train``，即 plate-25-main 主训练任务；
- **分布外（OOD）**：13 个变体环境 + ``obj_set=test``，按 Vision / Semantic / Execution 划分；
- **补充评测**：``mani-ood`` 模式还会在 3 个 Semantic 任务上以 ``obj_set=train`` 各跑一轮。

每条评测轨迹由 ``episode_id`` （即 ``reset_state_id``）唯一确定，对应一组物体、盘子、位姿与（部分场景的）视觉扰动组合。``use_fixed_reset_state_ids=True`` 时环境按 ``episode_id`` 加载确定初始条件；``auto_reset=True`` 时 episode 结束后按序分配下一个 ``episode_id``。

OOD 场景列表
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - 类别
     - 环境 ID（``env.eval.init_params.id``）
     - 说明
   * - Vision
     - ``PutOnPlateInScene25VisionImage-v1``
     - 背景图像扰动
   * -
     - ``PutOnPlateInScene25VisionTexture03-v1`` / ``PutOnPlateInScene25VisionTexture05-v1``
     - 纹理扰动（强度 0.3 / 0.5）
   * -
     - ``PutOnPlateInScene25VisionWhole03-v1`` / ``PutOnPlateInScene25VisionWhole05-v1``
     - 整体视觉扰动（强度 0.3 / 0.5）
   * - Semantic
     - ``PutOnPlateInScene25Carrot-v1``
     - 未见胡萝卜物体
   * -
     - ``PutOnPlateInScene25Plate-v1``
     - 未见盘子
   * -
     - ``PutOnPlateInScene25Instruct-v1``
     - 语言指令变化
   * -
     - ``PutOnPlateInScene25MultiCarrot-v1`` / ``PutOnPlateInScene25MultiPlate-v1``
     - 多胡萝卜 / 多盘子
   * - Execution
     - ``PutOnPlateInScene25Position-v1``
     - 物体初始位置变化
   * -
     - ``PutOnPlateInScene25EEPose-v1``
     - 机械臂初始位姿变化
   * -
     - ``PutOnPlateInScene25PositionChangeTo-v1``
     - 目标位置动态变化

关键环境参数
~~~~~~~~~~~~

以下字段位于 ``env.eval.init_params`` 下：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - 作用
   * - ``id``
     - ManiSkill 环境注册名，决定 OOD 变体类型（见上表）。默认模板为 ``PutOnPlateInScene25Main-v3`` （训练场景）。
   * - ``obj_set``
     - 物体集合：``train`` （训练集物体）、``test`` （测试集物体）、``all`` （全部）。OOD 场景通常设为 ``test``；训练场景评测设为 ``train``。
   * - ``obs_mode``
     - 观测模式，VLA 评测使用 ``rgb+segmentation``。
   * - ``sim_backend``
     - 仿真后端，默认 ``gpu``，需要 NVIDIA GPU。
   * - ``policy_setup``
     - 动作空间配置，OpenVLA-OFT 使用 ``widowx_bridge`` （在 ``maniskill_ood_template`` 中设置）。

``env.eval`` 通用字段（``total_num_envs``、``max_episode_steps``、``auto_reset`` 等）说明见 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`。ManiSkill 评测示例通常使用 ``max_episode_steps=80``、``max_steps_per_rollout_epoch=80``、``ignore_terminations=True``。

OpenVLA-OFT 模型字段
~~~~~~~~~~~~~~~~~~~~

除 ``model_path`` 外，``maniskill_ood_openvlaoft_eval.yaml`` 还需正确设置：

.. code-block:: yaml

   rollout:
     model:
       model_type: openvla_oft
       unnorm_key: bridge_orig
       is_lora: True
       lora_path: /path/to/RLinf-OpenVLAOFT-ManiSkill-Base-Lora
       add_value_head: True
       max_prompt_length: 30

单场景评测
----------

在默认训练场景或任意 OOD 场景上单独评测时，通过 Hydra 覆盖 ``init_params``：

**训练分布内（plate-25-main）**

.. code-block:: bash

   bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
     env.eval.init_params.id=PutOnPlateInScene25Main-v3 \
     env.eval.init_params.obj_set=train \
     rollout.model.model_path=/path/to/model \
     rollout.model.lora_path=/path/to/lora

**单个 OOD 场景（以 Vision 为例）**

.. code-block:: bash

   bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
     env.eval.init_params.id=PutOnPlateInScene25VisionImage-v1 \
     env.eval.init_params.obj_set=test \
     rollout.model.model_path=/path/to/model \
     runner.ckpt_path=/path/to/checkpoint.pt

覆盖完整测试集
~~~~~~~~~~~~~~

每个场景的 ``total_num_trials`` 由物体数、盘子数与位姿组合数决定。资源充足时可增大 ``total_num_envs``；资源受限时在 ``auto_reset=True`` 下将 ``max_steps_per_rollout_epoch`` 设为 ``max_episode_steps`` 的整数倍，使每轮 ``rollout_epoch`` 串行覆盖更多 ``episode_id``：

.. code-block:: yaml

   env:
     eval:
       total_num_envs: 16
       max_episode_steps: 80
       max_steps_per_rollout_epoch: 320   # 4 * 80，每轮约评测 4 * total_num_envs 条轨迹
       auto_reset: True
       ignore_terminations: True
       use_fixed_reset_state_ids: True
       rollout_epoch: 1

批量 OOD 评测（``mani-ood`` 模式）
-----------------------------------

``mani-ood`` 模式会依次在上述 13 个 OOD 场景（``obj_set=test``）及 3 个 Semantic 场景（``obj_set=train``）上运行评测，合计 **16** 次，与训练文档中的完整 OOD 协议一致。

**必填环境变量**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 变量
     - 说明
   * - ``EVAL_NAME``
     - 本次批量评测名称，日志写入 ``logs/eval/<EVAL_NAME>/``
   * - ``CKPT_PATH``
     - RL 训练产出的 ``.pt`` 权重路径，覆盖 ``runner.ckpt_path``
   * - ``TOTAL_NUM_ENVS``
     - 并行环境数，对应 ``env.eval.total_num_envs``
   * - ``EVAL_ROLLOUT_EPOCH``
     - 评测轮次，对应 ``env.eval.rollout_epoch``

.. code-block:: bash

   export EVAL_NAME=my_ood_eval
   export CKPT_PATH=/path/to/checkpoint.pt
   export TOTAL_NUM_ENVS=16
   export EVAL_ROLLOUT_EPOCH=1
   bash evaluations/run_eval.sh mani-ood maniskill_ood_openvlaoft_eval

批量日志目录：``logs/eval/<EVAL_NAME>/<时间戳>-<env_id>-<obj_set>/run_ppo.log``

``mani-ood`` 模式会自动设置 ``HF_ENDPOINT`` （默认 ``https://hf-mirror.com``），可在运行前自行覆盖。

.. _maniskill-derive-from-train:

进阶用法
--------

**从训练配置派生评测**

ManiSkill 还支持 ``PickCube-v1``、``PutCarrotOnPlateInScene-v2`` 等训练任务（见 ``examples/embodiment/config/env/``），但尚无独立评测 YAML。可复制对应训练配置并设置：

- ``runner.task_type: embodied_eval``
- ``runner.only_eval: True``
- 删除 ``algorithm``、``actor`` 等训练段，保留 ``env.eval`` 与 ``rollout``

**调整并行规模**

.. code-block:: bash

   bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
     env.eval.total_num_envs=32 \
     rollout.model.model_path=/path/to/model

**加载 RL checkpoint**

.. code-block:: bash

   bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
     runner.ckpt_path=/path/to/checkpoint.pt \
     rollout.model.model_path=/path/to/model

常见问题
--------

- **资源路径：** 确保 ManiSkill assets 已下载到 ``rlinf/envs/maniskill/assets``。
- **GPU 仿真：** ``sim_backend: gpu`` 需要 NVIDIA GPU；headless 环境下 ``run_eval.sh`` 已设置 ``MUJOCO_GL=osmesa`` 等变量。
- **LoRA 路径：** OpenVLA-OFT 评测必须设置 ``lora_path``，否则无法正确加载 ManiSkill 策略。
- **checkpoint：** 批量模式通过 ``CKPT_PATH`` 传入 ``.pt`` 权重；单次评测使用 ``runner.ckpt_path``。
- **场景选择：** 默认 YAML 指向训练场景 ``PutOnPlateInScene25Main-v3``；评测 OOD 场景时需显式覆盖 ``env.eval.init_params.id`` 与 ``obj_set``，或使用 ``mani-ood`` 模式。
