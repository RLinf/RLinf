ManiSkill OOD 评测
==================

概览
----

ManiSkill OOD 评测用于检验 VLA policy 在分布外（Out-of-Distribution）场景上的泛化能力。评测基于 **Put-on-Plate** 任务族，沿用 `rl4vla <https://proceedings.neurips.cc/paper_files/paper/2025/hash/8c0fabe372177d2aded596be2d3b4544-Abstract-Conference.html>`_ 测试协议，覆盖训练分布内场景以及按 **Vision**、**Semantic**、**Execution** 划分的 13 个 OOD 变体。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 评测基准
      :text-align: center

      ManiSkill OOD（13 个 OOD + 训练分布内场景）

   .. grid-item-card:: 支持模型
      :text-align: center

      OpenVLA-OFT（示例） · OpenVLA / OpenPI（派生配置）

   .. grid-item-card:: 配置目录
      :text-align: center

      ``evaluations/maniskill/``

   .. grid-item-card:: 关联训练
      :text-align: center

      :doc:`ManiSkill 训练 <../../examples/embodied/maniskill>`

安装与运行
----------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. tab-set::

   .. tab-item:: 单场景评测

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

      **2. 下载资产与模型权重**

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets
         hf download RLinf/Openvla-oft-SFT-libero10-trajall --local-dir ./Openvla-oft-SFT-libero10-trajall
         hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir ./RLinf-OpenVLAOFT-ManiSkill-Base-Lora

      **3. 启动评测（默认训练分布内场景）**

      .. code-block:: bash

         bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
           rollout.model.model_path=./Openvla-oft-SFT-libero10-trajall \
           rollout.model.lora_path=./RLinf-OpenVLAOFT-ManiSkill-Base-Lora

      覆盖 ``env.eval.init_params.id`` 与 ``env.eval.init_params.obj_set`` 即可评测任意 OOD 场景；传入 ``runner.ckpt_path`` 可加载 RL 训练产出的 ``.pt`` 权重。

   .. tab-item:: 批量 16 场景 OOD（mani-ood）

      **1. 安装**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero && source .venv/bin/activate

      **2. 下载资产与模型权重**

      .. code-block:: bash

         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets
         hf download RLinf/Openvla-oft-SFT-libero10-trajall --local-dir ./Openvla-oft-SFT-libero10-trajall
         hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir ./RLinf-OpenVLAOFT-ManiSkill-Base-Lora

      在 ``evaluations/maniskill/maniskill_ood_openvlaoft_eval.yaml`` 中设置 ``rollout.model.model_path: ./Openvla-oft-SFT-libero10-trajall`` 与 ``rollout.model.lora_path: ./RLinf-OpenVLAOFT-ManiSkill-Base-Lora``。

      **3. 批量评测全部 16 个 OOD 场景**

      .. code-block:: bash

         export EVAL_NAME=my_ood_eval
         export CKPT_PATH=./logs/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt
         export TOTAL_NUM_ENVS=16
         export EVAL_ROLLOUT_EPOCH=1
         bash evaluations/run_eval.sh mani-ood maniskill_ood_openvlaoft_eval

      批量日志保存在 ``logs/eval/<EVAL_NAME>/<时间戳>-<env_id>-<obj_set>/run_ppo.log``；``mani-ood`` 默认将 ``HF_ENDPOINT`` 设为 ``https://hf-mirror.com``，提前导出即可覆盖。

``evaluations/maniskill/`` 提供 OpenVLA-OFT 示例配置（基座模型 `RLinf/Openvla-oft-SFT-libero10-trajall <https://huggingface.co/RLinf/Openvla-oft-SFT-libero10-trajall>`_ + ManiSkill LoRA `RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora <https://huggingface.co/RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora>`_）。OpenVLA、OpenPI 或其他 ManiSkill 任务可参考 :ref:`从训练配置派生评测 <maniskill-derive-from-train>`。

.. _maniskill-eval-config:

.. dropdown:: 示例配置与评测配置详解

   ``evaluations/maniskill/`` 已有示例：

   .. list-table::
      :header-rows: 1
      :widths: 40 35 25

      * - 配置文件
        - 说明
        - 模型
      * - ``maniskill_ood_openvlaoft_eval.yaml``
        - OOD 泛化评测模板（默认训练场景）
        - OpenVLA-OFT

   **评测协议**

   ManiSkill OOD 在 ``env.eval.init_params`` 下通过 ``id`` 与 ``obj_set`` 指定场景，并汇总 ``eval/success_once``：

   - **训练分布内：** ``PutOnPlateInScene25Main-v3`` + ``obj_set=train``。
   - **分布外（OOD）：** 13 个变体环境 + ``obj_set=test``，按 Vision、Semantic、Execution 分类。
   - **补充评测：** ``mani-ood`` 模式还会以 ``obj_set=train`` 运行 3 个 Semantic 任务（合计 16 轮）。

   每条轨迹由 ``episode_id``\ （即 ``reset_state_id``）确定初始条件；开启 ``use_fixed_reset_state_ids=True`` 可复现固定初始状态，开启 ``auto_reset=True`` 时回合结束后按序分配下一个 ``episode_id``。

   **OOD 场景列表**

   .. list-table::
      :header-rows: 1
      :widths: 18 44 38

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

   **关键环境参数（``env.eval.init_params``）**

   .. list-table::
      :header-rows: 1
      :widths: 26 74

      * - 字段
        - 作用
      * - ``id``
        - ManiSkill 环境注册名，默认 ``PutOnPlateInScene25Main-v3``。
      * - ``obj_set``
        - 物体集合：``train``、``test`` 或 ``all``。OOD 场景用 ``test``，训练分布内用 ``train``。
      * - ``obs_mode``
        - 观测模式，VLA 评测使用 ``rgb+segmentation``。
      * - ``sim_backend``
        - 仿真后端，默认 ``gpu``，需要 NVIDIA GPU。
      * - ``policy_setup``
        - 动作空间配置，OpenVLA-OFT 使用 ``widowx_bridge``。

   **批量 ``mani-ood`` 环境变量**

   .. list-table::
      :header-rows: 1
      :widths: 28 72

      * - 变量
        - 说明
      * - ``EVAL_NAME``
        - 批量评测名称，日志写入 ``logs/eval/<EVAL_NAME>/``。
      * - ``CKPT_PATH``
        - RL 训练产出的 ``.pt`` 权重路径，覆盖 ``runner.ckpt_path``。
      * - ``TOTAL_NUM_ENVS``
        - 并行环境数（``env.eval.total_num_envs``）。
      * - ``EVAL_ROLLOUT_EPOCH``
        - 评测轮次（``env.eval.rollout_epoch``）。

   通用 ``env.eval`` 字段见 :doc:`../reference/configuration` 中的 :ref:`env-eval-fields`\ （示例通常设 ``max_episode_steps=80``、``max_steps_per_rollout_epoch=80`` 与 ``ignore_terminations=True``）。OpenVLA-OFT 还须在 ``rollout.model`` 下配置 ``unnorm_key: bridge_orig``、``is_lora: True``、``lora_path``、``add_value_head: True`` 与 ``max_prompt_length: 30``。

.. _maniskill-derive-from-train:

.. dropdown:: 进阶用法（单个 OOD 场景、并行规模与从训练配置派生）

   **单个 OOD 场景与 RL checkpoint**

   .. code-block:: bash

      bash evaluations/run_eval.sh maniskill maniskill_ood_openvlaoft_eval \
        env.eval.init_params.id=PutOnPlateInScene25VisionImage-v1 \
        env.eval.init_params.obj_set=test \
        env.eval.total_num_envs=32 \
        rollout.model.model_path=./Openvla-oft-SFT-libero10-trajall \
        rollout.model.lora_path=./RLinf-OpenVLAOFT-ManiSkill-Base-Lora \
        runner.ckpt_path=./logs/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt

   资源受限时，可在 ``auto_reset: True`` 下将 ``max_steps_per_rollout_epoch`` 设为 ``max_episode_steps`` 的整数倍（例如 ``total_num_envs: 16``、``max_episode_steps: 80``、``max_steps_per_rollout_epoch: 320``），让每轮串行覆盖更多 ``episode_id``。

   **从训练配置派生评测**

   ManiSkill 还支持 ``PickCube-v1``、``PutCarrotOnPlateInScene-v2`` 等训练任务（位于 ``examples/embodiment/config/env/``）。复制对应训练 YAML，设置 ``runner.task_type: embodied_eval`` 与 ``runner.only_eval: True``，移除 ``algorithm``、``actor`` 等训练段，仅保留 ``env.eval`` 与 ``rollout``。

结果与常见问题
--------------

终端输出 ``eval/success_once``，单次评测日志与视频保存在 ``logs/<timestamp>-<config>/``，批量评测日志保存在 ``logs/eval/<EVAL_NAME>/``\ （详见 :doc:`../reference/results`）。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 现象
     - 处理
   * - 缺少 ManiSkill 资产
     - 按安装与运行中的资产下载命令下载 ``RLinf/maniskill_assets``。
   * - GPU 仿真报错
     - ``sim_backend: gpu`` 需要 NVIDIA GPU。
   * - OpenVLA-OFT 加载失败
     - 设置 ``rollout.model.lora_path`` 指向 ``./RLinf-OpenVLAOFT-ManiSkill-Base-Lora``。
   * - RL checkpoint 未生效
     - 批量 ``mani-ood`` 模式通过 ``CKPT_PATH`` 传入 ``.pt`` 权重，单场景评测使用 ``runner.ckpt_path``。
   * - 始终在训练场景评测而非 OOD
     - 默认 YAML 指向 ``PutOnPlateInScene25Main-v3``；显式覆盖 ``env.eval.init_params.id`` 与 ``obj_set``，或使用 ``mani-ood`` 模式。
