RoboDojo 评测
=============

`RoboDojo <https://robodojo-benchmark.github.io/>`__ 是基于 Isaac Sim 的双臂操作基准，面向 ARX-X5 等机器人提供桌面整理、抓取、放置和工具使用任务。RLinf 将 RoboDojo 接入为 Gym 环境，用于 VLA 策略的 rollout 与评估，暂不支持 RL 训练。

概览
----------------------------------------

在 Isaac Sim 上运行 RoboDojo，并通过 RLinf 的统一环境接口驱动双臂机器人。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅（Pi_05）

   .. grid-item-card:: 算法
      :text-align: center

      评估（π₀.₅）

   .. grid-item-card:: 任务
      :text-align: center

      Put Bottles Into Dustbin

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 1+ GPU

| **你将完成：** 安装环境 → 下载场景资产和 Pi_05 检查点 → 转换检查点 → 运行评估。
| **前置条件：** :doc:`基础安装 </rst_source/start/installation>` · NVIDIA CUDA GPU · 至少 120 GB 可用磁盘空间。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 提供 ``put_bottles_into_dustbin`` 的环境配置，用作接入和验证示例。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 任务
     - 描述
   * - ``put_bottles_into_dustbin``
     - 使用双臂抓取桌上的瓶子并放入垃圾桶。

对应配置位于 ``examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml``。RoboDojo 的其他任务使用相同的环境接口，修改 ``task_config.task_name`` 和对应的任务参数即可切换。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - ``main_images`` 为头部相机 RGB；``wrist_images`` 为可选的左右腕部相机 RGB；``states`` 为 ``[B, 14]`` float32 本体状态，包含左右机械臂关节和夹爪状态。
   * - 动作
     - 默认 14 维连续动作 ``actions``：左臂 6 维、右臂 6 维、左夹爪 1 维、右夹爪 1 维。
   * - 奖励
     - 默认根据任务成功状态生成稀疏奖励。
   * - 提示词
     - RoboDojo 根据当前场景生成的自然语言指令，存放在 ``task_descriptions``。

安装与运行
----------

.. include:: _setup_common.rst

RoboDojo 支持 Linux x86_64 与 NVIDIA CUDA GPU（暂不支持纯 CPU、macOS 或 AMD ROCm）；安装前先用 ``nvidia-smi``、``uname -m`` 与 ``df -h .`` 检查驱动和磁盘空间。安装前导出 ``ROBODOJO_ASSETS_ROOT``，安装器会将其写入 ``.venv/bin/activate``\ （未设置时默认为 RLinf 仓库根目录下的 ``data/robodojo``）。安装器会设置 ``OMNI_KIT_ACCEPT_EULA=YES``，继续安装即表示接受 NVIDIA Omniverse 许可条款；不要单独执行 ``pip install rlinf-robodojo-runtime``。

.. tab-set::

   .. tab-item:: OpenPI 评测
      :sync: openpi

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export ROBODOJO_ASSETS_ROOT=$PWD/data/robodojo
         bash requirements/install.sh embodied --model openpi --env robodojo && source .venv/bin/activate

      **2. 下载资源与模型**

      从 ``RoboDojo-Benchmark/RoboDojo`` 下载场景资产（``Assets/``，约 39 GB）与官方 Pi_05 检查点到 ``$ROBODOJO_ASSETS_ROOT``，并将 ``59999/`` 下的原始 JAX/orbax 快照转换为 OpenPI 格式（``model.safetensors`` 与 ``assets/norm_stats.json``）：

      .. code-block:: bash

         export ROBODOJO_ASSETS_ROOT=$PWD/data/robodojo
         hf download RoboDojo-Benchmark/RoboDojo \
           --repo-type dataset \
           --local-dir "$ROBODOJO_ASSETS_ROOT" \
           --include "Assets/**" \
           --include "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/**"

         CKPT=$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999
         python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
           --input-model "$CKPT" \
           --input-norm-stats "$CKPT/assets/arx_x5_sim/norm_stats.json" \
           --output-model "$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05_robodojo_arx_x5" \
           --output-norm-stats "$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05_robodojo_arx_x5/assets/norm_stats.json" \
           --action-dim 32 \
           --action-horizon 50

      ``ROBODOJO_ASSETS_ROOT`` 必须指向包含 ``Assets/`` 的父目录，不能指向 ``Assets/`` 本身。务必保留 ``--include`` 过滤且只下载单个 run：完整仓库包含数 TB 数据，三次 ``Pi_05`` run（``joint-0``/``1``/``2``）合计约 70 GB。``ckpt/RoboDojo/Pi_0/`` 下为对应的 π₀ 检查点。

      **3. 启动评测**

      ``evaluations/robodojo/robodojo_put_bottles_into_dustbin_pi05_eval.yaml`` 复用 ``examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml``\ （14 维动作 ``[6, 6, 1, 1]``、``state_action_order: aloha``），并自动从 ``rollout.model.model_path`` 推导 ``norm_stats_path``。启动评估：

      .. code-block:: bash

         export ROBODOJO_ASSETS_ROOT=$PWD/data/robodojo
         bash evaluations/run_eval.sh robodojo \
           robodojo_put_bottles_into_dustbin_pi05_eval \
           rollout.model.model_path="$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05_robodojo_arx_x5"

      日志写入 ``logs/<时间戳>-<配置名>/``，结果保存在 ``runner.logger.log_path``。

   .. tab-item:: 仅安装环境
      :sync: env-only

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         export ROBODOJO_ASSETS_ROOT=$PWD/data/robodojo
         bash requirements/install.sh embodied --env robodojo && source .venv/bin/activate

      **2. 下载资源**

      下载场景资产（``Assets/``，约 39 GB）到 ``$ROBODOJO_ASSETS_ROOT``\ （务必保留 ``--include``，避免拉取数 TB 完整仓库）：

      .. code-block:: bash

         export ROBODOJO_ASSETS_ROOT=$PWD/data/robodojo
         hf download RoboDojo-Benchmark/RoboDojo \
           --repo-type dataset \
           --local-dir "$ROBODOJO_ASSETS_ROOT" \
           --include "Assets/**"

      **3. 验证环境**

      运行以下 Python 脚本，启动单个环境验证 Isaac Sim、相机与资产：

      .. code-block:: python

         import numpy as np
         from robodojo_runtime.bridge import VectorEnv

         env = VectorEnv(
             {
                 "task_name": "put_bottles_into_dustbin",
                 "env_cfg_type": "arx_x5",
                 "headless": True,
                 "cuda_device": 0,
                 "max_episode_steps": 700,
             },
             n_envs=1,
             env_seeds=[0],
         )
         try:
             env.reset()
             obs = env.get_obs()[0]
             print(obs["full_image"].shape, obs["state"].shape)
             env.step(np.zeros((1, 1, 14), dtype=np.float32))
         finally:
             env.close()

.. dropdown:: 进阶配置（可选）

   - **固定版本与兼容性覆盖**：``requirements/install.sh`` 与 ``requirements/embodied/envs/robodojo-override.txt`` 统一锁定 Python 3.11、RoboDojo runtime 0.3.0（``6d76a0c``）、PyTorch 2.7.0（cu126，CUDA 12.8+ 用 cu128）、TorchVision 0.22.0 / TorchAudio 2.7.0、Isaac Sim 5.1.0 与 IsaacLab 0.54.3（``afca7b09``），不要手工升级存在重叠版本约束的包。

   - **扩展多环境并行**：默认运行单个环境；稳定后可通过 Hydra 覆盖提高并行度，例如 ``env.eval.total_num_envs=4``、``cluster.component_placement.env=0-3`` 与 ``cluster.component_placement.rollout=0-3``。

   - **使用 ModelScope 下载**：执行 ``uv pip install "modelscope>=1.40.0"`` 后，使用相同的 ``--repo-type dataset``、``--local-dir`` 与 ``--include`` 参数运行 ``modelscope download``。

   - **验证依赖与 CUDA**\ （不启动仿真器）：

     .. code-block:: bash

        python - <<'PY'
        import torch
        from robodojo_runtime import assets_root, source_root
        from robodojo_runtime.bridge import VectorEnv

        print("torch:", torch.__version__)
        print("cuda:", torch.cuda.is_available())
        print("source:", source_root())
        print("assets:", assets_root())
        PY

   - **单环境快速验证**：开启多环境前先用单环境验证 Isaac Sim、相机与资产（脚本见 **仅安装环境** 标签页的第 3 步）。

可视化与结果
----------------------------------------

执行 ``tensorboard --logdir ./logs --port 6006`` 启动 TensorBoard，关注任务成功率指标 ``eval/success_once``。传入 ``env.eval.video_cfg.save_video=True`` 可保存评估视频，完整指标定义见 :doc:`训练指标 <../../reference/metrics>`。

.. dropdown:: 常见问题

   .. list-table::
      :header-rows: 1
      :widths: 34 66

      * - 现象
        - 排查与修复
      * - 安装后找不到 ``robodojo_runtime``
        - 确认已经执行 ``source .venv/bin/activate``。如果安装时使用了 ``--venv``，激活对应的虚拟环境。
      * - 依赖解析失败
        - 不要手工安装 Isaac Sim、IsaacLab 或 RoboDojo runtime。删除失败的环境后重新执行 ``install.sh``，让安装器应用固定版本和兼容性覆盖。
      * - ``pip check`` 报告 Isaac Sim 与 IsaacLab 版本不一致
        - 这些包使用互相覆盖的精确版本 pin。保留安装器选定的运行验证组合，不要手工升级 ``starlette``、``uvicorn`` 或 ``typing-extensions``。
      * - ``torch.cuda.is_available()`` 为 ``False``
        - 先运行 ``nvidia-smi`` 检查驱动和 GPU 是否可见。RoboDojo 暂不支持纯 CPU、macOS 或 AMD ROCm。
      * - 找不到 ``Assets``、``cam_head`` 或任务布局
        - 检查 ``ROBODOJO_ASSETS_ROOT`` 是否指向 ``Assets/`` 的父目录，并确认四个必需子目录都存在。资产下载不完整时，删除未完成的目录后重新下载。
      * - 启动后显存不足
        - 先把环境数降为 1，并关闭其他评估或推理进程。RoboDojo 会为每个并行环境启动独立的 Isaac Sim 进程，多环境配置需要根据单环境的实际显存占用调整。
      * - 进程结束后显存未释放
        - 在代码中使用 ``try/finally`` 调用 ``env.close()``。RoboDojo 适配器只支持完整关闭和重建，不支持挂起后恢复。
