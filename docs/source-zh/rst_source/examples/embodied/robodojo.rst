基于 RoboDojo 的强化学习训练
========================================

`RoboDojo <https://robodojo-benchmark.github.io/>`__ 是基于 Isaac Sim 的双臂\
操作基准，面向 ARX-X5 等机器人提供桌面整理、抓取、放置和工具使用任务。RLinf\
将 RoboDojo 接入为 Gym 环境，可以直接用于 VLA 策略的 rollout 和评估。

概览
----------------------------------------

在 Isaac Sim 上运行 RoboDojo，并通过 RLinf 的统一步态接口驱动双臂机器人。

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

| **你将完成：** 安装环境 → 下载场景资产和 Pi_05 检查点 → 验证安装 → 启动单环境任务。
| **前置条件：** :doc:`基础安装 </rst_source/start/installation>` · NVIDIA CUDA GPU ·
  至少 120 GB 可用磁盘空间。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 当前提供 ``put_bottles_into_dustbin`` 的环境配置，用作接入和验证示例。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 任务
     - 描述
   * - ``put_bottles_into_dustbin``
     - 使用双臂抓取桌上的瓶子并放入垃圾桶。

对应配置位于：

.. code-block:: text

   examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml

RoboDojo 的其他任务使用相同的环境接口，只需修改 ``task_config.task_name`` 和对应的任务参数。

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

版本锁定
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 使用 ``requirements/install.sh`` 和 ``requirements/embodied/envs/robodojo-override.txt`` 统一锁定 RoboDojo 的模拟器、PyTorch 和兼容性依赖。当前组合如下：

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 组件
     - 固定版本
   * - Python
     - 3.11
   * - RoboDojo runtime
     - 0.3.0 · 提交 ``6d76a0c``
   * - PyTorch
     - 2.7.0 · 按驱动选择 wheel（默认 cu126，CUDA 12.8+ 用 cu128）
   * - TorchVision / TorchAudio
     - 0.22.0 / 2.7.0 · 与 PyTorch 使用同一 wheel 索引
   * - Isaac Sim
     - 5.1.0
   * - IsaacLab
     - 0.54.3 · 提交 ``afca7b09``

安装
----------------------------------------

RoboDojo 当前支持 Linux x86_64 和 NVIDIA CUDA GPU，暂不支持纯 CPU、macOS\
或 AMD ROCm。安装前先确认驱动和磁盘空间：

.. code-block:: bash

   nvidia-smi
   uname -m
   df -h .

**自定义环境**

在 RLinf 仓库根目录安装 RoboDojo 和所需模拟器依赖：

.. code-block:: bash

   bash requirements/install.sh embodied --env robodojo
   source .venv/bin/activate

国内网络可以添加 ``--use-mirror``。需要 OpenPI 时使用：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robodojo

安装器会自动安装 PyTorch 2.7、Isaac Sim 5.1 和固定版本的 IsaacLab，并对齐 Isaac Sim 与 IsaacLab 的依赖版本。不要单独执行 ``pip install rlinf-robodojo-runtime``，否则可能得到无法启动的依赖组合。

.. note::

   安装器会设置 ``OMNI_KIT_ACCEPT_EULA=YES``。继续执行安装即表示接受\
   NVIDIA Omniverse 的相关许可条款。

下载场景资产
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

依赖安装不会自动下载场景资产。请在执行安装前导出 ``ROBODOJO_ASSETS_ROOT``，安装器会将该值写入 ``.venv/bin/activate``；未设置时记录的是 RLinf 仓库根目录下的 ``data/robodojo``。然后下载约 39 GB 的 ``Assets/``：

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   source .venv/bin/activate
   uv pip install "huggingface_hub>=0.34,<2"
   hf download RoboDojo-Benchmark/RoboDojo \
     --repo-type dataset \
     --local-dir "$ROBODOJO_ASSETS_ROOT" \
     --include "Assets/**"

国内网络也可以改用 ModelScope 下载同一份资产：

.. code-block:: bash

   uv pip install "modelscope>=1.40.0"
   modelscope download RoboDojo-Benchmark/RoboDojo \
     --repo-type dataset \
     --local-dir "$ROBODOJO_ASSETS_ROOT" \
     --include "Assets/**"

.. note::

   仓库包含 ``ckpt/`` 和 ``data/``，必须保留 ``--include "Assets/**"``，否则会下载数 TB 的无关文件。

也可以把已有资产放到该目录。正确结构是：

.. code-block:: text

   /path/to/robodojo/
   └── Assets/
       ├── Robots/
       ├── Object/
       ├── Material/
       └── Eval_Layout/

.. note::

   ``ROBODOJO_ASSETS_ROOT`` 应指向包含 ``Assets/`` 的父目录，而不是 ``Assets/`` 本身。使用 Hugging Face 镜像时，可在下载前设置 ``HF_ENDPOINT``。

验证安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下命令检查 Python 包和 CUDA，不会启动仿真器：

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

下载模型
----------------------------------------

评估配置使用 RoboDojo 官方发布的 Pi_05 检查点，即基于 ``RoboDojo-sim-arx_x5`` 仿真数据训练的 π₀.₅ policy。它和场景资产放在同一个 ``RoboDojo-Benchmark/RoboDojo`` 仓库里，位于 ``ckpt/RoboDojo/Pi_05/``，每次训练 run 一个目录。示例配置使用其中的 ``joint-0``：

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   source .venv/bin/activate
   hf download RoboDojo-Benchmark/RoboDojo \
     --repo-type dataset \
     --local-dir "$ROBODOJO_ASSETS_ROOT" \
     --include "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/**"

下载得到的是 JAX/orbax 原始检查点：

.. code-block:: text

   $ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/
   └── 59999/
       ├── _CHECKPOINT_METADATA
       ├── assets/arx_x5_sim/norm_stats.json
       ├── params/
       └── train_state/

RLinf 的 OpenPI 加载器读取 OpenPI 格式的检查点（``model.safetensors``），因此需要先转换。转换器依赖 ``jax`` 和 ``orbax-checkpoint``，由 ``--model openpi`` 安装：

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   source .venv/bin/activate
   CKPT=$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model "$CKPT" \
     --input-norm-stats "$CKPT/assets/arx_x5_sim/norm_stats.json" \
     --output-model "$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05_robodojo_arx_x5" \
     --output-norm-stats "$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05_robodojo_arx_x5/arx_x5_sim/norm_stats.json" \
     --action-dim 32 \
     --action-horizon 50

转换后的目录包含 ``model.safetensors`` 和 ``arx_x5_sim/norm_stats.json``。``rollout.model.model_path`` 应指向该目录，norm stats 路径会由它推导。同级的 ``joint-1``、``joint-2`` 是同一检查点的另外两次训练 run，``ckpt/RoboDojo/Pi_0/`` 下是对应的 π₀ 检查点。

.. note::

   只下载一个 run。整个 ``Pi_05`` 目录包含三次 run，约 70 GB，而评估只加载其中一个。

运行
----------------------------------------

**单环境启动**

第一次运行建议使用一个环境，确认 Isaac Sim、相机和资产都正常：

.. code-block:: bash

   python - <<'PY'
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
   PY

首次启动时不要同时运行策略服务、视频录制和多个环境。确认单环境能够完成 reset、\
step 和 close 后，再提高 ``env.train.total_num_envs``。

**接入训练配置**

训练配置可以直接引用：

.. code-block:: text

   examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml

策略的 ``action_dim`` 应与 ``action_dims`` 的总和一致。示例配置使用 ``[6, 6, 1, 1]``，即 14 维动作。

**用 Pi_05 检查点做策略评估**

``evaluations/robodojo/robodojo_put_bottles_into_dustbin_pi05_eval.yaml`` 复用\
上面的环境配置，用 OpenPI Pi_05 检查点评估 ``put_bottles_into_dustbin``。\
策略依赖需要按上文一并安装：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robodojo
   source .venv/bin/activate

安装完成后启动评估：

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   bash evaluations/run_eval.sh robodojo \
     robodojo_put_bottles_into_dustbin_pi05_eval

脚本会在 ``logs/<时间戳>-<配置名>/`` 下记录日志，结果写入 ``runner.logger.log_path`` 指定的目录。默认只运行一个环境；确认稳定后可以用 Hydra 覆盖提高并行度，例如 ``env.eval.total_num_envs=4`` 配合 ``cluster.component_placement.env=0-3`` 和 ``cluster.component_placement.rollout=0-3``。

14 维状态/动作的布局由 ``state_action_order`` 决定。官方 Pi_05 检查点使用 ALOHA 顺序 ``[左臂, 左夹爪, 右臂, 右夹爪]``，因此评估配置设置 ``state_action_order: aloha``。

评估配置中的 ``rollout.model.model_path`` 只是占位路径。``rollout.model.openpi_data.norm_stats_path`` 默认由它推导，因此只需覆盖 ``model_path``：

.. code-block:: bash

   bash evaluations/run_eval.sh robodojo \
     robodojo_put_bottles_into_dustbin_pi05_eval \
     rollout.model.model_path="$ROBODOJO_ASSETS_ROOT/ckpt/RoboDojo/Pi_05_robodojo_arx_x5"

可视化与结果
----------------------------------------

训练启动方式与 RLinf 的其他 embodied 环境一致。关键指标为 ``env/success_once``，完整指标说明见 :doc:`训练指标 <../../reference/metrics>`。

如需保存 rollout 视频，可在环境配置中启用：

.. code-block:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

常见问题
----------------------------------------

**安装后找不到 ``robodojo_runtime``**

确认已经执行 ``source .venv/bin/activate``。如果安装时使用了 ``--venv``，\
请激活对应的虚拟环境。

**依赖解析失败**

不要手工安装 Isaac Sim、IsaacLab 或 RoboDojo runtime。删除失败的安装结果，\
重新执行本文的安装命令，让安装器应用固定版本和兼容性覆盖。

**``pip check`` 报告 Isaac Sim 与 IsaacLab 版本不一致**

这些包使用互相覆盖的精确版本 pin。安装器选择的是已经过运行验证的组合；\
不要手工升级 ``starlette``、``uvicorn``、``typing-extensions`` 等包。

**``torch.cuda.is_available()`` 为 ``False``**

先运行 ``nvidia-smi`` 检查驱动和 GPU 是否可见。RoboDojo 当前不支持纯 CPU、\
macOS 或 AMD ROCm。

**找不到 ``Assets``、``cam_head`` 或任务布局**

检查 ``ROBODOJO_ASSETS_ROOT`` 是否指向 ``Assets/`` 的父目录，并确认四个必需子目录\
都存在。资产下载不完整时，删除未完成的目录后重新下载。

**启动后显存不足**

先把环境数降为 1，并关闭其他训练或推理进程。RoboDojo 会为每个并行环境启动独立的\
Isaac Sim 进程，多环境配置需要根据单环境的实际显存占用调整。

**进程结束后显存未释放**

在代码中使用 ``try/finally`` 调用 ``env.close()``。RoboDojo 适配器只支持完整关闭，\
不支持挂起后恢复。
