LingbotVLA 2.0 模型强化学习
========================================

.. figure:: https://raw.githubusercontent.com/robbyant/lingbot-vla-v2/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/assets/lingbot_vla2_framework.png
   :align: center
   :width: 90%

   LingbotVLA 2.0 框架（图片来源：`Robbyant <https://github.com/robbyant/lingbot-vla-v2>`__）。

用 PPO 在 RoboTwin 2.0 上微调 `LingbotVLA 2.0 <https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin>`__ 的动作专家。RLinf **原生**\ 加载官方发布的 RoboTwin checkpoint，并以它作为训练起点。LingbotVLA 2.0 是基于 Qwen3-VL 底座的视觉-语言-动作模型：它的混合专家（MoE）动作专家通过 flow matching 在 55 维的 latent 动作空间中生成 50 步动作块，再由机器人配置映射为 RoboTwin 的 14 维关节指令。PPO 只更新动作专家，Qwen3-VL 底座和动作专家的 MoE 路由器保持冻结。

概览
----------------------------------------

用 PPO 对 LingbotVLA 2.0 进行强化学习，完成 RoboTwin 2.0 的 ``click_bell`` 任务。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      RoboTwin 2.0

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      Click Bell

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 4 GPU（48 GB）

| **你将完成：** 原生安装 → 克隆 RoboTwin 与资产 → 下载检查点 → 填写模型路径 → PPO → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · RoboTwin 仓库与资产 · LingbotVLA 2.0 与 Qwen3-VL 检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - RoboTwin
     - Click Bell
     - ``robotwin_click_bell_ppo_lingbotvlav2``
     - 从官方 RoboTwin checkpoint 出发，用 PPO 训练动作专家。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 38

   * - 字段
     - 说明
   * - Observation
     - 头部相机、左右腕部相机，以及 14 维关节状态。
   * - Action
     - 每步 14 维关节指令，由 50 步、每步 55 维的 latent 动作块解码得到。
   * - Reward
     - RoboTwin 任务成功信号。
   * - Prompt
     - RoboTwin episode 的自然语言任务指令。

安装
----------------------------------------

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

首先克隆 RLinf 仓库并进入主目录：

.. code-block:: bash

    git clone https://github.com/RLinf/RLinf.git
    cd RLinf
    export RLINF_PATH=$(pwd)

2. 安装依赖
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LingbotVLA 1.0 与 2.0 安装的 Python 包都叫 ``lingbotvla``，因此需要各自使用独立的虚拟环境。

**选项 1：Docker 镜像**

RoboTwin 镜像中包含 ``lingbotvlav2`` 环境。在仓库根目录构建镜像：

.. code-block:: bash

    docker buildx build --load -f docker/Dockerfile \
      --build-arg BUILD_TARGET=embodied-robotwin \
      -t rlinf:embodied-robotwin .

进入容器后，通过内置的 ``switch_env`` 工具切换环境：

.. code-block:: bash

    source switch_env lingbotvlav2

LingbotVLA 2.0 的源码位于 ``/opt/venv/lingbotvlav2/lingbot-vla-v2``。

**选项 2：自定义环境**

一条命令安装 LingbotVLA 2.0 环境与 RoboTwin 依赖：

.. code-block:: bash

    bash requirements/install.sh embodied --model lingbotvlav2 --env robotwin
    source .venv/bin/activate

安装脚本会把 `RLinf 维护的 LingbotVLA 2.0 fork <https://github.com/RLinf/lingbot-vla-v2/tree/rlinf>`__ 的 ``rlinf`` 分支克隆到 ``.venv/lingbot-vla-v2``，该分支补充了 RLinf PPO 所需的接口。如需放在其他位置，请在安装前设置 ``LINGBOT_VLA_V2_PATH``。

RoboTwin 仓库克隆与资产下载
----------------------------------------

RoboTwin Assets 是 RoboTwin 环境运行所需的资源文件，需要从 HuggingFace 下载。

.. code-block:: bash

   # 1. 克隆 RoboTwin 仓库
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support

   # 2. 下载并解压 Assets 文件
   cd RoboTwin
   bash script/_download_assets.sh

下载模型
----------------------------------------

从 HuggingFace 下载 LingbotVLA 2.0 的 RoboTwin checkpoint 与 Qwen3-VL 底座模型：

.. code-block:: bash

    hf download robbyant/lingbot-vla-v2-6b-robotwin --local-dir lingbot-vla-v2-6b-robotwin
    hf download Qwen/Qwen3-VL-4B-Instruct --local-dir Qwen3-VL-4B-Instruct

然后在 ``examples/embodiment/config/model/lingbotvlav2.yaml`` 中填写空白的路径字段。训练与独立评测都读取这个文件，任何路径留空时，模型都会在启动时报错并指出对应字段：

.. code-block:: yaml

    model_path: /path/to/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt
    tokenizer_path: /path/to/Qwen3-VL-4B-Instruct
    lingbotvlav2:
      training_config_path: /path/to/lingbot-vla-v2-6b-robotwin/lingbotvla_cli.yaml
      robot_config_path: /path/to/RLinf/.venv/lingbot-vla-v2/configs/robot_configs/robotwin.yaml
      stats_path: /path/to/RLinf/.venv/lingbot-vla-v2/assets/norm_stats/robotwin.json

在 Docker 镜像中，``robot_config_path`` 与 ``stats_path`` 改以 ``/opt/venv/lingbotvlav2/lingbot-vla-v2`` 开头。

运行
----------------------------------------

配置文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **PPO（强化学习）**：``examples/embodiment/config/robotwin_click_bell_ppo_lingbotvlav2.yaml``
* **模型默认配置**，与独立评测共用：``examples/embodiment/config/model/lingbotvlav2.yaml``

关键配置片段 (PPO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

训练 rollout 在每个动作 chunk 中随机选一个去噪步按 flow-SDE 采样，其余步沿 ODE 推进。PPO 只为这一步计算概率，且只计入机器人变换解码为 RoboTwin 控制量的 14 个 latent 坐标；每个坐标一个比率（``token_level``），并启用 dual-clip（``clip_ratio_c``）。奖励与价值为 chunk 级，``critic_warmup_steps`` 在 policy 更新前先单独训练新的价值头若干个优化器步。

actor 以 fp32 保存可训练的动作专家权重，因为在当前学习率下，bf16 存储会把优化器的更新舍入掉。FSDP2 仍以 bf16 计算，rollout 保留一份 bf16 权重。

.. code-block:: yaml

    algorithm:
      reward_type: chunk_level
      logprob_type: token_level
      clip_ratio_c: 3.0
      recompute_logprobs: true
      max_recompute_logprob_gap: 1.0e-3

    rollout:
      model:
        precision: bf16

    actor:
      model:
        precision: fp32
      optim:
        critic_warmup_steps: 20
      fsdp_config:
        strategy: "fsdp2"
        mixed_precision:
          param_dtype: bf16
          reduce_dtype: fp32
          cast_forward_inputs: false

每次更新前，actor 会重新为 rollout 采样的每个 SDE 步计算概率。若某个样本在 rollout 与 actor 之间的总 log-probability 差值超过 ``algorithm.max_recompute_logprob_gap``，训练会停止；``actor/rollout_train_joint_logprob_gap_max`` 记录最大差值。

启动命令
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**注意**：该任务使用双臂机器人，启动前请在终端中声明机器人平台为 ALOHA，否则环境将无法正确加载动作空间：

.. code-block:: bash

    export ROBOT_PLATFORM="ALOHA"
    export ROBOTWIN_PATH=/path/to/RoboTwin
    export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH

然后在 RoboTwin Click Bell 任务上启动 PPO 训练：

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh robotwin_click_bell_ppo_lingbotvlav2

续训时，在训练配置中将 ``runner.resume_dir`` 设为已保存的 checkpoint 目录。

独立评测
----------------------------------------

通过 :doc:`RoboTwin 评测指南 <../../evaluations/guides/robotwin>` 运行独立评测，配置为 ``robotwin_click_bell_lingbotvlav2_eval``。在新的终端中评测前，先导出 `启动命令`_ 中的三个环境变量。如需评测 PPO checkpoint 而不是官方权重，传入其完整权重文件：

.. code-block:: bash

    bash evaluations/run_eval.sh robotwin_click_bell_lingbotvlav2_eval \
      runner.ckpt_path=/path/to/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ./logs --port 6006

关键指标是 ``env/success_once``。完整指标说明见 :doc:`训练指标 <../../reference/metrics>`。
