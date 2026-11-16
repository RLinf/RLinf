LingbotVLA 2.0 模型强化学习
========================================

.. figure:: https://raw.githubusercontent.com/robbyant/lingbot-vla-v2/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/assets/lingbot_vla2_framework.png
   :align: center
   :width: 90%

   LingbotVLA 2.0 框架（图片来源：`Robbyant <https://github.com/robbyant/lingbot-vla-v2>`__）。

用 SFT、PPO 或 GRPO 在 RoboTwin 2.0 上微调 `LingbotVLA 2.0 <https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin>`__ 的动作专家。RLinf **原生**\ 加载官方发布的 RoboTwin checkpoint，并以它作为训练起点。LingbotVLA 2.0 是基于 Qwen3-VL 底座的视觉-语言-动作模型：它的混合专家（MoE）动作专家通过 flow matching 在 55 维的 latent 动作空间中生成 50 步动作块，再由机器人配置映射为 RoboTwin 的 14 维关节指令。三种方法都只更新动作专家，Qwen3-VL 底座和动作专家的 MoE 路由器保持冻结。

概览
----------------------------------------

用 SFT、PPO 或 GRPO 训练 LingbotVLA 2.0，完成 RoboTwin 2.0 的 ``click_bell`` 任务。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      RoboTwin 2.0

   .. grid-item-card:: 算法
      :text-align: center

      SFT, PPO, GRPO

   .. grid-item-card:: 任务
      :text-align: center

      Click Bell

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 4 GPU（48 GB）

| **你将完成：** 原生安装 → 克隆 RoboTwin 与资产 → 下载检查点 → 填写模型路径 → SFT、PPO 或 GRPO → 在 64 个 seeds 上评测。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · RoboTwin 仓库与资产 · LingbotVLA 2.0 与 Qwen3-VL 检查点 · SFT 还需要 ``click_bell`` 示范数据（见下文步骤）。

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
   * - RoboTwin
     - Click Bell
     - ``robotwin_click_bell_grpo_lingbotvlav2``
     - 从官方 checkpoint 或 SFT checkpoint 出发做 GRPO 训练。
   * - RoboTwin
     - Click Bell
     - ``robotwin_sft_lingbotvlav2``
     - 在 50 条干净的 ``click_bell`` 示范上做 SFT。

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

然后把所运行配置中 ``actor.model`` 下的 ``/path/to`` 占位路径换成本地路径；评测配置则填写 ``rollout.model`` 下的路径。任何路径不存在时，模型都会在启动时报错并指出对应字段：

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
* **GRPO（强化学习）**：``examples/embodiment/config/robotwin_click_bell_grpo_lingbotvlav2.yaml``
* **SFT（示范数据）**：``examples/sft/config/robotwin_sft_lingbotvlav2.yaml``

环境变量
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

该任务使用双臂机器人，启动强化学习训练或评测前，请在终端中声明机器人平台为 ALOHA，否则环境将无法正确加载动作空间：

.. code-block:: bash

    export ROBOT_PLATFORM="ALOHA"
    export ROBOTWIN_PATH=/path/to/RoboTwin
    export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH

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

从示范数据训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

通过 SFT 的 flow matching 目标拟合示范动作块。表中的 SFT 结果使用 `RoboTwin 2.0 <https://huggingface.co/datasets/TianxingChen/RoboTwin2.0>`__ 中 50 条干净的 ``click_bell`` 示范（``dataset/click_bell/aloha-agilex_clean_50.zip``），转换为 LeRobot v3.0 数据集后共 3,855 帧，包含 14 维关节状态与动作，以及 480×640 的头部和左右腕部相机画面。`tailong-wu/STARVLA_RoboTwin-Clean_click_bell_lerobotv30 <https://huggingface.co/datasets/tailong-wu/STARVLA_RoboTwin-Clean_click_bell_lerobotv30>`__ 格式相同。换用其他数据集时，相机、状态和动作特征必须与机器人配置一致。

.. code-block:: bash

    hf download tailong-wu/STARVLA_RoboTwin-Clean_click_bell_lerobotv30 --repo-type dataset \
      --local-dir click_bell_clean_50

将 SFT 配置的 ``data.train_data_paths`` 设为数据集目录，即包含 ``meta/info.json`` 的那一层，并填写 ``actor.model`` 下的模型路径。然后在四张 GPU 上开始训练：

.. code-block:: bash

    bash examples/sft/run_vla_sft.sh robotwin_sft_lingbotvlav2

该配置冻结底座和路由器，只训练动作专家：共 600 步，每步 64 个样本，学习率 1e-5，每 100 步保存一次 checkpoint。cosine 调度跨越 2,000 步，所以训练结束时学习率仍接近峰值。观察 ``train/flow_mse`` 与 ``train/grad_norm``；示范数据上的 loss 不能预测任务成功率，需要在 RoboTwin 上评测 checkpoint。

续训时，将 ``runner.resume_dir`` 设为已保存的 ``checkpoints/global_step_N`` 目录。续训沿用相同的数据顺序和随机状态，记录的 loss 与从未中断的训练一致。

用分组奖励训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GRPO 在每组八条 rollout 内比较回报。配置使用 ``algorithm.adv_type: grpo``、``algorithm.loss_type: actor`` 和 ``actor.model.add_value_head: false``。任务、初始模型路径、64 个训练环境、动作块长度及 actor 学习率与 PPO 对齐。

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh robotwin_click_bell_grpo_lingbotvlav2

执行前，先导出 `环境变量`_ 中的变量。检查组内 rollout 是否产生不同回报和非零 advantage：policy 更新来自组内回报的差异。比较 PPO 与 GRPO checkpoint 时，使用相同的评测 seeds 和轨迹数量。

从 SFT checkpoint 开始 GRPO 时，在 GRPO 配置中将 ``runner.ckpt_path`` 设为 SFT 保存的完整权重文件。SFT 将它保存在日志路径下的 experiment 目录中：

.. code-block:: yaml

    runner:
      ckpt_path: logs/<timestamp>-robotwin_sft_lingbotvlav2/robotwin_sft_lingbotvlav2/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt

SFT 与 GRPO 都以 ``add_value_head: false`` 构建模型，因此 actor 和 rollout 均可严格加载这份权重。

用 PPO 训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

导出 `环境变量`_ 中的变量后，在 RoboTwin Click Bell 任务上启动 PPO 训练：

.. code-block:: bash

    bash examples/embodiment/run_embodiment.sh robotwin_click_bell_ppo_lingbotvlav2

续训时，在训练配置中将 ``runner.resume_dir`` 设为已保存的 checkpoint 目录。

独立评测
----------------------------------------

通过 :doc:`RoboTwin 评测指南 <../../evaluations/guides/robotwin>` 运行独立评测，配置为 ``evaluations/robotwin/robotwin_click_bell_lingbotvlav2_eval.yaml``。在新的终端中评测前，先导出 `环境变量`_ 中的三个变量。下面的命令在结果表所用的 64 个固定 seeds 上评测 checkpoint，rollout 与环境放在同一张 GPU 上：

.. code-block:: bash

    bash evaluations/run_eval.sh robotwin_click_bell_lingbotvlav2_eval \
      runner.ckpt_path=/path/to/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt \
      env.eval.total_num_envs=64 cluster.component_placement.env=0

去掉 ``runner.ckpt_path`` 即评测官方权重。评测 SFT 或 GRPO checkpoint 时，同时传入 ``rollout.model.add_value_head=false``，使严格加载使用与训练时一致的模型结构。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ./logs --port 6006

关键指标是 ``env/success_once``。完整指标说明见 :doc:`训练指标 <../../reference/metrics>`。

在四张 A100 80 GB GPU 上，所有 checkpoint 使用相同的 64 个固定环境 seeds 评测，每条轨迹最多 400 步，``rollout.model.eval_seed=0``：

.. list-table:: LingbotVLA 2.0 在 RoboTwin 任务上的评测结果（n=64）
   :header-rows: 1

   * - 任务
     - 官方权重
     - SFT（第 600 步）
     - RLinf-GRPO（第 100 步）
     - RLinf-PPO（第 100 步）
   * - ``click_bell``
     - 76.56%
     - 92.19%
     - **100.00%**
     - **100.00%**

比较 checkpoint 时，保持环境 seeds、worker 放置、batch size 和 ``rollout.model.eval_seed`` 一致。评测的初始噪声由 ``eval_seed`` 决定，不受模型初始化影响。
