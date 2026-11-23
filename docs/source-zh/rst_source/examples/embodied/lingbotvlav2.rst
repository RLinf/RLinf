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

从 Hugging Face 下载 RoboTwin 后训练 checkpoint 和 Qwen3-VL tokenizer：

.. code-block:: bash

    hf download robbyant/lingbot-vla-v2-6b-robotwin --local-dir lingbot-vla-v2-6b-robotwin
    hf download Qwen/Qwen3-VL-4B-Instruct --local-dir Qwen3-VL-4B-Instruct

设置 RoboTwin 发布目录和 tokenizer 路径。PPO 与评测共用的配置会从该目录读取后训练权重和 ``lingbotvla_cli.yaml``：

.. code-block:: bash

    export LINGBOT_VLA_V2_ROBOTWIN_ROOT=/path/to/lingbot-vla-v2-6b-robotwin
    export LINGBOT_VLA_V2_PATH=/path/to/.venv/lingbot-vla-v2
    export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct

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

监督微调
----------------------------------------

从 Foundation Base 微调 RoboTwin 动作。SFT 更新动作专家和投影层，冻结 Qwen-VL 主干与 MoE 路由器。SFT 与推理使用同一份上游模型源码，并在单机 8 卡上运行。FSDP2 对主干使用 BF16，动作专家与梯度规约使用 FP32。

从 `Hugging Face <https://huggingface.co/robbyant/lingbot-vla-v2-6b>`__ 下载 Foundation Base：

这里下载的是 Foundation Base。上面的 PPO 示例使用单独的 RoboTwin 后训练 checkpoint。

.. code-block:: bash

    hf download robbyant/lingbot-vla-v2-6b \
      --local-dir /path/to/lingbot-vla-v2-6b

设置 Foundation Base、LingBot-VLA V2 源码、tokenizer 和数据清单路径：

.. code-block:: bash

    export LINGBOT_VLA_V2_PATH=/path/to/.venv/lingbot-vla-v2
    export LINGBOT_VLA_V2_MODEL_ROOT=/path/to/lingbot-vla-v2-6b
    export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
    export LINGBOT_VLA_SFT_DATASET=/path/to/robotwin_train.txt

``LINGBOT_VLA_V2_MODEL_ROOT`` 指向下载后的 Foundation Base。SFT 从源码目录读取 RoboTwin 模型配置、机器人配置和归一化统计。PPO 使用独立的 ``LINGBOT_VLA_V2_ROBOTWIN_ROOT`` 指向后训练 checkpoint。

转换解压后的 RoboTwin 数据集。每个输入目录需要包含 ``data/episode<i>.hdf5`` 和 ``instructions/episode<i>.json``：

.. code-block:: bash

    python toolkits/lerobot/convert_robotwin_to_lerobot.py \
      --raw-root /path/to/click_bell/aloha-agilex_clean_50 \
      --output-root /path/to/robotwin2_lerobot/click_bell_clean_50
    printf 'robotwin /path/to/robotwin2_lerobot/click_bell_clean_50\n' > "$LINGBOT_VLA_SFT_DATASET"

数据清单每行写一个 ``robot_name dataset_path``。启动 SFT：

.. code-block:: bash

    bash examples/sft/run_vla_sft.sh robotwin_sft_lingbotvlav2

SFT 配置把 actor rank 分配到单机 GPU 0–7，global batch size 设为 8。训练过程记录 ``train/l1_loss``。独立评测会从共享 V2 model 配置读取同一个 ``flex_cached`` attention 设置。加载 SFT 权重时，关闭 PPO value head 并启用 SFT 构造对齐修复，确保评测模型的参数集合与 SFT 一致：

.. code-block:: bash

    bash evaluations/run_eval.sh robotwin_click_bell_lingbotvlav2_eval \
      runner.ckpt_path=/path/to/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt \
      rollout.model.model_path="${LINGBOT_VLA_V2_MODEL_ROOT}" \
      rollout.model.lingbotvlav2.training_config_path="${LINGBOT_VLA_V2_PATH}/configs/vla/robotwin/robotwin.yaml" \
      rollout.model.add_value_head=false \
      rollout.model.sft_constructor_alignment_fix=true

Foundation Base 对照实验
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

图中的训练从 ModelScope Foundation 快照 ``Robbyant/lingbot-vla-v2-6b`` 初始化，ModelScope revision 为 ``edfff7638a31a5dbfaae87016477191735bd754e``。上方 Hugging Face 命令是当前文档的下载流程；本图不声称 Hugging Face 快照与该 ModelScope revision 的元数据或文件哈希相同。Official 和 RLinf 都在同一份 3,384 条 occurrence 清单上完成了 423 次 optimizer update，global batch size 为 8。

.. figure:: /_static/images/lingbotvlav2_foundation_action_loss_ma25.png
   :width: 90%

   Foundation Base 对照实验的逐步 Action Loss 及其 25 步移动平均曲线。

.. list-table:: Action Loss 统计
   :header-rows: 1
   :widths: 40 30 30

   * - 指标
     - Official
     - RLinf
   * - 平均值
     - 0.102265
     - 0.101771
   * - 最后 50 步平均值
     - 0.065720
     - 0.065274

逐步 Action Loss 的 MAE 为 0.002404，Pearson 相关系数为 0.9984，最大绝对差为 0.014081。曲线比较 Official ``VLA_Loss`` 与 RLinf ``train/l1_loss``。Total Loss 单独记录，因为两套实现可能包含不同的辅助项。这是受控对照实验，不能据此宣称严格 parity 或任务成功率提升。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ./logs --port 6006

关键指标是 ``env/success_once``。完整指标说明见 :doc:`训练指标 <../../reference/metrics>`。
