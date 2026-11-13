Prefix Fine-Tune：从已有 VLA 开始在线训练
=========================================
视觉语言动作模型（VLA）读取相机图像、语言指令和机器人状态，输出动作。图像理解、语言理解和动作生成都在同一份大模型里。

RL Token（RLT）分两个阶段使用这份预训练能力。Stage 1 用示范数据联合训练 VLA 和 token transformer，把视觉语言特征压成紧凑表示。Stage 2 冻结特征模型，用这个表示、机器人状态和 VLA 参考动作训练两个轻量网络：actor 输出动作，critic 估计后续回报。在线更新只经过这两个网络，大模型只做前向计算。

Prefix 框架在此基础上用配置选择特征来源、网络和算法，采集和参数更新仍走同一套流程。特征可以来自 RLT token，也可以直接池化冻结 VLA 的视觉语言特征，这段特征称为 prefix。policy 指根据观测生成动作的模型。

训练时特征模型保持冻结。actor 读取 prefix 特征、机器人状态和参考动作，给出要执行的动作。环境执行后，系统把特征、动作、奖励和终止写入 replay，更新时反复采样，反向传播只经过 actor 和 critic。VLA 继续提供特征和参考动作，权重保持不变。

本页使用已发布的 OpenPI checkpoint 和配好的 ManiSkill 插接配置，先把这条流程跑通。当前接入的 VLA 是 OpenPI；其他 VLA 需要适配器，步骤见 :doc:`Prefix 组件扩展 <../../extending/prefix>`。如果已经训练过 RL Token 表示，:doc:`RL Token <rlt>` 上的 Stage 2 配置走同一套更新流程。

概览
----

第一次运行使用特征池化这条 preset。它已经选好 TD3 和 STEAM 阶段切换。一份 preset 同时指定环境、模型、算法和设备分配，运行时用配置文件名选择。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      TD3，另有 AC preset

   .. grid-item-card:: 模型
      :text-align: center

      冻结的 OpenPI π₀.₅ · MLP actor/critic

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      ManiSkill 插接

   .. grid-item-card:: 训练
      :text-align: center

      在线 rollout · replay · STEAM 路由

训练怎样工作
------------

一次控制从环境观测开始。冻结的 VLA 把图像和语言编码成 prefix 特征，并给出一段参考动作，也就是已有模型会执行的动作。小型 actor 读取这些特征、机器人当前关节状态（本体状态）和参考动作，输出要执行的动作。critic 估计该动作可能带来的后续回报。actor 一次输出连续若干步，这一段称为动作块。本例每块 10 步，每步 8 个数值，含义由机器人控制方式决定。

环境执行动作块之后，系统保存提取出的特征、实际发出的动作、奖励和终止信息。这个存储区是经验回放缓冲区（replay buffer）。更新会反复从中采样，反向传播只经过小网络。VLA 继续负责特征提取和参考动作，它的权重不会被更新。

RLinf 把环境交互和动作采集称为 rollout，把参数更新称为训练。两者交替进行，更新后的 policy 会同步回采集端。在线训练留下的 checkpoint 是轻量 policy；冻结的 VLA 仍是下载的那一份。

示例还用 STEAM 根据不同时刻的图像判断任务进度。开始时由 VLA 参考动作控制。训练预热完成，并且 STEAM 判断任务进入关键阶段之后，actor 才可以接管。这种动作来源的选择称为路由。STEAM 决定何时切换，actor 学习如何动作，两边的设置可以分开改。

选择起始配置
------------

普通特征池化可以使用基座 VLA，也可以使用已经用示范动作做过监督微调（SFT）的模型。另一条表示是 RLT：Stage 1 训练一个压缩 VLA 特征的小网络，Stage 2 冻结它并训练 actor 和 critic。两条路线共用下面的更新流程。

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - 要完成的实验
     - 配置文件，位于 ``examples/embodiment/config/``
   * - VLM 特征池化、TD3 与 STEAM 阶段切换
     - ``maniskill_prefix_stage2_td3_mlp_steam.yaml``
   * - RLT token 表示与 TD3
     - ``maniskill_rlt_stage2_td3_mlp.yaml``
   * - RLT token 表示与 AC
     - ``maniskill_rlt_stage2_ac_mlp.yaml``
   * - 为 STEAM 收集校准数据
     - ``maniskill_prefix_stage2_td3_mlp_steam_calibration.yaml``

下面的步骤使用第一项。这条流程跑通后，再用 RLT 配置比较表示或算法；Stage 1 的准备见 :doc:`RL Token <rlt>`。任务、相机或控制器变了，需要重新拟合阶段判断时，使用校准配置。

安装
----

以下流程面向有 NVIDIA GPU 的 Linux 机器。需要能访问 GitHub 和 Hugging Face，并留出存放模型、依赖、日志和训练 checkpoint 的磁盘。显存里要同时放下 VLA、STEAM 和轻量网络。示例并行运行 64 个训练环境，仿真本身也会占用显存。

示例用已发布 checkpoint 启动在线训练。交互数据由 ManiSkill 在运行中产生，下载的 checkpoint 里带有归一化统计量，用来把机器人状态和动作变到 VLA 训练时的数值尺度。

克隆仓库：

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://gh-proxy.com/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

用发布的镜像或本机环境安装并激活依赖。

**方式一：Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero
      # 为提高国内下载速度，可以使用：
      # infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

进入容器后，切换到 OpenPI 虚拟环境：

.. code:: bash

   source switch_env openpi

**方式二：本机环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以给 install.sh 加上 --use-mirror。
   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

安装脚本准备具身训练、OpenPI 模型和 ManiSkill 仿真所需依赖。第二条命令激活它生成的环境。后续下载和训练都在这个终端执行。新开终端时，重新进入仓库并激活环境。

下载模型
--------

这次运行的设置集中在 ``examples/embodiment/config/maniskill_prefix_stage2_td3_mlp_steam.yaml``。先保留网络和算法参数，填好模型路径，再检查路由和日志，然后启动。

示例加载四组模型。特征 VLA 生成表示和参考动作。Expert VLA 是另一份已训练模型，供以后启用接管时使用。STEAM value model 从图像判断进度。phase head 是配套的小型分类模型，用来判断是否进入关键阶段。第一次运行沿用这四组和 preset 里的设置。

.. list-table::
   :header-rows: 1
   :widths: 28 40 32

   * - 资源
     - Hugging Face 仓库
     - 配置字段
   * - 特征 VLA，step 1000
     - `ChikaYokoyama/maniskill_rlt_1000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_1000_new>`_
     - ``rollout.prefix_feature_model.model_path``
   * - Expert VLA，step 3000
     - `ChikaYokoyama/maniskill_rlt_3000_new <https://huggingface.co/ChikaYokoyama/maniskill_rlt_3000_new>`_
     - ``rollout.expert_model.model_path``
   * - STEAM value model，step 5000
     - `ChikaYokoyama/rlt_steam_value_model_sft_step_5000 <https://huggingface.co/ChikaYokoyama/rlt_steam_value_model_sft_step_5000>`_
     - ``rollout.routing_gate.model.model_path``
   * - Phase head 与其中保存的建议值
     - `ChikaYokoyama/rlt_steam_phase_head <https://huggingface.co/ChikaYokoyama/rlt_steam_phase_head>`_
     - ``rollout.routing_gate.actor_switch.phase_head_path``

把四个仓库下载到 ``data/steam_peg``，同时下载 STEAM 处理图像和语言所用的 SigLIP 与 Gemma backbone。使用 hf-mirror 时，先执行 ``export HF_ENDPOINT=https://hf-mirror.com``。

.. code:: bash

   CHECKPOINT_ROOT="$(pwd)/data/steam_peg"

   hf download ChikaYokoyama/maniskill_rlt_1000_new \
     --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
     --exclude '*/dcp_checkpoint/*'
   hf download ChikaYokoyama/maniskill_rlt_3000_new \
     --local-dir "$CHECKPOINT_ROOT/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints" \
     --exclude '*/dcp_checkpoint/*'
   hf download ChikaYokoyama/rlt_steam_value_model_sft_step_5000 \
     --local-dir "$CHECKPOINT_ROOT/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor" \
     --exclude 'dcp_checkpoint/*'
   hf download ChikaYokoyama/rlt_steam_phase_head \
     --local-dir "$CHECKPOINT_ROOT"

   hf download google/siglip-so400m-patch14-384 \
     --local-dir "$CHECKPOINT_ROOT/siglip-so400m-patch14-384"
   hf download google/gemma-3-270m \
     --local-dir "$CHECKPOINT_ROOT/gemma-3-270m"

前两个仓库已经包含 ``global_step_1000/actor`` 和 ``global_step_3000/actor``，所以下载到同一个 ``checkpoints`` 目录。STEAM 仓库的根目录直接存放 actor 文件，因此下载目标要带上 ``global_step_5000/actor``。这些命令跳过分布式训练状态，保留推理权重、归一化统计量和 processor 文件。

填写模型路径
------------

打开 ``examples/embodiment/config/maniskill_prefix_stage2_td3_mlp_steam.yaml``，把四个 checkpoint 字段改成上一节目录下的绝对路径。下面的 ``<CHECKPOINT_ROOT>`` 表示展开后的绝对路径，不要把尖括号里的文字留在文件中。

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - 字段
     - 本地路径
   * - ``rollout.prefix_feature_model.model_path``
     - ``<CHECKPOINT_ROOT>/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_1000/actor``
   * - ``rollout.expert_model.model_path``
     - ``<CHECKPOINT_ROOT>/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_3000/actor``
   * - ``rollout.routing_gate.model.model_path``
     - ``<CHECKPOINT_ROOT>/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor``
   * - ``rollout.routing_gate.actor_switch.phase_head_path``
     - ``<CHECKPOINT_ROOT>/steam_phase_head.pt``

保留 ``openpi_data.repo_id: maniskill_peginsertionside_joint`` 和 ``norm_stats_path: null``。这样每个 OpenPI checkpoint 会读取自己目录里的 ``maniskill_peginsertionside_joint/norm_stats.json``。

然后把 STEAM checkpoint 里的 backbone 路径指到本机下载目录。编辑 ``<CHECKPOINT_ROOT>/rlt_steam_value_model_sft/checkpoints/global_step_5000/actor/config.json``：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``config.json`` 字段
     - 本地路径
   * - ``vision_repo_id``
     - ``<CHECKPOINT_ROOT>/siglip-so400m-patch14-384``
   * - ``language_repo_id``
     - ``<CHECKPOINT_ROOT>/gemma-3-270m``

tokenizer 和 image processor 文件留在同一个 ``actor`` 目录。gate 从这里加载它们。

运行训练
--------

路径填好之后，保留示例的路由，检查设备和日志，再从仓库根目录启动。

保留示例的路由设置
~~~~~~~~~~~~~~~~~~

运行时要在 VLA、actor 和 expert 的动作之间选择。STEAM 给出的阶段或接管判断称为 gate 信号。保留下面这段配置时，actor 切换会实际生效；expert 使用 shadow 模式，只记录判断，不执行接管。

.. code:: yaml

   rollout:
     routing_gate:
       actor_switch:
         enable: True
         mode: active
         enter_threshold: null
         patience_chunks: null
       expert_takeover:
         enable: True
         mode: shadow
         warmup_chunks: null
         patience_chunks: null
         enter_threshold: null

这是完整配置里的一段，不要用它替换整个文件。``enter_threshold`` 是触发判断的门限，``patience_chunks`` 是判断需要连续满足的动作块数，``warmup_chunks`` 是开始检查 expert 接管前等待的动作块数。它们保持 ``null`` 时，使用 phase head 文件里保存的校准值；写成具体数字则会覆盖保存值。

.. warning::

   当前发布的 expert 门限在校准验证集中没有触发接管，因此不能据此打开自动接管。第一次运行保留 shadow。用 actor 实际控制的 episode 重新采集并核对门限之后，再考虑把 ``expert_takeover.mode`` 改成 ``active``。

检查设备与日志
~~~~~~~~~~~~~~

preset 里的 ``cluster.component_placement`` 把 actor 放在 GPU 0，把环境和 rollout 放在全部设备（``all``）上。启动前确认这些编号在本机存在。``env.train.total_num_envs`` 是并行训练环境数，示例为 64；更换硬件时结合显存和仿真负载一起调整。

示例默认把指标记到 Weights & Biases。使用这个后端时，先在同一个运行环境里完成账号配置。如果要在本机查看指标，把 ``runner.logger.logger_backends`` 设为 ``["tensorboard"]``。校准 preset 已经使用这个本机后端。

启动训练
~~~~~~~~

在已激活环境的仓库根目录执行：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam

命令会加载模型、启动仿真、收集交互数据并更新轻量网络。控制台输出写到 ``logs/<YYYYMMDD-HH:MM:SS>-maniskill_prefix_stage2_td3_mlp_steam/run_embodiment.log``。每次启动都有独立目录，用来区分不同实验。

可视化与结果
------------

训练启动后，先确认数据采集和参数更新在推进，再用定期评估查看任务是否完成。

训练进度
~~~~~~~~

训练先收集数据，再完成预热更新。示例要求 replay 至少积累 10,000 条记录，并完成 30,000 次预热更新，之后才允许 actor 接管。是否进入关键阶段仍由 STEAM 判断。看到机器人继续执行 VLA 参考动作，只说明接管还没发生，不说明训练停了。

在所选日志后端里，进度指标带有 ``train/prefix/`` 前缀：

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - 指标
     - 如何判断
   * - ``train/prefix/global_total_transitions_added``
     - 持续增加，说明交互数据正在写入 replay
   * - ``train/prefix/global_min_replay_size``
     - 达到预热容量后，才具备开始更新的条件
   * - ``train/prefix/update_step``
     - 增加说明 learner 正在执行参数更新
   * - ``train/prefix/ready_for_online``
     - 值为 ``1`` 表示预热条件已满足；actor 是否接管还取决于阶段 gate

这些指标说明流程在推进，不代表任务已经学会。任务效果看定期评估里的成功率。这个 runner 把一次 rollout 加一次更新算作一个 step，本 preset 每个 epoch 走一个这样的 step。每 25 个 step 评估一次，每 50 个 step 保存一次 checkpoint，分别由 ``runner.val_check_interval`` 和 ``runner.save_interval`` 控制。保留训练 checkpoint，以及它所配套的 VLA checkpoint 和归一化文件，供以后恢复或评估。

查看训练曲线
~~~~~~~~~~~~

使用 TensorBoard 时，把 ``runner.logger.logger_backends`` 设为 ``["tensorboard"]``，然后在已激活环境的仓库根目录启动：

.. code:: bash

   tensorboard --logdir logs --port 6006

在运行 TensorBoard 的机器上打开 ``http://localhost:6006``，选择本次运行。远程训练时通过端口转发访问。使用 Weights & Biases 时，在对应实验里查看同一组指标名。

记录评估结果
~~~~~~~~~~~~

比较不同特征或算法时，保持任务、评估初始状态、episode 数量和动作块长度一致，并记下所用 checkpoint。训练进度用来确认运行是否正常，成功率用来比较任务表现。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 记录项
     - 内容
   * - 实验设置
     - VLA checkpoint、特征池化、actor/critic 和算法配置
   * - 训练进度
     - 采集的交互记录数和参数更新次数
   * - 评估规模
     - 评估 episode 总数和初始状态设置
   * - 任务成功率
     - 成功 episode 数除以评估 episode 总数
   * - 动作表现
     - 抓取、接近孔位、插入过程，以及 actor 或 expert 的接管情况

需要保存视频时，把 ``env.eval.video_cfg.save_video`` 设为 ``true``。评估视频写到该次运行日志目录下的 ``video/eval``。成功率以这次运行的评估输出为准。

通过配置替换组件
----------------

确认数据采集和训练更新都在正常推进之后，再改实验，并且每次只改一个因素。先比较表示，再比较算法，这样更容易判断结果变化来自哪一次修改。

常见修改落在三组字段上。``rollout.prefix_feature_model`` 选择冻结的 VLA 及其特征，``actor.model`` 选择可训练网络，``algorithm`` 选择更新规则。下面的字段都是在已经启动过的完整配置里修改。

切换 VLA checkpoint 与特征表示
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

更换同一套 OpenPI 配置所兼容的 checkpoint 时，修改 ``rollout.prefix_feature_model.model_path``，并检查任务配置和归一化统计量是否匹配。更换 VLA 架构则需要适配器。当前接入的是 OpenPI，注册步骤见 :doc:`Prefix 组件扩展 <../../extending/prefix>`。

VLA 为图像块和语言 token 各产生一个向量。policy 需要固定长度的输入，因此要把这些向量收成一个向量，这一步称为池化。填充和其他无效位置由 mask 标出。``masked_mean`` 会跳过被 mask 的位置。用 ``rollout.prefix_feature_model.prefix.pool`` 选择聚合方式：

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - ``pool``
     - 计算内容
     - 使用条件
   * - ``masked_mean``
     - 对有效 prefix token 求均值
     - 直接使用 base 或 SFT VLA 特征，包括本例
   * - ``mean``
     - 对选定的全部 prefix token 求均值，包括被 mask 的位置
     - 需要比较完整序列池化的实验
   * - ``last``
     - 读取最后一个有效 token；没有 mask 时读取序列末端
     - 需要比较末端 token 的实验
   * - ``rlt_token``
     - 使用训练好的 RLT token transformer
     - checkpoint 包含匹配的 Stage 1 RLT 模块

``prefix.image_only: true`` 只用图像 token 构建表示，``false`` 同时使用图像和语言 token。比较表示时把这两个字段都写明确。使用 ``rlt_token`` 还要设置 ``openpi.use_rlt: true``，并让 Stage 1 模块参数与 checkpoint 一致。这条路线从已有的 RLT preset 改起。

切换 Actor
~~~~~~~~~~

AC 和 TD3 是两种更新 actor 与 critic 的算法，它们使用的动作生成方式不同，所以切换算法时要同时切换 actor。两份已发布 preset 的 critic 都对同一动作给出两个 Q 估计，用来构造训练目标。

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - 字段
     - AC preset
     - TD3 preset
   * - ``algorithm.loss_type``
     - ``prefix_off_policy``
     - ``prefix_off_policy``
   * - ``algorithm.name``
     - ``ac``
     - ``td3``
   * - ``actor.model.model_type``
     - ``prefix_policy``
     - ``prefix_policy``
   * - ``actor.model.actor_head.name``
     - ``fixed_std_mlp``
     - ``deterministic_mlp``
   * - ``algorithm.actor_agg_q``
     - ``q1``
     - ``min``、``q1`` 或 ``mean``
   * - ``actor.model.critic_head.name``
     - ``multi_q_mlp``，``num_q_heads: 2``
     - ``twin_q_mlp``

AC 使用随机 actor，preset 里的 ``fixed_std: 0.002`` 是固定标准差。TD3 使用带动作噪声的确定性 actor，preset 里的 ``noise_sigma: 0.1`` 是噪声尺度。切换时从对应 preset 复制整个 head 配置，包括这些尺度。

两种算法都保留 ``algorithm.entropy_tuning.alpha_type: fixed_alpha``，熵温度使用配置值，不再另外学习。critic 训练目标取前两个 Q 估计中的较小值，因此省略 ``algorithm.agg_q``，或把它设为 ``min``。``actor_agg_q`` 管的是另一件事：actor 更新时如何合并 Q 值。AC 只接受 ``q1``。

调整网络与本体状态历史
~~~~~~~~~~~~~~~~~~~~~~

配置里的 head 就是接在冻结特征后面的小网络。要改容量，修改 ``actor.model.actor_head.hidden_dims`` 和 ``actor.model.critic_head.hidden_dims``。``[256, 256]`` 表示两个宽度为 256 的隐藏层，这也是 TD3 preset 的设置。

先沿用该 preset 的 critic head。需要研究其他价值估计时，再用 ``multi_q_mlp`` 配置 Q 网络数量，或用 ``cross_q_mlp`` 联合处理当前状态和下一状态。配置校验会检查 Q 网络数量以及算法要求的其他条件。

如果当前本体状态不足以描述运动，可以给 policy 加上一段近期历史。设置 ``actor.model.state_history.enable``、历史步数 ``steps``，以及填充方式 ``pad``，取 ``zero`` 或 ``repeat``。历史按 policy 每次决策更新，并在对应环境重置时清理。``actor.model.z_dim`` 仍填写原始特征宽度，历史增加的宽度由配置计算。

调整训练目标与更新频率
~~~~~~~~~~~~~~~~~~~~~~

actor 一方面要提高预计回报，另一方面要接近 VLA 的参考动作，后者称为行为克隆（BC）。``algorithm.q_weight`` 和 ``algorithm.bc_weight`` 分别是两项的权重。如果某一步被标成人工干预，这一步模仿的是干预动作，而不是 VLA 参考动作。

示例打开了 ``algorithm.actor_weight_schedule``，训练过程中这两个权重会变化。如果希望全程使用固定权重，先把该 schedule 的 ``enable`` 设为 ``false``，再修改 ``q_weight`` 和 ``bc_weight``。

更新节奏由 ``algorithm.rlt_schedule`` 控制。``warmup_min_size`` 是开始训练所需的回放数据量，``warmup_post_collect_updates`` 是允许 actor 在线接管前的预热更新次数。之后可以按新增交互记录或完成的任务轮次安排更新。一次任务从重置到结束称为一个 episode，一条观测、动作和结果称为一个 transition。

critic 会考虑动作块内的奖励，以及动作块执行后状态的预计价值；越往后的奖励，权重越低。第一次运行先保留这些学习参数，等采集和更新都正常后再调节。

为新任务检查输入与动作
----------------------

把同一份 preset 迁到其他任务之前，先让冻结的 VLA、环境和 policy 对输入与动作的理解一致。VLA 通过三个字段把观测交给 policy：

.. list-table::
   :header-rows: 1
   :widths: 16 34 50

   * - 字段
     - 特征模型输出形状
     - 需要确认的内容
   * - ``z_rl``
     - ``[B, z_dim]``
     - 原始特征宽度与 ``actor.model.z_dim`` 一致
   * - ``proprio``
     - ``[B, proprio_dim]``
     - 本体状态的维度、含义和归一化一致
   * - ``ref_chunk``
     - ``[B, ref_num_action_chunks, action_dim]``
     - 动作维度匹配环境，参考长度覆盖实际执行长度

``B`` 是 batch 大小，``num_action_chunks`` 是 policy 执行的动作块长度。policy 把 ``z_rl`` 和当前 ``proprio`` 拼成状态，并用 ``ref_chunk`` 的前 ``num_action_chunks`` 步作为 actor 的条件输入和默认 BC 目标。

新任务第一次启动前，检查环境图像字段、语言指令、OpenPI data config 和 ``norm_stats.json``。使用 STEAM 时，还要检查 ``rollout.routing_gate.cameras`` 是否把每个 STEAM 相机键映射到该相机实际看到的环境图像。本例把外部相机映射到 ``main_images``，把手腕相机映射到 ``wrist_images``。更换相机后，让 STEAM 读取新的图像，并重新检查阶段判断。

为自己的任务校准 STEAM
----------------------

任务、相机或控制行为变化后，需要重新检查关键阶段判断和 expert 接管门限。先用校准 preset 收集特征和标签，再离线拟合 phase head。

在仓库根目录、模型路径已经配好的环境中运行：

.. code:: bash

   STEPS=20 SAVE_INTER=-1 bash examples/embodiment/run_embodiment.sh maniskill_prefix_stage2_td3_mlp_steam_calibration

该 preset 使用环境几何 gate 做路由，打开 ``collect_phase_features`` 和 ``algorithm.rlt_gate_calibration.enable``，把特征、几何标签和诊断信息写到 ``<run_log_dir>/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration/rank_<rank>/trace_*.pt``。每次运行的 ``<run_log_dir>`` 位于 ``logs/`` 下。

这 20 个 step 用来检查采集是否走通。出现正负几何标签后可以拟合初版 phase head，但这次短运行完不成训练 preset 的 30,000 次预热更新，也不能据此验证 expert 接管。把下面的 ``/path/to/run_log_dir`` 换成这次启动的日志目录。``CHECKPOINT_ROOT`` 仍是下载模型时的目录。

.. code:: bash

   TRACE_DIR=/path/to/run_log_dir/maniskill_prefix_stage2_td3_mlp_steam_calibration/gate_calibration
   python toolkits/rlt/calibrate_steam_gates.py "$TRACE_DIR" \
     --phase-head-output "$CHECKPOINT_ROOT/steam_phase_head.pt" \
     --yaml-output "$CHECKPOINT_ROOT/steam_gate_recommendation.yaml" --device cpu

工具写出 ``steam_phase_head.pt`` 和 ``steam_gate_recommendation.yaml``。检查验证指标后，把训练 preset 的 phase-head 路径指到新文件。gate 门限保持 ``null``，运行时就会读取该 checkpoint 里保存的建议值。expert 门限还需要用 actor 实际控制的 episode 再做一次校准。日常训练使用训练 preset，两个采集开关都保持关闭。

现有配置能覆盖哪些修改
----------------------

使用已发布模型、改变特征池化、调整网络宽度，或在 AC 与 TD3 之间切换，都可以在本页的配置里完成。更换 VLA 架构或实现新的学习目标则需要适配代码。只把模型名或算法名改成任意字符串，并不会把它接进来。

其他 VLA 需要把观测转换成 ``z_rl``、``proprio`` 和 ``ref_chunk``，并让动作的单位、范围和长度与环境一致。新网络要读取同一组特征，并输出动作或价值。新算法要定义 actor 和 critic 的训练目标。注册完成之后，仍然在配置里选择这些组件，并继续使用现有的采集和更新流程。

需要编写适配代码时，再看 :doc:`Prefix 组件扩展 <../../extending/prefix>` 里的实现和注册示例。运行本页示例、调整已有组件，不必先读这些内部接口。

常见问题
--------

启动或行为异常时，先对照对应配置和日志检查，不要同时改模型、算法和环境。

找不到模型或归一化文件时，确认四个 checkpoint 都指向实际的 ``actor`` 目录，并且每个 OpenPI checkpoint 都带有自己的 ``norm_stats.json``。如果 STEAM 仍在打开生成该 checkpoint 的机器上的路径，修改它的 ``config.json`` 里的 ``vision_repo_id`` 和 ``language_repo_id``，并检查推荐 YAML 里的 phase-head 路径。

``z_rl`` 宽度不匹配时，对照 VLA 声明的特征宽度和 ``actor.model.z_dim``。启用 history 后，增加的宽度由历史步数和 ``proprio_dim`` 决定，不要再加进 ``z_dim``。切换算法后如果配置校验失败，确认 AC 或 TD3 使用了对应的 actor，并检查 ``actor_agg_q``、Q 网络数量和固定熵温度。动作幅度或方向异常时，核对环境动作定义、OpenPI data config、单步动作维度和归一化统计量是否一致。

actor 没有接管时，确认数据预热和更新预热都已完成，阶段 gate 已经触发，并且 ``rollout.routing_gate.actor_switch.mode`` 为 ``active``。expert gate 有记录但机器人没有被 expert 接管时，``expert_takeover.mode`` 仍是 ``shadow``。校准结果核对通过之后，再把它改成 ``active``。
