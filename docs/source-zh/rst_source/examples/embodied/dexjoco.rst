基于 DexJoCo 的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
   :align: center
   :width: 90%

   使用 Allegro 灵巧手浇花（图片来源：`DexJoCo <https://github.com/brave-eai/dexjoco>`__）。

使用 RLinf 在 `DexJoCo <https://github.com/brave-eai/dexjoco>`__ 官方 ``water_plant`` 示范数据上微调 CNN 和 π₀.₅ policy，再用 PPO 继续训练。这个 MuJoCo 基准包含 Franka 机械臂与 Allegro 灵巧手的单臂和双臂任务。本页介绍数据准备、训练和评测的完整流程。

概览
----------------------------------------

基于相机图像和本体状态，在 DexJoCo 任务上训练 Franka 与 Allegro policy。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      CNN · π₀.₅

   .. grid-item-card:: 算法
      :text-align: center

      SFT · PPO

   .. grid-item-card:: 任务
      :text-align: center

      6 个单臂 · 5 个双臂

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 1–8 GPU

| **你将完成：** 安装 → 下载数据集 + 模型 → 启动训练 → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 安装步骤中的 DexJoCo 包 · DexJoCo LeRobot 数据集 · 基座检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

训练和评测各运行一个任务。在 ``env.train`` 和 ``env.eval`` 中选择该任务的 ``env/dexjoco_<task>`` 配置组，一并加载任务名、提示词、相机以及默认回合和 rollout 时长。π₀.₅ 配方使用 ``dexjoco_water_plant`` 环境配置和 ``pi05_dexjoco`` 模型配置。环境支持以下任务，描述译自官方提示词：

.. list-table::
   :header-rows: 1
   :widths: 14 24 12 50

   * - 类别
     - 任务
     - 时间上限（步）
     - 描述
   * - 单臂
     - ``water_plant``
     - 1000
     - 抓取浇水壶并给植物浇水。
   * - 单臂
     - ``hammer_nail``
     - 1000
     - 用锤子将钉子钉入木板。
   * - 单臂
     - ``fold_glasses``
     - 1200
     - 折叠眼镜并放入眼镜盒。
   * - 单臂
     - ``pinch_tongs``
     - 1000
     - 抓取夹子并连续开合三次。
   * - 单臂
     - ``pick_bucket``
     - 1200
     - 将盒装食物放入桶中，再提起桶。
   * - 单臂
     - ``click_mouse``
     - 1200
     - 将鼠标移动到紫色鼠标垫上，再点击鼠标左键。
   * - 双臂
     - ``bimanual_assembly``
     - 1500
     - 左手抓取托盘、右手抓取插销，再将插销插入孔中。
   * - 双臂
     - ``bimanual_hanoi``
     - 1500
     - 完成三层汉诺塔的最后两步：右手将中盘从中间柱移到右柱，再用左手将小盘从左柱移到右柱。
   * - 双臂
     - ``bimanual_microwave_cook``
     - 1100
     - 打开微波炉门，将食物放入微波炉，关上门并按下启动键。
   * - 双臂
     - ``bimanual_photograph``
     - 1000
     - 左手抓取相机并对准标志，再用右手按下快门。
   * - 双臂
     - ``bimanual_unlock_ipad``
     - 1200
     - 抓取 iPad 并输入密码 123 解锁设备。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - ``camera_names`` 中各相机的 640×640 RGB 图像（CNN 配方中为 128×128），以及每条机械臂 23 维本体状态（7 维末端位姿和 16 个 Allegro 关节）。
   * - 动作
     - 每条机械臂 22 维连续动作：3D 末端执行器绝对位置、3D 旋转向量和 16 个 Allegro 关节目标；44 维双臂动作中右臂在前。
   * - 奖励
     - 稀疏 0/1 成功奖励。
   * - 提示词
     - DexJoCo 任务的自然语言指令，供 π₀.₅ 使用。

安装
----------------------------------------

.. include:: _setup_common.rst
   :end-before: 然后，使用

为你要运行的模型安装环境：

.. code:: bash

   # 国内用户可添加 --use-mirror。

   # CNN
   bash requirements/install.sh embodied --env dexjoco
   source .venv/bin/activate

   # OpenPI π₀.₅
   # bash requirements/install.sh embodied --model openpi --env dexjoco
   # source .venv/bin/activate

下载数据集
----------------------------------------

两类配方都从 `官方 LeRobot 示范数据 <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__ 出发。下载一个任务的数据并计算 π₀.₅ 的归一化统计量：

.. code:: bash

   # export HF_ENDPOINT=https://hf-mirror.com
   hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
     --include "dexjoco_lerobot_datasets/water_plant/**" \
     --local-dir /path/to/dexjoco
   PYTHONPATH=$(pwd) python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_dexjoco \
     --repo-id /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant

这些命令会：

1. 下载 ``water_plant`` 回合，包括前视和腕部视频、机器人状态、动作和任务文本。
2. 在数据集目录中写入 ``norm_stats.json``，供 SFT、PPO 和评测使用。

CNN 配方把同一份示范数据读成图像、状态和动作块数组。按训练分辨率导出一个任务：

.. code:: bash

   PYTHONPATH=$(pwd) python toolkits/lerobot/export_image_action_chunks.py \
     --repo-id /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant \
     --output /path/to/dexjoco/dexjoco_cnn_datasets/water_plant \
     --cameras front wrist --image-size 128 --chunk 10 --state-dim 23

这条命令会：

1. 解码一次前视和腕部视频，并把每一帧缩放到 128×128。
2. 把 ``main_images``、``extra_view_images``、``states``、``actions``\ （每帧之后的 10 个动作）和 ``episode_index`` 写成内存映射的 ``.npy`` 文件，并附带记录逐维动作统计量的 ``meta.json``。

双臂任务传入 ``--cameras ego wrist_left wrist_right --state-dim 46``。

下载模型
----------------------------------------

下载你要训练的模型检查点。

**CNN**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-ResNet10-pretrained

   # 或使用 huggingface-hub：
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-ResNet10-pretrained --local-dir RLinf-ResNet10-pretrained

然后在 ``examples/embodiment/config/dexjoco_ppo_cnn.yaml`` 中为 rollout 和 actor 设置相同的检查点路径：

.. code-block:: yaml

   rollout:
      model:
         model_path: /path/to/RLinf-ResNet10-pretrained
   actor:
      model:
         model_path: /path/to/RLinf-ResNet10-pretrained

**OpenPI π₀.₅**

为 ``water_plant`` 配方下载 DexJoCo 的 π₀.₅ 基座检查点，并转换为 RLinf OpenPI 格式，转换方法见 :doc:`sft_openpi`：

.. code-block:: bash

   # export HF_ENDPOINT=https://hf-mirror.com
   hf download DexJoCo/DexJoCo-Pi05 --include "pi05_base/params/**" \
     --local-dir /path/to/jax
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model /path/to/jax/pi05_base \
     --input-norm-stats /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json \
     --output-model /path/to/pi05_base_openpi \
     --output-norm-stats /path/to/pi05_base_openpi/dexjoco/norm_stats.json \
     --action-dim 32 --action-horizon 30 --max-token-len 250

运行
----------------------------------------

选择一个配方并启动训练：

.. list-table::
   :header-rows: 1
   :widths: 24 38 20 18

   * - 配方
     - 配置
     - 入口
     - 命令后缀
   * - CNN + SFT
     - ``examples/sft/config/dexjoco_sft_cnn.yaml``
     - ``run_vla_sft.sh``
     - ``dexjoco_sft_cnn``
   * - CNN + PPO
     - ``examples/embodiment/config/dexjoco_ppo_cnn.yaml``
     - ``run_embodiment.sh``
     - ``dexjoco_ppo_cnn``
   * - π₀.₅ + SFT
     - ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml``
     - ``run_vla_sft.sh``
     - ``dexjoco_sft_openpi_pi05``
   * - π₀.₅ + PPO
     - ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml``
     - ``run_embodiment.sh``
     - ``dexjoco_ppo_openpi_pi05``

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn

这条命令会：

1. 使用选定的 Hydra 配置启动 embodied 训练入口。
2. 为 actor、rollout 和 DexJoCo env 组件创建 Ray worker。
3. 运行 PPO rollout，计算稀疏成功奖励，并更新 CNN policy。

这些配方默认训练 ``water_plant``。用 CNN PPO 训练其他任务时，为训练和评测环境同时选择该任务的配置组，并使用对应的 SFT 检查点和动作统计量。例如，准备好 ``bimanual_assembly`` 数据并完成 CNN SFT 后，在仓库根目录运行：

.. code:: bash

   EMBODIED_PATH=$(pwd)/examples/embodiment PYTHONPATH=$(pwd) \
     MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python examples/embodiment/train_embodied_agent.py \
     --config-path config --config-name dexjoco_ppo_cnn \
     'env@env.train=dexjoco_bimanual_assembly' \
     'env@env.eval=dexjoco_bimanual_assembly' \
     actor.model.image_num=3 actor.model.state_dim=46 actor.model.action_dim=44 \
     actor.model.action_norm_stats=/path/to/bimanual_assembly_chunks/meta.json \
     runner.ckpt_path=/path/to/bimanual_assembly_sft/full_weights.pt

这两个配置组覆盖会一起加载任务的提示词、相机和默认时长。模型参数匹配该任务的 3 路相机、46 维状态和 44 维动作，CNN SFT 也使用相同的模型维度。独立评测 CNN 时，选择 ``env@env.eval=dexjoco_bimanual_assembly``，在 ``rollout.model`` 下设置对应的维度，并把 ``rollout.model.action_norm_stats`` 设为该任务的 ``meta.json``，``runner.ckpt_path`` 设为对应的 SFT 或 PPO 检查点。

以下 ``init_params`` 字段控制随机化和任务选项：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 配置项
     - 含义
   * - ``init_params.randomize``
     - 为 ``True`` 时随机化光照、桌面纹理和 ``front`` 相机位姿（双臂任务为 ``ego``）。
   * - ``init_params.randomize_dynamics``
     - 为 ``True`` 时随机化物体动力学。
   * - ``init_params.password``
     - ``bimanual_unlock_ipad`` 的解锁数字序列，默认 ``[1, 2, 3]``；任务的默认提示词写 123。

仿真器只在每个已执行动作块的最后一步渲染相机，供 policy 读取这帧观测并决策；设置 ``video_cfg.save_video: True`` 时每一步都会渲染。

CNN + SFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``examples/sft/config/dexjoco_sft_cnn.yaml`` 中设置以下字段：

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - 字段
     - 值
   * - ``data.train_data_paths``
     - 导出的动作块数据目录。
   * - ``actor.model.model_path``
     - ``RLinf-ResNet10-pretrained`` 目录。
   * - ``actor.model.action_norm_stats``
     - 导出的 ``meta.json``，其中逐维的动作均值和标准差用于归一化 policy 动作。

.. code:: bash

   bash examples/sft/run_vla_sft.sh dexjoco_sft_cnn

policy 用冻结的 ResNet-10 主干编码 128×128 的前视和腕部图像，拼接 23 维本体感知，预测之后的 10 个动作。损失是示范动作块的高斯负对数似然，因此 PPO 之后采样所用的逐维动作噪声来自对示范的拟合。20,000 步的配方在一张 48 GB GPU 上约需十分钟，保存 ``checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``。

双臂任务需设置 ``actor.model.image_num: 3``、``actor.model.state_dim: 46`` 和 ``actor.model.action_dim: 44``。

CNN + PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dexjoco_ppo_cnn`` 用同样的观测和动作布局运行 PPO：128 个环境、完整的 1000 步回合，每 5 轮在 64 个回合上评测。每轮执行 1 个 PPO update epoch，使用 action-level log-probability；前 50 个 optimizer step 只训练 value head。把 ``runner.ckpt_path`` 设为 SFT 的 ``full_weights.pt`` 即可从行为克隆 policy 出发；保持 ``null`` 则只从 ResNet-10 编码器开始训练。``actor.model.action_norm_stats`` 必须与 SFT 使用同一个 ``meta.json``。

SFT 的 encoder dropout 为 0.1；PPO 和评测设置 ``encoder_config.dropout: 0.0``，使用确定性的图像特征。

报告中的训练从 SFT 第 8000 步开始，使用 8 张 H100 GPU。要复现相同的 worker 分布和评测频率，在同一个配置文件中设置：

.. code:: yaml

   cluster:
     component_placement:
       actor,env,rollout: all
   runner:
     ckpt_path: /path/to/dexjoco_sft_cnn/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt
     val_check_interval: 1
     save_interval: 1
   env:
     train:
       seed: 42
     eval:
       seed: 10000
       video_cfg:
         save_video: False
   rollout:
     enable_torch_compile: False
     enable_cuda_graph: False
   actor:
     seed: 42

设置路径和运行配置后，启动训练：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn

独立评测请使用统一的 :doc:`Evaluation CLI <../../evaluations/reference/cli>` 运行 ``dexjoco_cnn_eval``，并把 ``runner.ckpt_path`` 设为 SFT 或 PPO 的 ``full_weights.pt``。

π₀.₅ + SFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pi05_dexjoco`` 模型配置读取 ``water_plant`` 的前视和腕部图像，以及 22 维动作。在 ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml`` 中设置以下字段：

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - 字段
     - 值
   * - ``data.train_data_paths``
     - 任务数据集目录。
   * - ``actor.model.model_path``
     - 转换得到的 ``pi05_base_openpi`` 目录。
   * - ``actor.model.openpi_data.norm_stats_path``
     - 该数据集的 ``norm_stats.json``。

.. code:: bash

   bash examples/sft/run_vla_sft.sh dexjoco_sft_openpi_pi05

该配方使用 FP32 参数、BF16 计算和 ``no_shard``，详见 :doc:`sft_openpi`。每张 actor GPU 都保存完整的模型、梯度和优化器状态，因此建议使用 80 GB 级别的 GPU。单张 48 GB GPU 上，设置 ``actor.micro_batch_size: 1``、``actor.global_batch_size: 1`` 和 ``actor.model.precision: bf16``；这会把参数和优化器精度从默认的 FP32 改为 BF16。全局批量必须能被单卡微批量与 actor 工作进程数的乘积整除。训练会保存 ``checkpoints/global_step_<N>`` 目录，PPO 和评测配置可以直接加载。

π₀.₅ + PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml`` 中，把 ``actor.model.model_path`` 和 ``rollout.model.model_path`` 设为 SFT 的 ``global_step_<N>`` 目录，并把 ``actor.model.openpi_data.norm_stats_path`` 设为 SFT 使用的统计量文件。

默认配方使用 token-level log-probability，policy 学习率为 5e-6，value head 学习率为 2e-4。报告中的训练从 SFT 第 5000 步开始，使用 8 张 H100 GPU，共训练 150 轮，每轮评测并保存检查点。学习率调度总步数（``actor.optim.total_training_steps``）为 1000，独立于 150 轮的训练预算。训练种子为 42，评测种子为 10000，并关闭视频录制和 Torch 编译。

设置模型和统计量路径后，启动训练：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_openpi_pi05

这条命令会：

1. 使用选定的 Hydra 配置启动 embodied 训练入口。
2. 为 actor、rollout 和 DexJoCo env 组件创建 Ray worker。
3. 运行 PPO rollout，计算稀疏成功奖励，并更新 OpenPI policy。

每次决策执行预测的 30 个动作中的前 10 个，用 5 步去噪采样。每轮采集 128 个环境 × 2 个 rollout epoch 的数据，执行一个 PPO epoch，全局批量为 1024，微批量为 64。前 50 个优化器步只训练 value head（``actor.optim.critic_warmup_steps``）。

独立评测请使用统一的 :doc:`Evaluation CLI <../../evaluations/reference/cli>` 运行 ``dexjoco_openpi_pi05_eval``。把 ``rollout.model.model_path`` 设为 SFT 或 PPO 检查点，把 ``rollout.model.openpi_data.norm_stats_path`` 设为训练时使用的统计量文件。

报告中的曲线使用 8 个 worker 评测 64 个回合，设置 ``env.eval.seed: 10000``，每块执行 10 个动作，去噪 5 步，``max_token_len: 200``。每个回合上限为 1000 个环境步，评测 rollout 预算为 1020 步。环境种子固定 reset 状态，动作采样使用各进程的随机数历史。

.. note::

   ``dexjoco_ppo_cnn`` 默认运行在 GPU ``0`` 上，π₀.₅ 配置把 ``actor,env,rollout: all`` 放在同一组 GPU 上。请根据硬件调整 ``cluster.component_placement``、``env.train.total_num_envs`` 和批量大小。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ./logs --port 6006

观察 ``env/success_once`` 中的训练 rollout 成功率，以及 ``eval/success_once`` 中的评测成功率。完整指标说明见 :doc:`训练指标 <../../reference/metrics>`。

若要把同一组指标保存到 Weights & Biases，先运行 ``wandb login`` 登录，再启用两个日志后端：

.. code:: yaml

   runner:
     logger:
       project_name: rlinf
       experiment_name: dexjoco_ppo_cnn
       logger_backends: ["tensorboard", "wandb"]

如需视频，请在环境配置中启用 video：

.. code:: yaml

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval

.. figure:: /_static/image/dexjoco_pi05_sft_ppo.png
   :align: center
   :width: 100%
   :alt: π₀.₅ 在 water_plant 上经过 5000 步 SFT 和 150 轮 PPO 的成功率。

   π₀.₅ 在 ``water_plant`` 上的 SFT + PPO 成功率（64 个评测回合）。

`SFT + PPO 报告 <https://forge.coreweave.com/matthewmzy-peking-university/rlinf-dexjoco/reports/DexJoCo-water_plant:-pi0.5-SFT-to-PPO-B-with-one-evaluation-protocol--VmlldzoxODI0MDM2Nw==>`__ 包含评测曲线和原始日志。下表的 PPO 结果来自检查点的独立评测。

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 配方
     - 结果描述
   * - CNN + SFT
     - 在 ``water_plant`` 上，20,000 步配方在一张 48 GB GPU 上于第 8000 步达到峰值，64 个评测回合成功 50 个（78%）；第 10,000–20,000 步保持在 66% 到 75% 之间。
   * - CNN + PPO
     - 从 SFT 第 8000 步继续使用 action-level PPO 在 8 张 H100 GPU 上训练，到预定终点第 200 轮时，成功率从 42/64（65.6%）升至 59/64（92.2%）。第 0/10/20/40/80/120/150/200 轮的 64 个回合分别成功 42/40/49/48/55/57/61/59 个。训练使用 1e-5 的 policy 学习率和 1600 的全局批量。所有检查点均使用 8 个 worker 评测相同的 64 个回合，设置 ``env.eval.seed: 10000``，每块执行 10 个动作。
   * - π₀.₅ + SFT
     - 在 ``water_plant`` 上，使用 16 张 H100 GPU、全局批量 32 训练到 SFT 第 5000 步，成功率达到 44/64（68.75%）。
   * - π₀.₅ + PPO
     - 使用 token-level PPO 在 8 张 H100 GPU 上训练，在预定终点第 150 轮达到 63/64（98.44%），第 120 轮达到峰值 64/64（100%）。
