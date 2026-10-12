基于 DexJoCo 的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
   :align: center
   :width: 90%

   使用 Allegro 灵巧手浇花（图片来源：`DexJoCo <https://github.com/brave-eai/dexjoco>`__）。

`DexJoCo <https://github.com/brave-eai/dexjoco>`__ 是基于 MuJoCo 的灵巧操作基准，包含 Franka 机械臂与 Allegro 灵巧手的单臂和双臂任务。你将使用 RLinf 通过 PPO 训练 CNN policy，或者先在官方示范数据上微调 π₀.₅，再用 PPO 继续训练。

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
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 安装步骤中的 DexJoCo 资产 · DexJoCo LeRobot 数据集 · 基座检查点（见下文步骤）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

训练和评测各运行一个任务。在 ``env.train`` 和 ``env.eval`` 中设置 ``task_name``，描述即该任务的官方提示词：

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
     - Grasp the watering can and apply water to the plant.
   * - 单臂
     - ``hammer_nail``
     - 1000
     - Use the hammer to drive the nail into the wooden board.
   * - 单臂
     - ``fold_glasses``
     - 1200
     - Fold the glasses and place them into the case.
   * - 单臂
     - ``pinch_tongs``
     - 1000
     - Grasp the tongs and perform three consecutive open-close motions.
   * - 单臂
     - ``pick_bucket``
     - 1200
     - Place the boxed food into the bucket and then lift the bucket.
   * - 单臂
     - ``click_mouse``
     - 1200
     - Move the mouse to the purple mouse pad and click the left mouse button.
   * - 双臂
     - ``bimanual_assembly``
     - 1500
     - Grasp the tray with the left hand and the peg with the right hand, then insert the peg into the hole.
   * - 双臂
     - ``bimanual_hanoi``
     - 1500
     - Execute the final two moves of the three-level Tower of Hanoi: move the medium disk from the middle peg to the right peg with the right hand, then move the small disk from the left peg to the right peg with the left hand.
   * - 双臂
     - ``bimanual_microwave_cook``
     - 1100
     - Open the microwave door, place the food inside the microwave, close the door, and press the start button.
   * - 双臂
     - ``bimanual_photograph``
     - 1000
     - Grasp the camera with the left hand, align it with the logo, and press the shutter button with the right hand.
   * - 双臂
     - ``bimanual_unlock_ipad``
     - 1200
     - Grasp the iPad and enter the password 123 to unlock the device.

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - ``camera_names`` 中各相机的 640×640 RGB 图像（CNN 配方渲染 128×128），以及单臂任务 23 维 ``[pose7, hand16]``、双臂任务 46 维 ``[right_pose7, left_pose7, right_hand16, left_hand16]`` 的本体状态；每个 pose 由位置和标量在前的四元数组成。单臂任务使用 ``[front, wrist]``\ （``click_mouse`` 使用 ``[ego_right, wrist]``），双臂任务使用 ``[ego, wrist_left, wrist_right]``。
   * - 动作
     - 每条机械臂 22 维连续动作：3D 末端执行器绝对位置、3D 旋转向量和 16 个 Allegro 关节目标；44 维双臂动作中右臂在前。
   * - 奖励
     - 稀疏 0/1 成功奖励。
   * - 提示词
     - DexJoCo 任务的自然语言指令，供 π₀.₅ 使用。

安装
----------------------------------------

克隆 RLinf 仓库，并为你要运行的模型安装环境：

.. code:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   # 国内用户可添加 --use-mirror。

   # CNN
   bash requirements/install.sh embodied --env dexjoco
   source .venv/bin/activate

   # OpenPI π₀.₅
   # bash requirements/install.sh embodied --model openpi --env dexjoco
   # source .venv/bin/activate

安装脚本会把 DexJoCo 克隆到 ``.venv/dexjoco`` 并检出 commit ``8d23b0f``，激活环境时会把 ``DEXJOCO_ASSET_DIR`` 设为其中的 ``dexjoco/dexjoco/sim/envs`` 目录，场景 XML、网格、纹理和相机预设都在该目录下。如需使用其他副本，请设置 ``env.train.assets_path`` 和 ``env.eval.assets_path``。

下载数据集
----------------------------------------

π₀.₅ 配方在 `官方 LeRobot 示范数据 <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__ 上微调。下载一个任务的数据并计算归一化统计量：

.. code:: bash

   # export HF_ENDPOINT=https://hf-mirror.com
   hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
     --include 'dexjoco_lerobot_datasets/water_plant/**' \
     --local-dir /path/to/dexjoco
   PYTHONPATH=$(pwd) python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_dexjoco \
     --repo-id /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant

这些命令会：

1. 下载 ``water_plant`` 回合，包括前视和腕部视频、机器人状态、动作和任务文本。
2. 在数据集目录中写入 ``norm_stats.json``。SFT、PPO 和评测使用同一份统计量文件。

双臂任务传入 ``--config-name pi05_dexjoco_dual_arm``；``click_mouse`` 的主视角相机是 ``ego_right``，传入 ``--config-name pi05_dexjoco_click_mouse``。

下载模型
----------------------------------------

下载你要训练的模型检查点。

**CNN**

CNN + PPO 配方需要下载 ResNet 检查点：

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

DexJoCo 发布了 π₀.₅ 基座检查点，以及一个动作投影覆盖 44 维双臂动作的变体。下载基座检查点并转换为 RLinf OpenPI 格式，转换方法见 :doc:`sft_openpi`：

.. code-block:: bash

   # export HF_ENDPOINT=https://hf-mirror.com
   hf download DexJoCo/DexJoCo-Pi05 --include 'pi05_base/params/**' \
     --local-dir /path/to/jax
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model /path/to/jax/pi05_base \
     --input-norm-stats /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json \
     --output-model /path/to/pi05_base_openpi \
     --output-norm-stats /path/to/pi05_base_openpi/dexjoco/norm_stats.json \
     --action-dim 32 --action-horizon 30 --max-token-len 250

双臂任务改为下载 ``pi05_base_action_dim_44/params/**``，并使用 ``--action-dim 44`` 转换到 ``/path/to/pi05_base_action_dim_44_openpi``。

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

这些配方默认训练 ``water_plant``。换成其他任务时，在 ``env.train`` 和 ``env.eval`` 中覆盖 ``task_name`` 和 ``camera_names``，并把 ``max_episode_steps`` 设为该任务的时间上限。CNN 配方还需要把 ``env.eval.max_steps_per_rollout_epoch`` 设为该上限；π₀.₅ 配方则把 ``env.train`` 和 ``env.eval`` 中的 ``max_steps_per_rollout_epoch`` 都设为该上限向上取整到 30 的倍数。双臂任务使用 46 维状态和 44 维动作：CNN 需设置 ``actor.model.state_dim: 46`` 和 ``actor.model.action_dim: 44``；π₀.₅ 使用 SFT 一节中的双臂设置。

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
     - ``bimanual_unlock_ipad`` 的解锁数字序列，默认 ``[1, 2, 3]``；π₀.₅ 的提示词始终写 123。

使用 ``run_embodiment.sh`` 启动一个配置：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn

这条命令会：

1. 加载 ``examples/embodiment/config/dexjoco_ppo_cnn.yaml``，并把 actor、rollout 和 env worker 放在 GPU ``0`` 上。
2. 启动 8 个渲染 128×128 图像的训练环境。
3. 基于主视角图像和本体状态运行 PPO。

.. admonition:: 进一步配置
   :class: note

   - 组件放置和吞吐调优 → :doc:`组件放置 <../../concepts/placement>` 与 :doc:`执行模式 <../../concepts/execution_modes>`
   - 全量配置项 → :doc:`配置 <../../guides/index>`
   - 指标定义和日志后端 → :doc:`训练指标 <../../reference/metrics>`
   - 从检查点恢复 → :doc:`恢复训练 <../../guides/resume>`

π₀.₅ + SFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml`` 中设置以下字段：

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

双臂任务需要把 ``actor.model.openpi.config_name`` 设为 ``pi05_dexjoco_dual_arm``，把 ``actor.model.action_dim`` 和 ``actor.model.openpi.model_action_dim`` 设为 ``44``，并从转换得到的 44 维基座检查点开始训练。``click_mouse`` 需要把 ``actor.model.openpi.config_name`` 设为 ``pi05_dexjoco_click_mouse``。

π₀.₅ + PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml`` 中，把 ``actor.model.model_path`` 和 ``rollout.model.model_path`` 设为 SFT 的 ``global_step_<N>`` 目录，并把 ``actor.model.openpi_data.norm_stats_path`` 设为 SFT 使用的统计量文件。

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_openpi_pi05

这条命令会：

1. 加载 SFT 权重，并为 PPO 添加价值头。
2. 使用 16 个并行环境和 30 步动作块采集 rollout。
3. 用 PPO 更新 policy。

独立评测请使用统一的 :doc:`Evaluation CLI <../../evaluations/reference/cli>` 运行 ``dexjoco_openpi_pi05_eval``，该配置按 DexJoCo 官方协议评测 50 个回合，可在 1、2 或 5 张 GPU 上均分。把 ``rollout.model.model_path`` 设为 SFT 或 PPO 检查点，把 ``rollout.model.openpi_data.norm_stats_path`` 设为训练时使用的统计量文件；评测双臂检查点时，在 ``rollout.model`` 下使用同样的双臂设置。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir logs --port 6006

关键指标是 ``env/success_once``。完整指标说明见 :doc:`训练指标 <../../reference/metrics>`。

如需视频，请在环境配置中启用 video：

.. code:: yaml

   env:
     eval:
       video_cfg:
         save_video: True
         video_base_dir: ${runner.logger.log_path}/video/eval

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 配方
     - 结果描述
   * - π₀.₅ + SFT
     - 在原始运行使用的设置中（16 张 H100 GPU），``water_plant`` 第 5000 步的检查点在 50 个评测回合中成功 28 个（56%）。
   * - π₀.₅ + PPO
     - 从该检查点出发，在一个 8 卡 H100 节点上用 128 个环境训练并在每轮后评测（``runner.val_check_interval: 1``），64 个评测回合的成功率在第 3 轮后为 64.1%，第 5 轮后为 57.8%。
