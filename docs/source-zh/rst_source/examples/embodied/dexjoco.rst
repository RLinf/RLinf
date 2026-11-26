DexJoCo 强化学习
================

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
   :align: center
   :width: 90%

   使用 Allegro 灵巧手浇花（图片来源：`DexJoCo <https://github.com/brave-eai/dexjoco>`__）。

使用 DexJoCo 官方 ``water_plant`` 示范数据微调 CNN 和 π₀.₅ 策略，再用 PPO 在线提升。DexJoCo 提供 11 项单臂与双臂灵巧手操作任务，本页覆盖 Franka 机械臂加 Allegro 灵巧手在 WaterPlant 上的数据准备、SFT、PPO 训练与评测。

概览
----

策略根据前视与腕部 RGB 图像抓取水壶并给植物浇水。

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

      WaterPlant

   .. grid-item-card:: 硬件
      :text-align: center

      NVIDIA GPU

| **你将完成：** 安装依赖 → 下载示范与基座权重 → 运行 SFT → 启动 PPO → 观察 ``env/success_once``。
| **前置条件：** :doc:`环境安装 </rst_source/start/installation>` · NVIDIA GPU（CNN SFT 1 卡，PPO/评测 8 × 80 GB，π₀.₅ SFT 8–16 × 80 GB）。

任务、观测与动作
~~~~~~~~~~~~~~~~

所有方案均运行 ``dexjoco_water_plant``\ （单回合上限 1000 步）：

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测
     - 前视与腕部 RGB 图像，加 23 维本体感知状态（7 维末端位姿 + 16 维 Allegro 关节位置）。CNN 图像缩放到 128×128。
   * - 动作
     - 22 维连续动作：末端 3D 绝对位置、3D 旋转向量及 16 个 Allegro 关节目标位置。
   * - 奖励
     - 稀疏 0/1 成功奖励。
   * - 提示词（仅 π₀.₅）
     - “Grasp the watering can and apply water to the plant.”

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: CNN（SFT + PPO）
      :sync: dexjoco-cnn

      先运行 ``dexjoco_sft_cnn``\ （1 × 48 GB GPU，20,000 步，高斯 NLL 损失），再运行 ``dexjoco_ppo_cnn``\ （8 × 80 GB GPU，200 轮，128 个并行环境）。

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --env dexjoco && source .venv/bin/activate

      **2. 下载并导出数据**

      下载 `官方 LeRobot 示范数据 <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__ 与 ResNet-10 预训练权重，并将示范导出为 10 步动作块（生成的 ``meta.json`` 供 SFT、PPO 与评测复用）：

      .. code-block:: bash

         hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
           --include "dexjoco_lerobot_datasets/water_plant/**" \
           --local-dir ./data/dexjoco
         hf download RLinf/RLinf-ResNet10-pretrained \
           --local-dir ./data/RLinf-ResNet10-pretrained
         PYTHONPATH=$(pwd) python toolkits/lerobot/export_image_action_chunks.py \
           --repo-id ./data/dexjoco/dexjoco_lerobot_datasets/water_plant \
           --output ./data/dexjoco/dexjoco_cnn_datasets/water_plant \
           --cameras front wrist --image-size 128 --chunk 10 --state-dim 23

      **3. 启动 SFT**

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh dexjoco_sft_cnn \
           data.train_data_paths="$(pwd)/data/dexjoco/dexjoco_cnn_datasets/water_plant" \
           actor.model.model_path="$(pwd)/data/RLinf-ResNet10-pretrained" \
           actor.model.action_norm_stats="$(pwd)/data/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json"

      **4. 启动 PPO**

      加载第 8000 步 SFT checkpoint 启动 PPO：

      .. code-block:: bash

         SFT_CKPT=$(ls -d "$(pwd)"/logs/*-dexjoco_sft_cnn/dexjoco_sft_cnn_nll/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt | tail -n 1)
         bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn \
           runner.ckpt_path="$SFT_CKPT" \
           actor.model.model_path="$(pwd)/data/RLinf-ResNet10-pretrained" \
           rollout.model.model_path="$(pwd)/data/RLinf-ResNet10-pretrained" \
           actor.model.action_norm_stats="$(pwd)/data/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json"

      .. dropdown:: 独立评测与 MSE 基线对照（可选）

         评测任意 SFT 或 PPO 的 ``full_weights.pt``\ （8 卡 64 个 episode；参数见 :doc:`/rst_source/evaluations/reference/cli`）：

         .. code-block:: bash

            CKPT=$(ls -d "$(pwd)"/logs/*-dexjoco_ppo_cnn/dexjoco_ppo_cnn_nll/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt | tail -n 1)
            bash evaluations/run_eval.sh dexjoco dexjoco_cnn_eval \
              runner.ckpt_path="$CKPT" \
              rollout.model.model_path="$(pwd)/data/RLinf-ResNet10-pretrained" \
              rollout.model.action_norm_stats="$(pwd)/data/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json" \
              runner.logger.experiment_name=dexjoco_cnn_nll_ppo200_eval

         如需复现 MSE 基线，在第 3 步追加 ``actor.model.sft_loss_type=mse runner.logger.experiment_name=dexjoco_sft_cnn_mse``，并在第 4 步加载其第 8000 步权重。

   .. tab-item:: π₀.₅（SFT + PPO）
      :sync: dexjoco-pi05

      先运行 ``dexjoco_sft_openpi_pi05``\ （默认 2 节点 × 8 H100、5000 步；多机见 :doc:`/rst_source/guides/multi_node`，单机 8 卡可追加 ``cluster.num_nodes=1``），再运行 ``dexjoco_ppo_openpi_pi05``\ （8 × 80 GB GPU，150 轮）。

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env dexjoco && source .venv/bin/activate

      **2. 下载并转换基座权重**

      下载 `官方 LeRobot 示范数据 <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__ 与 π₀.₅ 基座权重，计算归一化统计并转换为 RLinf OpenPI 格式：

      .. code-block:: bash

         hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
           --include "dexjoco_lerobot_datasets/water_plant/**" \
           --local-dir ./data/dexjoco
         hf download DexJoCo/DexJoCo-Pi05 --include "pi05_base/params/**" \
           --local-dir ./data/jax
         PYTHONPATH=$(pwd) python toolkits/lerobot/calculate_norm_stats.py \
           --config-name pi05_dexjoco \
           --repo-id ./data/dexjoco/dexjoco_lerobot_datasets/water_plant
         python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
           --input-model ./data/jax/pi05_base \
           --input-norm-stats ./data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json \
           --output-model ./data/pi05_base_openpi \
           --output-norm-stats ./data/pi05_base_openpi/dexjoco/norm_stats.json \
           --action-dim 32 --action-horizon 30 --max-token-len 250

      **3. 启动 SFT**

      .. code-block:: bash

         # 2 节点复现（16 卡）；单机 8 卡在末尾追加 cluster.num_nodes=1
         bash examples/sft/run_vla_sft.sh dexjoco_sft_openpi_pi05 \
           data.train_data_paths="$(pwd)/data/dexjoco/dexjoco_lerobot_datasets/water_plant" \
           actor.model.model_path="$(pwd)/data/pi05_base_openpi" \
           actor.model.openpi_data.norm_stats_path="$(pwd)/data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json"

      **4. 启动 PPO**

      加载第 5000 步 SFT checkpoint 启动 PPO：

      .. code-block:: bash

         SFT_CKPT=$(ls -d "$(pwd)"/logs/*-dexjoco_sft_openpi_pi05/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000 | tail -n 1)
         bash examples/embodiment/run_embodiment.sh dexjoco_ppo_openpi_pi05 \
           actor.model.model_path="$SFT_CKPT" \
           rollout.model.model_path="$SFT_CKPT" \
           actor.model.openpi_data.norm_stats_path="$(pwd)/data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json"

      .. dropdown:: 独立评测（可选）

         评测任意 SFT 或 PPO checkpoint 目录（8 卡 64 个 episode；参数见 :doc:`/rst_source/evaluations/reference/cli`）：

         .. code-block:: bash

            CKPT=$(ls -d "$(pwd)"/logs/*-dexjoco_ppo_openpi_pi05/dexjoco_ppo_openpi_pi05/checkpoints/global_step_150 | tail -n 1)
            bash evaluations/run_eval.sh dexjoco dexjoco_openpi_pi05_eval \
              rollout.model.model_path="$CKPT" \
              rollout.model.openpi_data.norm_stats_path="$(pwd)/data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json" \
              runner.logger.experiment_name=dexjoco_pi05_ppo150_eval

可视化与结果
------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``env/success_once`` 与 ``eval/success_once``，指标定义与 Weights & Biases / SwanLab 配置见 :doc:`训练指标 </rst_source/reference/metrics>`。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/3f186472f997ecd29977191f8e4da004cff4a302/pic/dexjoco_cnn_nll_vs_mse.png
   :align: center
   :width: 100%
   :alt: CNN 分别经过 MSE、NLL SFT 后在 water_plant 上的 PPO 成功率。

   CNN 分别经过 MSE、NLL SFT 后在 ``water_plant`` 上的 PPO 成功率（64 个评测 episode；第 0 轮为第 8000 步 SFT 独立评测）。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/dc2607a2769dfd49cca117cdf42e5ef3f70b8dcf/pic/dexjoco_pi05_sft_ppo.png
   :align: center
   :width: 100%
   :alt: π₀.₅ 在 water_plant 上经过 5000 步 SFT 和 150 轮 PPO 的成功率。

   π₀.₅ 在 ``water_plant`` 上的 SFT + PPO 成功率（64 个评测 episode）。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 方案
     - 成功率
   * - CNN + SFT 基线
     - 第 8000 步（独立评测）：MSE 为 46/64（71.9%），NLL 为 42/64（65.6%）。
   * - CNN + PPO（MSE SFT）
     - 第 200 轮（独立评测）：44/64（68.8%）；训练期间峰值（第 123 轮）：60/64（93.8%）。
   * - CNN + PPO（NLL SFT）
     - 第 200 轮（独立评测）：59/64（92.2%）；训练期间峰值（第 163 轮）：64/64（100%）。
   * - π₀.₅ + SFT
     - 第 5000 步（独立评测）：50/64（78.1%）。
   * - π₀.₅ + PPO
     - 第 150 轮（独立评测）：63/64（98.44%）；独立评测峰值（第 120 轮）：64/64（100%）。
