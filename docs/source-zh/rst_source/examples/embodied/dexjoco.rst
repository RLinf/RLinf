DexJoCo 强化学习
================

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
   :align: center
   :width: 90%

   使用 Allegro 灵巧手浇花（图片来源：`DexJoCo <https://github.com/brave-eai/dexjoco>`__）。

使用 DexJoCo 官方 ``water_plant`` 示范数据训练 CNN 和 π₀.₅ policy，再用 PPO 微调。本页覆盖 Franka 机械臂与 Allegro 灵巧手浇花任务的数据准备、SFT、PPO 训练与 checkpoint 评测。DexJoCo 提供 11 项单臂和双臂任务，本页配置与结果针对 WaterPlant。

概览
----

policy 根据前视与腕部 RGB 图像抓取水壶并给植物浇水。

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

| **前置条件：** :doc:`环境安装 </rst_source/start/installation>` · NVIDIA GPU（1–16 卡，按下方方案选择）。

任务、观测与动作
~~~~~~~~~~~~~~~~

所有配置均使用 ``dexjoco_water_plant``\ （单回合上限 1000 步）：

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
   :end-before: 然后，使用下列两种方式之一准备依赖

各阶段使用的仓库内置配置如下（SFT 的“步”指一次优化器更新；PPO 的“轮”指一次采样加多步更新；评测在 8 卡上跑 64 个固定初始状态 episode，随机种子 10000）：

.. list-table::
   :header-rows: 1
   :widths: 17 49 18 16

   * - 方案
     - 配置
     - GPU（单卡显存）
     - 运行规模
   * - CNN + SFT
     - ``examples/sft/config/dexjoco_sft_cnn.yaml``
     - 1 × RTX PRO 5000 Blackwell（48 GB）
     - 20,000 步 SFT
   * - CNN + PPO
     - ``examples/embodiment/config/dexjoco_ppo_cnn.yaml``
     - 8 × H100（80 GB）
     - 200 轮 PPO
   * - CNN 评测
     - ``evaluations/dexjoco/dexjoco_cnn_eval.yaml``
     - 8 × H100（80 GB）
     - 64 个 episode
   * - π₀.₅ + SFT
     - ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml``
     - 16 × H100（80 GB，2 节点）
     - 5000 步 SFT
   * - π₀.₅ + PPO
     - ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml``
     - 8 × H100（80 GB）
     - 150 轮 PPO
   * - π₀.₅ 评测
     - ``evaluations/dexjoco/dexjoco_openpi_pi05_eval.yaml``
     - 8 × H100（80 GB）
     - 64 个 episode

每次训练的 checkpoint 保存在 ``logs/<timestamp>-<config>/<experiment_name>/checkpoints/global_step_<N>``，具体时间戳可查看对应目录的 ``run_embodiment.log``。

.. tab-set::

   .. tab-item:: CNN（SFT + PPO）
      :sync: dexjoco-cnn

      **1. 安装环境**

      .. code-block:: bash

         bash requirements/install.sh embodied --env dexjoco
         source .venv/bin/activate

      **2. 下载并导出数据**

      下载 `官方 LeRobot 示范数据 <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__ 与 ResNet-10 预训练权重，并将示范导出为 10 步动作块：

      .. code-block:: bash

         hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
           --include "dexjoco_lerobot_datasets/water_plant/**" \
           --local-dir ./data/dexjoco
         hf download RLinf/RLinf-ResNet10-pretrained \
           --local-dir ./data/RLinf-ResNet10-pretrained

      .. code-block:: bash

         PYTHONPATH=$(pwd) python toolkits/lerobot/export_image_action_chunks.py \
           --repo-id ./data/dexjoco/dexjoco_lerobot_datasets/water_plant \
           --output ./data/dexjoco/dexjoco_cnn_datasets/water_plant \
           --cameras front wrist --image-size 128 --chunk 10 --state-dim 23

      导出目录会生成训练数组与 ``meta.json``\ （动作均值与标准差），后续 SFT、PPO 与评测需沿用同一份 ``meta.json``。

      **3. SFT 与基线评测**

      在 ``examples/sft/config/dexjoco_sft_cnn.yaml`` 中填写本地路径（默认高斯 NLL 损失，训练 20,000 步，每 2000 步保存一次），然后启动 SFT：

      .. code-block:: yaml

         data:
           train_data_paths: /abs/path/to/data/dexjoco/dexjoco_cnn_datasets/water_plant
         actor:
           model:
             model_path: /abs/path/to/data/RLinf-ResNet10-pretrained
             action_norm_stats: /abs/path/to/data/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

      .. code-block:: bash

         bash examples/sft/run_vla_sft.sh dexjoco_sft_cnn

      开始 PPO 前，先在 ``evaluations/dexjoco/dexjoco_cnn_eval.yaml`` 中设置 SFT 第 8000 步的 ``runner.ckpt_path``、``rollout.model.model_path`` 与 ``rollout.model.action_norm_stats``，并运行评测：

      .. code-block:: bash

         bash evaluations/run_eval.sh dexjoco dexjoco_cnn_eval \
           runner.logger.experiment_name=dexjoco_cnn_nll_sft8000_eval

      **4. PPO 与评测**

      在 ``examples/embodiment/config/dexjoco_ppo_cnn.yaml`` 中填写第 8000 步 SFT checkpoint 与模型路径（rollout 自动继承 ``actor.model.model_path``；128 个环境、动作级 log-prob、lr 1e-5、全局批大小 1600）：

      .. code-block:: yaml

         runner:
           ckpt_path: /abs/path/to/logs/<sft-timestamp>-dexjoco_sft_cnn/dexjoco_sft_cnn_nll/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt
         actor:
           model:
             model_path: /abs/path/to/data/RLinf-ResNet10-pretrained
             action_norm_stats: /abs/path/to/data/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn

      评测第 200 轮 PPO 时，将 ``evaluations/dexjoco/dexjoco_cnn_eval.yaml`` 中的 ``runner.ckpt_path`` 改为 ``global_step_200/.../full_weights.pt`` 后执行：

      .. code-block:: bash

         bash evaluations/run_eval.sh dexjoco dexjoco_cnn_eval \
           runner.logger.experiment_name=dexjoco_cnn_nll_ppo200_eval

      .. dropdown:: CNN MSE 损失对照（可选）

         复现 MSE 基线时，复用上面导出的数据，在 SFT 中切换为 MSE 损失（``actor.model.sft_loss_type=mse``）并使用独立实验名 ``dexjoco_sft_cnn_mse``。

         PPO 加载 MSE 第 8000 步权重并设实验名 ``dexjoco_ppo_cnn_mse``；评测分别使用 ``dexjoco_cnn_mse_sft8000_eval`` 与 ``dexjoco_cnn_mse_ppo200_eval``。

   .. tab-item:: π₀.₅（SFT + PPO）
      :sync: dexjoco-pi05

      **1. 安装环境**

      .. code-block:: bash

         bash requirements/install.sh embodied --model openpi --env dexjoco
         source .venv/bin/activate

      **2. 下载并转换基座权重**

      下载 `官方 LeRobot 示范数据 <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__ 与 π₀.₅ 基座模型，计算归一化统计并转换为 RLinf OpenPI 格式：

      .. code-block:: bash

         hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
           --include "dexjoco_lerobot_datasets/water_plant/**" \
           --local-dir ./data/dexjoco
         hf download DexJoCo/DexJoCo-Pi05 --include "pi05_base/params/**" \
           --local-dir ./data/jax

      .. code-block:: bash

         PYTHONPATH=$(pwd) python toolkits/lerobot/calculate_norm_stats.py \
           --config-name pi05_dexjoco \
           --repo-id ./data/dexjoco/dexjoco_lerobot_datasets/water_plant
         python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
           --input-model ./data/jax/pi05_base \
           --input-norm-stats ./data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json \
           --output-model ./data/pi05_base_openpi \
           --output-norm-stats ./data/pi05_base_openpi/dexjoco/norm_stats.json \
           --action-dim 32 --action-horizon 30 --max-token-len 250

      **3. SFT 与基线评测**

      参考 SFT 配置 ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml`` 默认使用 **2 节点 × 8 张 H100**\ （``cluster.num_nodes: 2``，多机搭建见 :doc:`/rst_source/guides/multi_node`）；单机快速验证可在命令末尾追加 ``cluster.num_nodes=1``。先在配置中填写本地路径：

      .. code-block:: yaml

         data:
           train_data_paths: /abs/path/to/data/dexjoco/dexjoco_lerobot_datasets/water_plant
         actor:
           model:
             model_path: /abs/path/to/data/pi05_base_openpi
             openpi_data:
               norm_stats_path: /abs/path/to/data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

      .. code-block:: bash

         # 2 节点复现（16 卡，5000 步）
         bash examples/sft/run_vla_sft.sh dexjoco_sft_openpi_pi05

         # 单机快速验证（8 卡）
         bash examples/sft/run_vla_sft.sh dexjoco_sft_openpi_pi05 cluster.num_nodes=1

      在 8 卡节点上评测第 5000 步 SFT checkpoint：将 ``evaluations/dexjoco/dexjoco_openpi_pi05_eval.yaml`` 中的 ``rollout.model.model_path`` 与 ``norm_stats_path`` 指向 ``global_step_5000`` 后运行：

      .. code-block:: bash

         bash evaluations/run_eval.sh dexjoco dexjoco_openpi_pi05_eval \
           runner.logger.experiment_name=dexjoco_pi05_sft5000_eval

      **4. PPO 与评测**

      在 ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml`` 中填写第 5000 步 SFT checkpoint 与归一化统计路径（rollout 自动继承 ``actor.model.model_path``；150 轮、token 级 log-prob、policy lr 5e-6、value lr 2e-4）：

      .. code-block:: yaml

         actor:
           model:
             model_path: /abs/path/to/logs/<sft-timestamp>-dexjoco_sft_openpi_pi05/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000
             openpi_data:
               norm_stats_path: /abs/path/to/data/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh dexjoco_ppo_openpi_pi05

      评测第 150 轮 PPO 时，通过 Hydra 参数传入 checkpoint 路径即可（其他参数见 :doc:`/rst_source/evaluations/reference/cli`）：

      .. code-block:: bash

         bash evaluations/run_eval.sh dexjoco dexjoco_openpi_pi05_eval \
           rollout.model.model_path=/abs/path/to/logs/<ppo-timestamp>-dexjoco_ppo_openpi_pi05/dexjoco_ppo_openpi_pi05/checkpoints/global_step_150 \
           runner.logger.experiment_name=dexjoco_pi05_ppo150_eval

可视化与结果
------------

运行 ``tensorboard --logdir ./logs --port 6006`` 查看 ``env/success_once`` 与 ``eval/success_once``，指标定义与 Weights & Biases / SwanLab 配置见 :doc:`训练指标 </rst_source/reference/metrics>`。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/3f186472f997ecd29977191f8e4da004cff4a302/pic/dexjoco_cnn_nll_vs_mse.png
   :align: center
   :width: 100%
   :alt: CNN 分别经过 MSE、NLL SFT 后在 water_plant 上的 PPO 成功率。

   CNN 分别经过 MSE、NLL SFT 后在 ``water_plant`` 上的 PPO 成功率（64 个评测 episode）。

CNN 图中的第 0 轮为 SFT 第 8000 步 checkpoint 的独立评测，第 1–200 轮为训练期间评测（TensorBoard step ``N - 1`` 对应第 ``N`` 轮）。单 worker 扫描 10 个 NLL SFT checkpoint 时，第 8000 步成功率最高，为 50/64（78.1%）；在统一对比使用的 8 worker 评测配置上为 42/64（65.6%）。

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
     - 第 200 轮（独立评测）：44/64（68.8%）；训练期间评测峰值（第 123 轮）：60/64（93.8%）。
   * - CNN + PPO（NLL SFT）
     - 第 200 轮（独立评测）：59/64（92.2%）；训练期间评测首次峰值（第 163 轮）：64/64（100%）。
   * - π₀.₅ + SFT
     - 第 5000 步（独立评测）：50/64（78.1%）。
   * - π₀.₅ + PPO
     - 第 150 轮（独立评测）：63/64（98.44%）；独立评测峰值（第 120 轮）：64/64（100%）。
