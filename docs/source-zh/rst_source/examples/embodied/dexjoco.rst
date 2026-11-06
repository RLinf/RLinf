基于 DexJoCo 的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
   :align: center
   :width: 90%

   使用 Allegro 灵巧手浇花（图片来源：`DexJoCo <https://github.com/brave-eai/dexjoco>`__）。

使用 DexJoCo 官方 ``water_plant`` 示范数据微调 CNN 和 π₀.₅ policy，再用 PPO 继续训练。本页介绍 Franka 机械臂与 Allegro 灵巧手浇花任务的数据准备、训练和检查点评测流程。

概览
----------------------------------------

policy 根据前视和腕部图像抓取水壶，再给植物浇水。

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

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

训练和评测配置均使用 ``dexjoco_water_plant``：

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - 任务
     - 回合上限
     - 描述
   * - ``water_plant``
     - 1000 步
     - 抓起水壶并给植物浇水。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

两类 policy 使用以下观测、动作和奖励：

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 说明
   * - 观测
     - 前视和腕部 RGB 图像，加上 23 维本体状态：7 维末端位姿和 16 个 Allegro 关节位置。CNN 图像缩放到 128×128。
   * - 动作
     - 22 维连续动作：末端的三维绝对位置、三维旋转向量和 16 个 Allegro 关节目标。
   * - 奖励
     - 稀疏的 0/1 成功奖励。
   * - 提示词
     - “Grasp the watering can and apply water to the plant.”

安装
----------------------------------------

.. include:: _setup_common.rst
   :end-before: 然后，使用下列两种方式之一准备依赖

按所选 policy 安装依赖，后续命令均在 RLinf 仓库根目录执行：

.. code:: bash

   # CNN
   bash requirements/install.sh embodied --env dexjoco

   # π₀.₅ 使用：
   # bash requirements/install.sh embodied --model openpi --env dexjoco
   source .venv/bin/activate

下载数据集
----------------------------------------

两类配方使用同一份 `官方 LeRobot 示范数据 <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__。先下载 ``water_plant`` 回合：

.. code:: bash

   hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
     --include "dexjoco_lerobot_datasets/water_plant/**" \
     --local-dir /path/to/dexjoco

数据集包含前视和腕部视频、机器人状态、动作和任务文本。下面分别准备 CNN 和 π₀.₅ 的训练输入。

运行
----------------------------------------

选择一种 policy，依次完成 SFT、PPO 和评测。报告中的实验使用以下训练配置：

.. list-table::
   :header-rows: 1
   :widths: 17 49 18 16

   * - 配方
     - 配置
     - GPU
     - 训练步数
   * - CNN + SFT
     - ``examples/sft/config/dexjoco_sft_cnn.yaml``
     - 1 × 48 GB
     - 20,000 次更新
   * - CNN + PPO
     - ``examples/embodiment/config/dexjoco_ppo_cnn.yaml``
     - 8 × H100
     - 200 轮
   * - π₀.₅ + SFT
     - ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml``
     - 16 × H100
     - 5000 次更新
   * - π₀.₅ + PPO
     - ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml``
     - 8 × H100
     - 150 轮

CNN：SFT 与 PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下载 ResNet-10 主干，并把示范数据导出为图像、状态和长度为 10 的动作块：

.. code:: bash

   hf download RLinf/RLinf-ResNet10-pretrained \
     --local-dir /path/to/RLinf-ResNet10-pretrained
   PYTHONPATH=$(pwd) python toolkits/lerobot/export_image_action_chunks.py \
     --repo-id /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant \
     --output /path/to/dexjoco/dexjoco_cnn_datasets/water_plant \
     --cameras front wrist --image-size 128 --chunk 10 --state-dim 23

导出器生成训练数组和记录动作归一化统计量的 ``meta.json``。在 ``examples/sft/config/dexjoco_sft_cnn.yaml`` 中设置这些路径：

.. code:: yaml

   data:
     train_data_paths: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant
   actor:
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
       action_norm_stats: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

启动 SFT，以均方误差拟合示范动作块：

.. code:: bash

   bash examples/sft/run_vla_sft.sh dexjoco_sft_cnn

SFT 保存 ``checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``。从第 8000 步继续训练，在 ``examples/embodiment/config/dexjoco_ppo_cnn.yaml`` 中设置：

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
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
     enable_torch_compile: False
     enable_cuda_graph: False
   actor:
     seed: 42
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
       action_norm_stats: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

PPO 使用 128 个环境、action-level log-probability、1e-5 学习率和 1600 的全局批量。启动训练：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn

评测时使用训练所用的动作统计量。下面的命令在 8 张 GPU 上运行 64 个回合：

.. code:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_cnn_eval \
     runner.ckpt_path=/path/to/dexjoco_ppo_cnn/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt \
     rollout.model.model_path=/path/to/RLinf-ResNet10-pretrained \
     rollout.model.action_norm_stats=/path/to/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json \
     env.eval.seed=10000 env.eval.video_cfg.save_video=False

π₀.₅：SFT 与 PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

计算 OpenPI 归一化统计量，并下载 π₀.₅ 基座模型：

.. code:: bash

   PYTHONPATH=$(pwd) python toolkits/lerobot/calculate_norm_stats.py \
     --config-name pi05_dexjoco \
     --repo-id /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant
   hf download DexJoCo/DexJoCo-Pi05 --include "pi05_base/params/**" \
     --local-dir /path/to/jax
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
     --input-model /path/to/jax/pi05_base \
     --input-norm-stats /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json \
     --output-model /path/to/pi05_base_openpi \
     --output-norm-stats /path/to/pi05_base_openpi/dexjoco/norm_stats.json \
     --action-dim 32 --action-horizon 30 --max-token-len 250

这些命令生成 ``norm_stats.json`` 和 RLinf OpenPI 格式的模型。按照 :doc:`/rst_source/guides/multi_node` 准备双节点集群，然后在 ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml`` 中替换 ``cluster`` 配置段，并设置以下训练字段：

.. code:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor: 0-15
   runner:
     max_steps: 5000
   data:
     train_data_paths: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant
   actor:
     model:
       model_path: /path/to/pi05_base_openpi
       openpi:
         max_token_len: 250
         torch_compile: False
       openpi_data:
         norm_stats_path: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

SFT 使用全局批量 32、micro batch 2 和 30,000 步的学习率调度。在头节点启动：

.. code:: bash

   bash examples/sft/run_vla_sft.sh dexjoco_sft_openpi_pi05

训练得到的 ``checkpoints/global_step_5000`` 目录用于初始化 PPO。在 ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml`` 中设置：

.. code:: yaml

   rollout:
     model:
       model_path: /path/to/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000
   actor:
     model:
       model_path: /path/to/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000
       openpi_data:
         norm_stats_path: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

在单节点上启动 PPO。默认配方训练 150 轮，使用 token-level log-probability、5e-6 的 policy 学习率和 2e-4 的 value 学习率，每轮评测并保存检查点：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_openpi_pi05

使用训练所用的统计量评测保存的检查点：

.. code:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_openpi_pi05_eval \
     rollout.model.model_path=/path/to/dexjoco_ppo_openpi_pi05/checkpoints/global_step_150 \
     rollout.model.openpi_data.norm_stats_path=/path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

评测配置在 8 个 worker 上运行 64 个回合，种子为 10000，每个动作块执行 10 步、去噪 5 步，回合上限为 1000 步。其他评测选项见 :doc:`/rst_source/evaluations/reference/cli`。

可视化与结果
----------------------------------------

用 TensorBoard 查看 ``env/success_once`` 和 ``eval/success_once``：

.. code:: bash

   tensorboard --logdir ./logs --port 6006

使用 Weights & Biases 时，先运行 ``wandb login``，再启用两个日志后端：

.. code:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard", "wandb"]

.. figure:: https://raw.githubusercontent.com/matthewmzy/misc/4b5dbf3db3a251fc33415113b6dbe504f6c08202/pic/dexjoco_cnn_sft_ppo.png
   :align: center
   :width: 100%
   :alt: CNN 在 water_plant 上经过 8000 步 SFT 和 200 轮 PPO 的成功率。

   CNN 在 ``water_plant`` 上的 SFT + PPO 成功率（64 个评测回合）。

`CNN PPO 运行日志 <https://wandb.ai/matthewmzy-peking-university/rlinf-dexjoco/runs/cnn-mse-ppo-20261106>`__。

.. figure:: https://raw.githubusercontent.com/matthewmzy/misc/11c9e5e117ae7231055ec4b561284e131df71c0a/pic/dexjoco_pi05_sft_ppo.png
   :align: center
   :width: 100%
   :alt: π₀.₅ 在 water_plant 上经过 5000 步 SFT 和 150 轮 PPO 的成功率。

   π₀.₅ 在 ``water_plant`` 上的 SFT + PPO 成功率（64 个评测回合）。

`SFT + PPO 报告 <https://forge.coreweave.com/matthewmzy-peking-university/rlinf-dexjoco/reports/DexJoCo-water_plant:-pi0.5-SFT-to-PPO-B-with-one-evaluation-protocol--VmlldzoxODI0MDM2Nw==>`__ 包含评测曲线和原始日志。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 配方
     - 成功率
   * - CNN + SFT
     - 第 8000 步达到 46/64（71.9%）。
   * - CNN + PPO
     - 从 SFT 第 8000 步继续训练：第 0 轮为 46/64（71.9%），第 200 轮为 44/64（68.8%）；峰值为第 123 轮的 60/64（93.8%）。
   * - π₀.₅ + SFT
     - 第 5000 步达到 44/64（68.75%）。
   * - π₀.₅ + PPO
     - 从 SFT 第 5000 步继续训练：第 150 轮达到 63/64（98.44%），第 120 轮达到峰值 64/64（100%）。
