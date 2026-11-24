基于 DexJoCo 的强化学习训练
========================================

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/main/docs/pics/water_plant_4.png
   :align: center
   :width: 90%

   使用 Allegro 灵巧手浇花（图片来源：`DexJoCo <https://github.com/brave-eai/dexjoco>`__）。

使用 DexJoCo 官方 ``water_plant`` 示范数据训练 CNN 和 π₀.₅ policy，再用 PPO 微调。本页介绍 Franka 机械臂与 Allegro 灵巧手浇花任务的数据准备、训练和 checkpoint 评测流程。DexJoCo 包含 11 项单臂和双臂任务，本页配置与实验结果均针对 WaterPlant。

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

训练和评测配置均使用 ``dexjoco_water_plant``\ ：

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - 任务
     - episode 步数上限
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
     - 前视和腕部 RGB 图像，加上 23 维本体感知状态：7 维末端位姿和 16 个 Allegro 关节位置。CNN 图像缩放到 128×128。
   * - 动作
     - 22 维连续动作：末端的三维绝对位置、三维旋转向量和 16 个 Allegro 关节的目标位置。
   * - 奖励
     - 稀疏的 0/1 成功奖励。
   * - 提示词（仅 π₀.₅）
     - “Grasp the watering can and apply water to the plant.”

安装
----------------------------------------

.. include:: _setup_common.rst
   :end-before: 然后，使用下列两种方式之一准备依赖

按所选 policy 安装依赖，后续命令均在 RLinf 仓库根目录执行。安装脚本会在仓库根目录的 ``.venv`` 中创建虚拟环境：

.. code:: bash

   # CNN
   bash requirements/install.sh embodied --env dexjoco

   # π₀.₅ 使用：
   # bash requirements/install.sh embodied --model openpi --env dexjoco
   source .venv/bin/activate

下载数据集
----------------------------------------

两类模型使用同一份 `官方 LeRobot 示范数据 <https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot>`__。下载 ``water_plant`` 的示范轨迹：

.. code:: bash

   hf download DexJoCo/DexJoCo-Datasets-LeRobot --repo-type dataset \
     --include "dexjoco_lerobot_datasets/water_plant/**" \
     --local-dir /path/to/dexjoco

数据集包含前视和腕部视频、机器人状态、动作和任务文本。下面分别准备 CNN 和 π₀.₅ 的训练输入。

运行
----------------------------------------

选择一种 policy，按下文步骤训练并评测 checkpoint。配置文件已包含复现实验所需的参数，填写下文列出的本地路径即可。各阶段使用以下配置：

.. list-table::
   :header-rows: 1
   :widths: 17 49 18 16

   * - 方案
     - 配置
     - GPU（括号内为每张卡的显存）
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
     - 16 × H100（80 GB）
     - 5000 步 SFT
   * - π₀.₅ + PPO
     - ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml``
     - 8 × H100（80 GB）
     - 150 轮 PPO
   * - π₀.₅ 评测
     - ``evaluations/dexjoco/dexjoco_openpi_pi05_eval.yaml``
     - 8 × H100（80 GB）
     - 64 个 episode

SFT 的“步”指一次优化器更新；PPO 的“轮”指一次轨迹采样及随后的训练，其中包含多次优化器更新。下文的训练进度和 checkpoint 分别按这两种方式计数，评测规模按 episode 数计。

每次启动训练，脚本都会在仓库根目录下创建 ``logs/<timestamp>-<config>``\ 。checkpoint 保存在该目录下的 ``<experiment_name>/checkpoints/global_step_<N>``\ 。下文中的 ``/path/to/RLinf`` 填写仓库根目录，``<sft-timestamp>`` 和 ``<ppo-timestamp>`` 分别填写对应训练目录的时间戳。每次运行的 ``run_embodiment.log`` 都记录了包含日志路径的启动命令。

CNN：SFT 与 PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下载 ResNet-10 预训练权重，并将示范数据导出为 CNN 训练所需的图像、状态和动作序列：

.. code:: bash

   hf download RLinf/RLinf-ResNet10-pretrained \
     --local-dir /path/to/RLinf-ResNet10-pretrained
   PYTHONPATH=$(pwd) python toolkits/lerobot/export_image_action_chunks.py \
     --repo-id /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant \
     --output /path/to/dexjoco/dexjoco_cnn_datasets/water_plant \
     --cameras front wrist --image-size 128 --chunk 10 --state-dim 23

导出命令会缩放图像，并为每帧观测配上示范中连续 10 步的动作。输出目录包含这些训练数据和 ``meta.json``\ ，后者记录动作的均值与标准差。SFT、PPO 和评测沿用同一份统计量，保证动作缩放一致。在 ``examples/sft/config/dexjoco_sft_cnn.yaml`` 中填写：

.. code:: yaml

   data:
     train_data_paths: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant
   actor:
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
       action_norm_stats: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

默认配置使用高斯负对数似然（NLL）训练 CNN。SFT 共训练 20,000 步，每 2000 步保存一个 checkpoint：

.. code:: bash

   bash examples/sft/run_vla_sft.sh dexjoco_sft_cnn

开始 PPO 前，先评测 SFT 第 8000 步的 checkpoint。在 ``evaluations/dexjoco/dexjoco_cnn_eval.yaml`` 中设置实验名，填写 checkpoint、ResNet 和动作统计量的路径：

.. code:: yaml

   runner:
     logger:
       experiment_name: dexjoco_cnn_nll_sft8000_eval
     ckpt_path: /path/to/RLinf/logs/<sft-timestamp>-dexjoco_sft_cnn/dexjoco_sft_cnn_nll/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt
   rollout:
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
       action_norm_stats: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

在 8 张 GPU 上运行评测。配置使用 8 个 worker 和同一组 64 个初始状态，随机种子为 10000。每次预测执行 10 个动作，每个 episode 最多执行 1000 步动作：

.. code:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_cnn_eval

在 ``examples/embodiment/config/dexjoco_ppo_cnn.yaml`` 中填写模型、归一化统计量和选定 checkpoint 的路径：

.. code:: yaml

   runner:
     ckpt_path: /path/to/RLinf/logs/<sft-timestamp>-dexjoco_sft_cnn/dexjoco_sft_cnn_nll/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt
   rollout:
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
   actor:
     model:
       model_path: /path/to/RLinf-ResNet10-pretrained
       action_norm_stats: /path/to/dexjoco/dexjoco_cnn_datasets/water_plant/meta.json

PPO 使用 128 个环境和动作级对数概率（action-level），policy 学习率为 1e-5，全局批大小为 1600。启动训练：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_cnn

PPO 将 checkpoint 保存在 ``logs/<ppo-timestamp>-dexjoco_ppo_cnn/dexjoco_ppo_cnn_nll/checkpoints`` 下。评测第 200 轮时，在同一份评测 YAML 中只修改 checkpoint 和实验名：

.. code:: yaml

   runner:
     logger:
       experiment_name: dexjoco_cnn_nll_ppo200_eval
     ckpt_path: /path/to/RLinf/logs/<ppo-timestamp>-dexjoco_ppo_cnn/dexjoco_ppo_cnn_nll/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt

运行 PPO checkpoint 评测：

.. code:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_cnn_eval

π₀.₅：SFT 与 PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

π₀.₅ 直接读取原始 LeRobot 数据。先计算状态和动作的归一化统计量，再下载并转换基座模型权重：

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

第一条命令在数据集目录下生成 ``norm_stats.json``\ ，SFT、PPO 和评测均使用这份统计量。权重转换命令将 PyTorch 权重按 RLinf 的 OpenPI 格式保存到 ``/path/to/pi05_base_openpi``\ ，供 SFT 加载。SFT 的训练数据仍使用原始 LeRobot 目录。

按照 :doc:`/rst_source/guides/multi_node` 准备两台各有 8 张 H100 的节点。配置文件已设置 16 张 GPU 和 5000 步 SFT，在 ``examples/sft/config/dexjoco_sft_openpi_pi05.yaml`` 中填写以下路径：

.. code:: yaml

   data:
     train_data_paths: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant
   actor:
     model:
       model_path: /path/to/pi05_base_openpi
       openpi_data:
         norm_stats_path: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

SFT 的全局批大小为 32，每张 GPU 的微批大小为 2，学习率按 30,000 步的调度计划变化。在头节点启动：

.. code:: bash

   bash examples/sft/run_vla_sft.sh dexjoco_sft_openpi_pi05

实验使用 SFT 结束时第 5000 步的 checkpoint 初始化 PPO。这个 checkpoint 保存在 ``logs/<sft-timestamp>-dexjoco_sft_openpi_pi05/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000``\ 。

后续评测和 PPO 均在有 8 张 GPU 的头节点上运行。确认头节点能读取这个 checkpoint 目录，以及数据集目录下的 ``norm_stats.json``\ 。先评测 SFT 基线，在 ``evaluations/dexjoco/dexjoco_openpi_pi05_eval.yaml`` 中填写这两个路径：

.. code:: yaml

   rollout:
     model:
       model_path: /path/to/RLinf/logs/<sft-timestamp>-dexjoco_sft_openpi_pi05/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000
       openpi_data:
         norm_stats_path: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

运行 SFT 基线评测：

.. code:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_openpi_pi05_eval \
     runner.logger.experiment_name=dexjoco_pi05_sft5000_eval

评测配置使用 8 个 worker 和同一组 64 个初始状态，随机种子为 10000。每次预测执行 10 个动作，模型采样使用 5 步去噪，每个 episode 最多执行 1000 步动作。下文评测 PPO checkpoint 时沿用这套配置。

用同一个 SFT checkpoint 初始化 PPO。在 ``examples/embodiment/config/dexjoco_ppo_openpi_pi05.yaml`` 中设置：

.. code:: yaml

   rollout:
     model:
       model_path: /path/to/RLinf/logs/<sft-timestamp>-dexjoco_sft_openpi_pi05/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000
   actor:
     model:
       model_path: /path/to/RLinf/logs/<sft-timestamp>-dexjoco_sft_openpi_pi05/dexjoco_sft_openpi_pi05/checkpoints/global_step_5000
       openpi_data:
         norm_stats_path: /path/to/dexjoco/dexjoco_lerobot_datasets/water_plant/norm_stats.json

在同一头节点上启动 PPO。默认配置训练 150 轮，使用 token 级对数概率（token-level），policy 学习率为 5e-6，价值网络学习率为 2e-4。每轮评测并保存 checkpoint：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh dexjoco_ppo_openpi_pi05

PPO 将第 150 轮的 checkpoint 保存在 ``logs/<ppo-timestamp>-dexjoco_ppo_openpi_pi05/dexjoco_ppo_openpi_pi05/checkpoints/global_step_150``\ 。在上面的评测配置中，仅将 ``rollout.model.model_path`` 改为这个 checkpoint：

.. code:: yaml

   rollout:
     model:
       model_path: /path/to/RLinf/logs/<ppo-timestamp>-dexjoco_ppo_openpi_pi05/dexjoco_ppo_openpi_pi05/checkpoints/global_step_150

用单独的实验名运行 PPO 评测：

.. code:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_openpi_pi05_eval \
     runner.logger.experiment_name=dexjoco_pi05_ppo150_eval

其他评测选项见 :doc:`/rst_source/evaluations/reference/cli`。

CNN：复现 MSE 对照
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

复现 MSE 对照时，复用上面导出的数据，重新运行 SFT 和 PPO。在 SFT 配置中设置 ``actor.model.sft_loss_type: mse`` 和 ``runner.logger.experiment_name: dexjoco_sft_cnn_mse``\ 。PPO 使用单独的实验名，并加载 MSE 训练第 8000 步的权重：

.. code:: yaml

   runner:
     logger:
       experiment_name: dexjoco_ppo_cnn_mse
     ckpt_path: /path/to/RLinf/logs/<sft-timestamp>-dexjoco_sft_cnn/dexjoco_sft_cnn_mse/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt

评测 MSE 的 SFT 基线时，在评测 YAML 中改用 ``dexjoco_sft_cnn_mse`` 的第 8000 步 checkpoint，实验名设为 ``dexjoco_cnn_mse_sft8000_eval``\ ，再运行同一条评测命令。

PPO 评测的实验名设为 ``dexjoco_cnn_mse_ppo200_eval``\ ，``runner.ckpt_path`` 设为 ``/path/to/RLinf/logs/<ppo-timestamp>-dexjoco_ppo_cnn/dexjoco_ppo_cnn_mse/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt``\ 。两处时间戳分别填写 MSE SFT 和 MSE PPO 运行目录的时间戳，其余设置保持一致。

可视化与结果
----------------------------------------

用 TensorBoard 查看 ``env/success_once`` 和 ``eval/success_once``\ ：

.. code:: bash

   tensorboard --logdir ./logs --port 6006

使用 Weights & Biases 时，先运行 ``wandb login``\ ，再启用两个日志后端：

.. code:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard", "wandb"]

.. figure:: https://raw.githubusercontent.com/RLinf/misc/3f186472f997ecd29977191f8e4da004cff4a302/pic/dexjoco_cnn_nll_vs_mse.png
   :align: center
   :width: 100%
   :alt: CNN 分别经过 MSE、NLL SFT 后在 water_plant 上的 PPO 成功率。

   CNN 分别经过 MSE、NLL SFT 后在 ``water_plant`` 上的 PPO 成功率（64 个评测 episode）。

CNN 图中的第 0 轮来自 SFT 第 8000 步 checkpoint 的独立评测，第 1–200 轮来自训练期间的评测。TensorBoard 的 step ``N - 1`` 记录完成第 ``N`` 轮 PPO 后的评测结果。

CNN 的 SFT 评测逐个检查了 10 个 NLL checkpoint，每次用一个 worker 运行 64 个 episode。第 8000 步的成功率最高，为 50/64（78.1%），因此选它作为 PPO 起点。使用 8 个 worker 的评测配置后，这个起点的成功率为 42/64（65.6%）。SFT 与 PPO checkpoint 的对比沿用这套配置。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/dc2607a2769dfd49cca117cdf42e5ef3f70b8dcf/pic/dexjoco_pi05_sft_ppo.png
   :align: center
   :width: 100%
   :alt: π₀.₅ 在 water_plant 上经过 5000 步 SFT 和 150 轮 PPO 的成功率。

   π₀.₅ 在 ``water_plant`` 上的 SFT + PPO 成功率（64 个评测 episode）。

下表中 π₀.₅ 的 SFT 和 PPO checkpoint 均在同一组 64 个初始状态上独立评测。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 方案
     - 成功率
   * - CNN + SFT 基线
     - 第 8000 步（独立评测）：MSE 为 46/64（71.9%），NLL 为 42/64（65.6%）。
   * - CNN + PPO（MSE SFT）
     - 第 200 轮（独立评测）：44/64（68.8%）；训练期间的评测峰值（第 123 轮）：60/64（93.8%）。
   * - CNN + PPO（NLL SFT）
     - 第 200 轮（独立评测）：59/64（92.2%）；训练期间的评测首次峰值（第 163 轮）：64/64（100%）。
   * - π₀.₅ + SFT
     - 第 5000 步（独立评测）：50/64（78.1%）。
   * - π₀.₅ + PPO
     - 第 150 轮（独立评测）：63/64（98.44%）；独立评测峰值（第 120 轮）：64/64（100%）。
