LingbotVLA 2.0 监督微调
========================================

使用 RLinf 从 LingbotVLA 2.0 Foundation Base 开始，在 RoboTwin 上训练动作模型。

概览
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      LingbotVLA 2.0

   .. grid-item-card:: 方法
      :text-align: center

      Action-only SFT

   .. grid-item-card:: 数据
      :text-align: center

      RoboTwin multi-dataset 清单

   .. grid-item-card:: 硬件
      :text-align: center

      单机 8 卡

任务
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 26 28 22

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 训练目标
   * - RoboTwin
     - Click Bell
     - ``robotwin_sft_lingbotvlav2``
     - 从 Foundation Base 更新动作专家和投影层。

准备数据
--------

从 `RoboTwin 数据集发布页 <https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset/click_bell>`__ 下载并解压 Click Bell 原始演示数据。转换脚本要求使用旧版目录格式：每个数据集目录包含 ``data/episode<i>.hdf5`` 和 ``instructions/episode<i>.json``。

下载模型
--------

下载 Foundation Base 和 tokenizer：

.. code-block:: bash

   hf download robbyant/lingbot-vla-v2-6b --local-dir lingbot-vla-v2-6b
   hf download Qwen/Qwen3-VL-4B-Instruct --local-dir Qwen3-VL-4B-Instruct

SFT 从 ``robbyant/lingbot-vla-v2-6b`` 开始训练。:doc:`PPO 示例 <lingbotvlav2>` 使用独立的 RoboTwin 后训练 checkpoint ``robbyant/lingbot-vla-v2-6b-robotwin``。

安装
----

按照 :doc:`LingbotVLA 2.0 <lingbotvlav2>` 的安装步骤准备并激活虚拟环境，同时获取 LingbotVLA 源码。以下命令都在 RLinf 仓库根目录运行。

运行
----

转换解压后的 RoboTwin 数据集：

.. code-block:: bash

   python toolkits/lerobot/convert_robotwin_to_lerobot.py \
     --raw-root /path/to/click_bell/aloha-agilex_clean_50 \
     --output-root /path/to/robotwin_lerobot/click_bell_clean_50

转换完成后，日志会显示写入的 episode 数和帧数。创建 ``robotwin_train.txt``，每行写一个 ``robot_name dataset_path``：

.. code-block:: text

   robotwin /path/to/robotwin_lerobot/click_bell_clean_50

在 ``examples/sft/config/robotwin_sft_lingbotvlav2.yaml`` 中填写以下路径：

.. code-block:: yaml

   actor:
     model:
       model_path: /path/to/lingbot-vla-v2-6b
       tokenizer_path: /path/to/Qwen3-VL-4B-Instruct
       lingbotvlav2:
         config_path: /path/to/lingbot-vla-v2
         training_config_path: /path/to/lingbot-vla-v2/configs/vla/robotwin/robotwin.yaml
         robot_config_path: /path/to/lingbot-vla-v2/configs/robot_configs/robotwin.yaml
         robot_config_root: /path/to/lingbot-vla-v2/configs/robot_configs
         stats_path: /path/to/lingbot-vla-v2/assets/norm_stats/robotwin.json
   data:
     train_data_paths: /path/to/robotwin_train.txt

训练配置使用单机 8 卡、global batch size 8 和 FSDP2。Qwen-VL 主干使用 BF16，动作专家使用 FP32。SFT 使用 ``flex_cached`` attention；PPO 和独立评测通过共享配置使用 ``eager``，以支持 cached key/value 解码。

运行 SFT：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh robotwin_sft_lingbotvlav2

日志和 checkpoint 保存在 ``logs/<timestamp>-robotwin_sft_lingbotvlav2/``。训练时可在 TensorBoard 中查看 ``train/l1_loss``，完整指标见 :doc:`训练指标 <../../reference/metrics>`。

评测前，按照 :doc:`RoboTwin 评测指南 <../../evaluations/guides/robotwin>` 准备仿真环境和资产，再设置环境路径：

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/RoboTwin
   export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH
   export ROBOT_PLATFORM=ALOHA

在 ``evaluations/robotwin/robotwin_click_bell_lingbotvlav2_sft_eval.yaml`` 中填写 Foundation、tokenizer 和 LingbotVLA 源码路径，再评测保存的 SFT checkpoint：

.. code-block:: bash

   bash evaluations/run_eval.sh robotwin_click_bell_lingbotvlav2_sft_eval \
     runner.ckpt_path=/path/to/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt

可视化与结果
------------

图中的对照实验使用 Click Alarmclock 数据和 ModelScope 上的 Foundation 权重快照 ``Robbyant/lingbot-vla-v2-6b``，版本号为 ``edfff7638a31a5dbfaae87016477191735bd754e``。两组训练使用同一份含 3,384 条样本记录的数据清单，全局批量大小为 8，共完成 423 次优化器更新。这个 ModelScope 版本号标识实验所用权重；前面的 Hugging Face 下载命令不固定到这一版本。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/9f22f0d7c6ed5e4b27c0212b3da848831d1a3c2a/pic/lingbotvlav2_foundation_action_loss_ma25.png
   :width: 90%

   Foundation Base 训练的 Action Loss 与 25 步移动平均。

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

逐步 Action Loss 的 MAE 为 0.002404，Pearson 相关系数为 0.9984，最大绝对差为 0.014081。图中比较 Official ``VLA_Loss`` 与 RLinf ``train/l1_loss``。两套实现的 Total Loss 可能包含不同辅助项，需要单独评估。该结果属于受控对照实验，不能证明严格 parity 或任务成功率提升。
