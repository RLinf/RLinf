快速上手
========

四步在 `ManiSkill3 <https://github.com/haosulab/ManiSkill>`__ 上用 PPO 训练 **OpenVLA** 或 **OpenVLA-OFT**。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 1. 安装环境
      :text-align: center

      ``install.sh`` 或 Docker Hub

   .. grid-item-card:: 2. 下载资源
      :text-align: center

      模型权重与 ManiSkill 资产

   .. grid-item-card:: 3. 启动训练
      :text-align: center

      一行命令直接跑通

   .. grid-item-card:: 4. 查看曲线
      :text-align: center

      TensorBoard 监控指标

前置条件
--------

先按 :doc:`安装说明 <installation>` 准备好 ManiSkill 环境。中国大陆用户可先执行 ``export HF_ENDPOINT=https://hf-mirror.com`` 加速下载。

下载环境资产
------------

在仓库根目录下载 ManiSkill3 场景资产（只需下载一次）：

.. code-block:: bash

   hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets

下载模型并启动训练
------------------

切换下方标签选择模型。两个快速上手配置默认使用所有可用加速卡；需要指定设备时，修改 ``cluster.component_placement``，详见 :doc:`../guides/basic_config`。

.. tab-set::

   .. tab-item:: OpenVLA

      下载预训练权重（`论文 <https://proceedings.neurips.cc/paper_files/paper/2025/hash/8c0fabe372177d2aded596be2d3b4544-Abstract-Conference.html>`__）：

      .. code-block:: bash

         hf download gen-robot/openvla-7b-rlvla-warmup --local-dir ./checkpoints/openvla-7b-rlvla-warmup

      启动 PPO 训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart \
           actor.model.model_path=$PWD/checkpoints/openvla-7b-rlvla-warmup

   .. tab-item:: OpenVLA-OFT

      下载基座模型与 LoRA 权重：

      .. code-block:: bash

         hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main --local-dir ./checkpoints/openvlaoft-base && \
         hf download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir ./checkpoints/openvlaoft-lora

      启动 PPO 训练：

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft_quickstart \
           actor.model.model_path=$PWD/checkpoints/openvlaoft-base \
           actor.model.lora_path=$PWD/checkpoints/openvlaoft-lora

rollout 会自动沿用 ``actor.model.model_path``，只需传一次权重路径。

查看训练结果
------------

日志与权重保存在 ``./logs``。启动 TensorBoard，重点关注 ``rollout/env_info/return`` 和 ``rollout/env_info/success_once``：

.. code-block:: bash

   tensorboard --host 0.0.0.0 --logdir ./logs

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/embody-quickstart-metric.jpg
   :align: center
   :width: 90%

   ManiSkill3 快速上手的 TensorBoard 训练曲线。

下一步
------

- :doc:`浏览示例库 <../examples/index>`，查看 LIBERO、RoboTwin、OpenPI 与真机示例。
- :doc:`运行独立评测 <../evaluations/index>`，评估训练后的权重。
- :doc:`性能优化指南 <../guides/performance/index>`，了解 :doc:`训练流水线模式 <../guides/training_pipeline>` 与 :doc:`执行模式 <../concepts/execution_modes>`。
