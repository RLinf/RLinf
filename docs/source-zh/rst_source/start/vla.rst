快速上手 1：在 Maniskill3 上使用 PPO 训练 VLA 模型
=================================================

本快速教程将带你使用 **RLinf** 框架，在  
`ManiSkill3 <https://github.com/haosulab/ManiSkill>`_ 环境中训练视觉-语言-动作模型（VLA），包括  
`OpenVLA <https://github.com/openvla/openvla>`_。

为简化流程，你可以在单卡 GPU 上直接运行以下脚本完成训练。

环境简介
--------------------------

ManiSkill3 是一个基于 GPU 加速的机器人研究仿真平台，  
专注于复杂接触操作和具身智能任务。  
该基准涵盖多个领域，包括机械臂、移动操作器、人形机器人以及灵巧手，  
支持抓取、组装、绘图、移动等多种任务。

我们还针对 GPU 仿真器进行了系统级优化（详见 :doc:`../tutorials/mode/hybrid`）。

启动训练
--------------------------

**步骤 1：下载预训练模型**

若使用 **OpenVLA** 模型，请运行以下命令：

.. code-block:: bash

   # 下载 OpenVLA 预训练模型
   hf download gen-robot/openvla-7b-rlvla-warmup \
   --local-dir /path/to/model/openvla-7b-rlvla-warmup/

该模型已在论文中引用：`paper <https://arxiv.org/abs/2505.19789>`_

**步骤 2：运行官方提供的训练脚本**

为方便使用，我们提供的配置文件默认支持单卡训练。  
如果你有多张 GPU 并希望加快训练速度，  
建议你修改配置文件  
``./examples/embodiment/config/maniskill_ppo_openvla_quickstart.yaml`` 中的参数  
``cluster.num_gpus_per_node``。

你可以根据实际资源设置为 **1、2、4 或 8**。

.. code-block:: yaml

   cluster:
     num_nodes: 1
     num_gpus_per_node: 1
     component_placement:
        actor,rollout: all

运行脚本之前，请根据你下载的模型和数据集路径，修改 YAML 文件中的以下字段：

- ``rollout.model_dir``  
- ``actor.checkpoint_load_path``  
- ``actor.tokenizer.tokenizer_model``  

完成上述修改后，运行以下命令启动训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart

**步骤 3：查看训练结果**

- 最终模型与指标保存路径：``../results``  
- TensorBoard 可视化日志路径：``../results/tensorboard``  
  启动方式如下：

  .. code-block:: bash

     tensorboard --logdir ../results/tensorboard/ --port 6006

打开 TensorBoard 后，你会看到类似下图的界面。  
建议重点关注以下指标：

- ``rollout/env_info/return``  
- ``rollout/env_info/success_once``  

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/embody-quickstart-metric.jpg" width="800"/>
