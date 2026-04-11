多节点训练
===================

本指南将带你启动一个 **4 节点的 Ray 集群** （每个节点有 **8 块 GPU** ），  
并使用  
`DeepSeek-R1-Distill-Qwen-1.5B <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`_  
模型在 *math* 任务上运行分布式强化学习训练。

只要你根据实际情况修改 YAML 配置文件，这一套流程也可以扩展到任意数量的节点和 GPU。

准备工作
-------------

开始前请确保以下几点已完成：

* 已将 RLinf 仓库克隆到所有节点都能访问的共享文件系统中。
* 每个节点都已启动对应的容器镜像。

步骤 1：启动 Ray 集群
----------------------------

首先清除旧的缓存状态：

.. code-block:: bash

   rm -f ray_utils/ray_head_ip.txt

然后在 **每个节点** 的 shell 中运行以下命令：

==========================================  ==========================
节点编号                                     命令
==========================================  ==========================
0（head 节点）                               ``RANK=0 bash ray_utils/start_ray.sh``
1                                           ``RANK=1 bash ray_utils/start_ray.sh``
2                                           ``RANK=2 bash ray_utils/start_ray.sh``
3                                           ``RANK=3 bash ray_utils/start_ray.sh``
==========================================  ==========================

当脚本成功运行后 **head 节点** 上的终端会输出如下内容（为简洁起见，下图为 2 节点、16 GPU 的示例）：

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/start-0.jpg" width="800"/>

每个 **worker 节点** 上的终端输出如下：

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/start-1.jpg" width="800"/>

当所有四个启动脚本都打印出 *Ray started*，**保持在 head 节点终端中**，并检查集群是否已正确连接（此例为 ``4 × 8 = 32`` 张 GPU）：

.. code-block:: bash

   bash ray_utils/check_ray.sh 32

.. note::

   ``check_ray.sh`` 的参数必须等于集群中GPU等加速器的总数。

如果一切正常，你将看到如下输出：

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/check.jpg" width="800"/>

注意：为简洁起见，本文使用的是 2 节点 16 GPU 的截图。

步骤 2：启动训练任务
------------------------------------

我们提供了两种模式的启动示例：**共享式模式** 和 **分离式模式**

共享式模式
^^^^^^^^^^^^^^^^^^^^^^^^^^

所有训练阶段（rollout、inference、actor）共享 **所有 GPU**。  
修改示例 YAML：

.. code-block:: yaml

   # examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron.yaml
   cluster:
     num_nodes: 4          # 根据你的集群情况修改
     component_placement:
       actor,rollout: all  # “all” 表示使用所有可见 GPU

在 head 节点上运行：

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh \
        qwen2.5-1.5b-grpo-megatron

分离式模式
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

不同阶段使用不同 GPU 范围，支持更细粒度的流水线并行。  
修改流水线 YAML 配置：

.. code-block:: yaml

   # examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron-pipeline.yaml
   cluster:
     num_nodes: 4
     component_placement:
       rollout:    0-19        # 使用 20 块 GPU
       inference:  20-23       # 使用 4 块 GPU
       actor:      24-31       # 使用 8 块 GPU

* 注意：``rollout + inference + actor`` 使用的 GPU 总数必须等于总 GPU 数（此例中为 ``32``）。
* 范围是 **闭区间** （即包含起止编号）。

启动任务：

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh \
        qwen2.5-1.5b-grpo-megatron-pipeline
