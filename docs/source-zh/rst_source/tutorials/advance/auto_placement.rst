自动放置
=====================

RL 训练前的自动放置
--------------------------------

该工具会自动为 RL 训练流程生成最优的组件放置配置。  
它会分析不同组件（rollout、inference、training 等）的计算开销，并确定最佳放置策略，从而最小化整体训练时间。

概览
~~~~~~~~~

自动放置工具由 `toolkits/auto_placement` 下的三个主要组件构成：

- **scheduler_task.py**：主调度器，执行时间与空间的分时复用以寻找最优放置方案  
- **resource_allocator.py**：负责不同组件的资源分配  
- **workflow.py**：管理工作流图和成本计算  

使用方法
~~~~~~~~~~~~~~~

步骤 1：收集 Profile 数据
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在运行自动放置工具之前，你需要收集组件的 Profile 数据。  
这包括测量各组件（rollout、inference、training 等）在共享式模式下一次迭代的计算时间。

将 Profile 数据添加到 YAML 配置文件的 ``profile_data`` 部分：

.. code-block:: yaml

   profile_data:
     actor_cost: 95.7      # Training 组件耗时（每次迭代秒数）
     inference_cost: 30.8  # Inference 组件耗时（每次迭代秒数）
     rollout_cost: 59.9    # Rollout 组件耗时（每次迭代秒数）

**如何收集 Profile 数据：**

1. 使用原始集群在共享式模式下运行训练若干次迭代  
2. 使用分析工具测量每个组件每次迭代的耗时  
3. 记录每个组件的平均迭代耗时  

步骤 2：运行自动放置
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用提供的 shell 脚本运行自动放置工具：

.. code-block:: bash

   cd examples/reasoning
   ./run_placement_autotune.sh [config_name]

其中 ``config_name`` 是你的配置文件名称。

脚本的输出类似如下：

.. code-block:: text

   Best placement for this task is:

   cluster:
     num_nodes: 1
     component_placement:
       rollout,actor: all

步骤 3：应用结果
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

该工具会输出一个新的包含优化组件放置的配置。  
将输出中的 ``cluster.component_placement`` 部分复制，替换掉原始 YAML 文件中的对应部分。

即用优化后的 ``cluster.component_placement`` 替换原始配置文件里的这一部分。

故障排查
~~~~~~~~~~~~~~~~~~~~~

1. **缺少 Profile 数据错误**：确保 YAML 文件包含 ``profile_data`` 部分，并包含三个组件的耗时数值。  

2. **无效放置**：检查 GPU 总分配是否超过集群容量。  
