检查点恢复
=================

意外情况 —— 网络错误、断电、节点被抢占 —— 都可能中断一个长时间运行的分布式任务。  
为了解决这一问题，RLinf 会在每隔 ``runner.save_interval`` 步时保存一个完整的检查点，  
并允许你从最近的快照恢复，最大限度减少工作损失。  

检查点布局
-----------------

假设有如下 YAML 片段：

.. code-block:: yaml

   runner:
     task_type: math
     logger:
       log_path: ${runner.output_dir}/${runner.experiment_name}
       project_name: rlinf
       experiment_name: ${runner.experiment_name}

     save_interval: 50          
     experiment_name: grpo-1.5b
     output_dir: ./logs

检查点会出现在  
``./logs/grpo-1.5b/checkpoints/`` 下：

.. code-block:: text

   logs/grpo-1.5b/checkpoints/
   ├── global_step_50/
   │   ├── actor/
   │   │   ├── iter_0000050/
   │   │   │   ├── mp_rank_00/
   │   │   │   │   ├── distrib_optim.pt
   │   │   │   │   └── model_optim_rng.pt
   │   │   │   └── mp_rank_01/                 
   │   │   │       ├── distrib_optim.pt
   │   │   │       └── model_optim_rng.pt
   │   │   └── latest_checkpointed_iteration.txt
   │   └── data/
   │       └── data.pt                         
   └── global_step_100/
       └── …

关键点
~~~~~~~~~~

* **分片权重** —— ``mp_rank_*`` 中的文件遵循 Megatron 的张量并行布局；每个 GPU 只会重新加载属于自己的分片。  
* **优化器 / RNG 状态** —— *同时* 保存了 Adam 参数（``distrib_optim.pt``）和随机数生成器，确保恢复后可以比特级复现。  
* **数据采样器** —— ``data.pt`` 存储了 dataloader，保证不会遗漏或重复样本。  

恢复训练
-----------------

1. **选择最新的检查点**

   如果 ``global_step_150/`` 是编号最高的目录，它就是最新的快照。  

2. **修改 YAML**

   .. code-block:: yaml

      runner:
        resume_dir: ${runner.output_dir}/${runner.experiment_name}/checkpoints/global_step_150

3. **完全按原方式重新启动**

   启动 Ray，然后运行相同的 ``run_main_*.sh`` 启动脚本。  
   RLinf 会自动检测到 ``resume_dir`` 并：  

   * 在每个节点/rank 上恢复模型分片、优化器、RNG 和 dataloader 状态。  
   * 从 ``global_step_150`` 继续计数 —— 下一个保存的检查点将是 ``global_step_200`` （因为 ``save_interval`` 为 50）。  

.. tip::

   想验证恢复是否成功，可以查看日志行。  
   如果下一次训练从 step 150 开始，就说明恢复正常！  
