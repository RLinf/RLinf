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


如果使用 Megatron 作为训练后端，其检查点会出现在 `output_dir/experiment_name/checkpoints/` 下,
而如果使用 FSDP/FSDP2 作为训练后端，其检查点会出现在 `log_path/experiment_name/checkpoints/` 下。

Megatron 检查点
~~~~~~~~~~~~~~~~

Megatron检查点文件结构如下：

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
^^^^^^^^^^^^^^^

* **分片权重** —— ``mp_rank_*`` 中的文件遵循 Megatron 的张量并行布局；每个 GPU 只会重新加载属于自己的分片。  
* **优化器 / RNG 状态** —— *同时* 保存了 Adam 参数（``distrib_optim.pt``）和随机数生成器，确保恢复后可以比特级复现。  
* **数据采样器** —— ``data.pt`` 存储了 dataloader，保证不会遗漏或重复样本。  

FSDP/FSDP2 检查点
~~~~~~~~~~~~~~~~~~

FSDP/FSDP2 检查点文件结构如下：

.. code-block:: text

   -- global_step_2
      -- actor
         |-- __0_0.distcp
         |-- __1_0.distcp
         |-- __2_0.distcp
            -- __3_0.distcp


FSDP/FSDP2 通过 DCP (torch.distributed.checkpoint) 保存和加载检查点，其结果为一组分布式检查点文件(.distcp)。  
每个文件包含模型参数、优化器状态和 RNG 状态的分片。

检查点文件向Pytorch State Dict文件转换
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果你需要将 FSDP/FSDP2 检查点转换为标准的 Pytorch State Dict 文件用于模型评估或是其他用途，可以使用toolkit文件夹下的工具
``toolkits/ckpt_convertor/convert_dcp_to_state_dict.py``, 使用方法如下：

.. code-block:: bash

   convert_dcp_to_state_dict.py [-h] --dcp_path DCP_PATH --output_path OUTPUT_PATH


其中 ``DCP_PATH`` 是包含 DCP 文件的目录，``OUTPUT_PATH`` 是保存转换后模型 State Dict 文件的路径。

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
