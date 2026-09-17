检查点恢复
=================

将 ``runner.resume_dir`` 指向已保存的检查点，即可恢复中断的训练。RLinf 每隔 ``runner.save_interval`` 步保存一次检查点。先根据下文的后端目录结构找到检查点，再使用相同配置重新启动训练。

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

   experiment_name/checkpoints/
   ├── global_step_10/
   │   └── actor/
   │       ├── dcp_checkpoint/
   │       │   ├── .metadata
   │       │   ├── __0_0.distcp
   │       │   ├── __1_0.distcp
   │       │   ├── __2_0.distcp
   │       │   └── __3_0.distcp
   │       └── model_state_dict/
   │           └── full_weights.pt
   └── global_step_20/
       └── …


FSDP/FSDP2 通过 DCP（``torch.distributed.checkpoint``）保存模型参数、优化器状态、学习率调度器和随机数生成器（RNG）状态。复制检查点时，保留全部 ``.distcp`` 文件以及 ``.metadata`` 文件。

DCP 检查点分别保存每个 actor rank 的 Python、NumPy 和 PyTorch CPU RNG 状态；worker 有加速设备时，也保存当前设备的 RNG 状态。恢复时保持 actor world size 不变，才能让各 rank 从原来的位置继续生成随机数。如果 world size 改变，已有 rank 恢复各自保存的序列，新增 rank 保留初始化后的 RNG 状态。加载会继续执行并输出警告；改变拓扑后不能复现原来的训练序列。这一机制不会恢复环境状态或单独创建的 generator，因此仅恢复这些 RNG 状态不能保证训练曲线完全一致。

旧版 DCP 检查点中保存的单个 RNG 字典仍可加载，但 DCP 可能已将不同 rank 的 RNG 状态去重，恢复时无法找回被丢弃的序列。``local_shard`` 格式仍在每个 rank 自己的文件中保存 RNG 状态。


恢复训练
-----------------

1. **选择最新的检查点**

   如果 ``global_step_10/`` 是编号最高的目录，它就是最新的快照。  

2. **修改 YAML**

   .. code-block:: yaml

      runner:
        resume_dir: ${runner.output_dir}/${runner.experiment_name}/checkpoints/global_step_10

3. **完全按原方式重新启动**

   启动 Ray，然后运行相同的 ``run_main_*.sh`` 启动脚本。  
   RLinf 会自动检测到 ``resume_dir`` 并：  

   * 在每个节点/rank 上恢复模型分片、优化器、RNG 和 dataloader 状态。  
   * 从 ``global_step_10`` 继续计数 —— 下一个保存的检查点将是 ``global_step_20`` （因为 ``save_interval`` 为 10）。  

.. tip::

   想验证恢复是否成功，可以查看日志行。  
   如果下一次训练从 step 30 开始，就说明恢复正常！  
