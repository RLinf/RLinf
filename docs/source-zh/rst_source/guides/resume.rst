断点续训
========

意外情况（如网络错误、断电或节点被抢占）可能中断长时间运行的分布式任务。为此，RLinf 每隔 ``runner.save_interval`` 步保存一次完整 checkpoint，可以从最近的快照恢复训练，减少进度损失。

Checkpoint 布局
---------------

假设有如下 YAML 片段：

.. code-block:: yaml

   runner:
     task_type: reasoning
     logger:
       log_path: ${runner.output_dir}/${runner.experiment_name}
       project_name: rlinf
       experiment_name: ${runner.experiment_name}

     save_interval: 50          
     experiment_name: grpo-1.5b
     output_dir: ./logs

推理与智能体 runner 会将 checkpoint 保存在 ``<runner.output_dir>/<runner.experiment_name>/checkpoints/`` 下（例如 ``logs/grpo-1.5b/checkpoints/``）。具身 runner 则保存在 ``<runner.logger.log_path>/<runner.logger.experiment_name>/checkpoints/`` 下（具身配置通常将 ``runner.logger.log_path`` 设为 ``"../results"``，不会在 ``log_path`` 中重复拼接 ``experiment_name``）。

Megatron Checkpoint
~~~~~~~~~~~~~~~~~~~

Megatron checkpoint 文件结构如下：

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

FSDP/FSDP2 根据 actor worker 的实现，使用 DCP（``torch.distributed.checkpoint``）或按 rank 保存的 ``local_shard`` 检查点。默认 DCP 格式的目录结构如下：

.. code-block:: text

   experiment_name/checkpoints/
   ├── global_step_50/
   │   └── actor/
   │       ├── dcp_checkpoint/
   │       │   ├── __0_0.distcp
   │       │   ├── __1_0.distcp
   │       │   ├── __2_0.distcp
   │       │   └── __3_0.distcp
   │       └── model_state_dict/
   │           └── full_weights.pt
   └── global_step_100/
       └── …


DCP 将训练状态保存到一组分布式 checkpoint 文件（``.distcp``）中。可选导出的 ``model_state_dict/full_weights.pt`` 只包含模型权重；恢复训练时应使用完整的 checkpoint 目录，同时恢复优化器、scheduler 和 RNG 状态。

部分 worker 为每个 rank 单独保存一个文件。SAC 和 DAgger 在 ``actor.fsdp_config.use_orig_params`` 为 true 时选择 ``local_shard`` 格式，IQL 则使用此格式保存 policy、critic 和 value 模型。例如，两个 rank 的 SAC actor 目录包含：

.. code-block:: text

   global_step_50/actor/
   └── local_shard_checkpoint/
       ├── checkpoint_rank_0.pt
       └── checkpoint_rank_1.pt

每个文件保存对应 rank 的模型分片、优化器状态、scheduler 状态和 RNG 状态。FSDP2 会将文件中的局部张量恢复到当前模型的分布式参数中，也支持各 rank 分片大小不等的参数。恢复时必须保持模型、FSDP 版本、world size、分片配置以及 rank 与分片的对应关系一致；``local_shard`` 不会为不同拓扑重新分配分片。请保留所有 rank 的文件，使用相同的启动脚本和配置，并按下文设置 ``runner.resume_dir``；worker 会自动选择对应的 checkpoint 格式。


恢复训练
--------

1. **选择最新的 checkpoint**

   如果 ``global_step_50/`` 是编号最高的目录，它就是最新的快照。

2. **修改 YAML**

   .. code-block:: yaml

      runner:
        resume_dir: ${runner.output_dir}/${runner.experiment_name}/checkpoints/global_step_50

3. **按原方式重新启动**

   运行与原任务相同的启动脚本（例如推理任务使用 ``run_main_*.sh``，具身任务使用 ``run_embodiment.sh``；多机集群需先确保各节点已启动 Ray）。RLinf 会自动检测 ``resume_dir`` 并：

   * 在每个节点/rank 上恢复模型分片、优化器、RNG 和 dataloader 状态。
   * 从 ``global_step_50`` 继续计数，下一个保存的 checkpoint 将是 ``global_step_100``\ （因为 ``save_interval`` 为 50）。

.. tip::

   查看训练日志即可验证恢复是否生效：恢复后下一次训练应从 step 51 开始（紧接 ``global_step_50``）。
