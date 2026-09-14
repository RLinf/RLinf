ManiSkill 的 PhysX 仿真可以在 CPU 上运行，不依赖承载模型的 accelerator。请在所选 ManiSkill 配置的 ``env.train`` 与 ``env.eval`` 中使用以下设置：

.. code-block:: yaml

   env:
     train:
       total_num_envs: 2
       init_params:
         sim_backend: cpu
         render_backend: "pci:0000:00:00.0"
     eval:
       total_num_envs: 2
       init_params:
         sim_backend: cpu
         render_backend: "pci:0000:00:00.0"

``sim_backend: cpu`` 会将 PhysX 移到 CPU，VLA 仍在所选 accelerator 上运行。``render_backend`` 通过完整 PCI 地址选择 SAPIEN Vulkan renderer；RLinf 会在创建环境前保留 ``pci:<domain>:<bus>:<slot>.<function>`` 格式。若容器内的 renderer 使用其他地址，请替换示例值。

.. warning::

   CPU simulation backend 无法在单个进程内向量化多个环境。请将训练与评测的 ``total_num_envs`` 分别设为对应 env worker 的 rank 数量，使每个 worker 只运行一个环境。
