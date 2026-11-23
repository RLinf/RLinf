新增硬件与系统后端
==================

面向底层系统扩展：将 RLinf 移植到新的 GPU / NPU 加速卡，或定制 Actor 到 Rollout 的权重同步链路。

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: 新增加速卡后端
      :link: ../new_accelerator
      :link-type: doc

      为 RLinf 的硬件调度层与安装脚本适配新的 GPU / NPU 加速卡。

   .. grid-item-card:: 自定义权重同步
      :link: ../weight_syncer
      :link-type: doc

      定制并优化具身训练中的 Actor 到 Rollout 权重传输机制。

.. toctree::
   :hidden:

   新增加速卡后端 <../new_accelerator>
   自定义权重同步 <../weight_syncer>
