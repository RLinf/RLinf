Dual-Arm Franka
===============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka.jpg
   :align: center
   :width: 80%
   :alt: Dual-Arm Franka

   双 Franka 机器人平台。

本节汇总 RLinf 支持的双 Franka 数据采集、监督微调、部署与 DAgger
训练流程。请根据使用场景选择对应教程。

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Dual-Arm Collect-SFT-Deploy (OpenPI_RLinf)
      :link: dual_franka_openpi_rlinf
      :link-type: doc

      使用 OpenPI_RLinf 完成双 Franka π0 与 π0.5 训练和部署。

   .. grid-item-card:: HG-DAgger via VR
      :link: dual_franka_pico_dagger
      :link-type: doc

      使用 PICO 进行双臂数据采集与 DAgger 训练。

.. toctree::
   :hidden:
   :maxdepth: 1

   Dual-Arm Collect-SFT-Deploy (OpenPI_RLinf) <dual_franka_openpi_rlinf>
   HG-DAgger via VR <dual_franka_pico_dagger>
