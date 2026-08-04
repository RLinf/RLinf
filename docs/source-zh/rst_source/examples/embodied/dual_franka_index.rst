Dual-Arm Franka
===============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka.jpg
   :align: center
   :width: 80%
   :alt: Dual-Arm Franka

   双 Franka 机器人平台。

本节汇总 RLinf 支持的双 Franka 数据采集、监督微调、部署与 DAgger
训练流程。请根据使用场景选择对应教程。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Collect-SFT-Deploy
      :link: dual_franka
      :link-type: doc

      双 Franka GELLO 数据采集、数据转换、训练与部署基础流程。

   .. grid-item-card:: Collect-SFT-Deploy (RLinf-pytorch)
      :link: dual_franka_openpi_pytorch
      :link-type: doc

      使用 OpenPI PyTorch 完成双 Franka 策略微调与部署。

   .. grid-item-card:: HG-DAgger via VR
      :link: dual_franka_pico_dagger
      :link-type: doc

      使用 PICO 进行双臂数据采集与 DAgger 训练。

.. toctree::
   :hidden:
   :maxdepth: 1

   Collect-SFT-Deploy <dual_franka>
   Collect-SFT-Deploy (RLinf-pytorch) <dual_franka_openpi_pytorch>
   HG-DAgger via VR <dual_franka_pico_dagger>
