双臂 Franka
===============

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka.jpg
   :align: center
   :width: 80%
   :alt: 双臂 Franka

   双臂 Franka 机器人平台。

按需求选择双臂 Franka 的数据采集、监督微调、部署与 DAgger 训练指南。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: 采集、SFT 与部署
      :link: dual_franka
      :link-type: doc

      使用 GELLO 采集双臂演示数据，完成数据转换、微调与部署。

   .. grid-item-card:: VR HG-DAgger
      :link: dual_franka_pico_dagger
      :link-type: doc

      使用 PICO 进行双臂数据采集与 DAgger 训练。

   .. grid-item-card:: 采集、SFT 与部署（OpenPI PyTorch）
      :link: dual_franka_openpi_pytorch
      :link-type: doc

      使用 OpenPI PyTorch 完成双臂 Franka policy 微调与部署。

.. toctree::
   :hidden:
   :maxdepth: 1

   采集、SFT 与部署 <dual_franka>
   VR HG-DAgger <dual_franka_pico_dagger>
   采集、SFT 与部署（OpenPI PyTorch） <dual_franka_openpi_pytorch>
