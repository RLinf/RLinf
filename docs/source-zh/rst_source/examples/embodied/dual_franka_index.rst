双臂 Franka
===========

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dual-franka.jpg
   :align: center
   :width: 80%
   :alt: 双臂 Franka

   用于双臂数据采集、SFT、部署与在线 HG-DAgger 的双臂 Franka 平台。

按需求选择双臂 Franka 的数据采集、OpenPI π₀.₅ 微调部署或在线 HG-DAgger 指南。

.. note::

   **推荐流程**：1. 采集示教数据（GELLO 或 PICO）→ 2. SFT 并部署 OpenPI π₀.₅ → 3. 用 PICO HG-DAgger 在线迭代。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: 采集、SFT 与部署
      :link: dual_franka
      :link-type: doc

      用 GELLO 采集双臂示教数据，完成数据转换、微调与部署。

   .. grid-item-card:: VR HG-DAgger
      :link: dual_franka_pico_dagger
      :link-type: doc

      用 PICO 采集双臂数据并运行在线 Human-Gated DAgger。

   .. grid-item-card:: 采集、SFT 与部署（OpenPI PyTorch）
      :link: dual_franka_openpi_pytorch
      :link-type: doc

      用 OpenPI PyTorch 完成双臂 Franka policy 微调与部署。

.. toctree::
   :hidden:
   :maxdepth: 1

   采集、SFT 与部署 <dual_franka>
   VR HG-DAgger <dual_franka_pico_dagger>
   采集、SFT 与部署（OpenPI PyTorch） <dual_franka_openpi_pytorch>
