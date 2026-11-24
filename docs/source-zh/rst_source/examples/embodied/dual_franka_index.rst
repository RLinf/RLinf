Franka（双臂）
==============

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

   .. grid-item-card:: OpenPI PyTorch
      :link: dual_franka_openpi_pytorch
      :link-type: doc

      用 OpenPI PyTorch 完成双臂 Franka policy 微调与部署。

   .. grid-item-card:: OpenPI JAX
      :link: dual_franka
      :link-type: doc

      用 GELLO 采集双臂示教数据，完成数据转换、微调与部署。

   .. grid-item-card:: PICO VR HG-DAgger
      :link: dual_franka_pico_dagger
      :link-type: doc

      用 PICO 采集双臂数据并运行在线 Human-Gated DAgger。

.. toctree::
   :hidden:
   :maxdepth: 1

   OpenPI PyTorch <dual_franka_openpi_pytorch>
   OpenPI JAX <dual_franka>
   PICO VR HG-DAgger <dual_franka_pico_dagger>
