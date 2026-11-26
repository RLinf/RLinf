YAM
===

选择双臂 YAM 的数据采集、OpenPI π₀.₅ 微调与真机评估指南。

.. note::

   **推荐流程**：1. 用 PICO 采集双臂示教数据 → 2. 训练并评估 OpenPI π₀.₅。

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: PICO 数据采集
      :link: yam
      :link-type: doc

      用 PICO 双手柄和三路 RGB 相机采集 14 维绝对位置示教数据。

   .. grid-item-card:: π₀.₅ SFT 与评测
      :link: yam_openpi
      :link-type: doc

      用 YAM 示教数据微调 OpenPI π₀.₅，并在双臂机器人上评估 checkpoint。

.. toctree::
   :hidden:
   :maxdepth: 1

   PICO 数据采集 <yam>
   π₀.₅ SFT 与评测 <yam_openpi>
