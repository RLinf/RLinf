世界模型
========

用学习得到的视频世界模型作为模拟器，闭环强化学习训练 VLA policy。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: BWM 世界模型
      :img-top: https://raw.githubusercontent.com/boundless-large-model/boundless-world-model/main/assets/images/teaser.png
      :img-alt: BWM 世界模型
      :link: embodied/bwm
      :link-type: doc

      在 BWM 世界模型中用 PPO 训练单视角 π₀.₅ policy 完成 RoboTwin ``adjust_bottle`` 任务。

   .. grid-item-card:: OpenSora 世界模型
      :img-top: https://raw.githubusercontent.com/hpcaitech/Open-Sora-Demo/main/readme/icon.png
      :img-alt: OpenSora 世界模型
      :link: embodied/opensora
      :link-type: doc

      在 OpenSora 视频世界模型中用 GRPO 微调 OpenVLA-OFT。

   .. grid-item-card:: Wan 世界模型
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
      :img-alt: Wan 世界模型
      :link: embodied/wan
      :link-type: doc

      用 Wan 视频世界模型作为模拟器，通过 GRPO 微调 OpenVLA-OFT。

.. toctree::
   :hidden:
   :maxdepth: 2

   BWM <embodied/bwm>
   OpenSora <embodied/opensora>
   Wan <embodied/wan>
