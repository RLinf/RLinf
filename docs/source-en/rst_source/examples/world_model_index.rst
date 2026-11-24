World Models
============

Pick this section to train VLA policies with closed-loop RL using learned video world models as simulators.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: BWM World Model
      :img-top: https://raw.githubusercontent.com/boundless-large-model/boundless-world-model/main/assets/images/teaser.png
      :img-alt: Boundless World Model
      :link: embodied/bwm
      :link-type: doc

      Train a single-view π₀.₅ policy with PPO on RoboTwin ``adjust_bottle`` inside BWM.

   .. grid-item-card:: OpenSora World Model
      :img-top: https://raw.githubusercontent.com/hpcaitech/Open-Sora-Demo/main/readme/icon.png
      :img-alt: OpenSora world model
      :link: embodied/opensora
      :link-type: doc

      Fine-tune OpenVLA-OFT with GRPO inside the OpenSora video world model.

   .. grid-item-card:: Wan World Model
      :img-top: https://raw.githubusercontent.com/RLinf/misc/main/pic/wan.png
      :img-alt: Wan world model
      :link: embodied/wan
      :link-type: doc

      Fine-tune OpenVLA-OFT with GRPO using the Wan video world model as simulator.

.. toctree::
   :hidden:
   :maxdepth: 2

   BWM <embodied/bwm>
   OpenSora <embodied/opensora>
   Wan <embodied/wan>
