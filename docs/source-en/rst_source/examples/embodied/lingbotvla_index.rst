LingbotVLA
==========

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/lingbotvla.png
   :align: center
   :width: 80%
   :alt: LingbotVLA on RoboTwin

   LingbotVLA on RoboTwin.

Pick the LingbotVLA generation you want to train on RoboTwin 2.0. The two
generations use different backbones and install into separate environments.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: LingbotVLA 1.0
      :link: lingbotvla
      :link-type: doc

      Qwen2.5-VL-based model: SFT and GRPO on RoboTwin tasks.

   .. grid-item-card:: LingbotVLA 2.0
      :link: lingbotvlav2
      :link-type: doc

      Qwen3-VL-based model with a mixture-of-experts action expert: PPO on
      RoboTwin ``click_bell``.

.. toctree::
   :hidden:
   :maxdepth: 1

   LingbotVLA 1.0 <lingbotvla>
   LingbotVLA 2.0 <lingbotvlav2>
