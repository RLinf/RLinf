LingbotVLA
==========

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/lingbotvla.png
   :align: center
   :width: 80%
   :alt: RoboTwin 上的 LingbotVLA

   RoboTwin 上的 LingbotVLA。

选择要在 RoboTwin 2.0 上训练的 LingbotVLA 版本。两个版本的底座模型不同，需要安装到不同的环境中。

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: LingbotVLA 1.0
      :link: lingbotvla
      :link-type: doc

      基于 Qwen2.5-VL 的模型：在 RoboTwin 任务上进行 SFT 与 GRPO 训练。

   .. grid-item-card:: LingbotVLA 2.0
      :link: lingbotvlav2
      :link-type: doc

      基于 Qwen3-VL、带混合专家（MoE）动作专家的模型：在 RoboTwin ``click_bell`` 上进行 PPO 训练。

.. toctree::
   :hidden:
   :maxdepth: 1

   LingbotVLA 1.0 <lingbotvla>
   LingbotVLA 2.0 <lingbotvlav2>
