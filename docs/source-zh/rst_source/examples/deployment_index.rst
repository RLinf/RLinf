部署加速
========

本节以部署加速为主线，汇总 RLinf 在策略推理、动作执行与实时控制方面的
优化方法。

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: RTC：实时控制推理延迟隐藏
      :link: embodied/rtc
      :link-type: doc

      异步重叠 action chunk 执行与策略推理，加速仿真与真机部署。

.. toctree::
   :hidden:
   :maxdepth: 1

   RTC <embodied/rtc>
