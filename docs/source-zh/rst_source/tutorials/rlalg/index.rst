支持的强化学习算法
=================================

本节将概述每种算法，包括其核心特点、损失函数以及在 RLinf 中高效运行所需的关键配置选项。  

每种算法的实现都考虑了灵活性，研究人员和实践者可以将它们应用到各种强化学习任务中。无论是探索标准基准任务，还是设计自定义环境，RLinf 都提供了简化的接口来支持训练和评估。  

目前，RLinf 支持以下七种常用的强化学习算法：  

- :doc:`近端策略优化 (PPO) <ppo>`  
- :doc:`组相对策略优化 (GRPO) <grpo>`  
- :doc:`解耦裁剪与动态采样策略优化 (DAPO) <dapo>`  
- :doc:`REINFORCE++ <reinforce>` 
- :doc:`SAC <sac>` 
- :doc:`Cross-Q <crossq>` 
- :doc:`RLPD <rlpd>`

我们会持续扩展未来版本中支持的算法，敬请期待后续的更新！  

.. toctree::
   :maxdepth: 1
   :hidden:

   ppo
   grpo
   dapo
   reinforce
   sac
   crossq
   rlpd
