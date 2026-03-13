Math推理的强化学习训练（PPO）
=============================

在示例 `Math推理的强化学习训练 <reasoning.html>`_ 中，我们已经介绍了使用 GRPO 来训练数学推理模型，而 RLinf 同样支持使用 PPO 算法来训练同样的任务，本文将介绍如何使用PPO来进行这一任务的训练。由于 GRPO 可以看做是标准 PPO 的一种变体，而且我们尽量使得 PPO 与 GRPO 复用大部分代码和配置项，因此本文略去了较为重复的部分，阅读本文前建议先阅读 `Math推理的强化学习训练 <reasoning.html>`_ 示例。

数据集
-------------

我们同样采用 boba 数据集，细节请参考 `Math推理的强化学习训练 <reasoning.html>`_。

算法
---------

我们采用标准的 PPO（Proximal Policy Optimization）算法，关于该算法的详细介绍，请参考 `PPO <../../tutorials/rlalg/ppo.html>`_。

运行脚本
---------------------

**1. 配置文件**

推荐配置示例：  

- ``examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron-dynamicbatch-4gpu.yaml``

**2. 启动命令**

PPO 训练与 GRPO 训练的启动命令基本相同，也是使用 ``run_main_grpo_math.sh`` 作为入口脚本，RLinf 会通过 yaml 配置文件中是否存在 critic 相关配置以及 adv_type 的取值（PPO 通常使用 gae 作为优势函数）来自动判断是否使用 PPO 训练。


训练曲线
~~~~~~~~~~~~~~

我们基于 Qwen2.5-1.5B-Instruct 模型，使用 PPO 算法进行了训练，下面展示训练曲线。橙色为RLinf，蓝色为作为对照的VeRL，两者运行相同的算法配置。

由于 Qwen2.5-1.5B-Instruct 模型基础能力较弱，所以整体 reward 数值较低，但是随着训练的进行，reward 数值明显上升。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/ppo_rlinf_vs_verl.jpg" style="width: 50%;"/>
       <p><em>MATH 1.5B PPO</em></p>
     </div>
   </div>

