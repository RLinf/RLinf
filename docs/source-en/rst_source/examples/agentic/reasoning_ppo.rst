PPO training for Math Reasoning
==================================

Use this recipe when you want actor-critic PPO for the same math-reasoning task covered by :doc:`reasoning`. PPO and GRPO share most launch and data settings, so this page only lists the PPO-specific differences.

Overview
--------

Use this recipe when you want actor-critic PPO for the same math-reasoning setup
used by the GRPO example.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-1.5B

   .. grid-item-card:: Algorithm
      :text-align: center

      PPO with GAE advantages and a critic

   .. grid-item-card:: Data
      :text-align: center

      AReaL-boba math reasoning data

   .. grid-item-card:: Hardware
      :text-align: center

      Multi-GPU Megatron training

Dataset
-------

We also use the boba dataset. For details, see :doc:`reasoning`.

How PPO Works
-------------

Use standard PPO (Proximal Policy Optimization) with GAE advantages and a critic. For the algorithm reference, see :doc:`PPO <../../reference/algorithms/ppo>`.

Run It
------

**1. Config file**

For common path, cluster, and runner fields, see :doc:`Training configuration <../../reference/configuration>`. Recommended config example:

- ``examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml``

**2. Launch command**

The launch command for PPO training is basically the same as for GRPO training. We also use ``run_main_grpo_math.sh`` as the entry script. RLinf automatically determines whether to use PPO training based on whether there are critic-related configurations in the YAML config file and the value of ``adv_type`` (PPO typically uses ``gae`` as the advantage function).


Visualization and Results
-------------------------

We fine-tune the Qwen2.5-1.5B-Instruct model with PPO. The orange line is RLinf and the blue line is VeRL; both use the same algorithm configuration. For common metric meanings, see :doc:`Training metrics <../../reference/metrics>`.

Since the base capability of the Qwen2.5-1.5B-Instruct model is relatively weak, the overall reward values are low. However, as training progresses, the reward values increase significantly.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/ppo_rlinf_vs_verl.jpg" style="width: 50%;"/>
       <p><em>MATH 1.5B PPO</em></p>
     </div>
   </div>
