Qwen2.5 Math PPO
================

Use this recipe when you want actor-critic PPO for the same math-reasoning task covered by :doc:`reasoning`. PPO and GRPO share most launch and data settings, so this page only lists the PPO-specific differences.

Overview
--------

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

How PPO Works
-------------

Use standard PPO (Proximal Policy Optimization) with GAE advantages and a critic. For the algorithm reference, see :doc:`PPO <../../../reference/algorithms/ppo>`.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__. For common path, cluster, and runner fields, see :doc:`Training configuration <../../../reference/configuration>`.

.. tab-set::

   .. tab-item:: Qwen2.5-1.5B-Instruct (Megatron PPO)

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      Download the base model and the boba dataset (see :doc:`reasoning` for dataset format details):

      .. code-block:: bash

         hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./Qwen2.5-1.5B-Instruct
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. Launch**

      Update paths in ``examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml``. PPO training uses the same entry script (``run_main_grpo_math.sh``) as GRPO training; RLinf selects PPO automatically from the critic configuration and ``adv_type`` (typically ``gae``):

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-ppo-megatron


Visualization and Results
-------------------------

We fine-tune the Qwen2.5-1.5B-Instruct model with PPO. The orange line is RLinf and the blue line is VeRL; both use the same algorithm configuration. For common metric meanings, see :doc:`Training metrics <../../../reference/metrics>`.

Since the base capability of the Qwen2.5-1.5B-Instruct model is relatively weak, the overall reward values are low. However, as training progresses, the reward values increase significantly.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/ppo_rlinf_vs_verl.jpg" style="width: 50%;"/>
       <p><em>MATH 1.5B PPO</em></p>
     </div>
   </div>
