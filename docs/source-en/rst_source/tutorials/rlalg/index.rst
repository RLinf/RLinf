Supported RL Algorithms
=================================

In this section, we provide an overview of each algorithm, including their core characteristics, loss functions, and key configuration options required to run them effectively using RLinf.

Each algorithm is implemented with flexibility in mind, allowing researchers and practitioners to apply them to a variety of reinforcement learning tasks. Whether you're exploring standard benchmarks or designing custom environments, RLinf offers streamlined interfaces for training and evaluation.

As of now, RLinf supports seven widely-used reinforcement learning algorithms:

- :doc:`Proximal Policy Optimization (PPO) <ppo>`
- :doc:`Group Relative Policy Optimization (GRPO) <grpo>`
- :doc:`Domain-Adaptive Policy Optimization (DAPO) <dapo>`
- :doc:`REINFORCE++ <reinforce>` 
- :doc:`Soft Actor-Critic (SAC) <sac>`
- :doc:`Cross-Q <crossq>`
- :doc:`RLPD <rlpd>`

We are continuously working to expand the selection of supported algorithms in future releases. Stay tuned for upcoming additions!

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
