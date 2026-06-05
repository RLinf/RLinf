Reinforcement Learning with Prior Data (RLPD)
================================================

1. Introduction
------------------

RLPD (Reinforcement Learning with Prior Data) is a highly sample efficient algorithm designed to leverage offline datasets to accelerate online reinforcement learning. Built upon the :doc:`Soft Actor-Critic (SAC) <sac>` framework, RLPD introduces three minimal but critical design choices to stabilize training and improve sample efficiency:

- Symmetric Sampling: A balanced sampling strategy that constructs training batches using a 50/50 split between the agent's online replay buffer and the offline demo dataset.

- Layer Normalization: Integrates Layer Normalization into the critic networks to prevent catastrophic value overestimation and extrapolation errors when learning from static datasets.

- Stabilize sample-efficient updates (High UTD or asynchronous): RLPD takes a large critic ensemble (e.g., 10 networks) and random subsetting (Random Ensemble Distillation) to stabilize the training with high update-to-data (UTD) ratios or async updates.

RLPD demonstrates that standard off-policy RL algorithms can effectively utilize offline data without complex pre-training. It has been widely used in real-world RL.

For more details, see the original `RLPD <https://arxiv.org/abs/2302.02948>`_ paper.

2. Objective Function
------------------------

RLPD retains the maximum entropy objective of SAC. The policy :math:`\pi` is trained to maximize both the expected return and the entropy of the policy.

The core difference lies in the Critic update. RLPD utilizes an ensemble of :math:`E` critic networks (e.g., :math:`E=10`). The loss function for each critic :math:`Q_{\theta_i}` is computed over a hybrid batch :math:`\mathcal{B}` composed equally of online data :math:`\mathcal{D}_{\text{online}}` and offline data :math:`\mathcal{D}_{\text{offline}}`:

.. math::

   L(\theta_i, \mathcal{B}) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{B}} \left[ \left( Q_{\theta_i}(s, a) - y \right)^2 \right]

The target :math:`y` is calculated using **Random Ensemble Distillation** (REDQ), where a random subset :math:`\mathcal{Z}` of the target critics is selected to compute the pessimistic estimate:

.. math::

   y = r + \gamma \left( \min_{j \in \mathcal{Z}} Q_{\theta'_j}(s', a') - \alpha \log \pi_{\phi}(a'|s') \right)

where :math:`a' \sim \pi_{\phi}(\cdot|s')`, :math:`\theta'` denotes the target network parameters, and :math:`\mathcal{Z} \subset \{1, \dots, E\}`.

The actor loss remains similar to SAC, updating the policy :math:`\pi_{\phi}` to maximize the expected minimum Q-value (averaged over the ensemble or a subset) and the entropy term:

.. math::

   L(\phi, \mathcal{B}) = \mathbb{E}_{s \sim \mathcal{B}, a \sim \pi_{\phi}} \left[ \alpha \log \pi_{\phi}(a|s) - \frac{1}{E} \sum_{i=1}^{E} Q_{\theta_i}(s, a) \right]

3. Specific Designs
---------------------

RLPD relies on specific architectural and procedural choices to handle the distribution shift caused by offline data:

- Symmetric Sampling: RLPD samples distinct minibatches from the online replay buffer :math:`\mathcal{D}_{\text{online}}` and the offline dataset :math:`\mathcal{D}_{\text{offline}}` and concatenates them to form a single training batch. The standard ratio is 50% online and 50% offline. This ensures the agent retains the stability of offline data while adapting to new online experiences.

- Layer Normalization: To mitigate the "unbounded extrapolation" issue where Q-values for out-of-distribution actions diverge, RLPD applies Layer Normalization after the first layer of the MLP in the Q-networks. This implicitly bounds the Q-values by the norm of the weight matrix, stabilizing learning in sparse-reward or complex settings.

- Ensemble Q: To improve sample efficiency, our RLPD performs asynchronous update. To prevents overfitting often associated with frequent updates, it employs an ensemble of critics (e.g., :math:`E=10` or :math:`E=20`) and subsets them during target calculation, like REDQ.

4. Configuration
-------------------

RLPD builds on the SAC configuration with additional parameters for the offline dataset, ensemble size, and normalization.

.. code-block:: yaml

   algorithm:
     update_epoch: 30
     group_size: 1
     agg_q: mean

     backup_entropy: False # remove entropy term
     critic_subsample_size: 2 # Number of critics to subsample for target calculation
     edac_eta: 0.0 # Experimental EDAC critic diversity coefficient. Disabled by default.
     edac_grad_eps: 1.0e-6 # Numerical epsilon for EDAC gradient normalization.
     eval_rollout_epoch: 1

     demo_buffer: # add offline demo data
       enable_cache: True
       cache_size: 200 # number of trajectories cached in memory
       sample_window_size: 200 # number of latest trajectories to sample from for demo buffer
       min_buffer_size: 1
       load_path: "/path/to/demo_data"

     adv_type: embodied_sac
     loss_type: embodied_sac
     loss_agg_func: "token-mean"
     
     bootstrap_type: standard
     gamma: 0.96
     tau: 0.005

   actor:
     model:
       num_q_heads: 10 # Number of Q-networks in the critic ensemble

   rollout:
     group_name: "RolloutGroup"
     backend: "huggingface"
     enable_offload: False
     pipeline_stage_num: 1

     model:
       model_path: "/path/to/model"
       precision: ${actor.model.precision}

5. Optional EDAC Critic Diversity
---------------------------------

RLPD configurations commonly use a critic ensemble. When
``actor.model.num_q_heads > 1`` and ``algorithm.critic_subsample_size > 0``,
random critic subsetting is used for target calculation. RLinf also has a
unit-tested EDAC-style critic diversity loss helper for SAC-N/EDAC-style
experiments.

Because EDAC differentiates Q-values with respect to actions and keeps that
gradient graph for the critic update, wiring it into critic training would
increase critic-step memory and compute cost. The current FSDP SAC/RLPD worker
does not wire this helper into the critic update path, and positive
``algorithm.edac_eta`` fails fast during config validation. Import
``rlinf.algorithms.critic_regularizers.compute_edac_critic_diversity_loss``
from a custom worker if you want to experiment with the helper directly. Use it
as an experimental building block rather than a default training setting.
