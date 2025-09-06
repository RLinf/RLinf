REINFORCE++ 
=====================

1. Introduction
---------------

**REINFORCE++** is a lightweight method for RL post-training.  
It starts from the classical REINFORCE algorithm and borrows two key ideas from PPO — *per-token KL penalties* and *advantage normalization* — 
while **avoiding the extra policy-clipping loss** that can bias gradient estimates.

Core design choices conclude:

* **Single-response training** (`group_size = 1`): one sampled answer per prompt.  
* **Per-token KL penalty** (default *k₂*): KL is subtracted from the scalar reward instead of being added as an extra loss term.  
* **Global advantage normalization** across the whole batch.  
* **REINFORCE++ baseline**: when `group_size > 1`, mean reward within each prompt group is used as a baseline *before* global normalization.

2. Objective Function
----------------------

Let :math:`q` be the prompt, :math:`o_{1:T}` the generated tokens, and :math:`\pi_{\theta}^{\text{RL}}` the current policy.  
The per-token advantage at time step :math:`t` is

.. math::

   A_{q,0,t} \;=\; r(o_{1:T}, q)\;-\;\beta
   \sum_{i=t}^{T} \operatorname{KL}(i) \tag{8}

where

.. math::

   \operatorname{KL}(t) \;=\;
   \log\!\left(
     \frac{\pi^{\text{RL}}_{\theta_{\text{old}}}(o_t \mid q,\,o_{<t})}
          {\pi^{\text{SFT}}(o_t \mid q,\,o_{<t})}
   \right) \tag{9}

:math:`\pi^{\text{SFT}}` is the frozen supervised-finetuned reference policy and  
:math:`\beta` controls the strength of the KL penalty.

To stabilise training, the advantages are normalised **across the global batch**:

.. math::

   A^{\text{norm}}_{q,o_t} \;=\;
   \frac{
     A_{q,o_t} \;-\;
     \operatorname{mean}\ \bigl(A_{q,o_t}\,\mid\,A_{q,o_t}\in\mathcal{D}_{\text{batch}}\bigr)
   }{
     \operatorname{std}\ \bigl(A_{q,o_t}\,\mid\,A_{q,o_t}\in\mathcal{D}_{\text{batch}}\bigr)
   } \tag{10}

The policy is then updated with the standard REINFORCE gradient  
:math:`\nabla_{\theta}\,\log\pi_{\theta}(o_t\!\mid\!q,o_{<t})\,A^{\text{norm}}_{q,o_t}` .

3. Configuration
-----------------

REINFORCE++

.. code-block:: yaml

   algorithm:
     adv_type:      "reinpp"       # use REINFORCE++
     group_size:    1              # one response per prompt
     kl_beta:       0.0001
     normalize_advantages: False   # global norm already applied

   data:
     rollout_batch_size: 8192

REINFORCE++ baseline

.. code-block:: yaml

   algorithm:
     adv_type:      "reinpp_baseline"
     group_size:    16             # multiple responses per prompt
     kl_beta:       0.0001

   data:
     rollout_batch_size: 512

4. Notes
---------

REINFORCE++ adopts the so-called :math:`k_1` KL.  
The GRPO algorithm uses a :math:`k_3` form that mixes on-policy and reference probabilities, 
but that estimator is biased.  
Using :math:`k_1` KL keeps the update unbiased while still discouraging large policy shifts.
