Custom Environment Rewards
==========================

This page shows how to customize the per-step reward an embodied
environment feeds into RL training (GRPO, PPO, ...): dense rewards,
hierarchical (staged) reward structures, and fully custom reward
functions — all selected from YAML, without modifying environment code.

How step rewards flow into training
-----------------------------------

Each embodied environment converts the simulator's raw reward and the
per-step ``info`` dict into the training reward inside
``_calc_step_reward`` (e.g. ``rlinf/envs/maniskill/maniskill_env.py``).
That reward feeds trajectory returns, which the advantage function
selected by ``algorithm.adv_type`` (e.g. ``grpo``, ``gae``) consumes.

The reward function is chosen by ``env.<split>.reward_mode``. For the
ManiSkill environment:

- ``default`` — the built-in hardcoded hierarchy
  (grasp ``0.1`` + consecutive grasp ``0.1`` + gated success ``1.0``);
- ``raw`` — pass through the simulator's raw (typically dense) reward;
- ``only_success`` — sparse ``1.0`` on task success;
- any other name — looked up in the step-reward registry
  (``rlinf/envs/rewards``), described next.

Weighted hierarchical dense rewards from YAML
---------------------------------------------

The built-in ``weighted_components`` reward composes a dense, layered
reward purely from config. Each component references a tensor in the
env's step ``info`` dict (the special name ``raw`` references the
simulator's raw reward):

.. code-block:: yaml

   env:
     train:
       reward_mode: weighted_components
       reward_components:
         # Stage 1: contact / grasp detection
         is_src_obj_grasped: 0.3
         # Stage 2: dense pose-alignment shaping from a distance signal
         gripper_carrot_dist:
           weight: 0.5
           transform: one_minus_tanh
           scale: 5.0
         # Stage 3: task completion, gated on the grasp stage
         success:
           weight: 1.0
           requires: [is_src_obj_grasped]

Each entry is either ``name: weight`` or a dict with:

- ``weight`` (required) — the component's coefficient;
- ``transform`` — reshapes the signal before weighting:
  ``none`` (default), ``neg`` (``-scale * v``, for penalties),
  ``neg_exp`` (``exp(-scale * v)``) and ``one_minus_tanh``
  (``1 - tanh(scale * v)``) — the latter two turn distances into dense
  rewards that peak at zero;
- ``scale`` — transform sharpness (default ``1.0``);
- ``requires`` — a list of boolean ``info`` keys that must all hold for
  the component to count, which is how hierarchical / staged structures
  are expressed.

The available component names are whatever the task's ``evaluate()``
puts into ``info``: for the ManiSkill put-on tasks these include
``is_src_obj_grasped``, ``consecutive_grasp``, ``success``,
``gripper_carrot_dist``, ``carrot_plate_dist``, and more. A component
name missing from ``info`` raises an error listing the available keys.

Registering a fully custom reward function
------------------------------------------

When weighted components are not expressive enough, register your own
function. It receives the raw reward, the step ``info`` dict, and the
env config node, and returns a ``[num_envs]`` float tensor:

.. code-block:: python

   # my_pkg/my_rewards.py
   import torch
   from rlinf.envs.rewards import register_env_reward

   @register_env_reward("staged_pick_place")
   def staged_pick_place(*, raw_reward, info, cfg=None):
       reward = 0.3 * info["is_src_obj_grasped"].float()
       reward += 0.5 * (1.0 - torch.tanh(5.0 * info["carrot_plate_dist"]))
       reward += 1.0 * (info["success"] & info["is_src_obj_grasped"]).float()
       return reward

Then select it from YAML. If the module lives outside the RLinf tree,
point ``reward_fn_module`` at it so the decorator runs before lookup:

.. code-block:: yaml

   env:
     train:
       reward_mode: staged_pick_place
       reward_fn_module: my_pkg.my_rewards

Notes
-----

- ``env.<split>.use_rel_reward: True`` applies to every reward mode: the
  env emits the per-step *difference* of your reward, so a monotone
  staged reward becomes a one-time bonus per achieved stage.
- Group-based advantages (``algorithm.adv_type: grpo``) normalize
  trajectory returns within each group, so only relative reward scale
  across trajectories matters; with ``gae``, per-step dense shaping
  directly affects the value target.
- For environments other than ManiSkill, route their
  ``_calc_step_reward`` through
  ``rlinf.envs.rewards.get_env_reward_fn`` the same way to adopt the
  registry.
