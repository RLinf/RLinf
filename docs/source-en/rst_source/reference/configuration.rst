Training Configuration
======================

RLinf recipes are Hydra YAML configs composed from reusable presets. Use this page as the one-page configuration map; example pages link here instead of repeating key tables.

Annotated Minimal YAML
----------------------

Core embodied recipes compose ``base/embodied_rl``, an ``env`` preset, a ``model`` preset, and an ``algorithm`` preset, so the experiment file lists only what differs for your task:

.. code-block:: yaml

   defaults:
     - base/embodied_rl                              # cluster (all GPUs), runner, FSDP, patch_syncer
     - env/manikill_put_carrot_on_plate_in_scene@env.train
     - env/manikill_put_carrot_on_plate_in_scene@env.eval
     - model/openvla@actor.model                     # policy architecture & default precision
     - algorithm/ppo@algorithm                       # PPO / GAE hyperparameters
     - _self_

   runner:
     logger:
       experiment_name: "maniskill_ppo_openvla_quickstart"

   env:
     train:
       total_num_envs: 32                            # parallel simulator instances
       rollout_epoch: 1

   rollout:
     pipeline_stage_num: 2                           # pipeline stages overlapping env & rollout

   actor:
     micro_batch_size: 20                            # per-GPU forward/backward batch
     global_batch_size: 160                          # must be divisible by micro_batch_size * world_size
     model:
       model_path: "gen-robot/openvla-7b-rlvla-warmup"  # HF repo ID or local dir; rollout inherits it
       add_value_head: True
       is_lora: True

Override any field on the command line without editing the repository file:

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart \
     actor.model.model_path=/path/to/local-checkpoint \
     env.train.total_num_envs=8 \
     actor.micro_batch_size=4 \
     actor.global_batch_size=16 \
     runner.max_epochs=10

Five Keys to Tune on One GPU
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Key
     - Effect
   * - ``actor.model.model_path``
     - Accepts a Hugging Face ``owner/repo`` ID (downloaded/cached automatically) or a local directory. ``rollout.model.model_path`` inherits it automatically during training.
   * - ``cluster.component_placement``
     - Defaults to ``actor,env,rollout: all`` via ``cluster/collocated``. Restrict devices with e.g. ``actor,env,rollout: 0`` or split components across GPUs (:doc:`../concepts/overview`).
   * - ``env.train.total_num_envs``
     - Parallel environment count. Lower this first if simulator rendering or RAM is tight.
   * - ``actor.micro_batch_size`` / ``global_batch_size``
     - Lower ``micro_batch_size`` to fix actor OOM; keep ``global_batch_size`` divisible by ``micro_batch_size * actor_world_size``.
   * - ``actor.enable_offload`` / ``rollout.enable_offload``
     - Defaults to ``True`` in ``base/embodied_rl`` so collocated single-GPU runs fit in VRAM; set ``False`` on disaggregated multi-GPU clusters.

Hydra Presets and Config Locations
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Preset / Workload
     - Location
     - Role / Launcher
   * - Embodied base & cluster
     - ``examples/embodiment/config/{base,cluster}/``
     - Shared ``base/embodied_rl`` scaffolding and ``cluster/collocated`` placement (override with ``override /cluster@cluster: null`` for custom layouts).
   * - Algorithm presets
     - ``examples/embodiment/config/algorithm/``
     - Reusable ``ppo`` and ``grpo`` defaults mounted at ``@algorithm``.
   * - Env, model, backend
     - ``examples/embodiment/config/{env,model,hybrid_engines}/``
     - Simulator/robot presets, model templates, and FSDP defaults.
   * - Embodied RL recipes
     - ``examples/embodiment/config/*.yaml``
     - ``bash examples/embodiment/run_embodiment.sh <config_name>``
   * - Reasoning & Agentic RL
     - ``examples/{reasoning,agent}/**/config/*.yaml``
     - ``run_main_grpo_{math,vqa}.sh`` or recipe-specific launcher under ``examples/agent/<recipe>/``
   * - SFT
     - ``examples/sft/config/*.yaml``
     - ``bash examples/sft/run_vla_sft.sh <config_name>`` (or ``run_vlm_sft.sh``)
   * - Standalone Evaluation
     - ``evaluations/<benchmark>/*.yaml``
     - ``bash evaluations/run_eval.sh <benchmark> <config_name>``

Top-Level Sections
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Section
     - Ownership
   * - ``cluster``
     - Node count, hardware node groups, and ``component_placement`` for actor, rollout, env, reward, or agent workers.
   * - ``runner``
     - Task type, ``logger.log_path`` (TensorBoard, videos, checkpoints), epoch/step budget, validation/save cadence, and resume path.
   * - ``algorithm``
     - Advantage estimator, policy loss, clip ratios, KL settings, and off-policy/demo buffer options.
   * - ``env`` / ``data``
     - Simulator or real-robot settings (``env``) or dataset paths and prompt/answer keys (``data``).
   * - ``rollout``
     - Inference backend (HuggingFace, SGLang, vLLM), sampling parameters, and optional model override.
   * - ``actor``
     - Training backend (FSDP, Megatron), model path, LoRA, optimizer, batch sizes, and FSDP/Megatron sharding.

Full Key References
-------------------

- :doc:`Architecture Overview <../concepts/overview>` — five-minute visual map of components, placement modes, and data formats.
- :doc:`Basic Configuration <../guides/basic_config>` — shared ``hydra``, ``cluster``, ``runner``, ``algorithm``, and ``actor`` keys.
- :doc:`Embodied Configuration <../guides/embodiment_config>` — complete embodied ``env``, ``rollout``, and ``model`` reference.
- :doc:`Agentic Configuration <../guides/agentic_config>` — LLM/VLM reasoning and agent loop keys.
- :doc:`Training Metrics <metrics>` and :doc:`Evaluation Configuration <../evaluations/reference/configuration>`.
