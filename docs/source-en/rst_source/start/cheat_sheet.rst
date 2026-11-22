Cheat Sheet
===========

Run common RLinf workflows with the shortest commands.

Install
-------

Each virtual environment holds one model + environment stack — pass ``--venv <dir>`` when keeping multiple stacks side by side. In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. For prebuilt images, visit `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.

.. tab-set::

   .. tab-item:: Embodied Training

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env maniskill_libero && source .venv/bin/activate

   .. tab-item:: Embodied Eval

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero --venv .venv-eval && source .venv-eval/bin/activate

   .. tab-item:: Agentic / Reasoning

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic --engine sglang --venv .venv-agentic && source .venv-agentic/bin/activate

   .. tab-item:: Docs

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh docs --venv .docs-venv

Run Training
------------

Single-node runs start Ray automatically. Config names map directly to YAML files on disk: embodied RL in ``examples/embodiment/config/``, SFT in ``examples/sft/config/``, reasoning and agentic in ``examples/reasoning/config/`` and ``examples/agent/``, and evaluation in ``evaluations/<benchmark>/``.

.. tab-set::

   .. tab-item:: MLP

      .. code-block:: bash

         source .venv/bin/activate
         bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

   .. tab-item:: VLA PPO

      .. code-block:: bash

         source .venv/bin/activate
         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets
         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart

   .. tab-item:: Agentic / Reasoning

      Pass your local model and math dataset paths on the command line:

      .. code-block:: bash

         source .venv-agentic/bin/activate
         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-single-gpu \
           rollout.model.model_path=./Qwen2.5-1.5B-Instruct \
           actor.model.model_path=./Qwen2.5-1.5B-Instruct \
           actor.tokenizer.tokenizer_model=./Qwen2.5-1.5B-Instruct \
           data.train_data_paths=./math_data

Pass ``actor.model.model_path=/path/to/checkpoint`` (or a Hugging Face ``owner/repo``) to override embodied model weights; rollout inherits it automatically. For multi-node runs, export ``RLINF_NODE_RANK`` before ``ray start`` on each node (:doc:`../guides/multi_node`).

Evaluate
--------

Run standalone embodied evaluation (configs live in ``evaluations/<benchmark>/<config_name>.yaml``):

.. code-block:: bash

   source .venv-eval/bin/activate

   # Evaluate the default Hugging Face base model (8 parallel envs)
   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval env.eval.total_num_envs=8

   # Evaluate an RL/SFT .pt checkpoint on top of the base model
   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
     env.eval.total_num_envs=8 \
     runner.ckpt_path=./logs/<run>/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt

Operations and Overrides
------------------------

.. code-block:: bash

   # Watch training and eval curves in TensorBoard
   tensorboard --host 0.0.0.0 --logdir ./logs

   # Resume training from a saved checkpoint directory
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart \
     runner.resume_dir=./logs/<run>/checkpoints/global_step_50

   # Restrict a multi-GPU config to GPU 0
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart \
     cluster.component_placement.actor,rollout,env=0

   # Clean up a stale local Ray session (check `ray status` first on shared hosts)
   ray stop

Next Steps
----------

- :doc:`Installation <installation>` — targets, flags, and Docker Hub link.
- :doc:`Quick Start <vla>` — four-step ManiSkill3 walkthrough.
- :doc:`Launch & Scale <../guides/launch-scale/index>` — multi-node and real-world runs.
- :doc:`Evaluation <../evaluations/index>` — supported benchmarks and eval configs.
