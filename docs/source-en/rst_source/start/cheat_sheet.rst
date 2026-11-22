Cheat Sheet
===========

Run common RLinf workflows with the shortest commands.

Install
-------

Install locally with ``install.sh`` (append ``--use-mirror`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` in mainland China), or pull a prebuilt image from `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__:

.. tab-set::

   .. tab-item:: Embodied

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env maniskill_libero && source .venv/bin/activate

   .. tab-item:: Agentic / Reasoning

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic --engine sglang && source .venv/bin/activate

   .. tab-item:: Docs

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh docs --venv .docs-venv

Run Training
------------

Single-node runs start Ray automatically. Run a 1-GPU zero-download MLP check, or launch a VLA quickstart (downloads its default Hugging Face checkpoint on first run after fetching ManiSkill assets once):

.. code-block:: bash

   # 1-GPU zero-download MLP check (install with: bash requirements/install.sh embodied --env maniskill_libero)
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

   # 7B VLA PPO quickstart
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart

Pass ``actor.model.model_path=/path/to/checkpoint`` (or another Hugging Face ``owner/repo``) to override the model; rollout inherits it automatically. Quickstart configs default to ``all`` accelerators; change ``cluster.component_placement`` to target specific devices (:doc:`../reference/configuration`). For multi-node runs, set ``RLINF_NODE_RANK`` before running ``ray start`` on each node (:doc:`../guides/multi_node`).

Evaluate
--------

Run standalone embodied evaluation (requires an ``openpi`` + ``libero`` environment; pass ``--venv`` to ``install.sh`` if keeping a separate training venv):

.. code-block:: bash

   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=RLinf/RLinf-Pi05-LIBERO-SFT

Replace ``RLinf/RLinf-Pi05-LIBERO-SFT`` with a local checkpoint directory if you already downloaded the weights.

Next Steps
----------

- :doc:`Installation <installation>` — targets, flags, and Docker Hub link.
- :doc:`Quick Start <vla>` — four-step ManiSkill3 walkthrough.
- :doc:`Launch & Scale <../guides/launch-scale/index>` — multi-node and real-world runs.
- :doc:`Evaluation <../evaluations/index>` — supported benchmarks and eval configs.
