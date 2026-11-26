Qwen2.5 Math (PPO)
==================

Train Qwen2.5-1.5B-Instruct on AReaL-boba math problems with actor-critic :doc:`PPO </rst_source/reference/algorithms/ppo>` (GAE advantages and a Megatron critic). This recipe shares the launch script and dataset setup with :doc:`Qwen2.5 Math GRPO <reasoning>`.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-1.5B-Instruct

   .. grid-item-card:: Algorithm
      :text-align: center

      PPO (GAE + critic, Megatron + SGLang)

   .. grid-item-card:: Task
      :text-align: center

      Math reasoning (AReaL-boba)

   .. grid-item-card:: Hardware
      :text-align: center

      Single node (1–8 GPUs)

| **What you'll do:** install ``agentic`` deps → download model and AReaL-boba data → launch ``run_main_grpo_math.sh`` with the PPO config → monitor reward curves.
| **Prerequisites:** :doc:`Installation & Hardware </rst_source/start/installation>` · NVIDIA GPU node with SGLang and Megatron support.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.

.. tab-set::

   .. tab-item:: Qwen2.5-1.5B-Instruct

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      Download the model and `AReaL-boba-Data <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`__:

      .. code-block:: bash

         hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./Qwen2.5-1.5B-Instruct
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. Launch**

      PPO reuses ``run_main_grpo_math.sh``; ``examples/reasoning/config/math/qwen2.5-1.5b-ppo-megatron.yaml`` enables the critic and GAE advantages (``critic.use_critic_model: true``, ``algorithm.adv_type: gae``). Pass local model, dataset, and output paths via Hydra overrides:

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-ppo-megatron \
           rollout.model.model_path=./Qwen2.5-1.5B-Instruct \
           data.train_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
           data.val_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
           runner.output_dir=./results

.. dropdown:: Advanced configuration & multi-node (optional)

   - **Dataset keys:** see :doc:`Qwen2.5 Math GRPO <reasoning>` for ``data.prompt_key``, ``data.answer_key``, and ``data.apply_chat_template``.
   - **Multi-node scaling:** start a Ray cluster following :doc:`/rst_source/guides/multi_node`, increase ``cluster.num_nodes``, and run the launch command on the head node. For common cluster and runner fields, see :doc:`Training configuration </rst_source/reference/configuration>`.

Visualization and Results
-------------------------

Watch training progress with ``tensorboard --logdir ./results --port 6006``; see :doc:`Training metrics </rst_source/reference/metrics>` for metric definitions and W&B/SwanLab logging.

The figure compares RLinf (orange) and VeRL (blue) fine-tuning Qwen2.5-1.5B-Instruct under identical PPO settings. Reward rises steadily over training from the weak base model baseline.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/ppo_rlinf_vs_verl.jpg" style="width: 50%;"/>
       <p><em>MATH 1.5B PPO</em></p>
     </div>
   </div>
