Qwen2.5 Math (GRPO)
===================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

Train DeepSeek-R1-Distill-Qwen 1.5B or 7B models with :doc:`GRPO </rst_source/reference/algorithms/grpo>` on AReaL-boba math problems, reaching 40.84 (1.5B) and 56.23 (7B) average across AIME 24, AIME 25, and GPQA-diamond.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      DeepSeek-R1-Distill-Qwen-1.5B · 7B

   .. grid-item-card:: Algorithm
      :text-align: center

      GRPO (token-level loss, Megatron + SGLang)

   .. grid-item-card:: Task
      :text-align: center

      Math reasoning (AReaL-boba)

   .. grid-item-card:: Hardware
      :text-align: center

      16 nodes default (single-node quick start ≥ 2 / 4 GPUs)

| **What you'll do:** install ``agentic`` deps → download model and AReaL-boba data → launch ``run_main_grpo_math.sh`` → track curves and evaluate.
| **Prerequisites:** :doc:`Installation & Hardware </rst_source/start/installation>` · 2+ GPUs (1.5B) or 4+ GPUs (7B) for single-node quick start; 16 nodes for the default config.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__.

.. tab-set::

   .. tab-item:: DeepSeek-R1-Distill-Qwen-1.5B

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      Download the model and `AReaL-boba-Data <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`__:

      .. code-block:: bash

         hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./DeepSeek-R1-Distill-Qwen-1.5B
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. Launch**

      .. warning::

         ``examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron.yaml`` defaults to **16 nodes** (``cluster.num_nodes: 16``). Override ``cluster.num_nodes=1`` on a single machine (requires at least 2 GPUs for TP=2).

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron \
           cluster.num_nodes=1 \
           rollout.model.model_path=./DeepSeek-R1-Distill-Qwen-1.5B \
           actor.tokenizer.tokenizer_model=./DeepSeek-R1-Distill-Qwen-1.5B \
           data.train_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
           data.val_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]'

   .. tab-item:: DeepSeek-R1-Distill-Qwen-7B

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      Download the model and `AReaL-boba-Data <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`__:

      .. code-block:: bash

         hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir ./DeepSeek-R1-Distill-Qwen-7B
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. Launch**

      .. warning::

         ``examples/reasoning/config/math/qwen2.5-7b-grpo-megatron.yaml`` defaults to **16 nodes** (``cluster.num_nodes: 16``). Override ``cluster.num_nodes=1`` on a single machine (requires at least 4 GPUs for TP=4).

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-7b-grpo-megatron \
           cluster.num_nodes=1 \
           rollout.model.model_path=./DeepSeek-R1-Distill-Qwen-7B \
           actor.tokenizer.tokenizer_model=./DeepSeek-R1-Distill-Qwen-7B \
           data.train_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
           data.val_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]'

.. dropdown:: Advanced configuration: multi-node, vLLM & custom datasets (optional)

   - **Full 16-node run:** start a Ray cluster following :doc:`/rst_source/guides/multi_node`, then run the launch command on the head node without ``cluster.num_nodes=1`` using shared storage paths.

   - **vLLM rollout backend:** install with ``bash requirements/install.sh agentic --engine vllm`` and append ``rollout.rollout_backend=vllm``. See :doc:`Rollout Engine Versions </rst_source/guides/version>` and :doc:`Training configuration </rst_source/reference/configuration>`.

   - **Custom dataset format:** the default config reads ``prompt`` and ``solutions`` (boxed answers, rewarded +5 / -5). Override ``data.prompt_key`` and ``data.answer_key`` for other schemas, and set ``data.apply_chat_template=True`` when prompts are chat message lists.

Visualization and Results
-------------------------

Watch training progress with ``tensorboard --logdir ./logs --port 6006``; see :doc:`Training metrics </rst_source/reference/metrics>` for metric definitions and W&B/SwanLab logging.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/1.5b-loss-curve.jpg" style="width: 100%;"/>
       <p><em>MATH 1.5B</em></p>
     </div>
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/7b-loss-curve.jpg" style="width: 100%;"/>
       <p><em>MATH 7B</em></p>
     </div>
   </div>

Evaluate checkpoints with `LLMEvalKit <https://github.com/RLinf/LLMEvalKit>`__ on AIME 24, AIME 25, and GPQA-diamond. Trained checkpoints are published at `RLinf-math-1.5B <https://huggingface.co/RLinf/RLinf-math-1.5B>`__ and `RLinf-math-7B <https://huggingface.co/RLinf/RLinf-math-7B>`__.

.. list-table:: **1.5 B model results**
   :header-rows: 1
   :widths: 45 15 15 25 15

   * - Model
     - AIME 24
     - AIME 25
     - GPQA-diamond
     - Average
   * - |huggingface| `DeepSeek-R1-Distill-Qwen-1.5B (base model) <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`_
     - 28.33
     - 24.90
     - 27.45
     - 26.89
   * - |huggingface| `DeepMath-1.5B <https://huggingface.co/zwhe99/DeepMath-1.5B>`_
     - 37.80
     - 30.42
     - 32.11
     - 33.44
   * - |huggingface| `DeepScaleR-1.5B-Preview <https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview>`_
     - 40.41
     - 30.93
     - 27.54
     - 32.96
   * - |huggingface| `AReaL-1.5B-Preview-Stage-3 <https://huggingface.co/inclusionAI/AReaL-1.5B-Preview-Stage-3>`_
     - 40.73
     - 31.56
     - 28.10
     - 33.46
   * - AReaL-1.5B-retrain\*
     - 44.42
     - 34.27
     - 33.81
     - 37.50
   * - |huggingface| `FastCuRL-1.5B-V3 <https://huggingface.co/Nickyang/FastCuRL-1.5B-V3>`_
     - 43.65
     - 32.49
     - 35.00
     - 37.05
   * - |huggingface| `RLinf-math-1.5B <https://huggingface.co/RLinf/RLinf-math-1.5B>`_
     - **48.44**
     - **35.63**
     - **38.46**
     - **40.84**

\* Retrained with the default settings for 600 steps.

.. list-table:: **7 B model results**
   :header-rows: 1
   :widths: 45 15 15 25 15

   * - Model
     - AIME 24
     - AIME 25
     - GPQA-diamond
     - Average
   * - |huggingface| `DeepSeek-R1-Distill-Qwen-7B (base model) <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>`_
     - 54.90
     - 40.20
     - 45.48
     - 46.86
   * - |huggingface| `AReaL-boba-RL-7B <https://huggingface.co/inclusionAI/AReaL-boba-RL-7B>`_
     - 61.66
     - 49.38
     - 46.93
     - 52.66
   * - |huggingface| `Skywork-OR1-7B <https://huggingface.co/Skywork/Skywork-OR1-7B>`_
     - 66.87
     - 52.49
     - 44.43
     - 54.60
   * - |huggingface| `Polaris-7B-Preview <https://huggingface.co/POLARIS-Project/Polaris-7B-Preview>`_
     - **68.55**
     - 51.24
     - 43.88
     - 54.56
   * - |huggingface| `AceMath-RL-Nemotron-7B <https://huggingface.co/nvidia/AceMath-RL-Nemotron-7B>`_
     - 67.30
     - **55.00**
     - 45.57
     - 55.96
   * - |huggingface| `RLinf-math-7B <https://huggingface.co/RLinf/RLinf-math-7B>`_
     - 68.33
     - 52.19
     - **48.18**
     - **56.23**
