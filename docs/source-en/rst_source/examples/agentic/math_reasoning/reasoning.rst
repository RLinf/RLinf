Qwen2.5 Math GRPO
=================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

Train a Qwen-based reasoning model with GRPO on math data. Compared with supervised fine-tuning, RL encourages diverse reasoning paths while optimizing final-answer correctness.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      Qwen2.5-1.5B and Qwen2.5-7B

   .. grid-item-card:: Algorithm
      :text-align: center

      GRPO with token-level loss and minibatch early-stop

   .. grid-item-card:: Data
      :text-align: center

      AReaL-boba math reasoning data

   .. grid-item-card:: Hardware
      :text-align: center

      16 nodes default (single-node quick start supported)


Dataset
-------------

We use the dataset from `AReaL-boba-Data <https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/>`_.
This dataset integrates data from DeepScaleR, Open-Reasoner-Zero, Light-R1, DAPO, NuminaMath (AoPS/Olympiad subsets), and ZebraLogic.
Overly simple problems are filtered out to ensure dataset quality and effectiveness.

An example training sample looks like:

.. code-block:: json

   {
      "prompt": "<｜User｜>\nProblem description... Please reason step by step, and put your final answer within \\boxed{}.<｜Assistant｜><think>\n",
      "task": "math",
      "query_id": "xx",
      "solutions": ["\\boxed{x}"]
   }

.. note::

  Ensure the dataset is structured as the above using the default configurations.
  Otherwise, read below configuration guidelines carefully to adapt RLinf to your dataset.

We also support importing other dataset types.
To support different dataset formats, you can adjust the configuration as needed.

- **Prompt key and answer key configurations**

  The default configuration expects the dataset to have the ``prompt`` and ``solutions`` keys for retrieving the prompt and answer like in the Boba dataset.

  However, different datasets may have different key names or structures. You can customize the configuration to match your dataset's format.
  Change ``prompt_key`` and ``answer_key`` in the configuration YAML file to point to the correct fields in your dataset.

  For example, if your dataset uses ``prompt`` and ``label`` as keys, set:

  .. code-block:: yaml

      prompt_key: "prompt"
      answer_key: "label"

- **apply_chat_template configuration**

  Some datasets require a chat template for the prompt. Enable ``apply_chat_template`` in the configuration:

  .. code-block:: yaml

      apply_chat_template: true

  For example, if your dataset stores chat messages as a list:

  .. code-block:: json

      {
          "prompt": [{"content": "<str>", "role": "<str>"},],
          "label": "<str>",
      }

  When enabled, RLinf formats each prompt with ``tokenizer.apply_chat_template()`` into an input string.

How GRPO Works
--------------

We adopt GRPO (Group Relative Policy Optimization) with the following modifications:

- Token-level loss: Instead of averaging loss over the entire response sequence, we compute the average over tokens, as in DAPO.
  This prevents excessively long responses from dominating training and reduces their gradient impact.

- Minibatch early-stop: If the importance ratio within a minibatch becomes too large, we discard that minibatch to stabilize training.

Reward function:

- +5 if the final boxed/numeric answer is correct;
- -5 if incorrect.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__. For common path, cluster, and runner fields, see :doc:`Training configuration <../../../reference/configuration>`.

.. tab-set::

   .. tab-item:: DeepSeek-R1-Distill-Qwen-1.5B

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./DeepSeek-R1-Distill-Qwen-1.5B
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. Launch**

      .. warning::

         ``examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron.yaml`` defaults to **16 nodes** (``cluster.num_nodes: 16``), which fails on a single machine unless overridden.

      **Single-node quick start** (overrides node count, model path, and dataset path via Hydra CLI; requires at least 2 GPUs for ``tensor_model_parallel_size=2``):

      .. code-block:: bash

         PYTHONPATH=$(pwd):/opt/Megatron-LM:$PYTHONPATH \
         python examples/reasoning/main_grpo.py \
             --config-path examples/reasoning/config/math \
             --config-name qwen2.5-1.5b-grpo-megatron \
             cluster.num_nodes=1 \
             rollout.model.model_path=./DeepSeek-R1-Distill-Qwen-1.5B \
             actor.tokenizer.tokenizer_model=./DeepSeek-R1-Distill-Qwen-1.5B \
             data.train_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
             data.val_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]'

      **Full 16-node run**: update ``runner.output_dir``, ``rollout.model.model_path``, ``actor.tokenizer.tokenizer_model``, and ``data.train_data_paths`` in ``examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron.yaml``, start the Ray cluster following :doc:`/rst_source/guides/multi_node`, and run on the head node:

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron

   .. tab-item:: DeepSeek-R1-Distill-Qwen-7B

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir ./DeepSeek-R1-Distill-Qwen-7B
         hf download --repo-type dataset inclusionAI/AReaL-boba-Data --local-dir ./AReaL-boba-Data

      **3. Launch**

      .. warning::

         ``examples/reasoning/config/math/qwen2.5-7b-grpo-megatron.yaml`` defaults to **16 nodes** (``cluster.num_nodes: 16``), which fails on a single machine unless overridden.

      **Single-node quick start** (overrides node count, model path, and dataset path via Hydra CLI; requires at least 4 GPUs for ``tensor_model_parallel_size=4``):

      .. code-block:: bash

         PYTHONPATH=$(pwd):/opt/Megatron-LM:$PYTHONPATH \
         python examples/reasoning/main_grpo.py \
             --config-path examples/reasoning/config/math \
             --config-name qwen2.5-7b-grpo-megatron \
             cluster.num_nodes=1 \
             rollout.model.model_path=./DeepSeek-R1-Distill-Qwen-7B \
             actor.tokenizer.tokenizer_model=./DeepSeek-R1-Distill-Qwen-7B \
             data.train_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]' \
             data.val_data_paths='["./AReaL-boba-Data/AReaL-boba-106k.jsonl"]'

      **Full 16-node run**: update ``runner.output_dir``, ``rollout.model.model_path``, ``actor.tokenizer.tokenizer_model``, and ``data.train_data_paths`` in ``examples/reasoning/config/math/qwen2.5-7b-grpo-megatron.yaml``, start the Ray cluster following :doc:`/rst_source/guides/multi_node`, and run on the head node:

      .. code-block:: bash

         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-7b-grpo-megatron

.. note::

   **Switch rollout backend to vLLM:** install the vLLM environment with ``bash requirements/install.sh agentic --engine vllm``, then pass ``rollout.rollout_backend=vllm`` (or set it in the YAML). See :doc:`Rollout Engine Versions </rst_source/guides/version>` for supported engine versions.

Visualization and Results
-------------------------

We trained 1.5B and 7B models from DeepSeek-R1-Distill-Qwen.

Monitor training with TensorBoard:

.. code-block:: bash

   tensorboard --logdir ./logs --port 6006

For common metric meanings, see :doc:`Training metrics <../../../reference/metrics>`. The following plots show training curves.

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


Final Performance
~~~~~~~~~~~~~~~~~

We provide an evaluation `toolkit <https://github.com/RLinf/LLMEvalKit>`_.

Evaluation results on AIME 24, AIME 25, and GPQA-diamond:

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

\* We retrain the model using the default settings for 600 steps.

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


Public Checkpoints
------------------

We release trained models on Hugging Face for public use:

- `RLinf-math-1.5B <https://huggingface.co/RLinf/RLinf-math-1.5B>`_
- `RLinf-math-7B <https://huggingface.co/RLinf/RLinf-math-7B>`_
