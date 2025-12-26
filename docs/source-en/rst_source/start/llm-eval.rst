How to evaluate? Reasoner Scenarios
=====================================

Introduction
------------

We provide an integrated evaluation toolkit for long chain-of-thought (CoT) mathematical reasoning tasks.  
The `toolkit <https://github.com/RLinf/LLMEvalKit>`_ includes both code and datasets,  
making it convenient for researchers to evaluate trained large language models on mathematical reasoning.

**Acknowledgements:** This evaluation toolkit is adapted from the `Qwen2.5-Math <https://github.com/QwenLM/Qwen2.5-Math>`_ project.

Environment Setup
-----------------
First, clone the repository:

.. code-block:: bash

   git clone https://github.com/RLinf/LLMEvalKit.git 

Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt 

If you are using our Docker image, you only need to additionally install:

.. code-block:: bash

   pip install Pebble
   pip install timeout-decorator

Quick Start
-----------------

Model Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^
During training, models are saved in Megatron format. You can use the conversion scripts located at ``RLinf/toolkits/ckpt_convertor/`` to convert them to Huggingface format.

You have two ways to use the scripts:

**Method 1: Edit the script files**

Manually open ``mg2hf_7b.sh`` or ``mg2hf_1.5b.sh``, and set the following variables to your desired paths.

1. ``CKPT_PATH_MG`` (Megatron checkpoint path, e.g., ``results/run_name/checkpoints/global_step_xx/actor/``), 
2. ``CKPT_PATH_HF`` (Huggingface target path, any path), and
3. ``CKPT_PATH_ORIGINAL_HF`` (base model checkpoint used for initializing training, e.g., ``/path/to/DeepSeek-R1-Distill-Qwen-1.5B``) 

**Method 2: Command-line arguments**

A more flexible approach is to pass paths directly through command-line arguments.
.. code-block:: bash

   # For 1.5B models
   bash mg2hf_1.5b.sh /path/to/megatron_checkpoint /target/path/to/huggingface_checkpoint /path/to/base_model_checkpoint

   # For 7B models
   bash mg2hf_7b.sh /path/to/megatron_checkpoint /target/path/to/huggingface_checkpoint /path/to/base_model_checkpoint

Run Evaluation Script
^^^^^^^^^^^^^^^^^^^^^^

If you want to run evaluation on a **single dataset**, you can execute the following command:

.. code-block:: bash

   MODEL_NAME_OR_PATH=/model/path  # Replace with your model path
   OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval
   SPLIT="test"
   NUM_TEST_SAMPLE=-1
   export CUDA_VISIBLE_DEVICES="0"

   DATA_NAME="aime24"  # Options include: aime24, aime25, gpqa_diamond
   PROMPT_TYPE="r1-distilled-qwen"
   # NOTE:
   # for aime24 and aime25, use PROMPT_TYPE="r1-distilled-qwen";
   # for gpqa_diamond, use PROMPT_TYPE="r1-distilled-qwen-gpqa".

   TOKENIZERS_PARALLELISM=false \
   python3 -u math_eval.py \
       --model_name_or_path ${MODEL_NAME_OR_PATH} \
       --data_name ${DATA_NAME} \
       --output_dir ${OUTPUT_DIR} \
       --split ${SPLIT} \
       --prompt_type ${PROMPT_TYPE} \
       --num_test_sample ${NUM_TEST_SAMPLE} \
       --use_vllm \
       --save_outputs

For **batch evaluation**, you can run the ``main_eval.sh`` script. This script will sequentially evaluate the model on the AIME24, AIME25, and GPQA-diamond datasets.

.. code-block:: bash

   bash LLMEvalKit/evaluation/main_eval.sh /path/to/model_checkpoint

You can specify ``CUDA_VISIBLE_DEVICES`` in the script for more flexible GPU management.  


Evaluation Results
------------------------------

Results will be printed in the terminal and saved in ``OUTPUT_DIR``. Batch evaluation defaults to saving in the ``LLMEvalKit/evaluation/outputs`` directory.  
The results include:

1. Metadata (``xx_metrics.json``): statistical summary  
2. Complete model outputs (``xx.jsonl``): includes complete reasoning process and prediction results  

Metadata example:

.. code-block:: javascript

   {
       "num_samples": 30,
       "num_scores": 960,
       "timeout_samples": 0,
       "empty_samples": 0,
       "acc": 42.39375,
       "time_use_in_second": 3726.008672475815,
       "time_use_in_minite": "62:06"
   }

The field ``acc`` represents the **average accuracy across all sampled responses**, which is the main evaluation metric.

Model output example:

.. code-block:: javascript

   {
      "idx": 0, 
      "question": "Find the number of...", 
      "gt_cot": "None", 
      "gt": "204", // ground truth answer
      "solution": "... . Thus, we have the equation $(240-t)(s) = 540$ ..., ", // standard solution
      "answer": "204", // ground truth answer
      "code": ["Alright, so I need to figure out ... . Thus, the number of ... is \\(\\boxed{204}\\)."], // generated reasoning chains
      "pred": ["204"], // extracted answers from reasoning chains
      "report": [null], 
      "score": [true] // whether the extracted answers are correct
   }

Supported Datasets
------------------------------

The toolkit currently supports the following evaluation datasets:

.. list-table:: Supported Datasets
   :header-rows: 1
   :widths: 20 80

   * - Dataset
     - Description
   * - ``aime24``
     - Problems from **AIME 2024** (American Invitational Mathematics Examination), focusing on high-school Olympiad-level mathematical reasoning.
   * - ``aime25``
     - Problems from **AIME 2025**, same format as AIME24 but with a different test set.
   * - ``gpqa_diamond``
     - The most challenging subset (Diamond split) of **GPQA (Graduate-level Google-Proof Q&A)**,  
       containing cross-disciplinary problems (e.g., mathematics, physics, computer science) that require deep reasoning capabilities rather than memorization.

Parameter Configuration
------------------------------

The main configurable parameters are as follows:

.. list-table:: Configuration Parameter Description
   :header-rows: 1
   :widths: 20 80

   * - Name
     - Description
   * - ``data_name``
     - Dataset to evaluate. Supported: ``aime24``, ``aime25``, ``gpqa_diamond``.
   * - ``prompt_type``
     - Prompt template. Use ``r1-distilled-qwen`` for AIME datasets, ``r1-distilled-qwen-gpqa`` for GPQA.
   * - ``temperature``
     - Sampling temperature. Recommended: ``0.6`` for 1.5B models, ``1.0`` for 7B models.
   * - ``top_p``
     - Nucleus sampling parameter. Default: ``0.95``.
   * - ``n_sampling``
     - Number of responses sampled per question, used to compute average accuracy. Default: ``32``.
   * - ``max_tokens_per_call``
     - Maximum tokens generated per call. Default: ``32768``.
   * - ``output_dir``
     - Output directory for results. Default: ``./outputs``.


