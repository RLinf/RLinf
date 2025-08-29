Model Evaluation
================

Introduction
------------
We provide an integrated evaluation toolkit for long chain-of-thought (CoT) mathematical reasoning.  
The `toolkit <https://github.com/RLinf/LLMEvalKit>`_ includes both code and datasets, allowing researchers to benchmark trained LLMs on math-related reasoning tasks.  

**Acknowledgements:** This evaluation toolkit is adapted from `Qwen2.5-Math <https://github.com/QwenLM/Qwen2.5-Math>`_.

Environment Setup
-----------------
First, clone the repository:

.. code-block:: bash

   git clone https://github.com/RLinf/LLMEvalKit.git 

To use the package, install the required dependencies:

.. code-block:: bash

   pip install -r requirements.txt 

If you are using our Docker image, you only need to additionally install:

.. code-block:: bash

   pip install Pebble
   pip install timeout-decorator

Quick Start
-----------
To run evaluation on a single dataset:

.. code-block:: bash

   MODEL_NAME_OR_PATH=/model/path  # replace with your model path
   OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval
   SPLIT="test"
   NUM_TEST_SAMPLE=-1
   export CUDA_VISIBLE_DEVICES="0"

   DATA_NAME="aime24"  # options: aime24, aime25, gpqa_diamond
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

For **batch evaluation**, run:

.. code-block:: bash

   bash main_eval.sh

You must set ``MODEL_NAME_OR_PATH`` and ``CUDA_VISIBLE_DEVICES`` in the script.  
This will sequentially evaluate the model on AIME24, AIME25, and GPQA-diamond datasets.  

Results
-------
The results are printed to the console and stored in ``OUTPUT_DIR``.  
Stored outputs include:

1. Metadata (``xx_metrics.json``): summary statistics.  
2. Full model outputs (``xx.jsonl``): complete reasoning traces and predictions.  

Example Metadata:

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

``acc`` reports the **average accuracy across all sampled responses**, which serves as the main evaluation metric.  

Example Model Output:

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

Datasets
--------
The toolkit currently supports the following evaluation datasets:

.. list-table:: Supported Datasets
   :header-rows: 1
   :widths: 20 80

   * - Dataset
     - Description
   * - ``aime24``
     - Problems from the **American Invitational Mathematics Examination (AIME) 2024**, focusing on high-school Olympiad-level mathematics reasoning.
   * - ``aime25``
     - Problems from the **AIME 2025**, same format as AIME24 but with different test set.
   * - ``gpqa_diamond``
     - A subset of **GPQA (Graduate-level Google-Proof Q&A)** with the most challenging questions (Diamond split). Covers multi-disciplinary topics (e.g., mathematics, physics, computer science) requiring deep reasoning beyond memorization.

Configuration
-------------
The main configurable parameters are:

.. list-table:: Configuration Parameters
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


