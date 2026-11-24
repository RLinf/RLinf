LingbotVLA 2.0 Supervised Fine-Tuning
=====================================

Run action-only RoboTwin supervised fine-tuning (SFT) from the LingbotVLA 2.0 Foundation Base with RLinf.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      LingbotVLA 2.0

   .. grid-item-card:: Methods
      :text-align: center

      Action-only SFT

   .. grid-item-card:: Data
      :text-align: center

      RoboTwin multi-dataset manifest

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 26 28 22

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - RoboTwin
     - Click Bell
     - ``robotwin_sft_lingbotvlav2``
     - Train the action expert and projection layers from the Foundation Base.

Prepare the Data
----------------

Download and extract the raw Click Bell demonstrations from the `RoboTwin dataset release <https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset/click_bell>`__. This converter requires the legacy layout: ``data/episode<i>.hdf5`` and ``instructions/episode<i>.json`` in each dataset directory.

Download the Model
------------------

Download the Foundation Base and tokenizer:

.. code-block:: bash

   hf download robbyant/lingbot-vla-v2-6b --local-dir lingbot-vla-v2-6b
   hf download Qwen/Qwen3-VL-4B-Instruct --local-dir Qwen3-VL-4B-Instruct

Use ``robbyant/lingbot-vla-v2-6b`` as the SFT starting checkpoint. The :doc:`PPO recipe <lingbotvlav2>` uses the separate ``robbyant/lingbot-vla-v2-6b-robotwin`` post-training checkpoint.

Installation
------------

Follow the installation steps on :doc:`LingbotVLA 2.0 <lingbotvlav2>` to install and activate the LingbotVLA environment and obtain its source checkout. Run the following commands from the RLinf repository root.

Run It
------

Convert each extracted RoboTwin dataset:

.. code-block:: bash

   python toolkits/lerobot/convert_robotwin_to_lerobot.py \
     --raw-root /path/to/click_bell/aloha-agilex_clean_50 \
     --output-root /path/to/robotwin_lerobot/click_bell_clean_50

The converter reports the number of episodes and frames written. Create ``robotwin_train.txt`` with one ``robot_name dataset_path`` entry per dataset:

.. code-block:: text

   robotwin /path/to/robotwin_lerobot/click_bell_clean_50

Edit these paths in ``examples/sft/config/robotwin_sft_lingbotvlav2.yaml``:

.. code-block:: yaml

   actor:
     model:
       model_path: /path/to/lingbot-vla-v2-6b
       tokenizer_path: /path/to/Qwen3-VL-4B-Instruct
       lingbotvlav2:
         config_path: /path/to/lingbot-vla-v2
         training_config_path: /path/to/lingbot-vla-v2/configs/vla/robotwin/robotwin.yaml
         robot_config_path: /path/to/lingbot-vla-v2/configs/robot_configs/robotwin.yaml
         robot_config_root: /path/to/lingbot-vla-v2/configs/robot_configs
         stats_path: /path/to/lingbot-vla-v2/assets/norm_stats/robotwin.json
   data:
     train_data_paths: /path/to/robotwin_train.txt

Run SFT on one node with 8 GPUs. The task config uses global batch size 8, FSDP2, a BF16 Qwen-VL backbone, and an FP32 action expert. SFT uses ``flex_cached`` attention; PPO and standalone evaluation keep the shared ``eager`` backend for cached-key/value decoding.

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh robotwin_sft_lingbotvlav2

The launcher writes logs and checkpoints under ``logs/<timestamp>-robotwin_sft_lingbotvlav2/``. Track ``train/l1_loss`` in TensorBoard; see :doc:`Training metrics <../../reference/metrics>` for the metric list.

Before evaluation, prepare the simulator and assets using the :doc:`RoboTwin evaluation guide <../../evaluations/guides/robotwin>`, then export the environment paths:

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/RoboTwin
   export ROBOTWIN_ASSETS_PATH=$ROBOTWIN_PATH
   export ROBOT_PLATFORM=ALOHA

Set the Foundation, tokenizer, and upstream LingbotVLA paths in ``evaluations/robotwin/robotwin_click_bell_lingbotvlav2_sft_eval.yaml``, then evaluate a saved SFT checkpoint:

.. code-block:: bash

   bash evaluations/run_eval.sh robotwin_click_bell_lingbotvlav2_sft_eval \
     runner.ckpt_path=/path/to/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt

Visualization and Results
-------------------------

The plotted comparison used Click Alarmclock data and the ModelScope Foundation snapshot ``Robbyant/lingbot-vla-v2-6b`` at revision ``edfff7638a31a5dbfaae87016477191735bd754e``. Official and RLinf each completed 423 optimizer updates on the same 3,384-occurrence manifest with global batch size 8. This ModelScope revision identifies the experiment weights; the Hugging Face download above has its own revision history.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/9f22f0d7c6ed5e4b27c0212b3da848831d1a3c2a/pic/lingbotvlav2_foundation_action_loss_ma25.png
   :width: 90%

   Action Loss for the Foundation Base runs and their 25-update moving average.

.. list-table:: Action Loss Statistics
   :header-rows: 1
   :widths: 40 30 30

   * - Metric
     - Official
     - RLinf
   * - Mean
     - 0.102265
     - 0.101771
   * - Last 50 updates mean
     - 0.065720
     - 0.065274

The per-update Action Loss has an MAE of 0.002404, Pearson correlation of 0.9984, and maximum absolute difference of 0.014081. The chart compares Official ``VLA_Loss`` with RLinf ``train/l1_loss``. Total Loss may include different auxiliary terms and is evaluated separately. This is a controlled comparison; it does not establish strict parity or task success improvement.
