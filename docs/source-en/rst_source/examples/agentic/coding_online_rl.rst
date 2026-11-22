Coding Online RL
================

Use this recipe to connect Continue to RLinf, collect accept/reject feedback on code completions, and update a Qwen coder model online.

Related reading: :doc:`A First Look at the "Last Mile" of Agent Deployment: Cursor Online Reinforcement Learning <../../resources/blog/build_a_coding_online_rl_case>`.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-Coder-1.5B

   .. grid-item-card:: Algorithm
      :text-align: center

      PPO online RL or GRPO offline validation

   .. grid-item-card:: Feedback
      :text-align: center

      Continue accept / reject events or LLM-as-judge labels

   .. grid-item-card:: Services
      :text-align: center

      Inference on ``8081`` and feedback ingestion on ``8082``

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Step
     - Component
     - Outcome
   * - Real-time interaction
     - Continue extension
     - Sends code-completion requests to RLinf
   * - Model inference
     - RLinf inference service
     - Returns code-completion suggestions
   * - User feedback
     - Continue tracking callback
     - Records accepted or rejected completions
   * - Online learning
     - RLinf training service
     - Updates the policy from feedback

This recipe includes both an online RL path (with Continue) and an offline validation path that simulates human preference with an LLM judge (no Continue extension required).

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__. For common path, runner, rollout, and cluster fields, see :doc:`Training configuration <../../reference/configuration>`.

.. tab-set::

   .. tab-item:: Online RL (PPO + Continue)

      **1. Install**

      Install RLinf and the HTTP client dependencies, then install the modified Continue VSIX from `RLinf/continue releases <https://github.com/RLinf/continue/releases>`_ in VS Code (``code --install-extension /path/to/continue-1.3.9.vsix``, or via Command Palette ``Extensions: Install from VSIX``):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install httpx

      Add the autocomplete model and feedback tracking settings to ``~/.continue/config.yaml`` (replace ``http://xxx:xx/`` with your RLinf service host), then set **Autocomplete Model** to ``my-autocomplete`` in the Continue **Models** settings:

      .. dropdown:: Continue configuration (``~/.continue/config.yaml``)

         .. code-block:: yaml

            # Add a model for code completion
            models:
              - name: my-autocomplete
                provider: openai
                model: Qwen2.5-Coder-1.5B
                apiBase: http://xxx:8081/v1
                apiKey: xxx
                roles:
                  - autocomplete

            # Send user accept/reject feedback on code completions
            tabAutocompleteOptions:
              enableCompletionTracking: true
              completionTrackingUrl: http://xxx:8082/api/training/submit
              completionTrackingHeaders:
                Authorization: Bearer test-token
                X-Project-ID: test-project
              maxPromptTokens: 1024
              debounceDelay: 350
              multilineCompletions: auto

      **2. Download**

      .. code-block:: bash

         hf download Qwen/Qwen2.5-Coder-1.5B --local-dir ./Qwen2.5-Coder-1.5B

      **3. Launch**

      Set ``runner.output_dir`` and ``rollout.model.model_path`` in ``examples/agent/coding_online_rl/config/qwen2.5-1.5b-ppo.yaml``:

      .. code-block:: yaml

         runner:
           output_dir: /path/to/your/logs

         rollout:
           model:
             model_path: /path/to/your/model

      Start the online RL service (inference API on port ``8081``, feedback ingestion on port ``8082``), then open VS Code with Continue and start coding:

      .. code-block:: bash

         bash examples/agent/coding_online_rl/run_main_coding_online_rl.sh

   .. tab-item:: Offline Validation (GRPO + LLM Judge)

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install httpx

      **2. Download**

      .. code-block:: bash

         uv pip install modelscope
         hf download Qwen/Qwen2.5-Coder-1.5B --local-dir ./Qwen2.5-Coder-1.5B
         modelscope download --dataset "paxionfruit/code-fim-v2-python-filtered" --local_dir ./code-fim-v2-python-filtered

      **3. Launch**

      Set ``runner.output_dir``, ``rollout.model.model_path``, ``data.train_data_paths``, and ``data.val_data_paths`` in ``examples/agent/coding_online_rl/config/qwen2.5-1.5b-grpo-llm_judge.yaml``:

      .. code-block:: yaml

         runner:
           output_dir: /path/to/your/logs

         rollout:
           model:
             model_path: /path/to/your/model

         data:
           train_data_paths: ["/path/to/your/dataset/code-fim-v2-python-filtered_formatted_train_3k.jsonl"]
           val_data_paths: ["/path/to/your/dataset/code-fim-v2-python-filtered_formatted_test_1k.jsonl"]

      Export the LLM-as-judge credentials and launch offline validation:

      .. code-block:: bash

         export LLMASJUDGE_API_URL=your_api_url
         export LLMASJUDGE_API_KEY=your_api_key
         export LLMASJUDGE_MODEL=your_model  # optional; ensure the prompt fits your model
         bash examples/agent/coding_online_rl/run_main_coding_rl_llm_judge.sh

Visualization and Results
-------------------------

Monitor logs, TensorBoard, and checkpoints. For common metric meanings, see :doc:`Training metrics <../../reference/metrics>`.

1. **View logs**

   .. code-block:: bash

      tail -f results/online-ppo-1.5b-pipeline/train.log

2. **Launch TensorBoard**

   .. code-block:: bash

      tensorboard --logdir results/Qwen2.5-Coder-1.5B-grpo-llm_judge

3. **Check checkpoints**

   Model checkpoints are saved periodically under ``results/Qwen2.5-Coder-1.5B-grpo-llm_judge/checkpoints/`` (or ``results/online-ppo-1.5b-pipeline/checkpoints/`` for online PPO).

Verify the Client
-----------------

Run the test client to simulate Continue requests and feedback submission:

.. code-block:: bash

   python examples/agent/coding_online_rl/simple_online_coding_client.py

Troubleshooting
---------------

1. **Port conflicts**

   If ports ``8081`` or ``8082`` are occupied, change the port settings in the YAML config.

2. **Model loading failure**

   Verify that ``rollout.model.model_path`` points to an accessible local model directory.

3. **Continue connection failure**

   Check the API endpoints in ``~/.continue/config.yaml`` and network reachability, or run ``simple_online_coding_client.py`` to verify feedback delivery.
