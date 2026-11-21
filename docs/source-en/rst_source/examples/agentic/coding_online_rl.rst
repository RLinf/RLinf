Online RL for Code Completion Agent
=======================================================

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

Installation
------------

Install RLinf first, then add the HTTP client dependencies:

.. code-block:: bash

   pip install httpx asyncio

For offline validation, download the dataset:

.. code-block:: bash

   modelscope download --dataset "paxionfruit/code-fim-v2-python-filtered" --local_dir code-fim-v2-python-filtered

Run It
------

Configure Continue Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Install the modified Continue extension**

   Upstream Continue does not upload code-completion preference feedback. Download the modified VSIX from `RLinf/continue releases <https://github.com/RLinf/continue/releases>`_ (or build it from source), then install it in VS Code:

   - CLI: ``code --install-extension /path/to/continue-1.3.9.vsix``
   - Command Palette: press ``Cmd+Shift+P``, run ``Extensions: Install from VSIX``, and select the file.

2. **Configure Continue settings**

   Add the following to ``~/.continue/config.yaml`` (replace ``http://xxx:xx/`` with your RLinf service host):

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

   After saving, open the Continue sidebar, click the Settings gear icon, and set **Autocomplete Model** to ``my-autocomplete`` on the **Models** page.

Start Training Service
~~~~~~~~~~~~~~~~~~~~~~

For common path, runner, rollout, and cluster fields, see :doc:`Training configuration <../../reference/configuration>`.

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - Mode
     - Config
     - Launch script
   * - Online RL (PPO)
     - ``examples/agent/coding_online_rl/config/qwen2.5-1.5b-ppo.yaml``
     - ``examples/agent/coding_online_rl/run_main_coding_online_rl.sh``
   * - Offline validation (GRPO)
     - ``examples/agent/coding_online_rl/config/qwen2.5-1.5b-grpo-llm_judge.yaml``
     - ``examples/agent/coding_online_rl/run_main_coding_rl_llm_judge.sh``

1. **Edit the config**

   Set ``runner.output_dir`` and ``rollout.model.model_path`` in the chosen YAML:

   .. code-block:: yaml

      runner:
        output_dir: /path/to/your/logs

      rollout:
        model:
          model_path: /path/to/your/model

   For offline validation, also set ``data.train_data_paths`` and ``data.val_data_paths`` in ``qwen2.5-1.5b-grpo-llm_judge.yaml`` and export the LLM-as-judge credentials:

   .. code-block:: yaml

      data:
        train_data_paths: ["/path/to/your/dataset/code-fim-v2-python-filtered_formatted_train_3k.jsonl"]
        val_data_paths: ["/path/to/your/dataset/code-fim-v2-python-filtered_formatted_test_1k.jsonl"]

   .. code-block:: bash

      export LLMASJUDGE_API_URL=your_api_url
      export LLMASJUDGE_API_KEY=your_api_key
      export LLMASJUDGE_MODEL=your_model  # optional; ensure the prompt fits your model

2. **Launch the training service**

   From the RLinf repository root, start online RL (exposes the inference API on port ``8081`` and feedback ingestion on port ``8082``):

   .. code-block:: bash

      bash examples/agent/coding_online_rl/run_main_coding_online_rl.sh

   Or start offline validation with LLM-as-judge feedback:

   .. code-block:: bash

      bash examples/agent/coding_online_rl/run_main_coding_rl_llm_judge.sh

Use Continue
~~~~~~~~~~~~

Open VS Code with the configured Continue extension and start coding. Continue sends completion requests to port ``8081`` and submits accepted completions as positive feedback and rejected completions as negative feedback to port ``8082``, updating the policy online.

Visualization and Results
-------------------------

Monitor logs, TensorBoard, and checkpoints. For common metric meanings, see :doc:`Training metrics <../../reference/metrics>`.

1. **View logs**

   .. code-block:: bash

      tail -f results/ppo-1.5b/train.log

2. **Launch TensorBoard**

   .. code-block:: bash

      tensorboard --logdir results/grpo-1.5b

3. **Check checkpoints**

   Model checkpoints are saved periodically under ``results/grpo-1.5b/checkpoints/``.

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
