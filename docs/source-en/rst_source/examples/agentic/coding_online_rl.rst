Coding Online RL
================

Connect the Continue VS Code extension to RLinf to collect live accept/reject code-completion feedback and update Qwen2.5-Coder-1.5B online with PPO, or run offline GRPO validation with an LLM judge. See the companion blog :doc:`A First Look at the "Last Mile" of Agent Deployment: Cursor Online Reinforcement Learning <../../resources/blog/build_a_coding_online_rl_case>`.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      Qwen2.5-Coder-1.5B

   .. grid-item-card:: Algorithm
      :text-align: center

      PPO (online) · GRPO (offline judge)

   .. grid-item-card:: Task
      :text-align: center

      IDE code completion (Continue / FIM)

   .. grid-item-card:: Hardware
      :text-align: center

      8 GPUs (online) · 1–8 GPUs (offline)

| **What you'll do:** install deps and Continue VSIX → download ``Qwen2.5-Coder-1.5B`` → launch online PPO or offline GRPO → verify with the test client.
| **Prerequisites:** :doc:`Installation & Hardware </rst_source/start/installation>` · VS Code + Continue VSIX (online) or LLM-as-judge API key (offline).

Online PPO uses a disaggregated 8-GPU single-node layout (rollout 0–3, inference 4–5, actor 6–7, reward 0–3; inference on port 8081, feedback on port 8082) in ``qwen2.5-1.5b-ppo.yaml``. Offline GRPO collocates all workers on 1–8 GPUs in ``qwen2.5-1.5b-grpo-llm_judge.yaml``.

Setup and Run
-------------

In mainland China, append ``--use-mirror`` to ``install.sh`` and run ``export HF_ENDPOINT=https://hf-mirror.com`` before ``hf download``. Prebuilt images are available on `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__; common config fields live in :doc:`Training configuration </rst_source/reference/configuration>`.

.. tab-set::

   .. tab-item:: Online RL (PPO + Continue)

      **1. Install**

      Install RLinf and ``httpx``, then install the modified Continue VSIX from `RLinf/continue releases <https://github.com/RLinf/continue/releases>`_ in VS Code (``code --install-extension ./continue-1.3.9.vsix``, or Command Palette ``Extensions: Install from VSIX``):

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install httpx

      Configure ``~/.continue/config.yaml`` with your RLinf host and select ``my-autocomplete`` as **Autocomplete Model** in Continue:

      .. dropdown:: Continue configuration (``~/.continue/config.yaml``)

         .. code-block:: yaml

            models:
              - name: my-autocomplete
                provider: openai
                model: Qwen2.5-Coder-1.5B
                apiBase: http://xxx:8081/v1
                apiKey: xxx
                roles:
                  - autocomplete

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

      Start the online RL service using ``examples/agent/coding_online_rl/config/qwen2.5-1.5b-ppo.yaml``, then open VS Code with Continue and code:

      .. code-block:: bash

         bash examples/agent/coding_online_rl/run_main_coding_online_rl.sh \
           rollout.model.model_path=./Qwen2.5-Coder-1.5B \
           runner.output_dir=./results

   .. tab-item:: Offline Validation (GRPO + LLM Judge)

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate
         uv pip install httpx modelscope

      **2. Download**

      Download the base model and the Python FIM dataset from ModelScope:

      .. code-block:: bash

         hf download Qwen/Qwen2.5-Coder-1.5B --local-dir ./Qwen2.5-Coder-1.5B
         modelscope download --dataset "paxionfruit/code-fim-v2-python-filtered" --local_dir ./code-fim-v2-python-filtered

      **3. Launch**

      Export your LLM-as-judge API credentials and launch offline validation with ``examples/agent/coding_online_rl/config/qwen2.5-1.5b-grpo-llm_judge.yaml``:

      .. code-block:: bash

         export LLMASJUDGE_API_URL=your_api_url
         export LLMASJUDGE_API_KEY=your_api_key
         export LLMASJUDGE_MODEL=your_model  # optional; ensure the prompt fits your model
         bash examples/agent/coding_online_rl/run_main_coding_rl_llm_judge.sh \
           rollout.model.model_path=./Qwen2.5-Coder-1.5B \
           runner.output_dir=./results \
           data.train_data_paths='["./code-fim-v2-python-filtered/code-fim-v2-python-filtered_formatted_train_3k.jsonl"]' \
           data.val_data_paths='["./code-fim-v2-python-filtered/code-fim-v2-python-filtered_formatted_test_1k.jsonl"]'

.. dropdown:: Verify the client and troubleshooting

   Run the test client to simulate Continue completion requests and feedback submission:

   .. code-block:: bash

      python examples/agent/coding_online_rl/simple_online_coding_client.py

   - **Port conflicts:** if port 8081 or 8082 is in use, update the ports in ``qwen2.5-1.5b-ppo.yaml``.
   - **Model loading failure:** confirm ``rollout.model.model_path`` points to an accessible local model directory.
   - **Continue connection failure:** check the API endpoints in ``~/.continue/config.yaml`` and network reachability, or run ``simple_online_coding_client.py`` to verify feedback delivery.

Visualization and Results
-------------------------

Track logs and metrics in TensorBoard, or see :doc:`Training metrics </rst_source/reference/metrics>` for W&B/SwanLab and metric definitions.

.. code-block:: bash

   tail -f results/online-ppo-1.5b-pipeline/train.log
   tensorboard --logdir results/Qwen2.5-Coder-1.5B-grpo-llm_judge

Checkpoints are saved under ``results/online-ppo-1.5b-pipeline/checkpoints/`` (online PPO) or ``results/Qwen2.5-Coder-1.5B-grpo-llm_judge/checkpoints/`` (offline GRPO).
