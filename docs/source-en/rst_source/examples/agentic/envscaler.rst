Train a Tool-Calling Policy with EnvScaler
==========================================

Train Qwen3 with GRPO on EnvScaler's 2,550 reinforcement learning scenarios. Each
sample starts with a fresh business database. The policy can call one tool per
turn to inspect or modify it. When the episode ends, EnvScaler runs the task's
checklist against the final state. The fraction of checks passed supplies the
reward for each policy turn.

Installation
------------

Start from an RLinf agentic environment with SGLang and a local Qwen3-1.7B
checkpoint. Run these commands in the RLinf repository. Make the trusted EnvScaler
checkout and its bundled JSON files available at the same paths on every node.
EnvScaler executes Python environment definitions and checks from these files.

.. code-block:: bash

   git clone https://github.com/RUC-NLPIR/EnvScaler.git ../EnvScaler
   git -C ../EnvScaler checkout 7ab81e557c5992d62d4533e81bfec3d2ca0ae24b
   export ENVSCALER_ROOT="$(cd ../EnvScaler && pwd)"
   export PYTHONPATH="$PWD:$ENVSCALER_ROOT/interact_with_env:${PYTHONPATH:-}"
   python -m pip install openai python-dotenv

Run Training
------------

In the default non-conversational mode the policy receives the task at the start,
executes tools locally and submits a final reply to trigger scoring.

.. code-block:: bash

   python examples/agent/envscaler/train.py rollout.model.model_path=/path/to/Qwen3-1.7B

For conversations, supply an OpenAI-compatible chat service for the simulated
user. Export its settings on each node before starting Ray. Only the simulator
sees the private goal. It gives the opening request, answers questions and ends
the episode with ``###STOP###``. The same local checklist scores both modes.

.. code-block:: bash

   export USER_OPENAI_BASE_URL=http://user-model-host:8000/v1
   export USER_OPENAI_API_KEY='YOUR_KEY'
   python examples/agent/envscaler/train.py \
       rollout.model.model_path=/path/to/Qwen3-1.7B \
       agentloop.mode=conversational agentloop.user_model=YOUR_SERVED_MODEL
