ActiveVLN with EmbodiInfer
==========================

Run ActiveVLN GRPO training with a Hugging Face actor, EmbodiInfer rollout,
and the VLN-CE environment server. The example below uses GPU 0 for training
and inference and GPU 1 for Habitat rendering, with batch size 2 and BF16 weights.

Installation
------------

Use separate Python environments for RLinf and the Habitat server. Pin the
ActiveVLN source that supplies both the HTTP client and server:

.. code-block:: bash

   export ACTIVEVLN_ROOT="$HOME/ActiveVLN"
   git clone https://github.com/arvillion/ActiveVLN.git "$ACTIVEVLN_ROOT"
   git -C "$ACTIVEVLN_ROOT" checkout 3a0c63b00e4f42c828cc74c3554afce17641da60
   export PYTHONPATH="$ACTIVEVLN_ROOT:$ACTIVEVLN_ROOT/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"

Install the RLinf training environment, then the pinned EmbodiInfer runtime
and ActiveVLN client. Keep the source checkout on ``PYTHONPATH`` in each terminal
because the upstream client uses a namespace package.

.. code-block:: bash

   # Run from the RLinf checkout.
   bash requirements/install.sh --no-flash-attn embodied --env dummy --venv .venv-activevln
   source .venv-activevln/bin/activate
   uv pip install -r requirements/embodiinfer.txt
   uv pip install -e "$ACTIVEVLN_ROOT/vlnce_server" httpx
   python -c 'from vlnce_server.client import BatchEnvClient; from vlnce_server.env_config import VLNCEEnvConfig'

``requirements/embodiinfer.txt`` pins EmbodiInfer to
``2c6b06835ac41bba80501dbf2bbbb4e1281aa2c5`` and Transformers to ``4.51.3``.

Prepare and Start the Server
----------------------------

In a separate ``server`` environment, follow the pinned
`ActiveVLN server installation <https://github.com/arvillion/ActiveVLN/blob/3a0c63b00e4f42c828cc74c3554afce17641da60/README.md#setup-the-vln-ce-environemnt>`_
for Python 3.10, Habitat-Sim and Habitat-Lab v0.1.7. Follow its
`data preparation <https://github.com/arvillion/ActiveVLN/blob/3a0c63b00e4f42c828cc74c3554afce17641da60/README.md#data-preparation>`_
to place Matterport3D scenes under ``$ACTIVEVLN_ROOT/data/scene_datasets/mp3d``
and R2R/RxR annotations under ``$ACTIVEVLN_ROOT/data/datasets``.
Include the training annotations and NDTW ground-truth files for both datasets;
this server revision loads both even when the RxR worker count is zero.

Start two R2R simulators on physical GPU 1 from the ActiveVLN checkout:

.. code-block:: bash

   conda activate server
   python -m pip install ray flask hydra-core av
   cd "$ACTIVEVLN_ROOT"
   export PYTHONPATH="$ACTIVEVLN_ROOT:$ACTIVEVLN_ROOT/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"
   python -m vlnce_server.server --config-path "$ACTIVEVLN_ROOT/vlnce_server/config" \
     server.host=127.0.0.1 server.port=5001 \
     'vlnce.gpus=[1]' 'vlnce.r2r_gpu_plan=[2]' 'vlnce.rxr_gpu_plan=[0]'

Keep this process running. ``vlnce.r2r_gpu_plan`` controls simulator capacity;
raise it when increasing ``env.train.total_num_envs`` or ``env.eval.total_num_envs``.

Download the Model
------------------

In the RLinf training environment, download the checkpoint and tokenizer at a
fixed revision:

.. code-block:: bash

   export ACTIVEVLN_MODEL="$HOME/models/activevln-r2r"
   hf download Arvil/Qwen2.5-VL-3B_rl_r2r_4000 \
     --revision 160987313e3e869705f42400d1b8f28177044518 \
     --local-dir "$ACTIVEVLN_MODEL"

Run Training
------------

From the RLinf checkout, with the client ``PYTHONPATH`` exported, run:

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   export PYTHONPATH="$PWD:$PYTHONPATH"
   CUDA_VISIBLE_DEVICES=0 python examples/embodiment/train_embodied_agent.py \
     --config-name activevln_r2r_grpo_embodiinfer \
     actor.model.model_path="$ACTIVEVLN_MODEL" \
     env.train.tokenizer_path="$ACTIVEVLN_MODEL" \
     env.eval.tokenizer_path="$ACTIVEVLN_MODEL"

The configuration enables CUDA graphs, fused operations, split attention,
GPU image preprocessing and episode-history reuse. It runs two training steps
and repeats R2R episode 1 by default. Set ``env.train.episode_ids`` and
``env.eval.episode_ids`` to select more episodes, and set ``runner.max_steps``
and ``runner.max_epochs`` for longer training. For a remote server, override
``env.train.server_url`` and ``env.eval.server_url``.

Select inference with ``rollout.generation_backend=embodiinfer`` and set runtime
options under ``rollout.embodiinfer``. The actor is selected independently with
``actor.model.model_backend=huggingface``. Set ``rollout.generation_backend=huggingface``
to use HF generation with the same actor and environment.

Repetition penalties include the prompt and generated response. Both rollout
log probabilities and actor replay use the distribution after top-p filtering.
