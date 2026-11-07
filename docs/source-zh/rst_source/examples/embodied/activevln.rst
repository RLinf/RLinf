使用 EmbodiInfer 训练 ActiveVLN
===============================

本例使用 Hugging Face actor、EmbodiInfer rollout 和 VLN-CE 环境服务进行 ActiveVLN GRPO 训练。下面的配置将训练和推理放在 GPU 0，Habitat 渲染放在 GPU 1，batch 为 2，权重采用 BF16。

安装依赖
--------

为 RLinf 和 Habitat 服务分别准备 Python 环境。先固定提供 HTTP client 和 server 的 ActiveVLN 源码版本：

.. code-block:: bash

   export ACTIVEVLN_ROOT="$HOME/ActiveVLN"
   git clone https://github.com/arvillion/ActiveVLN.git "$ACTIVEVLN_ROOT"
   git -C "$ACTIVEVLN_ROOT" checkout 3a0c63b00e4f42c828cc74c3554afce17641da60
   export PYTHONPATH="$ACTIVEVLN_ROOT:$ACTIVEVLN_ROOT/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"

安装 RLinf 训练环境，再安装固定版本的 EmbodiInfer 和 ActiveVLN client。上游 client 使用 namespace package，因此每个终端都需要将源码目录加入 ``PYTHONPATH``。

.. code-block:: bash

   # Run from the RLinf checkout.
   bash requirements/install.sh --no-flash-attn embodied --env dummy --venv .venv-activevln
   source .venv-activevln/bin/activate
   uv pip install -r requirements/embodiinfer.txt
   uv pip install -e "$ACTIVEVLN_ROOT/vlnce_server" httpx
   python -c 'from vlnce_server.client import BatchEnvClient; from vlnce_server.env_config import VLNCEEnvConfig'

``requirements/embodiinfer.txt`` 将 EmbodiInfer 固定到 ``2c6b06835ac41bba80501dbf2bbbb4e1281aa2c5``，Transformers 固定为 ``4.51.3``。

准备数据并启动服务
------------------

在独立的 ``server`` 环境中，按固定版本的 `ActiveVLN 服务端安装说明 <https://github.com/arvillion/ActiveVLN/blob/3a0c63b00e4f42c828cc74c3554afce17641da60/README.md#setup-the-vln-ce-environemnt>`_ 安装 Python 3.10、Habitat-Sim v0.1.7 和 Habitat-Lab v0.1.7。按其 `数据准备说明 <https://github.com/arvillion/ActiveVLN/blob/3a0c63b00e4f42c828cc74c3554afce17641da60/README.md#data-preparation>`_ 将 Matterport3D 场景放到 ``$ACTIVEVLN_ROOT/data/scene_datasets/mp3d``，将 R2R/RxR 标注放到 ``$ACTIVEVLN_ROOT/data/datasets``。两个数据集的训练标注和 NDTW ground-truth 文件都需要准备；这个服务端版本在 RxR worker 数量为零时也会加载 RxR 数据。

从 ActiveVLN 源码目录启动服务，在物理 GPU 1 上创建两个 R2R simulator：

.. code-block:: bash

   conda activate server
   python -m pip install ray flask hydra-core av
   cd "$ACTIVEVLN_ROOT"
   export PYTHONPATH="$ACTIVEVLN_ROOT:$ACTIVEVLN_ROOT/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"
   python -m vlnce_server.server --config-path "$ACTIVEVLN_ROOT/vlnce_server/config" \
     server.host=127.0.0.1 server.port=5001 \
     'vlnce.gpus=[1]' 'vlnce.r2r_gpu_plan=[2]' 'vlnce.rxr_gpu_plan=[0]'

保持服务运行。``vlnce.r2r_gpu_plan`` 控制 simulator 数量；增大 ``env.train.total_num_envs`` 或 ``env.eval.total_num_envs`` 时，相应提高服务端容量。

下载模型
--------

在 RLinf 训练环境中，下载固定版本的权重和 tokenizer：

.. code-block:: bash

   export ACTIVEVLN_MODEL="$HOME/models/activevln-r2r"
   hf download Arvil/Qwen2.5-VL-3B_rl_r2r_4000 \
     --revision 160987313e3e869705f42400d1b8f28177044518 \
     --local-dir "$ACTIVEVLN_MODEL"

运行训练
--------

回到 RLinf 源码目录，保留 client 的 ``PYTHONPATH`` 设置，然后运行：

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   export PYTHONPATH="$PWD:$PYTHONPATH"
   CUDA_VISIBLE_DEVICES=0 python examples/embodiment/train_embodied_agent.py \
     --config-name activevln_r2r_grpo_embodiinfer \
     actor.model.model_path="$ACTIVEVLN_MODEL" \
     env.train.tokenizer_path="$ACTIVEVLN_MODEL" \
     env.eval.tokenizer_path="$ACTIVEVLN_MODEL"

配置默认开启 CUDA graphs、融合算子、split attention、GPU 图像预处理和 episode 历史复用，运行两个训练步并重复使用 R2R episode 1。通过 ``env.train.episode_ids`` 和 ``env.eval.episode_ids`` 选择更多 episode，通过 ``runner.max_steps`` 和 ``runner.max_epochs`` 调整训练长度。服务部署在远端时，设置 ``env.train.server_url`` 和 ``env.eval.server_url``。

通过 ``rollout.generation_backend=embodiinfer`` 选择推理后端，在 ``rollout.embodiinfer`` 下设置运行时选项。Actor 通过 ``actor.model.model_backend=huggingface`` 独立选择。将 ``rollout.generation_backend`` 改为 ``huggingface``，即可使用相同的 actor 和环境进行 HF 生成。

重复惩罚作用于 prompt 和已生成的 response。Rollout 返回的 logprob 与 actor 重算均使用 top-p 截断后的分布。
