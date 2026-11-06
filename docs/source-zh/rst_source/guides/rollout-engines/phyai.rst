PhyAI 具身 Rollout 后端
=======================

本页介绍如何用 `PhyAI <https://github.com/mingti-org/phyai>`_ 在进程内训练 PI0.5 policy，或通过外部 Gateway 执行评估。PhyAI 是面向 VLA 等物理 AI 模型的推理框架。进程内路径由每个 RLinf rollout worker 管理一个 engine，LIBERO 环境、FSDP actor、观测变换和权重同步仍由 RLinf 负责。

下面先说明进程内 PPO 的安装与训练流程。需要独立管理模型服务时，请阅读 :ref:`phyai-http-evaluation-zh`。该评估路径使用 ``PhyaiEmbodiedWorker``，Gateway 和模型 Server 运行在另一套环境中。

准备工作
--------

需要一台配备 NVIDIA GPU 的 Linux 机器和 Python 3.12。目前该后端支持在 LIBERO 上用同步 PPO 训练 PI0.5，每个 rollout worker 使用一张 GPU。训练流程已使用 RLinf PI0.5 LIBERO SFT checkpoint，在两张 NVIDIA H800 80 GB GPU 上完成 smoke test。

安装
----

在 RLinf 仓库根目录下，将 RLinf、OpenPI、LIBERO 和 PhyAI 安装到同一个虚拟环境，然后激活该环境：

.. code-block:: bash

   bash requirements/install.sh \
     --python 3.12 \
     --torch 2.11.0 \
     --install-rlinf \
     embodied \
     --model openpi \
     --env libero \
     --engine phyai
   source .venv/bin/activate

这条命令做了两件事：

1. 按 Hugging Face rollout 的方式安装 RLinf、OpenPI 和 LIBERO。
2. 按已安装 Torch 的 CUDA 版本，安装 RLinf 测试过的 PhyAI 版本。PhyAI 自带的依赖会替换 OpenPI 所需的 Torch 和 Transformers 版本，因此安装器保留原有版本；如果 PhyAI 的 CUDA 版本与 Torch 不一致，安装会直接停止。

准备 Checkpoint
---------------

先下载 PI0.5 LIBERO SFT checkpoint：

.. code-block:: bash

   # For mainland China users: export HF_ENDPOINT=https://hf-mirror.com
   hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir RLinf-Pi05-LIBERO-SFT

然后在 ``examples/embodiment/config/libero_spatial_ppo_openpi_pi05_phyai.yaml`` 中把 ``actor.model.model_path`` 设为下载目录。rollout 读取同一路径，因此 PhyAI 与 actor 加载相同的权重，并用同一份统计量归一化观测。其他 PI0.5 checkpoint 及下载镜像见 :doc:`π₀ 与 π₀.₅ 模型强化学习训练 <../../examples/embodied/pi0>`。

如果要从 RL 训练得到的 checkpoint 继续，还需将 ``runner.ckpt_path`` 设为对应的 ``.pt`` 文件。每个 PhyAI engine 会在下载的 checkpoint 之上加载这份 state dict，因此第一次 rollout 使用的权重与 actor 一致，冻结参数也不例外。

配置后端
--------

示例配置已经选择了 PhyAI，关键的几行如下：

.. code-block:: yaml

   defaults:
     - hybrid_engines/phyai@rollout.phyai

   algorithm:
     recompute_logprobs: True
     importance_sampling_fix: True
     importance_sampling_clip: 1.0

   rollout:
     rollout_backend: phyai
     phyai:
       max_batch_size: null

``rollout.rollout_backend: phyai`` 让训练脚本启动 PhyAI rollout worker，而不是 Hugging Face rollout worker。``hybrid_engines/phyai`` 提供 kernel 策略，固定 PhyAI 在 PI0.5 上使用的 attention 和激活函数 kernel，使其数值在两种实现允许的范围内尽量接近 actor。``max_batch_size`` 限制单个 engine 一次处理的观测数量；设为 ``null`` 时，按分配给该 worker 的环境数量确定。

``algorithm`` 中的三个配置项用于处理 PhyAI 与 actor 之间剩余的数值差异，下一节说明原因。

修正 Rollout 与 Actor 的差异
----------------------------

PhyAI 和 FSDP actor 使用同一份权重，但计算所用的 kernel 不同，例如 paged attention、融合的激活函数和不同的归约顺序。因此，actor 为 PhyAI 采样出的动作计算 log-probability 时，结果会与 PhyAI 报告的值略有出入。

PPO 以产生数据的 policy 为基准衡量每次更新。如果这个基准来自 PhyAI，kernel 带来的差异就会进入每一个 PPO ratio，甚至单独触发 clipping。开启 ``recompute_logprobs`` 后，actor 会在训练前为每个采样动作重新计算 log-probability，并以自己的结果作为基准。``importance_sampling_fix`` 再按 actor 与 PhyAI 对同一动作的概率之比为每个样本加权，权重上限为 ``importance_sampling_clip``，从而让 actor 能正确使用并非由自己采样的数据。这三个配置项需要一起开启：只开启 ``importance_sampling_fix`` 而未开启 ``recompute_logprobs`` 时，启动阶段会报错。

修正方法及其指标的详细说明见 :doc:`PPO <../../reference/algorithms/ppo>`。

启动训练
--------

设置好 checkpoint 路径后，在仓库根目录启动示例：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh \
     libero_spatial_ppo_openpi_pi05_phyai

该命令通过具身训练入口启动 FSDP actor、PhyAI rollout worker 和 LIBERO 环境 worker。每个训练 step 先由 PhyAI 收集 rollout，再由 actor 重新计算 log-probability 并更新参数，最后在下一轮 rollout 前把新权重同步到所有 PhyAI engine。

检查训练结果
------------

启动脚本将日志写入 ``logs/<timestamp>-<config-name>/``。``env/success_once`` 反映任务成功率。``actor/importance_sampling_weight`` 接近 1、``actor/importance_sampling_clip_fraction`` 较小，说明 PhyAI 与 actor 的结果仍然一致；如果权重逐渐偏离 1，说明二者的差异在扩大。其余指标的定义见 :doc:`训练指标 <../../reference/metrics>`。

.. _phyai-http-evaluation-zh:

通过外部 Gateway 评估
---------------------

如果模型推理由独立服务管理，使用这条路径。评估入口启动 RLinf 环境 worker 和 HTTP rollout worker，你需要先启动模型 Server 和 Gateway。``serving_mode: worker_http`` 选择这条仅用于评估的路径；上面的 PPO 流程使用进程内的 ``PhyAIRolloutWorker``。

.. code-block:: text

   RLinf EnvWorker -> PhyaiEmbodiedWorker
       -> HTTP MessagePack -> PhyAI Gateway
       -> gRPC -> PI0.5 model server

安装评估客户端
~~~~~~~~~~~~~~

复用 RLinf 已有的 SGLang/LIBERO 安装入口，创建独立客户端环境。官方公共 ``InferenceHTTPClient`` 未作修改；即使请求目标是 PhyAI Gateway，MessagePack 分支仍从 SGLang action protocol 导入编解码函数：

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh      --python 3.12      --venv .venv-phyai-http      --install-rlinf      embodied --model sglang --sglang 0.5.19 --env libero
   source .venv-phyai-http/bin/activate

这条安装命令准备官方 codec 及其导入依赖，不修改 RLinf 的公共 HTTP Client 或项目依赖元数据。安装软件与启动服务是两回事，下面实际启动的推理服务只有 PhyAI 模型 Server 和 Gateway。客户端与服务端仍使用独立环境，因为 SGLang、Torch 和 Transformers 的依赖可能不同。

.. warning::

   为这些命令创建新环境。安装器会同步所选依赖，不要指向还需要保留额外软件包的混合实验环境。

准备服务环境和模型资源
~~~~~~~~~~~~~~~~~~~~~~

在独立环境中复用进程内后端已有的 ``--engine phyai`` 安装入口。它选择 CUDA 12/13 依赖，将 PhyAI 固定为 ``0.2.0rc1``，并检查 PhyAI 安装是否保留 RLinf 所用的 Torch 和 Transformers 版本。随后只补 Gateway、processor 及其 HTTP 专用依赖：

.. code-block:: bash

   cd /path/to/RLinf
   bash requirements/install.sh      --python 3.12      --torch 2.11.0      --venv .venv-phyai-serving      --install-rlinf      embodied --model openpi --env libero --engine phyai
   source .venv-phyai-serving/bin/activate
   env -u UV_TORCH_BACKEND uv pip install      -r requirements/agentic/phyai_0.2.0rc1_http.txt
   env -u UV_TORCH_BACKEND uv pip install --no-deps      "phyai-gateway==0.2.0rc1"      "phyai-utils-tools==0.2.0rc1"      --index https://mingti-org.github.io/phyai/simple/

HTTP 依赖清单仅补充已有安装器，不另行维护一套 PhyAI runtime 配方。``--no-deps`` 避免新增软件包替换已验证的 Torch/Transformers 运行栈。CI 在补包前后检查这两个版本，再检查 Server 配置。环境的 ``bin`` 目录应位于 ``PATH`` 中，供 ``ninja`` 等 kernel 构建工具使用；激活环境和 Shell e2e 脚本都会处理这一点。补充安装仍需在 A800 上做干净环境验证，进程内安装通过不能代替 HTTP 服务和 checkpoint processor 的验证。

准备本地 LeRobot 格式的 PI0.5 LIBERO checkpoint，包括预处理、归一化及动作后处理文件，以及完整的本地 PaliGemma tokenizer 快照。checkpoint 应对应 8 维 state、7 维 action，并能返回至少 16 步的动作块。上面的进程内 RLinf/OpenPI checkpoint 使用另一种格式，替换前需要确认 ``PI05Backend`` 是否兼容。模型和 tokenizer 的获取及访问授权由部署方另行准备。

在 RLinf 仓库根目录中创建 ``pi05-http.yaml``，内容如下，资源路径填写模型所在机器上的实际位置：

.. code-block:: yaml

   plugin: pi05
   plugin_args:
     checkpoint_dir: /path/to/lerobot-pi05-libero
     max_batch_size: 8
     inputs_image_shape:
       - [360, 360, 3]
       - [360, 360, 3]
   config:
     device:
       target: cuda
       params_dtype: bfloat16
     runtime:
       use_cuda_graph: true
       flashinfer_workspace_bytes: 2147483648
   deployment:
     mode: auto
   server:
     host: 127.0.0.1
     port: 50063
     model_name: pi05
     workers: 1
   adapter:
     factory: phyai.server.backends.pi05:PI05Backend
     args:
       tokenizer_dir: /path/to/paligemma-tokenizer
       image_names: [main_images, wrist_images]
       state_dim: 8
       action_dim: 7

这里的相机名匹配 RLinf HTTP 请求，不是 LeRobot 客户端使用的 LIBERO 原始相机 key。``max_batch_size: 8`` 覆盖下面单 worker、8 个环境的示例。增大请求 batch 前，同步调整该上限，并检查显存和 100 MiB 消息限制。FlashInfer workspace 显式设为 2 GiB，避免使用较小的运行时默认值。

启动模型 Server 和 Gateway
~~~~~~~~~~~~~~~~~~~~~~~~~~

选择空闲 GPU 和未占用端口。在第一个服务终端中激活 PhyAI 环境，启动模型 Server：

.. code-block:: bash

   cd /path/to/RLinf
   source .venv-phyai-serving/bin/activate
   export GPU_ID=0
   CUDA_VISIBLE_DEVICES="$GPU_ID" python -m phyai server pi05-http.yaml

等待 ``Serving model pi05 on 127.0.0.1:50063``，这条日志出现在模型初始化和 warmup 完成之后。保持该进程运行，在第二个服务终端激活同一环境并启动 Gateway：

.. code-block:: bash

   cd /path/to/RLinf
   source .venv-phyai-serving/bin/activate
   python -m phyai_gateway.server \
     --gateway 127.0.0.1:50111 \
     --http-host 127.0.0.1 \
     --http-port 30000 \
     --backend pi05 127.0.0.1:50063
   # In another shell:
   curl --noproxy '*' http://127.0.0.1:30000/v1/backends

Gateway 主动调用已登记 Server 的 ``CheckHealth``。等待 ``pi05``、``127.0.0.1:50063`` 对应记录显示 ``healthy: true``。Server YAML 的 ``server.model_name`` 和 Gateway 的后端名都应为 ``pi05``，当前 HTTP worker 每次请求会发送这个名字。登记只在 Gateway 进程内有效，重启时再次提供 ``--backend``。

这里的回环地址要求服务与 RLinf worker 共享网络命名空间。若运行在不同机器或容器中，将服务绑定到可达接口，在 ``--backend`` 中填写模型主机地址，在 ``rollout.phyai.gateway_url`` 中填写 Gateway 主机地址。这些 HTTP/gRPC 接口面向可信调用方，应由部署环境实施网络访问控制。

运行 HTTP 评估
~~~~~~~~~~~~~~

两项服务就绪后，在第三个终端使用 RLinf 客户端环境。为 LIBERO 渲染选择空闲 GPU，并让 placement 使用相同编号。模型 Server 是独立进程，可以使用另一张 GPU。

.. code-block:: bash

   cd /path/to/RLinf
   source .venv-phyai-http/bin/activate
   export GPU_ID=0
   export CUDA_VISIBLE_DEVICES="$GPU_ID"
   bash evaluations/run_eval.sh libero libero_spatial_pi05_eval_phyai \
     "cluster.component_placement.env=$GPU_ID" \
     "cluster.component_placement.rollout=$GPU_ID" \
     rollout.phyai.gateway_url=http://127.0.0.1:30000 \
     env.eval.total_num_envs=8 \
     env.eval.rollout_epoch=1 \
     env.eval.max_episode_steps=480 \
     env.eval.max_steps_per_rollout_epoch=480 \
     env.eval.video_cfg.save_video=false

``run_eval.sh`` 提供标准 LIBERO/EGL 默认环境变量。HTTP YAML 选择 ``worker_http`` 并请求 16 步动作块。这里把原配置的较大评估规模覆盖为 8 个环境、1 轮 rollout，保留 360x360 相机输入。权重和 tokenizer 路径由模型 Server YAML 管理，客户端命令只选择 Gateway 地址。

在最终评估指标中检查 ``success_once``、``success_at_end`` 和 ``num_trajectories``。此前这条 HTTP 链路在本地 A40 完成过 8 条轨迹，``success_once=0.875``、``success_at_end=0.75``。这是历史运行记录，不是本次修改后新的干净环境验证或 A800 CI 结果。

评估结束后，在对应终端中用 Ctrl+C 停止 Gateway 和模型 Server。更多后端管理和多机部署说明见 `PhyAI 服务指南 <https://phyai.mintlify.app/zh/deployment/server-gateway>`_。

运行 HTTP Smoke Test
~~~~~~~~~~~~~~~~~~~~

使用仓库中的 Shell 脚本启动并清理整条链路，复用相同的评估 YAML。先按上文准备客户端和服务环境，再传入兼容模型资源与空闲 GPU 编号：

.. code-block:: bash

   cd /path/to/RLinf
   export PHYAI_PYTHON="$PWD/.venv-phyai-serving/bin/python"
   export RLINF_VENV="$PWD/.venv-phyai-http"
   export PHYAI_HTTP_CHECKPOINT=/path/to/lerobot-pi05-libero
   export PHYAI_HTTP_TOKENIZER=/path/to/paligemma-tokenizer
   SERVER_GPU_ID=0 GPU_ID=1 NUM_ENVS=8 MIN_SUCCESS_ONCE=0.5 \
     bash tests/e2e_tests/evaluations/run_phyai_gateway.sh

脚本默认读取 ``tests/e2e_tests/evaluations/phyai_gateway_server.yaml``，把两个资源变量写入本次运行的配置副本，检查端口是否空闲，等待模型和 Gateway 就绪，再运行 8 环境、480 步且关闭视频的评估。显存允许时，模型和渲染可以使用同一张 GPU。若另一份 Server YAML 的 batch、相机及动作契约与客户端匹配，可以将它作为第一个参数传入。

脚本创建独立 Ray 实例，只让本次评估连接它。退出时清理本次启动的进程组，不停止已有服务或其他 Ray 集群。日志、解析后的 Server YAML 和成功时生成的 ``result.json`` 保留在 ``logs/ci/phyai-gateway.<suffix>/``。

``MIN_SUCCESS_ONCE`` 默认是零，用于检查整条链路能否完成。上面的命令另加了至少 50% 成功率的要求；应与 CI 维护者根据固定 checkpoint 和 reset 状态确定质量门槛。本次修改未执行更新后的脚本或新增回归测试，之前的 A40 结果不代表这些修改已通过运行验证。

启用 HTTP CI 任务
~~~~~~~~~~~~~~~~~

``.github/workflows/embodied-e2e-tests.yml`` 中的 ``embodied-libero-phyai-http-test`` 复用进程内任务的 ``embodied`` runner、``--engine phyai`` 安装入口和“组件测试后运行 e2e”的结构。另一个环境通过现有 SGLang 安装器准备官方 HTTP Client 的 codec。服务端只在共享 PhyAI runtime 配方上补 Gateway、processor 和 HTTP 依赖。模型推理使用 GPU 0，渲染使用 GPU 1，评估规模为 8 个环境、480 步并关闭视频。

启用前，在所有可能接收该任务的 runner 上准备兼容的 LeRobot checkpoint 和完整 PaliGemma tokenizer，并将实际绝对路径配置为下面两个 GitHub 仓库变量：

.. list-table:: HTTP CI 资源
   :header-rows: 1

   * - 仓库变量
     - 要求
   * - ``PHYAI_HTTP_CHECKPOINT``
     - 包含兼容 PI0.5 权重和 processor 文件的目录。
   * - ``PHYAI_HTTP_TOKENIZER``
     - 包含完整 tokenizer 快照的目录。

任一变量未设置时，任务会跳过；跳过不代表 HTTP 链路已有 CI 覆盖。配置齐全后，任务检查资源目录，执行 ``test_models.py -k phyai_http``，再调用 Shell 脚本运行上文相同的评估 YAML。两个 Python 环境使用任务独占的临时目录，结束后删除。Shell 脚本管理并关闭本次服务进程和独立 Ray 实例，不停止其他 Ray 集群。评估日志在任务失败时也会上传。

CI 默认成功率门槛为零，使用完成轨迹、合法指标和进程零退出码检查整条链路。确定固定 checkpoint 和稳定的质量门槛后，再将 job 中的 ``MIN_SUCCESS_ONCE`` 设为正数。服务端补充安装和这项新 workflow 尚未在官方 A800 runner 上执行，仍需维护者按仓库正常策略批准并启用 PR 的 CI 运行。
