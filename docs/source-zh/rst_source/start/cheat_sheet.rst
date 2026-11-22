命令速查表
==========

RLinf 常用工作流的最短命令。

安装
----

每个虚拟环境只安装一套模型与环境组合；需要共存多套环境时，通过 ``--venv <dir>`` 指定独立目录。中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。

.. tab-set::

   .. tab-item:: 具身训练

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env maniskill_libero && source .venv/bin/activate

   .. tab-item:: 具身评测

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openpi --env libero --venv .venv-eval && source .venv-eval/bin/activate

   .. tab-item:: 智能体 / 推理

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic --engine sglang --venv .venv-agentic && source .venv-agentic/bin/activate

   .. tab-item:: 文档构建

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh docs --venv .docs-venv

启动训练
--------

单机运行会自动启动本地 Ray。配置名对应仓库中的 YAML 文件：具身强化学习位于 ``examples/embodiment/config/``，SFT 位于 ``examples/sft/config/``，推理与智能体位于 ``examples/reasoning/config/`` 与 ``examples/agent/``，评测位于 ``evaluations/<benchmark>/``。

.. tab-set::

   .. tab-item:: MLP（单卡 · 零权重下载）

      .. code-block:: bash

         source .venv/bin/activate
         bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp \
           runner.max_epochs=3 runner.val_check_interval=-1

   .. tab-item:: VLA PPO（OpenVLA）

      .. code-block:: bash

         source .venv/bin/activate
         hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets
         bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart

   .. tab-item:: 智能体 / 推理（单卡）

      在 ``examples/reasoning/config/math/qwen2.5-1.5b-single-gpu.yaml`` 中填写 ``rollout.model.model_path``、``actor.model.model_path``、``actor.tokenizer.tokenizer_model`` 与 ``data.train_data_paths`` 后启动：

      .. code-block:: bash

         source .venv-agentic/bin/activate
         bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-single-gpu

具身训练如需替换为本地权重或其他 Hugging Face ``owner/repo``，传入 ``actor.model.model_path=...`` 即可（rollout 自动继承）。多机训练前，先在每个节点导出 ``RLINF_NODE_RANK`` 再执行 ``ray start``\ （详见 :doc:`../guides/multi_node`）。

运行评测
--------

运行具身独立评测（配置文件位于 ``evaluations/<benchmark>/<config_name>.yaml``）：

.. code-block:: bash

   source .venv-eval/bin/activate

   # 评测 Hugging Face 或本地基座模型
   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=RLinf/RLinf-Pi05-LIBERO-SFT

   # 在基座模型上加载 RL / SFT 产出的 .pt 权重
   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=RLinf/RLinf-Pi05-LIBERO-SFT \
     runner.ckpt_path=./logs/<run>/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt

常用操作与参数覆盖
------------------

.. code-block:: bash

   # 在 TensorBoard 中查看训练与评测曲线
   tensorboard --host 0.0.0.0 --logdir ./logs

   # 从保存的 checkpoint 目录断点续训
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart \
     runner.resume_dir=./logs/<run>/checkpoints/global_step_50

   # 将多卡配置限制到 0 号 GPU 运行
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart \
     cluster.component_placement.actor,rollout,env=0

   # 清理残留的本地 Ray 会话（共享机器上先用 ray status 确认无他人任务）
   ray stop

下一步
------

- :doc:`安装说明 <installation>`：安装目标、常用参数与 Docker Hub 入口。
- :doc:`训练快速开始 <vla>`：单卡 MLP 与 ManiSkill3 VLA 入门示例。
- :doc:`启动与扩展 <../guides/launch-scale/index>`：多机与真机部署。
- :doc:`评测 <../evaluations/index>`：支持的 benchmark 与评测配置。
