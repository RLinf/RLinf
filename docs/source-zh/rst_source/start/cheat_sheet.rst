命令速查表
==========

RLinf 常用工作流的最短命令。

安装
----

使用 ``install.sh`` 本地安装（中国大陆用户在末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``），或前往 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__ 拉取预构建镜像：

.. tab-set::

   .. tab-item:: 具身智能（Embodied）

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model openvla --env maniskill_libero && source .venv/bin/activate

   .. tab-item:: 智能体 / 推理（Agentic）

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic --engine sglang && source .venv/bin/activate

   .. tab-item:: 文档构建（Docs）

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh docs --venv .docs-venv

启动训练
--------

单机运行会自动启动本地 Ray。可以先跑单卡零下载的 MLP 快速验证，或直接启动 VLA 快速开始（下载一次 ManiSkill 场景资产后首次运行自动缓存默认 Hugging Face 权重）：

.. code-block:: bash

   # 单卡零下载 MLP 验证（环境安装：bash requirements/install.sh embodied --env maniskill_libero）
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_mlp

   # 7B VLA PPO 快速开始
   hf download --repo-type dataset RLinf/maniskill_assets --local-dir rlinf/envs/sim/maniskill/assets
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart

如需替换为本地权重或其他 Hugging Face ``owner/repo``，传入 ``actor.model.model_path=...`` 即可（rollout 自动继承）。快速开始配置默认使用全部加速卡；需要指定设备时修改 ``cluster.component_placement``，详见 :doc:`../reference/configuration`。多机训练前，先在每个节点设置 ``RLINF_NODE_RANK``，再执行 ``ray start``，详见 :doc:`../guides/multi_node`。

运行评测
--------

运行具身独立评测（需使用 ``openpi`` + ``libero`` 环境；如需与训练环境隔离，安装时用 ``--venv`` 指定独立目录）：

.. code-block:: bash

   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=RLinf/RLinf-Pi05-LIBERO-SFT

如已提前下载到本地，可将 ``RLinf/RLinf-Pi05-LIBERO-SFT`` 替换为本地权重目录。

下一步
------

- :doc:`安装说明 <installation>`：安装目标、常用参数与 Docker Hub 入口。
- :doc:`训练快速开始 <vla>`：单卡 MLP 与 ManiSkill3 VLA 入门示例。
- :doc:`启动与扩展 <../guides/launch-scale/index>`：多机与真机部署。
- :doc:`评测 <../evaluations/index>`：支持的 benchmark 与评测配置。
