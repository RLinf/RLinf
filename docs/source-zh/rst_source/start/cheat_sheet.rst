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

单机运行会自动启动本地 Ray，快速上手配置首次运行时会自动缓存下载默认 Hugging Face 权重：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvla_quickstart

如需替换为本地权重或其他 Hugging Face ``owner/repo``，传入 ``actor.model.model_path=...`` 即可（rollout 自动继承）。快速上手配置默认使用全部加速卡；需要指定设备时修改 ``cluster.component_placement``，详见 :doc:`../reference/configuration`。多机训练前，先在每个节点设置 ``RLINF_NODE_RANK``，再执行 ``ray start``，详见 :doc:`../guides/multi_node`。

运行评测
--------

运行具身独立评测：

.. code-block:: bash

   bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
     rollout.model.model_path=/path/to/checkpoint

下一步
------

- :doc:`安装说明 <installation>`：安装目标、常用参数与 Docker Hub 入口。
- :doc:`快速上手 <vla>`：ManiSkill3 四步入门示例。
- :doc:`启动与扩展 <../guides/launch-scale/index>`：多机与真机部署。
- :doc:`评测 <../evaluations/index>`：支持的 benchmark 与评测配置。
