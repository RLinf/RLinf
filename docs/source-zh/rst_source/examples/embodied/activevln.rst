使用 EmbodiInfer 训练 ActiveVLN
===============================

本例用 Hugging Face actor、EmbodiInfer rollout 和 VLN-CE 环境服务运行两步 ActiveVLN GRPO 冒烟测试。配置重复使用 R2R episode 1，batch size 为 2，权重采用 BF16。

概览
----

.. list-table::
   :header-rows: 1

   * - 模型
     - 环境
     - 算法
     - 硬件布局
   * - ActiveVLN（Qwen2.5-VL-3B）
     - VLN-CE / Habitat
     - GRPO
     - GPU 0：训练与推理；GPU 1：渲染

任务
~~~~

预设在 Matterport3D 场景中执行 R2R 导航指令。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 观测
     - 动作
     - 奖励
   * - RGB 图像、导航指令和 episode 历史
     - 文本导航指令
     - 导航成功奖励

安装
----

.. include:: _setup_common.rst
   :start-after: 首先，克隆 RLinf 仓库：
   :end-before: 然后，使用下列两种方式

安装训练环境和 VLN-CE 客户端：

.. code-block:: bash

   bash requirements/install.sh embodied --model activevln --env vlnce_server --venv .venv-activevln
   source .venv-activevln/bin/activate

安装脚本将 ActiveVLN 源码下载到 ``.venv-activevln/ActiveVLN``\ 。Habitat 需要单独配置 Python 环境。

准备数据并启动服务
~~~~~~~~~~~~~~~~~~

在 RLinf 仓库目录设置源码路径：

.. code-block:: bash

   export ACTIVEVLN_ROOT="$PWD/.venv-activevln/ActiveVLN"

按固定版本的 `ActiveVLN 服务端安装说明 <https://github.com/arvillion/ActiveVLN/blob/3a0c63b00e4f42c828cc74c3554afce17641da60/README.md#setup-the-vln-ce-environemnt>`_ 创建 ``server`` 环境，安装 Python 3.10、Habitat-Sim v0.1.7 和 Habitat-Lab v0.1.7。

按其 `数据准备说明 <https://github.com/arvillion/ActiveVLN/blob/3a0c63b00e4f42c828cc74c3554afce17641da60/README.md#data-preparation>`_ 将 Matterport3D 场景放到 ``$ACTIVEVLN_ROOT/data/scene_datasets/mp3d``\ ，将 R2R/RxR 标注放到 ``$ACTIVEVLN_ROOT/data/datasets``\ 。

准备 R2R 和 RxR 的训练标注及 NDTW 真实路径文件。即使 RxR worker 数量为零，服务端也会加载 RxR 数据。

补充安装服务端依赖，只需执行一次：

.. code-block:: bash

   conda activate server
   python -m pip install ray flask hydra-core av

在物理 GPU 1 上启动两个 R2R 模拟器：

.. code-block:: bash

   cd "$ACTIVEVLN_ROOT"
   export PYTHONPATH="$ACTIVEVLN_ROOT:$ACTIVEVLN_ROOT/vlnce_server${PYTHONPATH:+:$PYTHONPATH}"
   python -m vlnce_server.server --config-path "$ACTIVEVLN_ROOT/vlnce_server/config" \
     server.host=127.0.0.1 server.port=5001 \
     'vlnce.gpus=[1]' 'vlnce.r2r_gpu_plan=[2]' 'vlnce.rxr_gpu_plan=[0]'

保持服务运行。在另一个终端确认以下请求返回 ``200`` 后，再启动训练：

.. code-block:: bash

   curl -sS -X OPTIONS http://127.0.0.1:5001/batch/reset -o /dev/null -w '%{http_code}\n'

``vlnce.r2r_gpu_plan`` 控制模拟器数量。增大训练或评估环境数量时，需要相应提高服务端容量。

下载模型
~~~~~~~~

另开终端，回到 RLinf 仓库目录，下载模型权重：

.. code-block:: bash

   source .venv-activevln/bin/activate
   hf download Arvil/Qwen2.5-VL-3B_rl_r2r_4000 \
     --revision 160987313e3e869705f42400d1b8f28177044518 \
     --local-dir "$HOME/models/activevln-r2r"

运行
----

将以下 YAML 字段设为下载模型的绝对路径：

.. list-table::
   :header-rows: 1

   * - 文件
     - 字段
   * - ``examples/embodiment/config/model/activevln.yaml``
     - ``model_path``
   * - ``examples/embodiment/config/env/vlnce_server_r2r.yaml``
     - ``tokenizer_path``

从 RLinf 仓库目录启动训练。\ ``RAY_ADDRESS=local`` 让 RLinf 使用独立的 Ray 集群，避免连接到服务端集群。

.. code-block:: bash

   RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=0 \
     bash examples/embodiment/run_embodiment.sh activevln_r2r_grpo_embodiinfer

通过 ``rollout.rollout_backend`` 选择 ``embodiinfer`` 或 ``huggingface`` 推理，在 ``rollout.embodiinfer`` 下配置 EmbodiInfer 选项。

扩展训练时，在预设中修改 ``env.train.episode_ids``\ 、``runner.max_steps`` 和 ``runner.max_epochs``\ 。服务部署在远端时，设置 ``env.train.server_url`` 和 ``env.eval.server_url``\ 。

结果与可视化
------------

训练日志和 TensorBoard 事件保存在 ``logs/<timestamp>-activevln_r2r_grpo_embodiinfer`` 下。启动 TensorBoard 查看训练指标：

.. code-block:: bash

   tensorboard --logdir logs

训练指标见 :doc:`/rst_source/reference/metrics`，评估流程见 :doc:`/rst_source/evaluations/get_started/index`。
