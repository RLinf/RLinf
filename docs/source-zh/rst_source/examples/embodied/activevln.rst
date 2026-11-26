使用 EmbodiInfer 训练 ActiveVLN
===============================

本例用 FSDP actor、EmbodiInfer rollout 和 VLN-CE 环境服务运行两步 ActiveVLN GRPO 快速验证。默认配置在两个环境中重复运行 R2R episode 1，每个 GRPO group 包含两个环境，全局训练 batch size 为 16，权重采用 BF16。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      VLN-CE / Habitat

   .. grid-item-card:: 算法
      :text-align: center

      GRPO

   .. grid-item-card:: 任务
      :text-align: center

      R2R 导航

   .. grid-item-card:: 硬件
      :text-align: center

      GPU 0：训练与推理；GPU 1：渲染

**前置条件：**\ :doc:`安装指南 </rst_source/start/installation>`\ 和下文的 Habitat 环境配置。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 目标
   * - VLN-CE / Habitat
     - R2R
     - ``activevln_r2r_grpo_embodiinfer``
     - 在 Matterport3D 场景中按指令导航

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 说明
   * - 观测
     - RGB 图像与 episode 历史
   * - 动作
     - 文本导航动作
   * - 奖励
     - 导航成功奖励
   * - Prompt
     - 自然语言导航指令

安装与运行
----------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: NVIDIA CUDA
      :sync: cuda

      **1. 安装**

      安装训练环境和 VLN-CE 客户端：

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model activevln --env vlnce_server --venv .venv-activevln
         source .venv-activevln/bin/activate

      安装脚本将 ActiveVLN 源码下载到 ``.venv-activevln/ActiveVLN``\ 。Habitat 需要单独配置 Python 环境。

      **2. 准备数据并启动服务**

      在 RLinf 仓库目录设置源码路径：

      .. code-block:: bash

         export ACTIVEVLN_ROOT="$PWD/.venv-activevln/ActiveVLN"

      按以下指定版本的\ `ActiveVLN 服务端安装说明 <https://github.com/arvillion/ActiveVLN/blob/3a0c63b00e4f42c828cc74c3554afce17641da60/README.md#setup-the-vln-ce-environemnt>`_\ 创建 ``server`` 环境，安装 Python 3.10、Habitat-Sim v0.1.7 和 Habitat-Lab v0.1.7。

      按\ `数据准备说明 <https://github.com/arvillion/ActiveVLN/blob/3a0c63b00e4f42c828cc74c3554afce17641da60/README.md#data-preparation>`_\ 将 Matterport3D 场景放到 ``$ACTIVEVLN_ROOT/data/scene_datasets/mp3d``\ ，将 R2R/RxR 标注放到 ``$ACTIVEVLN_ROOT/data/datasets``\ 。

      准备 R2R 和 RxR 的训练标注及 NDTW 真实路径文件。即使 RxR worker 数量为零，服务端也会加载 RxR 数据。

      补充安装服务端依赖，只需执行一次：

      .. code-block:: bash

         deactivate
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

      ``vlnce.r2r_gpu_plan`` 控制模拟器数量。增大训练或评测环境数量时，需要相应提高服务端容量。

      **3. 下载模型**

      在刚才检查服务的终端中，回到 RLinf 仓库目录，下载模型权重：

      .. code-block:: bash

         source .venv-activevln/bin/activate
         hf download Arvil/Qwen2.5-VL-3B_rl_r2r_4000 \
           --revision 160987313e3e869705f42400d1b8f28177044518 \
           --local-dir ./models/activevln-r2r

      **4. 启动训练**

      在 RLinf 仓库目录执行以下命令，使用下载的模型启动训练。默认使用 GPU 0 训练和推理，GPU 1 渲染。\ ``RAY_ADDRESS=local`` 为 RLinf 启动独立的 Ray 集群。

      .. code-block:: bash

         RAY_ADDRESS=local \
           bash examples/embodiment/run_embodiment.sh activevln_r2r_grpo_embodiinfer \
           actor.model.model_path="$PWD/models/activevln-r2r" \
           env.train.tokenizer_path="$PWD/models/activevln-r2r" \
           env.eval.tokenizer_path="$PWD/models/activevln-r2r"

      .. dropdown:: 高级配置（可选）

         **延长训练并保存 checkpoint**

         在 ``examples/embodiment/config/activevln_r2r_grpo_embodiinfer.yaml`` 中修改以下字段，再执行上面的启动命令，即可训练 1,000 步，每 100 步保存一次 checkpoint。训练和评测会轮换运行各自列表中的 episode。从服务端已加载的标注中选择 episode ID。

         .. code-block:: yaml

            runner:
              max_epochs: 1000
              max_steps: 1000
              save_interval: 100
              val_check_interval: 100
            env:
              train:
                episode_ids: [1, 2, 3, 4]
                max_episode_steps: 32
                max_steps_per_rollout_epoch: 32
              eval:
                episode_ids: [5, 6]
                max_episode_steps: 32
                max_steps_per_rollout_epoch: 32

         每个 episode 最多运行 32 轮模型交互，每轮最多执行三个导航动作。训练和评测轮流使用服务端的两个模拟器。checkpoint 保存在 ``logs/<timestamp>-activevln_r2r_grpo_embodiinfer/activevln_r2r_grpo_embodiinfer/checkpoints/global_step_<step>/actor`` 下。

         **独立评测**

         保持服务运行，在 RLinf 仓库目录评测下载的模型：

         .. code-block:: bash

            export EMBODIED_PATH="$PWD/examples/embodiment"
            RAY_ADDRESS=local python -m evaluations.eval_embodied_agent \
              --config-path "$EMBODIED_PATH/config" \
              --config-name activevln_r2r_grpo_embodiinfer \
              +model@rollout.model=activevln \
              rollout.model.model_path="$PWD/models/activevln-r2r" \
              env.eval.tokenizer_path="$PWD/models/activevln-r2r"

         **推理后端与服务地址**

         设置 ``rollout.rollout_backend=huggingface`` 可使用 Hugging Face rollout。EmbodiInfer 选项放在 ``rollout.embodiinfer`` 下。服务部署在远端时，设置 ``env.train.server_url`` 和 ``env.eval.server_url``\ 。

可视化与结果
------------

两步验证完成后，日志会显示训练到达第 2 步，并记录 ``env/success_once``\ 、``actor/policy_loss`` 和 ``actor/grad_norm``\ 。

在 ``logs/<timestamp>-activevln_r2r_grpo_embodiinfer`` 下查看 ``run_embodiment.log`` 中的控制台输出，或查看 ``tensorboard/`` 中的训练曲线。启动 TensorBoard 查看指标：

.. code-block:: bash

   tensorboard --logdir logs

指标含义见 :doc:`/rst_source/reference/metrics`\ 。
