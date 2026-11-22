Qwen2.5-3B-Instruct 的 Search-R1 训练
========================================

训练 Qwen2.5-3B-Instruct 通过调用搜索工具回答问题。请先按 :doc:`index` 搭建 RLinf 环境和本地 wiki 检索服务。

概述
----------------------------------------

使用本配方结合本地 wiki 检索服务训练搜索增强推理模型。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Qwen2.5-3B-Instruct

   .. grid-item-card:: 算法
      :text-align: center

      GRPO 或 PPO

   .. grid-item-card:: 工具
      :text-align: center

      FAISS 或 Qdrant 本地 wiki server

   .. grid-item-card:: 硬件
      :text-align: center

      参考运行使用 8×H100

一键上手
----------------------------------------

中国大陆用户在 ``install.sh`` 末尾追加 ``--use-mirror``，并在 ``hf download`` 前执行 ``export HF_ENDPOINT=https://hf-mirror.com``。预构建镜像见 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__。启动训练前，请先按 :doc:`index` 启动本地 wiki 检索服务。

.. tab-set::

   .. tab-item:: Megatron GRPO

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      下载基础模型和 `训练集 <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__：

      .. code-block:: bash

         hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./Qwen2.5-3B-Instruct
         hf download --repo-type dataset RLinf/Search-R1-Data --local-dir ./Search-R1-Data

      **3. 启动训练**

      在 ``examples/agent/searchr1/config/train_qwen2.5.yaml`` 中填写 ``data.train_data_paths`` 和 ``rollout.model.model_path``\ （使用 ``sampling_params.stop`` 时设置 ``rollout.detokenize: True``，并保持 ``rollout.return_logprobs`` 等于 ``not algorithm.recompute_logprobs``）：

      .. code-block:: yaml

         data:
           train_data_paths: ["/path/to/train.jsonl"]

         rollout:
           model:
             model_path: /path/to/model/Qwen2.5-3B-Instruct

      启动训练：

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_qwen2.5

   .. tab-item:: Megatron PPO

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      .. code-block:: bash

         hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./Qwen2.5-3B-Instruct
         hf download --repo-type dataset RLinf/Search-R1-Data --local-dir ./Search-R1-Data

      **3. 启动训练**

      在 ``examples/agent/searchr1/config/train_qwen2.5_ppo.yaml`` 中填写 ``data.train_data_paths`` 和 ``rollout.model.model_path``\ （PPO 直接使用 rollout logprobs，保持 ``rollout.return_logprobs`` 等于 ``not algorithm.recompute_logprobs``）：

      .. code-block:: yaml

         data:
           train_data_paths: ["/path/to/train.jsonl"]

         rollout:
           model:
             model_path: /path/to/model/Qwen2.5-3B-Instruct

      启动训练：

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_qwen2.5_ppo

   .. tab-item:: MA-FSDP GRPO

      **1. 安装环境**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh agentic && source .venv/bin/activate

      **2. 下载模型与数据**

      .. code-block:: bash

         hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./Qwen2.5-3B-Instruct
         hf download --repo-type dataset RLinf/Search-R1-Data --local-dir ./Search-R1-Data

      **3. 启动训练**

      在 ``examples/agent/searchr1/config/train_qwen2.5_fsdp.yaml`` 中填写相同的模型与数据路径（设置 ``agentloop.is_dynamic_rollout_batch: True`` 和 ``actor.training_backend: fsdp`` 后，入口使用 ``MAFSDPActor`` 并保持 ``actor.pack_traj: True`` 打包同轨迹多轮样本，避免重复计算共享前缀）：

      .. code-block:: bash

         bash examples/agent/searchr1/run_train.sh train_qwen2.5_fsdp

Checkpoint 导出与评测
----------------------------------------

checkpoint 的导出方式取决于训练后端。

对于 Megatron，运行以下命令将 checkpoint 转换为 HuggingFace model：

.. code-block:: bash

   CKPT_PATH_MG={your_output_dir}/{exp_name}/checkpoints/global_step_xxx/actor
   CKPT_PATH_HF={path/to/save/huggingface/model}
   CKPT_PATH_ORIGINAL_HF={path/to/model/Qwen2.5-3B-Instruct}
   CKPT_PATH_MF="${CKPT_PATH_HF}_middle_file"

   python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_mg_to_middle_file \
       --load-path "${CKPT_PATH_MG}" \
       --save-path "${CKPT_PATH_MF}" \
       --model qwen_2.5_3b \
       --tp-size 1 --ep-size 1 --pp-size 1 \
       --te-ln-linear-qkv true --te-ln-linear-mlp_fc1 true \
       --te-extra-state-check-none true --use-gpu-num 0 --process-num 16

   python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_middle_file_to_hf \
       --load-path "${CKPT_PATH_MF}" \
       --save-path "${CKPT_PATH_HF}" \
       --model qwen_2.5_3b \
       --use-gpu-num 0 --process-num 16

   rm -rf "${CKPT_PATH_MF}"
   rm -f "${CKPT_PATH_HF}"/*.done
   shopt -s extglob
   cp "${CKPT_PATH_ORIGINAL_HF}"/!(*model.safetensors.index).json "${CKPT_PATH_HF}"

PPO 训练会在每个 ``global_step_xxx`` 下保存 ``actor`` 和 ``critic`` 两个目录。导出模型时，用上面的命令转换 ``actor`` 目录。恢复训练时，将 ``runner.resume_dir`` 设为该 ``global_step_xxx`` 目录，RLinf 会同时恢复 actor 和 critic。

对于 FSDP，请使用内置的 FSDP 转换器，而不是上面的 Megatron 转换命令。将保存的 FSDP checkpoint 与对应训练运行生成的解析后配置文件传入转换器，并指定 HuggingFace 模型输出目录：

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
       convertor.train_config_path=/path/to/config.yaml \
       convertor.ckpt_path=/path/to/checkpoints/global_step_xxx/actor/model_state_dict/full_weights.pt \
       convertor.save_path=/path/to/hf_model

``config.yaml`` 会随训练运行写出；如果它位于 ``convertor.ckpt_path`` 附近，转换器也可以自动查找。将导出的 HuggingFace 模型用于独立评测。

将转换得到的 HuggingFace model 路径填入 ``examples/agent/searchr1/config/eval_qwen2.5.yaml``：

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.8
     model:
       model_path: /path/to/eval/model
       model_type: qwen2.5

修改测试数据集路径：

.. code-block:: yaml

   data:
     ……
     val_data_paths: ["/path/to/eval.jsonl"]

运行 ``bash examples/agent/searchr1/run_eval.sh`` 启动测试。

可视化与结果
----------------------------------------

下面展示 reward 曲线和训练时间曲线。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1.png" style="width: 100%;"/>
       <p><em>Qwen2.5-3B-Instruct in RLinf</em></p>
     </div>
   </div>

相较于原版性能（response length 稳定后，单 step 133 s），我们加速了 55%，同时 reward 曲线和 eval 结果保持一致。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/searchr1_orig_impl_time.png" style="width: 35%;"/>
       <p><em>Qwen2.5-3B-Instruct in original implementation at PeterGriffinJin/Search-R1</em></p>
     </div>
   </div>

下图比较 GRPO 和 PPO 的训练 reward：

.. raw:: html

   <div style="text-align: center;">
     <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/search_r1/qwen_searchr1_ppo_grpo_reward.png" alt="Qwen2.5-3B Search-R1 reward：橙色为 GRPO，蓝色为 PPO" style="width: 100%; max-width: 410px;"/>
     <p><em>橙色为 GRPO，蓝色为 PPO。两组实验均使用 8×A800 80 GB，有效 batch size 均为 1024，TensorBoard 平滑系数为 0.6。</em></p>
   </div>
