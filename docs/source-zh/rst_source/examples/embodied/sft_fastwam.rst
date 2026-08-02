FastWAM 评测与监督微调
========================

.. figure:: https://yuantianyuan01.github.io/FastWAM/static/images/teaser_main.png
   :align: center
   :width: 90%

   Fast-WAM 保留视频协同训练，但评测时无需执行未来视频去噪即可生成动作。

在 LIBERO 或 LIBERO-Plus 上运行发布的
`FastWAM <https://github.com/yuantianyuan01/FastWAM>`__ 模型，并通过 RLinf 的
FSDP SFT 流水线监督微调其世界/动作专家。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO · LIBERO-Plus

   .. grid-item-card:: 算法
      :text-align: center

      评测 · SFT

   .. grid-item-card:: 任务
      :text-align: center

      LIBERO Spatial

   .. grid-item-card:: 硬件
      :text-align: center

      CUDA GPU · 多 GPU SFT

| **你将完成：** 安装 → 下载检查点与统计信息 → 评测 → 准备 LeRobot 数据和文本 embedding → 启动 SFT。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · CUDA GPU · 能从 Hugging Face 获取 Wan2.2 组件。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 24 30 24

   * - 环境
     - 任务 / 套件
     - 配置 / 权重
     - 重点
   * - LIBERO
     - LIBERO-Spatial
     - ``libero_spatial_fastwam_eval``
     - 批量、仅动作的评测。
   * - LIBERO-Plus
     - Spatial 扰动
     - 同一配置加 ``LIBERO_TYPE=plus``
     - 评测全部或单一扰动类型。
   * - 离线数据
     - LIBERO LeRobot
     - ``libero_sft_fastwam``
     - 对 MoT 专家执行全参数 FSDP SFT。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 字段
     - 说明
   * - 观测
     - 主视角和腕部 RGB 图像，以及 8 维 LIBERO 机器人状态。
   * - 提示词
     - 由 FastWAM 文本编码器处理的 LIBERO 自然语言指令。
   * - 动作
     - 长度为 32、每步 7 维的动作预测；RLinf 每次执行
       ``num_action_chunks`` 步后重新规划。
   * - 训练目标
     - 来自 FastWAM ``training_loss`` 的视频流匹配和动作流匹配 loss。

安装
----

.. include:: _setup_common.rst

评测时，一起安装 FastWAM 和 LIBERO：

.. code-block:: bash

   bash requirements/install.sh embodied --model fastwam --env libero
   source .venv/bin/activate

只做离线 SFT、不需仿真器时：

.. code-block:: bash

   bash requirements/install.sh embodied --model fastwam
   source .venv/bin/activate

安装器会 clone 固定版本的 FastWAM，安装其非 Torch 依赖，并通过 RLinf
平台感知的 Torch 覆盖机制默认选择 Torch 2.7.1（TorchCodec 0.5 所需）。
显式传入的 ``--torch`` 仍然优先。要复用现有 checkout，请在安装前设置
``FASTWAM_PATH=/path/to/FastWAM``。

LIBERO-Plus 请使用 ``--env liberoplus``。还需按
:ref:`zh-liberopro-plus-benchmark` 安装附加 assets。

下载模型
--------

如果要评测发布策略或从发布策略继续微调，可以下载 checkpoint
及匹配的归一化统计信息：

.. code-block:: bash

   huggingface-cli download yuanty/fastwam \
     libero_uncond_2cam224.pt \
     libero_uncond_2cam224_dataset_stats.json \
     --local-dir /workspace/checkpoints/fastwam

对于下面介绍的官方 base-model SFT，保持 ``model_path: null``。
SFT 配置会加载官方 Wan2.2 video DiT，并使用官方插值后的 ActionDiT
backbone payload 初始化动作分支。归一化统计信息仍通过
``dataset_stats_path`` 指定：

.. code-block:: yaml

   model_type: fastwam
   model_path: null
   dataset_stats_path: /workspace/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json

FastWAM 与 RLinf 配置
------------------------

RLinf 通过 OmegaConf 组合 FastWAM 上游 YAML，不会修改 Hydra 全局状态。
两层配置的职责分工如下：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 层级
     - 职责
   * - ``model.fastwam.config_name``
     - 选择上游架构、processor、数据形状、scheduler 和训练 loss 默认值。
       RLinf 默认使用 ``sim_libero``。
   * - ``model.fastwam.overrides``
     - 应用兼容上游的 dot-list 覆盖，例如在使用缓存文本 embedding 的 SFT 中设置
       ``model.load_text_encoder=false``。
   * - RLinf 模型字段
     - ``model_path``、``dataset_stats_path``、动作分块、采样参数与可选的未来视频可视化。
       这些值优先于 FastWAM 的评测默认值。
   * - RLinf FSDP 配置
     - 管理混合精度与梯度 checkpoint。SFT 的模型 preset 保持
       ``precision: bf16``；FSDP2 不额外 cast，由 worker 使用 bf16 autocast，
       与上游 Accelerator 路径一致。

.. note::

   FSDP 通用启动日志可能建议将模型参数设为 fp32，以适配常规模型的优化器策略。
   这条建议不适用于 FastWAM：其官方 SFT 路径就是 bf16 模型参数加 bf16 autocast，
   且 FSDP2 不应再额外进行参数 dtype cast。看到该通用提示不表示 FastWAM 配置错误，
   不需要为消除提示而改为 fp32。

FastWAM 检查点只使用 ``model_path``，不支持 ``checkpoint_path`` 别名。

评测
----

完整 SFT checkpoint 的评测使用 evaluations/libero/libero_fastwam_full_eval.yaml。
启动前直接将其中的两个占位路径替换为本机文件路径；它们不是环境变量，也不会自动下载
或解析。请保留已有 checkpoint、dataset stats 与数据集文件。

.. code-block:: yaml

   model_path: your_path_to/FASTWAM_SFT_CKPT
   dataset_stats_path: your_path_to/FASTWAM_DATASET_STATS

单一的 ``libero_spatial_fastwam_eval.yaml`` 取代了原先分开的小规模、
大规模、LIBERO-Plus、仅语言扰动和未来视频 YAML。

**标准 LIBERO smoke 评测：**

.. code-block:: bash

   MUJOCO_GL=egl bash evaluations/run_eval.sh \
     libero libero_spatial_fastwam_eval

**更大规模评测：** 将单个 rollout worker 的并行环境槽位增至 8，运行 80 条轨迹，并关闭录像：

.. code-block:: bash

   MUJOCO_GL=egl bash evaluations/run_eval.sh \
     libero libero_spatial_fastwam_eval \
     env.eval.total_num_envs=8 \
     env.eval.max_steps_per_rollout_epoch=2400 \
     env.eval.video_cfg.save_video=false

**LIBERO-Plus：** 通过环境变量选择全部扰动或单一类型，YAML 保持不变：

.. code-block:: bash

   LIBERO_TYPE=plus LIBERO_SUFFIX=all MUJOCO_GL=egl \
     bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval

   LIBERO_TYPE=plus LIBERO_SUFFIX=language MUJOCO_GL=egl \
     bash evaluations/run_eval.sh libero libero_spatial_fastwam_eval \
     env.eval.total_num_envs=8 env.eval.video_cfg.save_video=false

**未来视频可视化：** 动作生成仍为批量执行；可选的未来想象只针对第一个样本，
并受 ``max_video_saves`` 限制。

.. code-block:: bash

   MUJOCO_GL=egl bash evaluations/run_eval.sh \
     libero libero_spatial_fastwam_eval \
     env.eval.total_num_envs=2 \
     env.eval.video_cfg.save_video=false \
     rollout.model.visualize_future_video=true \
     rollout.model.future_video_dir=/workspace/future_video_demo

监督微调
--------

本节包含准备 SFT 的全部命令，不需要额外的准备脚本。当前配置从官方 Wan2.2
base model 和插值后的 ActionDiT backbone 开始训练，不会从发布的 FastWAM
checkpoint 继续训练。下方代码会下载必需权重与 dataset stats、准备 ActionDiT、
下载并解压四套 LIBERO 数据，并预计算 T5 text embedding。先激活环境，在仓库根目录
将代码逐行粘贴到终端执行；长时间下载和训练建议在 tmux 中进行：

.. code-block:: bash

   #!/usr/bin/env bash
   set -euo pipefail

   SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
   REPO_PATH="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
   PYTHON_BIN="${REPO_PATH}/.venv/bin/python"
   FASTWAM_PATH="${FASTWAM_PATH:-${REPO_PATH}/.venv/FastWAM}"
   CHECKPOINT_DIR="${FASTWAM_CHECKPOINT_DIR:-${REPO_PATH}/checkpoints/fastwam_release}"
   MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_PATH}/checkpoints}"
   DATASET_ROOT="${FASTWAM_DATASET_ROOT:-${REPO_PATH}/data/libero_mujoco3.3.2}"
   if [[ -n "${FASTWAM_DATASET_DIR:-}" ]]; then
       # Keep a one-suite override for smoke/debug runs.
       DATASET_DIRS=("${FASTWAM_DATASET_DIR}")
   else
       # Match the official task/libero_uncond_2cam224_1e-4.yaml.
       DATASET_DIRS=(
           "${DATASET_ROOT}/libero_spatial_no_noops_lerobot"
           "${DATASET_ROOT}/libero_object_no_noops_lerobot"
           "${DATASET_ROOT}/libero_goal_no_noops_lerobot"
           "${DATASET_ROOT}/libero_10_no_noops_lerobot"
       )
   fi
   DATASET_DIR="${DATASET_DIRS[0]}"
   TEXT_CACHE_DIR="${FASTWAM_TEXT_EMBEDDING_CACHE_DIR:-${DATASET_ROOT}/text_embeds_cache/libero}"
   DATASET_DIRS_HYDRA=$(IFS=,; echo "${DATASET_DIRS[*]}")
   ACTION_DIT_BACKBONE_PATH="${FASTWAM_ACTION_DIT_BACKBONE_PATH:-${MODEL_BASE_PATH}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
   DOWNLOAD_DATA="${FASTWAM_DOWNLOAD_DATA:-1}"

   export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"

   if command -v hf >/dev/null 2>&1; then
       HF_BIN=hf
   elif command -v huggingface-cli >/dev/null 2>&1; then
       HF_BIN=huggingface-cli
   else
       echo "The Hugging Face CLI is required. Activate .venv first." >&2
       exit 1
   fi
   if [ ! -x "${PYTHON_BIN}" ]; then
       echo "RLinf venv not found at ${PYTHON_BIN}; run requirements/install.sh first." >&2
       exit 1
   fi

   mkdir -p "${CHECKPOINT_DIR}" "${MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B" \
       "${MODEL_BASE_PATH}/Wan-AI/Wan2.1-T2V-1.3B" "${DATASET_ROOT}"

   echo "[FastWAM] Downloading the released LIBERO checkpoint and stats..."
   "${HF_BIN}" download yuanty/fastwam \
       libero_uncond_2cam224.pt \
       libero_uncond_2cam224_dataset_stats.json \
       --local-dir "${CHECKPOINT_DIR}"

   echo "[FastWAM] Downloading the VAE and T5 weights used by RLinf..."
   "${HF_BIN}" download Wan-AI/Wan2.2-TI2V-5B \
       Wan2.2_VAE.pth \
       models_t5_umt5-xxl-enc-bf16.pth \
       --local-dir "${MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B"
   "${HF_BIN}" download Wan-AI/Wan2.1-T2V-1.3B \
       --include "google/umt5-xxl/*" \
       --local-dir "${MODEL_BASE_PATH}/Wan-AI/Wan2.1-T2V-1.3B"

   echo "[FastWAM] Downloading the official Wan2.2 video DiT for base-model SFT..."
   if ! find "${MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B" -maxdepth 1 -type f \
       -name 'diffusion_pytorch_model*.safetensors' -print -quit | grep -q .; then
       "${HF_BIN}" download Wan-AI/Wan2.2-TI2V-5B \
           --include "diffusion_pytorch_model*.safetensors" \
           --local-dir "${MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B"
   fi

   if [ ! -s "${ACTION_DIT_BACKBONE_PATH}" ]; then
       echo "[FastWAM] Preprocessing the official ActionDiT backbone..."
       mkdir -p "$(dirname -- "${ACTION_DIT_BACKBONE_PATH}")"
       "${PYTHON_BIN}" "${FASTWAM_PATH}/scripts/preprocess_action_dit_backbone.py" \
           --model-config "${FASTWAM_PATH}/configs/model/fastwam.yaml" \
           --output "${ACTION_DIT_BACKBONE_PATH}" \
           --device "${FASTWAM_ACTION_DIT_DEVICE:-cuda}" \
           --dtype bfloat16
   fi

   if [ "${DOWNLOAD_DATA}" = "1" ]; then
       echo "[FastWAM] Downloading the official LIBERO LeRobot archive set..."
       "${HF_BIN}" download --repo-type dataset yuanty/LIBERO-fastwam \
           --local-dir "${DATASET_ROOT}"
       shopt -s nullglob
       archives=("${DATASET_ROOT}"/*.tar.gz)
       if [ "${#archives[@]}" -eq 0 ]; then
           echo "No LIBERO tar.gz archives found in ${DATASET_ROOT}." >&2
           exit 1
       fi
       for archive in "${archives[@]}"; do
           echo "[FastWAM] Extracting ${archive}"
           tar -xzf "${archive}" -C "${DATASET_ROOT}"
       done
   fi

   for dataset_dir in "${DATASET_DIRS[@]}"; do
       if [ ! -f "${dataset_dir}/meta/tasks.jsonl" ]; then
           echo "Expected LIBERO dataset at ${dataset_dir} was not found." >&2
           echo "Extract the official archive set or set FASTWAM_DATASET_DIR for a one-suite run." >&2
           exit 1
       fi
   done

   mkdir -p "${TEXT_CACHE_DIR}"
   export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"
   echo "[FastWAM] Precomputing T5 text embeddings..."
   "${PYTHON_BIN}" "${FASTWAM_PATH}/scripts/precompute_text_embeds.py" \
       task=libero_uncond_2cam224_1e-4 \
       "data.train.dataset_dirs=[${DATASET_DIRS_HYDRA}]" \
       "data.train.text_embedding_cache_dir=${TEXT_CACHE_DIR}" \
       +overwrite=false \
       model.redirect_common_files=true

   cat <<EOF
   FastWAM SFT assets are ready.
     FASTWAM_CHECKPOINT_DIR=${CHECKPOINT_DIR}
     DIFFSYNTH_MODEL_BASE_PATH=${MODEL_BASE_PATH}
     FASTWAM_DATASET_ROOT=${DATASET_ROOT}
     FASTWAM_DATASET_DIRS=${DATASET_DIRS_HYDRA}
     FASTWAM_DATASET_DIR=${DATASET_DIR}
     FASTWAM_TEXT_EMBEDDING_CACHE_DIR=${TEXT_CACHE_DIR}
     FASTWAM_ACTION_DIT_BACKBONE_PATH=${ACTION_DIT_BACKBONE_PATH}
   EOF

完成准备后，启动训练：

.. code-block:: bash

   tmux new -s fastwam-sft
   source .venv/bin/activate
   bash examples/sft/run_vla_sft.sh libero_sft_fastwam
   # Ctrl-b d 脱离；查看时执行：tmux attach -t fastwam-sft

官方 FastWAM 与 RLinf 的差异
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

官方流程使用 preprocess_action_dit_backbone.py、缓存的 T5 embedding，
以及 Accelerate + DeepSpeed ZeRO-1；官方 README 的 LIBERO 示例使用 8 张
GPU。RLinf 的差异是：

* 复用上游 RobotVideoDataset、FastWAMProcessor 和 training_loss；
* 使用通用的 train_vla_sft.py / FSDP2 worker，而不是上游 train_zero1.sh；
* 组合上游 sim_libero 配置但不修改 Hydra 全局状态，从官方 Wan2.2
  base 和插值后的 ActionDiT backbone 开始训练，只训练 MoT 和 proprio encoder；
* 模型 preset 保持 precision: bf16，并在 loss 外层启用 bf16 autocast，
  与上游 Accelerator 路径一致。FSDP2 不额外启用 cast，避免第二条混合精度路径；
  RLinf wrapper 仍会将 VAE、text context、action 和 proprio 的直连输入对齐到模型 dtype。

这是有意的集成差异：RLinf 不承诺逐字节复现上游 optimizer 和分布式
launcher，但保持上游模型 loss、数据变换、mask、归一化和文本 cache 格式兼容。
