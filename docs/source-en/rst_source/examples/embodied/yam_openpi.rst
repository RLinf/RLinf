Pi0.5 Training and Evaluation on YAM
====================================

Collect demonstrations with :doc:`yam`, compute normalization statistics,
fine-tune OpenPI Pi0.5, and evaluate the checkpoint on a dual-arm YAM setup.

Observation and Action
----------------------

``pi05_yam_joint`` uses 14-D states and absolute-position actions in the order
``[left joints (6), left gripper, right joints (6), right gripper]``.
Joints are in radians; grippers are normalized to ``[0, 1]``.
The three RGB views are the top, left-wrist, and right-wrist cameras, in that order.

Training uses state-relative deltas for the 12 joints and absolute gripper
targets. Shared OpenPI transforms normalize the 14-D vectors and then
pad them to the model's 32-D width. Pi0.5 also tokenizes the normalized 14-D
state with the task prompt. The model predicts 30 actions; evaluation
executes the first 5 before requesting the next prediction.

Installation
------------

On the GPU training node and the YAM controller, use the same RLinf revision and
install the model and robot dependencies in one environment:

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf
   bash requirements/install.sh embodied --model openpi --env yam
   source .venv/bin/activate

Configure the CAN interfaces and identify the three camera serials using
:doc:`yam`.

Dataset and Normalization
-------------------------

On the YAM controller, collect successful demonstrations following :doc:`yam`. The collector prints
the run directory containing ``lerobot_dataset/``. Set ``YAM_COLLECTION`` to
that directory. The following commands merge its shards into a new local dataset:

.. code-block:: bash

   export YAM_COLLECTION=/absolute/path/to/collection/run
   export YAM_DATASET="$HOME/yam-data/merged"
   mkdir -p "$(dirname "$YAM_DATASET")"
   python toolkits/lerobot/merge_lerobot_datasets.py \
       --source-dir "$YAM_COLLECTION/lerobot_dataset" --output-dir "$YAM_DATASET" \
       --only-success --dry-run
   python toolkits/lerobot/merge_lerobot_datasets.py \
       --source-dir "$YAM_COLLECTION/lerobot_dataset" --output-dir "$YAM_DATASET" \
       --only-success
   head -n 5 "$YAM_DATASET/meta/tasks.jsonl"
   PYTHONPATH=$(pwd) python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_yam_joint --repo-id "$YAM_DATASET"
   test -s "$YAM_DATASET/norm_stats.json"

Use the recorded task text from ``meta/tasks.jsonl`` as the evaluation prompt.
Keep the resulting statistics with this dataset and use the same file for
training and evaluation.

Prepare the dataset on the GPU node before converting the model or training.
With shared storage, set ``YAM_DATASET`` to the dataset's absolute path on that
node. To copy over SSH, obtain the username with ``whoami``, a reachable
controller address with ``hostname -I``, and the source directory with
``echo "$YAM_DATASET"`` on the controller. Replace the SSH username, controller
address and source path below with those values. From the RLinf directory on
the GPU node, copy the complete merged dataset, including its statistics, to
a new directory:

.. code-block:: bash

   source .venv/bin/activate
   export YAM_DATASET="$HOME/yam-data/merged"
   mkdir -p "$(dirname "$YAM_DATASET")"
   test ! -e "$YAM_DATASET" && \
       scp -r '<ssh-user>@<controller-ip>:/absolute/path/to/merged' "$YAM_DATASET" && \
       test -s "$YAM_DATASET/norm_stats.json"

Continue on the GPU node after the copy succeeds. The destination check preserves
an existing dataset; choose a different ``YAM_DATASET`` path if it already exists.

Pretrained Model
----------------

The common installer provides ``gsutil``. On the GPU node, download the official Pi0.5 base
checkpoint and convert it to the OpenPI PyTorch layout used by RLinf:

.. code-block:: bash

   export YAM_MODELS="$HOME/yam-models"
   mkdir -p "$YAM_MODELS/jax"
   gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base "$YAM_MODELS/jax/"
   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi \
       --input-model "$YAM_MODELS/jax/pi05_base" \
       --input-norm-stats "$YAM_DATASET/norm_stats.json" \
       --output-model "$YAM_MODELS/pi05_base_openpi" \
       --output-norm-stats "$YAM_MODELS/pi05_base_openpi/yam/norm_stats.json" \
       --action-dim 32 --action-horizon 30 --max-token-len 200
   test -s "$YAM_MODELS/pi05_base_openpi/model.safetensors"

Supervised Fine-Tuning
----------------------

On the GPU node, edit ``examples/sft/config/realworld_dual_yam_sft_openpi_pi05.yaml``
with the absolute paths obtained above:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Field
     - Value
   * - ``data.train_data_paths``
     - The merged ``YAM_DATASET`` directory.
   * - ``actor.model.model_path``
     - ``YAM_MODELS/pi05_base_openpi``.
   * - ``actor.model.openpi_data.norm_stats_path``
     - ``YAM_DATASET/norm_stats.json``.

The recipe uses FP32 parameters, BF16 computation and ``no_shard``;
see :doc:`sft_openpi`. Each actor GPU holds the
complete model, gradients and optimizer state. Allocating more GPUs increases
throughput rather than reducing this per-GPU memory requirement.

``cluster.component_placement.actor: all`` uses the allocated GPUs. Run
``nvidia-smi`` to check device memory and active processes. For a single 48 GB
GPU, start with ``actor.micro_batch_size: 1``, ``actor.global_batch_size: 1``
and ``actor.model.precision: bf16`` in the YAML. This changes parameter and
optimizer precision from the FP32 default; check training quality for your
task. Increase the batch only after a complete optimizer step fits. The global
batch must be divisible by the micro batch times the number of actor workers.

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh realworld_dual_yam_sft_openpi_pi05

The launcher prints the run's log directory. To monitor training, open another
terminal in the RLinf directory on the GPU node:

.. code-block:: bash

   source .venv/bin/activate
   tensorboard --logdir logs --port 6006

After a checkpoint is saved, list checkpoint directories from a separate
terminal in that same RLinf directory:

.. code-block:: bash

   find logs -type d -path '*/checkpoints/global_step_*'

Select a saved
``checkpoints/global_step_<N>`` directory from the ``find`` output: it contains
``actor/model_state_dict/full_weights.pt``, which the evaluation loader accepts
directly. ``<N>`` is the saved training step.

Evaluation
----------

.. warning::

   Evaluation connects to both CAN devices and all three cameras. If the SDK's
   ``gripper_limits`` are unset, connection automatically opens and closes each
   gripper for calibration; keep both grippers empty and their travel clear.
   Keep an operator at the emergency stop and supervise the entire run. Before
   the episode ends or the process is stopped, ensure both arms can safely remain
   supported when motor output is released. This recipe leaves reset and parking
   to the operator.

Use a GPU node with rank ``0`` and the YAM controller with rank ``1``. Install
the same dependencies on both nodes before starting Ray, as described in
:doc:`../../guides/multi_node`. Check ``ray status`` for an existing cluster;
use machines allocated to this run rather than stopping another user's cluster.

On both machines, run ``hostname -I`` and choose addresses reachable from each
other. Export ``GPU_IP`` and ``CONTROL_IP`` with those addresses. Check the
controller address with ``ping -c 1 "$CONTROL_IP"`` from the GPU node and the
GPU address with ``ping -c 1 "$GPU_IP"`` from the controller. Set
``YAM_TASK_DESCRIPTION`` to the task text in the dataset. On the controller,
export the CAN channels and top/left/right camera serials identified in
:doc:`yam` before starting Ray.

Configure the same authentication token on both nodes using the
`Ray self-managed cluster instructions
<https://docs.ray.io/en/latest/ray-security/token-auth.html#self-managed-clusters>`_.
Keep the token file private and use the allocated cluster's token rather than
replacing another run's credentials.

On the GPU node:

.. code-block:: bash

   source .venv/bin/activate
   export RAY_AUTH_MODE=token
   export YAM_TASK_DESCRIPTION='pick the block'
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address="$GPU_IP"

On the controller, use the same task description:

.. code-block:: bash

   source .venv/bin/activate
   export RAY_AUTH_MODE=token
   export YAM_TASK_DESCRIPTION='pick the block'
   export RLINF_NODE_RANK=1
   ray start --address="$GPU_IP:6379" --node-ip-address="$CONTROL_IP"

On the GPU node, ``ray status`` must list two alive nodes. Confirm the robot's
joint limits, camera views, emergency stop, and clear workspace using the
collection guide before enabling policy control. The operator positions both
arms for the task; reset and parking remain operator-controlled by default.

Set ``YAM_CHECKPOINT`` to the saved ``global_step_<N>`` directory selected above.
Verify ``test -s "$YAM_CHECKPOINT/actor/model_state_dict/full_weights.pt"``
and run evaluation on the GPU node:

.. code-block:: bash

   bash evaluations/run_eval.sh realworld realworld_dual_yam_openpi_pi05_eval \
       rollout.model.model_path="$YAM_CHECKPOINT" \
       rollout.model.openpi_data.norm_stats_path="$YAM_DATASET/norm_stats.json"

The launcher prints the evaluation log directory; videos are saved under that
run. The YAM environment reports execution metrics and a zero step reward;
the operator judges task completion from the robot execution and video.

RPent Deployment
----------------

RPent's YAM OpenPI integration uses this same ``pi05_yam_joint`` contract:
14-D action targets (12 joint positions and 2 gripper targets), top/left/right
cameras, a 30-step prediction, and 5 executed steps at 30 Hz. On both RPent
nodes, run the following from the RLinf repository directory used above:

.. code-block:: bash

   source .venv/bin/activate
   export RPENT_RLINF_ROOT="$(pwd)"
   git clone --branch main https://github.com/RLinf/RPent.git "$HOME/RPent-yam"
   uv pip install -e "$HOME/RPent-yam[yam]" "numpy<2"

This installs RPent's optional YAM dependencies in the existing model/robot environment while
preserving its dependency constraints. Use the fine-tuned checkpoint and its
dataset statistics with the RPC and controller startup sections of the
`RPent YAM deployment instructions
<https://github.com/RLinf/RPent/blob/main/docs/source-en/rst_source/real_world_robots/yam.rst>`_.
For their RLinf path settings, use ``RPENT_RLINF_ROOT`` exported above.
Stop the native evaluation before starting RPent's controller so only one
process owns the arms.
