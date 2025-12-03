Reward Model Workflow
=====================

This document provides a comprehensive guide for the reward model workflow, including data collection, reward model training, and using the trained reward model to replace environment success signals for training RL policies.

Algorithm Support
-----------------

**This workflow supports PPO (Proximal Policy Optimization)** algorithm:

- ✅ **PPO** (Proximal Policy Optimization) - Use config: ``maniskill_ppo_cnn.yaml``

**Key Benefits**:

- Simple and unified workflow
- All scripts and configs are located in ``examples/embodiment/`` directory

Overview
--------

The reward model workflow consists of three main stages:

1. **Data Collection**: Collect trajectories using an initial RL policy (PPO), storing images and success labels
2. **Reward Model Training**: Train a binary classifier to predict success from single frames (algorithm-agnostic)
3. **RL Policy Training**: Use the trained reward model to replace environment success signals for PPO training

**Key Points**:

- The workflow uses PPO algorithm for RL policy training
- Uses unified ``RewardWorker`` for reward computation

Key Files and Components
-------------------------

Data Collection
~~~~~~~~~~~~~~~

**``rlinf/models/reward_model/reward_data_collector.py``**

**Purpose**: Core data collection implementation

- **Main Class**: ``RewardDataCollector``
- **Responsibilities**:
  - Maintains separate trajectory buffers for each parallel environment
  - Collects observations (images), ``success_frame`` labels, and ``success_once`` flags
  - Saves complete trajectories as ``.npy`` files
  - Classifies trajectories as positive/negative based on success criteria
  - Ensures data format consistency (``[C, H, W]`` images, normalized to ``[0, 1]``)

**Key Methods**:

- ``add_env_output(env_output, env_info, is_last_step)``: Add environment output to buffers
- ``save_trajectory(env_idx)``: Save completed trajectory to disk
- ``_extract_image(image, key_offset)``: Extract and format images from observations

**``rlinf/workers/env/env_worker.py``**

**Purpose**: Environment worker that integrates data collection

- **Integration Point**: Initializes ``RewardDataCollector`` in ``__init__`` if ``cfg.reward.collect_data=True``
- **Integration Method**: Calls ``data_collector.add_env_output()`` after each environment interaction in ``interact()`` method
- **Key Logic**: Extracts ``success_frame`` from ``env_output`` and ``success_once`` from ``env_info``

Reward Model Training
~~~~~~~~~~~~~~~~~~~~~

**``examples/embodiment/main_reward.py``**

**Purpose**: Training script for reward model using Ray and WorkerGroup

- **Main Function**: ``main(cfg: DictConfig)`` (uses Hydra decorator)
- **Configuration**: Uses Hydra and OmegaConf, config file at ``examples/embodiment/config/train_reward_model.yaml``
- **Architecture**: Uses Ray for distributed execution, ``RewardWorker`` with ``fit()`` method for training
- **Responsibilities**:
  - Creates ``RewardWorker`` group using Ray
  - Initializes workers and datasets
  - Trains ``BinaryRewardClassifier`` with BCE loss using simple PyTorch training (no FSDP)
  - Saves checkpoints at regular intervals (by step, not epoch)

**``rlinf/models/reward_model/reward_classifier.py``**

**Purpose**: Reward model architecture definition

- **Main Class**: ``BinaryRewardClassifier``
- **Architecture**:
  - **Encoder**: ``ResNetEncoderWrapper`` (ResNet10 backbone)
  - **Classifier Head**: MLP (Linear → LayerNorm → Tanh → Linear → Sigmoid)
  - **Input**: Single frame image ``[B, C, H, W]``
  - **Output**: Binary logit (success probability via sigmoid)

Reward Model Usage
~~~~~~~~~~~~~~~~~~

**``rlinf/workers/reward/reward_worker.py``**

**Purpose**: Unified reward worker that supports both text-based reasoning tasks and embodied tasks

- **Main Class**: ``RewardWorker``
- **Task Type Detection**: Automatically detects task type (embodied vs text-based) based on config
- **Responsibilities**:
  - For embodied tasks: Loads ``BinaryRewardClassifier`` from checkpoint and computes frame-based rewards
  - For text-based tasks: Supports rule-based rewards (reward model for text tasks not yet implemented)
  - Handles batch processing across parallel environments
  - Replaces environment success signals with model predictions for embodied tasks

Stage 1: Data Collection
-------------------------

Configuration
~~~~~~~~~~~~~

The reward model workflow uses PPO algorithm. Use the config file:

- **PPO**: ``examples/embodiment/config/maniskill_ppo_cnn.yaml``

Example configuration:

.. code-block:: yaml

   reward:
     group_name: "RewardGroup"
     use_reward_model: False  # Disabled during data collection
     collect_data: True  # Enable data collection
     reward_model:
       # These settings are not used during data collection, but should match actor settings
       image_keys: ["base_camera"]
       image_size: [3, 64, 64]
     data_collection:
       # Directory to save positive trajectories (success_once=1 or any success_frame>=0.5)
       positive_dir: "./reward_data/positive"
       # Directory to save negative trajectories (success_once=0 and all success_frame<0.5)
       negative_dir: "./reward_data/negative"
       # Image keys to collect (must match actor.model.image_keys)
       image_keys: ["base_camera"]
       # Maximum number of trajectories to save (None = unlimited)
       max_positive_trajectories: 500
       max_negative_trajectories: 500

Startup Command
~~~~~~~~~~~~~~~

.. code-block:: bash

   # For PPO
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn

   # Or directly with python
   export EMBODIED_PATH="/path/to/RLinf/examples/embodiment"
   export PYTHONPATH="/path/to/RLinf:$PYTHONPATH"
   python examples/embodiment/train_embodied_agent.py \
       --config-path examples/embodiment/config/ \
       --config-name maniskill_ppo_cnn \
       reward.collect_data=True \
       reward.use_reward_model=False

Data Format
~~~~~~~~~~~

Each trajectory is saved as a ``.npy`` file containing a dictionary:

.. code-block:: python

   {
       'images': {
           'base_camera': np.array([T, C, H, W], dtype=np.float32)  # T frames, normalized [0, 1]
       },
       'labels': np.array([T], dtype=np.float32)  # T success_frame labels (0 or 1)
   }

**File Naming**: ``{counter:06d}.npy`` (e.g., ``000000.npy``, ``000001.npy``, ...)

**Save Locations**:

- Positive trajectories: ``{positive_dir}/000000.npy``, ``{positive_dir}/000001.npy``, ...
- Negative trajectories: ``{negative_dir}/000000.npy``, ``{negative_dir}/000001.npy``, ...

Trajectory Classification Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A trajectory is saved to ``positive_dir`` if:

- ``success_once=True`` **OR**
- Any frame has ``success_frame >= 0.5``

Otherwise, it is saved to ``negative_dir``.

Stage 2: Reward Model Training
--------------------------------

Prerequisites
~~~~~~~~~~~~~

1. **Pretrained ResNet10 Encoder** (Optional but recommended):
   - **File**: ``resnet10_pretrained.pt``
   - **Location**: Should be placed in the project root or specified via ``pretrained_encoder_path`` in config
   - **Purpose**: Provides pretrained visual features for better initialization
   - **Note**: If not provided (set to ``null``), the encoder will be randomly initialized

2. **Collected Data**:
   - Positive trajectories in ``positive_dir``
   - Negative trajectories in ``negative_dir``

Configuration
~~~~~~~~~~~~~

The training script uses Hydra and OmegaConf for configuration management. The default configuration file is located at ``examples/embodiment/config/train_reward_model.yaml``.

Default Configuration File Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   defaults:
     - override hydra/job_logging: stdout

   cluster:
     num_nodes: 1
     component_placement:
       reward: all

   runner:
     task_type: embodied
     output_dir: ../results
     experiment_name: reward-model-training

   reward:
     group_name: "RewardGroup"
     training_backend: simple  # Simple training mode (no FSDP)
     gpu_id: 0  # GPU device ID to use for training
     
     global_batch_size: 32
     num_epochs: 100
     log_interval: 100
     save_interval: 1000  # Save checkpoint every N steps
     save_dir: ${runner.output_dir}/${runner.experiment_name}/checkpoints
     
     model:
       image_keys: ["base_camera"]
       image_size: [3, 64, 64]
       hidden_dim: 256
       num_spatial_blocks: 8
       pretrained_encoder_path: null
       use_pretrain: true
       freeze_encoder: true
     
     optim:
       optimizer: adam
       lr: 1e-4
       clip_grad: 1.0
     
     data:
       positive_dir: ./reward_data/positive
       negative_dir: ./reward_data/negative
       image_key: base_camera
       num_workers: 4

Startup Command
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Using the provided script (recommended)
   bash examples/embodiment/train_reward_model.sh train_reward_model

   # Or directly with python
   python examples/embodiment/main_reward.py \
       --config-path examples/embodiment/config \
       --config-name train_reward_model

   # Override configuration parameters via command line
   bash examples/embodiment/train_reward_model.sh train_reward_model \
       reward.gpu_id=1 \
       reward.global_batch_size=64 \
       reward.num_epochs=50 \
       reward.save_interval=500

Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Configuration Parameters
   :header-rows: 1
   :widths: 28 36 66 36

   * - Parameter
     - Default
     - Description
     - Example
   * - ``reward.training_backend``
     - ``simple``
     - Training backend (use ``simple`` for small models)
     - ``simple``
   * - ``reward.gpu_id``
     - ``0``
     - GPU device ID to use for training
     - ``0``, ``1``, ``2``
   * - ``reward.global_batch_size``
     - ``32``
     - Training batch size
     - ``64``, ``128``
   * - ``reward.num_epochs``
     - ``100``
     - Number of training epochs
     - ``50``, ``100``
   * - ``reward.save_interval``
     - ``1000``
     - Save checkpoint every N steps
     - ``500``, ``1000``, ``2000``
   * - ``reward.log_interval``
     - ``100``
     - Log metrics every N steps
     - ``50``, ``100``
   * - ``reward.save_dir``
     - ``${runner.output_dir}/${runner.experiment_name}/checkpoints``
     - Directory to save checkpoints
     - ``./checkpoints``
   * - ``reward.data.positive_dir``
     - ``./reward_data/positive``
     - Directory containing positive ``.npy`` files
     - ``./reward_data/positive``
   * - ``reward.data.negative_dir``
     - ``./reward_data/negative``
     - Directory containing negative ``.npy`` files
     - ``./reward_data/negative``
   * - ``reward.model.image_keys``
     - ``["base_camera"]``
     - Image keys to use
     - ``["base_camera"]``
   * - ``reward.model.image_size``
     - ``[3, 64, 64]``
     - Image size as ``[C, H, W]``
     - ``[3, 64, 64]``
   * - ``reward.model.hidden_dim``
     - ``256``
     - Hidden dimension for classifier
     - ``256``, ``512``
   * - ``reward.model.num_spatial_blocks``
     - ``8``
     - Number of spatial blocks for pooling
     - ``8``, ``16``
   * - ``reward.model.pretrained_encoder_path``
     - ``null``
     - Path to pretrained ResNet10 encoder
     - ``./resnet10_pretrained.pt``
   * - ``reward.optim.lr``
     - ``1e-4``
     - Learning rate
     - ``1e-4``, ``5e-5``
   * - ``reward.data.num_workers``
     - ``4``
     - Number of data loading workers
     - ``4``, ``8``

Training Process
~~~~~~~~~~~~~~~~

1. **Initialization**:
   - Creates Ray cluster and ``RewardWorker`` group
   - Initializes dataset from ``reward.data.positive_dir`` and ``reward.data.negative_dir``
   - Loads all ``.npy`` files and expands trajectory-level data to frame-level samples
   - Reports statistics:

     * Number of trajectories and frames in each directory
     * Distribution of labels (label=1 vs label=0) in each directory

2. **Model Creation**:
   - Creates ``BinaryRewardClassifier`` with specified parameters in ``reward.model``
   - If ``pretrained_encoder_path`` is provided, loads pretrained ResNet10 weights
   - Encoder weights are frozen by default (``freeze_encoder=True``)
   - Moves model to specified GPU (``gpu_id``)

3. **Training Loop**:
   - Uses BCE loss with logits (``F.binary_cross_entropy_with_logits``)
   - AdamW optimizer (configured in ``reward.optim``)
   - Learning rate scheduler (configured in ``reward.lr_sched``)
   - Gradient clipping (if ``reward.optim.clip_grad > 0``)
   - Logs metrics every ``log_interval`` steps
   - Saves checkpoint every ``save_interval`` steps

4. **Checkpoint Saving**:
   - Checkpoints are saved by **step** (not epoch), controlled by ``reward.save_interval``
   - Save path format: ``{save_dir}/step_{global_step}/checkpoint.pt``
   - Example: ``../results/reward-model-training/checkpoints/step_1000/checkpoint.pt``

Stage 3: Training RL Policy with Reward Model
-----------------------------------------------

Configuration
~~~~~~~~~~~~~

The reward model workflow uses PPO algorithm. Use the config file:

- **PPO**: ``examples/embodiment/config/maniskill_ppo_cnn.yaml``

Example configuration:

.. code-block:: yaml

   reward:
     group_name: "RewardGroup"
     use_reward_model: True  # Enable reward model
     collect_data: False  # Disable data collection during policy training
     reward_model:
       # Path to trained reward model checkpoint
       checkpoint_path: "./checkpoints/reward_model.pt"
       # Image keys to use (must match actor.model.image_keys)
       image_keys: ["base_camera"]
       # Image size [C, H, W] (must match actor.model.image_size)
       image_size: [3, 64, 64]
       # Hidden dimension for classifier (must match training config)
       hidden_dim: 256
       # Number of spatial blocks for pooling (must match training config)
       num_spatial_blocks: 8
       # Path to pretrained ResNet10 encoder weights (optional, used during model creation)
       pretrained_encoder_path: "./resnet10_pretrained.pt"
       # Whether to use pretrained encoder (must match training config)
       use_pretrain: True
       # Whether to freeze encoder weights (typically True)
       freeze_encoder: True
       # Reward type: "binary" (0 or 1) or "continuous" (probability)
       reward_type: "binary"

**Important**: The ``image_keys``, ``image_size``, ``hidden_dim``, ``num_spatial_blocks``, ``use_pretrain``, and ``freeze_encoder`` settings **must match** the settings used during reward model training.

Startup Command
~~~~~~~~~~~~~~~

.. code-block:: bash

   # For PPO
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn

   # Or directly with python
   export EMBODIED_PATH="/path/to/RLinf/examples/embodiment"
   export PYTHONPATH="/path/to/RLinf:$PYTHONPATH"
   python examples/embodiment/train_embodied_agent.py \
       --config-path examples/embodiment/config/ \
       --config-name maniskill_ppo_cnn \
       reward.use_reward_model=True \
       reward.collect_data=False \
       reward.reward_model.checkpoint_path="./checkpoints/reward_model.pt"

Reward Model Loading Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Task Type Detection** (``RewardWorker._is_embodied_task()``):
   - Automatically detects if this is an embodied task based on config
   - For embodied tasks, initializes image-based reward model
   - For text-based tasks, initializes text-based reward handler

2. **Initialization for Embodied Tasks** (``RewardWorker.init_worker()``):
   - Creates ``BinaryRewardClassifier`` with config parameters
   - Loads checkpoint from ``checkpoint_path``
   - Automatically infers ``use_pretrain`` from checkpoint if there's a mismatch
   - Moves model to device (``torch.cuda.current_device()``)
   - Sets model to eval mode

3. **Checkpoint Compatibility**:
   - If ``use_pretrain`` in config doesn't match the training config, the code will automatically infer the correct value from the checkpoint's pooling layer kernel shape
   - Warning messages will be printed if adjustments are made

Reward Computation Process
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Unified Reward Computation** (``RewardWorker.compute_rewards()``):
   - Automatically detects input type (``RolloutResult`` vs ``EmbodiedRolloutResult``)
   - Routes to appropriate processing method based on input type

2. **For Embodied Tasks** (``RewardWorker._compute_embodied_rewards_with_model()``):
   - Receives ``EmbodiedRolloutResult`` from rollout worker
   - Extracts observations (images) from ``transitions`` or ``forward_inputs``
   - Processes each observation:

     - Extracts image using ``_extract_images_from_obs()``
     - Ensures images are in ``[B, C, H, W]`` format
     - Runs forward pass through reward model
     - Applies sigmoid to get success probability
     - Converts to reward based on ``reward_type``:

       - ``"binary"``: ``(probs > 0.5).float()`` → 0 or 1
       - ``"continuous"``: Uses probability directly

3. **Integration with RL Training**:
   - Reward model predictions replace or supplement environment ``success_frame`` signals
   - For PPO: Used for advantage computation and policy updates

Complete Workflow Summary
--------------------------

The complete workflow consists of three stages:

1. **Stage 1: Data Collection** - Collect trajectories using initial RL policy with ``reward.collect_data=True``
2. **Stage 2: Reward Model Training** - Train binary classifier using collected data
3. **Stage 3: RL Policy Training** - Use trained reward model with ``reward.use_reward_model=True``

Complete Workflow Example
-------------------------

.. code-block:: bash

   # Step 1: Collect data
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn \
       reward.collect_data=True \
       reward.use_reward_model=False

   # Step 2: Train reward model
   bash examples/embodiment/train_reward_model.sh train_reward_model \
       reward.data.positive_dir=./reward_data/positive \
       reward.data.negative_dir=./reward_data/negative

   # Step 3: Train with reward model
   bash examples/embodiment/run_embodiment.sh maniskill_ppo_cnn \
       reward.use_reward_model=True \
       reward.collect_data=False \
       reward.reward_model.checkpoint_path=./checkpoints/reward_model.pt

