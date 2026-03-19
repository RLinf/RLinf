RL with LIBERO-Pro & LIBERO-Plus Benchmark
===============================================

This update introduces full support for the LIBERO-Pro and LIBERO-Plus evaluation suites within the RLinf framework. By incorporating more complex task scenarios and longer manipulation horizons, these suites further challenge the generalization capabilities of VLA models (such as OpenVLA-OFT).

The primary objective is to develop a model capable of performing robotic manipulation by:

1. **Visual Understanding**: Processing RGB images from robot cameras.
2. **Language Comprehension**: Interpreting natural-language task descriptions under perturbations.
3. **Action Generation**: Producing precise robotic actions (position, rotation, gripper control).
4. **Reinforcement Learning**: Optimizing the policy via PPO/GRPO with environment feedback.

Environment
-----------
**Base Simulation Setup**

* **Environment:** Simulation benchmarks built on top of robosuite (MuJoCo), heavily extending the original LIBERO suites with rigorous perturbation tests.
* **Observation:** RGB images captured by both third-person and wrist-mounted cameras.
* **Action Space:** 7-dimensional continuous actions (3D position, 3D rotation, and 1D gripper control).

**LIBERO-Pro: Anti-Memorization Perturbations**
LIBERO-Pro systematically evaluates model robustness across four orthogonal dimensions to prevent rote memorization:

* **Object Attribute Perturbations:** Modifies non-essential attributes of target objects (e.g., color, texture, size) while preserving semantic equivalence.
* **Initial Position Perturbations:** Alters the absolute and relative spatial arrangements of objects at the start of the episode.
* **Instruction Perturbations:** Introduces semantic paraphrasing (e.g., "grab" instead of "pick up") and task-level modifications (e.g., replacing the target object in the instruction).
* **Environment Perturbations:** Randomly substitutes the background workspace/scene appearance.

**LIBERO-Plus: In-depth Robustness Perturbations**
LIBERO-Plus expands the evaluation into a massive suite of 10,030 tasks across 5 difficulty levels, applying perturbations across 7 physical and semantic dimensions:

* **Objects Layout:** Injects confounding distractor objects and shifts the target object's position/pose.
* **Camera Viewpoints:** Shifts the 3rd-person camera's distance, spherical position (azimuth/elevation), and orientation.
* **Robot Initial States:** Applies random perturbations to the robot arm's initial joint angles (qpos).
* **Language Instructions:** Rewrites task instructions using LLMs to add conversational distractions, common-sense reasoning, or complex reasoning chains.
* **Light Conditions:** Alters diffuse color, light direction, specular highlights, and shadow casting.
* **Background Textures:** Modifies scene themes (e.g., brick walls) and surface materials.
* **Sensor Noise:** Simulates real-world degradation by injecting motion blur, Gaussian blur, zoom blur, fog, and glass refraction distortions.

Algorithm
---------
**Core Algorithm Components**

* **PPO (Proximal Policy Optimization)**

  * Advantage estimation using GAE (Generalized Advantage Estimation).
  * Policy clipping with ratio limits.
  * Value function clipping.
  * Entropy regularization.

* **GRPO (Group Relative Policy Optimization)**

  * For every state / prompt the policy generates *G* independent actions.
  * Compute the advantage of each action by subtracting the group's mean reward.

**Vision-Language-Action Model**

* OpenVLA architecture with multimodal fusion.
* Action tokenization and de-tokenization.
* Value head for critic function.

Dependency Installation
-----------------------
To ensure full compatibility with the RLinf framework, please install the designated forks maintained under the RLinf organization.

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

Use Docker image for the experiment.

.. code:: bash

   # LIBERO-Pro
   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:embodied-liberopro

   # LIBERO-Plus
   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:embodied-liberoplus

**Option 2: Custom Environment**

.. code:: bash

    # For mainland China users, you can add the `--use-mirror` flag for better download speed.

    # Create an embodied environment with LIBERO-Pro support
    bash requirements/install.sh embodied --model openvla-oft --env liberopro

    # Create an embodied environment with LIBERO-Plus support
    bash requirements/install.sh embodied --model openvla-oft --env liberoplus

    # Activate the virtual environment
    source .venv/bin/activate

LIBERO-Plus Assets Download
---------------------------

LIBERO-Plus requires hundreds of new objects, textures, and other assets to function correctly. Download the ``assets.zip`` archive from the Hugging Face dataset ``Sylvest/LIBERO-plus`` and extract it into the installed ``liberoplus.liberoplus`` package directory.

.. code-block:: bash

    # Resolve the installed liberoplus package directory
    LIBERO_PLUS_PACKAGE_DIR=$(python -c "import pathlib; import liberoplus.liberoplus as l_plus; print(pathlib.Path(l_plus.__file__).resolve().parent)")

    # For mainland China users, you can use the following for better download speed:
    # export HF_ENDPOINT=https://hf-mirror.com

    # Download the assets archive from the Hugging Face dataset repo
    hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip \
        --local-dir "${LIBERO_PLUS_PACKAGE_DIR}"

    # Extract assets in place
    unzip -o "${LIBERO_PLUS_PACKAGE_DIR}/assets.zip" -d "${LIBERO_PLUS_PACKAGE_DIR}"

After extraction, ensure your directory structure matches the following layout:

.. code-block:: text

    <installed liberoplus package dir>/
    └── assets/
        ├── articulated_objects/
        ├── new_objects/
        ├── scenes/
        ├── stable_hope_objects/
        ├── stable_scanned_objects/
        ├── textures/
        ├── turbosquid_objects/
        ├── serving_region.xml
        ├── wall_frames.stl
        └── wall.xml

Model Download
--------------

For OpenVLA-OFT-based experiments on LIBERO-Pro and LIBERO-Plus, you can start from the same pretrained checkpoints used for standard LIBERO training:

.. code-block:: bash

    # Download the model (choose either method)
    # Method 1: Using git clone
    git lfs install
    git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
    git clone https://huggingface.co/RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

    # Method 2: Using huggingface-hub
    # For mainland China users, you can use the following for better download speed:
    # export HF_ENDPOINT=https://hf-mirror.com
    pip install huggingface-hub
    hf download RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
    hf download RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora --local-dir RLinf-OpenVLAOFT-LIBERO-130-Base-Lora

After downloading, make sure to correctly specify the model path in the configuration yaml file.

.. code-block:: yaml

    rollout:
       model:
          model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora
    actor:
       model:
          model_path: Pathto/RLinf/RLinf-OpenVLAOFT-LIBERO-90-Base-Lora

Running the Script
------------------
**1. Configuration Files**

The LIBERO-Pro and LIBERO-Plus suites reuse the standard LIBERO config family and switch the suite through the additional ``LIBERO_TYPE`` argument:

- **OpenVLA-OFT + GRPO**: ``examples/embodiment/config/libero_10_grpo_openvlaoft.yaml``

**2. Launch Commands**

**Training**

To start training a model on the newly integrated suites, use the ``run_embodiment.sh`` script:

.. code-block:: bash

    # Train on LIBERO-Pro
    export LIBERO_TYPE=pro
    bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft

    # Train on LIBERO-Plus
    export LIBERO_TYPE=plus
    bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft

**Evaluation**

To evaluate the trained models, use the ``eval_embodiment.sh`` script:

.. code-block:: bash

    # Evaluate on LIBERO-Pro
    export LIBERO_TYPE=pro
    bash examples/embodiment/eval_embodiment.sh libero_10_grpo_openvlaoft

    # Evaluate on LIBERO-Plus
    export LIBERO_TYPE=plus
    bash examples/embodiment/eval_embodiment.sh libero_10_grpo_openvlaoft