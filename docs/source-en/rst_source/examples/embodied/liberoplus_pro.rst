=================================================
LIBERO-Pro & LIBERO-Plus Integration
=================================================

Introduction
------------
This update introduces full support for the LIBERO-Pro and LIBERO-Plus evaluation suites within the RLinf framework. By incorporating more complex task scenarios and longer manipulation horizons, these suites further challenge the generalization capabilities of VLA models (such as OpenVLA-OFT).

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

Installation
------------
To ensure full compatibility with the RLinf framework, please install the designated forks maintained under the RLinf organization.

**1. Install LIBERO-Pro**

.. code-block:: bash

    git clone https://github.com/RLinf/LIBERO-PRO.git
    cd LIBERO-PRO
    pip install -r requirements.txt
    pip install -e .
    cd ..

**2. Install LIBERO-Plus**

LIBERO-Plus requires additional system-level dependencies for rendering and processing. 

.. code-block:: bash

    git clone https://github.com/RLinf/LIBERO-plus.git
    cd LIBERO-plus

    # Install system dependencies (requires root privileges)
    apt-get update
    apt-get install -y libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev

    # Install Python dependencies and the package
    pip install -r extra_requirements.txt
    pip install -e .

**3. Download LIBERO-Plus Assets**

LIBERO-Plus requires hundreds of new objects, textures, and other assets to function correctly. You must download the ``assets.zip`` file from the official LIBERO-plus collection and extract it into the specified path.

.. code-block:: bash

    # Navigate to the inner libero directory
    cd libero/libero/
    
    # Download and extract the assets (ensure you have the assets.zip file here)
    unzip assets.zip
    
    # Return to the workspace root
    cd ../../../../

After extraction, ensure your directory structure matches the following layout:

.. code-block:: text

    LIBERO-plus/
    └── libero/
        └── libero/
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

Usage
-----
**Training**

To start training a model on the newly integrated suites, use the ``run_embodiment.sh`` script:

.. code-block:: bash

    # Train on LIBERO-Pro
    bash run_embodiment.sh libero_10_grpo_openvlaoft LIBERO pro

    # Train on LIBERO-Plus
    bash run_embodiment.sh libero_10_grpo_openvlaoft LIBERO plus

**Evaluation**

To evaluate the trained models, use the ``eval_embodiment.sh`` script:

.. code-block:: bash

    # Evaluate on LIBERO-Pro
    bash eval_embodiment.sh libero_10_grpo_openvlaoft LIBERO pro

    # Evaluate on LIBERO-Plus
    bash eval_embodiment.sh libero_10_grpo_openvlaoft LIBERO plus