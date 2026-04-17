RL with EmbodiChain
===================

EmbodiChain (`<https://github.com/DexForce/EmbodiChain>`__) is an embodied
intelligence lab stack that exposes reinforcement learning tasks through a
Gym-style interface. RLinf integrates it as an embodied environment type
(``env_type: embodichain``).

The current RLinf integration is focused on simple reinforcement learning tasks. In
other words, RLinf already provides a stable environment entry point for
EmbodiChain tasks, and the validated example is **CartPole** with an MLP
actor-critic: configuration ``embodichain_ppo_cart_pole`` and environment
snippet ``env/embodichain_cart_pole``.

Current Support Scope
---------------------

RLinf currently validates EmbodiChain through a CartPole task:

- RLinf loads an EmbodiChain gym JSON via ``gym_config_path``
- RLinf extracts robot state fields and concatenates them into ``states``
- RLinf trains a standard RL policy such as ``mlp_policy``

At the moment, the bundled official example is:

- **CartPole + PPO + MLP** via ``embodichain_ppo_cart_pole``

The upstream EmbodiChain repository already contains richer task configs,
including manipulation-oriented tasks. However, those tasks are **not yet
packaged in RLinf as official recipes**, especially when they require camera
observations, language instructions, or VLA-style multimodal inputs.

Environment
-----------

**Environment Registration**

- **Env type**: ``embodichain``
- **Enum entry**: ``SupportedEnvType.EMBODICHAIN``
- **Implementation**:
  ``rlinf.envs.embodichain.embodichain_env.EmbodiChainEnv``

**Gym Config Resolution**

Set ``gym_config_path`` to an EmbodiChain task JSON file. RLinf resolves the
path in the following order:

1. Absolute path
2. If ``EMBODICHAIN_PATH`` is set: ``${EMBODICHAIN_PATH}/relative path`` (optional override for a local checkout)
3. A path next to the installed ``embodichain`` package (default after ``pip install``)

With a normal pip install, configs are resolved from the package without setting
``EMBODICHAIN_PATH``.

**Observation and Action Spaces**

- **Observation**: RLinf exposes a single tensor key ``states``.
- **State construction**: ``states`` is formed by concatenating the
  EmbodiChain ``robot`` fields listed in ``state_keys``.
- **Default state keys**: ``["qpos", "qvel", "qf"]``
- **Action space**: continuous Box actions

Make sure the policy config matches the task:

- ``actor.model.obs_dim`` must equal the flattened dimension of the selected
  state fields
- ``actor.model.action_dim`` must match the environment action dimension
- ``actor.model.policy_setup`` must match the task control mode

The CartPole example uses ``policy_setup: cartpole-delta-qpos``.

**Simulation Notes**

The following env fields are passed to EmbodiChain's
``SimulationManagerCfg``:

- ``headless``
- ``enable_rt``
- ``sim_device``

When running under RLinf placement, always use logical GPU id ``0`` inside the
worker process. RLinf sets ``CUDA_VISIBLE_DEVICES`` per worker, so
``cuda:0`` refers to the GPU assigned to that worker.

Algorithm
---------

The bundled EmbodiChain example uses:

- **Policy**: ``mlp_policy``
- **Algorithm**: PPO
- **Advantage estimation**: GAE
- **Loss type**: actor-critic

This setup is intended for low-dimensional state-based control and is aligned
with the general MLP workflow described in :doc:`mlp`.

Extension Path for Future VLA Fine-Tuning
-----------------------------------------

EmbodiChain is also a reasonable foundation for future RLinf **VLA
fine-tuning** on more complex embodied tasks, but that requires more than
changing ``gym_config_path``.

For VLA-oriented tasks, you will typically need to extend RLinf in the
following areas:

1. **Observation wiring**

   - Export camera images, masks, language instructions, or other multimodal
     inputs from EmbodiChain into the observation dictionary expected by the
     target VLA model.

2. **Model configuration**

   - Replace ``mlp_policy`` with the corresponding VLA model configuration
     (for example OpenVLA, OpenVLA-OFT, OpenPI, Dexbotic, or another supported
     model, depending on the task).

3. **Action semantics**

   - Align ``policy_setup``, ``action_dim``, and any action chunk settings with
     the concrete robot control interface exposed by the EmbodiChain task.

4. **Task recipe packaging**

   - Add dedicated RLinf config files for each validated EmbodiChain task,
     instead of relying on a single generic example page.

In other words, the current EmbodiChain integration already gives RLinf a
stable environment entry point, while richer visual or language-conditioned
tasks can be added incrementally on top of the same wrapper.

Dependency Installation
-----------------------

Install RLinf embodied dependencies and EmbodiChain without any VLA model
requirements:

.. code-block:: bash

   cd <path_to_RLinf_repository>
   bash requirements/install.sh embodied --env embodichain

This installs the latest EmbodiChain release from the project extra index.
Gym task configs are loaded from the installed package; you do not need
``EMBODICHAIN_PATH`` for typical use.

.. note::

   **Current Limitation**

   ``dexsim``, the simulator dependency of EmbodiChain, requires the
   ``libpython3.xx.so`` library. Currently, ``dexsim`` resolves this library
   through system paths (e.g., ``/usr/local/lib``) or Conda library paths via
   rpath, but does not fully support UV's Python runtime layout yet.

   If you encounter ``libpython3.11.so`` related runtime errors, we recommend
   using a Conda environment instead of UV:

   .. code-block:: bash

      # Create a Conda environment (recommended)
      conda create -n rlinf python=3.11
      conda activate rlinf
      pip install uv
      # Then run the installation script, but skip UV's Python management
      bash requirements/install.sh embodied --env embodichain --no-root

   We plan to improve UV support in the next ``dexsim`` release.

To point at a different EmbodiChain tree (for example a local git checkout),
export:

.. code-block:: bash

   export EMBODICHAIN_PATH=/path/to/EmbodiChain

The helper launcher ``examples/embodiment/run_embodiment.sh`` does not set a
default ``EMBODICHAIN_PATH``; leave it unset to use the installed package paths.

Quick Start
-----------

**1. Environment Config**

The reference env config is
``examples/embodiment/config/env/embodichain_cart_pole.yaml``:

.. code-block:: yaml

   env_type: embodichain
   gym_config_path: configs/agents/rl/basic/cart_pole/gym_config.json
   headless: true
   enable_rt: false
   sim_device: cuda
   state_keys: ["qpos", "qvel", "qf"]

**2. Training Config**

The top-level training config is
``examples/embodiment/config/embodichain_ppo_cart_pole.yaml``. It uses:

- ``model/mlp_policy@actor.model``
- PPO + GAE
- ``obs_dim: 6``
- ``action_dim: 2``
- ``policy_setup: cartpole-delta-qpos``

**3. Launch Training**

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh embodichain_ppo_cart_pole

**If you hit missing files or dataset download errors**

You do **not** need to download assets up front when everything already works. If a
run fails because EmbodiChain cannot find task or simulation resources, try:

1. Set the data root (where downloads should be stored):

   .. code-block:: bash

      export EMBODICHAIN_DATA_ROOT=/path/to/data

2. Download CartPole and shared simulation resources (same Python env as
   ``embodichain``):

   .. code-block:: bash

      python -m embodichain.data download --name CartPole
      python -m embodichain.data download --name SimResources

3. Retry the training command in step 3.

**4. Adapt to New Tasks**

To run another EmbodiChain task:

1. Point ``gym_config_path`` to another EmbodiChain gym JSON
2. Update ``state_keys`` if the task exposes different state fields
3. Update ``actor.model.obs_dim`` to match the flattened state dimension
4. Update ``actor.model.action_dim`` and ``policy_setup`` to match the task

For low-dimensional tasks, these changes are often sufficient. For visual or
language-conditioned tasks, you will likely also need additional RLinf
observation/model integration before training can run end to end.

Evaluation and CI
-----------------

EmbodiChain CartPole is also used by embodied end-to-end tests under
``tests/e2e_tests/embodied/``. After installing ``embodichain`` into the test
environment, configs resolve from the package; set ``EMBODICHAIN_PATH`` only if
you need a non-default checkout.
