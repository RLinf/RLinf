Stack Cube reward reference samples
==================================

Use these frozen simulated states to check reward and drop predicates without downloading model weights or starting Isaac Sim. ``boundary_trace.npz`` contains 1,055 height, XY, gripper and sampled cases. ``gpu_reward_trace.npz`` contains 3,040 measured states from eight EmbodiChain expert replays. ``provenance.json`` identifies the pinned IsaacLab source used to compute their expected rewards.

The component tests compare RLinf measurements against these expected values. To independently repeat the source comparison from the repository root, check out ``RLinf/IsaacLab`` at ``4246b6b4f4a3e74ee20e002ed7536b1c788d39f4`` and run:

.. code-block:: bash

   python toolkits/lerobot/compare_stack_cube_rewards.py \
     --isaaclab-root /path/to/RLinf-IsaacLab \
     --trace tests/fixtures/embodichain_stack_cube/boundary_trace.npz \
     --trace tests/fixtures/embodichain_stack_cube/gpu_reward_trace.npz \
     --output /tmp/stack_cube_reward_comparison.json

The output records the input and source hashes, comparison count and maximum absolute error. The tool executes unchanged manager and predicate definitions with lightweight configuration and scene objects. It validates numerical rewards and drop predicates; a native IsaacLab physics rollout is a separate integration check.
