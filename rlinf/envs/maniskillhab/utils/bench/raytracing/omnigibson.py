import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.object_states import Covered
from omnigibson.utils.constants import PrimType
from omnigibson.utils.profiling_utils import ProfilingEnv

import torch as th


OUTPUT_FP = "bench_results/og_raytracing.json"

PROFILING_FIELDS = [
    "FPS",
    "Omni step time",
    "Non-omni step time",
    "Memory usage",
    "Vram usage",
]
NUM_CLOTH = 5
NUM_SLICE_OBJECT = 3

SCENE_OFFSET = {
    "": [0, 0],
    "Rs_int": [0, 0],
    "Pomaria_0_garden": [0.3, 0],
    "grocery_store_cafe": [-3.5, 3.5],
    "house_single_floor": [-3, -1],
    "Ihlen_0_int": [-1, 2],
}


# NOTE (arth) remove argparse for same behavior each time
@dataclass
class Args:
    robot: int = 1
    scene: str = "Rs_int"


def main():
    args = Args()

    # Modify macros settings
    gm.HEADLESS = True
    gm.ENABLE_HQ_RENDERING = False
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = True
    gm.ENABLE_FLATCACHE = False  # True  # NOTE (arth): originally set to not arg.cloth, i.e. True, but I get error saying to set to False:
    # ValueError: Failed to initialize water system. Please set gm.USE_GPU_DYNAMICS=True and gm.ENABLE_FLATCACHE=False
    gm.USE_GPU_DYNAMICS = True

    cfg = {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 120,
        }
    }
    if args.robot > 0:
        cfg["robots"] = []
        for i in range(args.robot):
            cfg["robots"].append(
                {
                    "type": "Fetch",
                    "obs_modalities": ["rgb", "depth"],
                    "position": [
                        -1.3 + 0.75 * i + SCENE_OFFSET[args.scene][0],
                        0.5 + SCENE_OFFSET[args.scene][1],
                        0,
                    ],
                    "orientation": [0.0, 0.0, 0.7071, -0.7071],
                }
            )

    if args.scene:
        assert (
            args.scene in SCENE_OFFSET
        ), f"Scene {args.scene} not found in SCENE_OFFSET"
        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": args.scene,
        }
    else:
        cfg["scene"] = {"type": "Scene"}

    cfg["objects"] = [
        {
            "type": "DatasetObject",
            "name": "table",
            "category": "breakfast_table",
            "model": "rjgmmy",
            "fixed_base": True,
            "scale": [0.75] * 3,
            "position": [
                0.5 + SCENE_OFFSET[args.scene][0],
                -1 + SCENE_OFFSET[args.scene][1],
                0.3,
            ],
            "orientation": [0.0, 0.0, 0.7071, -0.7071],
        }
    ]

    cfg["objects"].extend(
        [
            {
                "type": "DatasetObject",
                "name": f"apple_{n}",
                "category": "apple",
                "model": "agveuv",
                "scale": [1.5] * 3,
                "position": [
                    0.5 + SCENE_OFFSET[args.scene][0],
                    -1.25 + SCENE_OFFSET[args.scene][1] + n * 0.2,
                    0.5,
                ],
                "abilities": {},
            }
            for n in range(NUM_SLICE_OBJECT)
        ]
    )
    cfg["objects"].extend(
        [
            {
                "type": "DatasetObject",
                "name": f"knife_{n}",
                "category": "table_knife",
                "model": "jxdfyy",
                "scale": [2.5] * 3,
            }
            for n in range(NUM_SLICE_OBJECT)
        ]
    )

    load_start = time.time()
    env = ProfilingEnv(configs=cfg)
    table = env.scene.object_registry("name", "table")
    apples = [
        env.scene.object_registry("name", f"apple_{n}") for n in range(NUM_SLICE_OBJECT)
    ]
    knifes = [
        env.scene.object_registry("name", f"knife_{n}") for n in range(NUM_SLICE_OBJECT)
    ]
    env.reset()

    for n, knife in enumerate(knifes):
        knife.set_position_orientation(
            position=apples[n].get_position_orientation()[0]
            + th.tensor([-0.15, 0.0, 0.1 * (n + 2)]),
            orientation=T.euler2quat(th.tensor([-math.pi / 2, 0, 0], dtype=th.float32)),
        )
        knife.keep_still()

    output, results = [], []

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=[SCENE_OFFSET[args.scene][0], -3 + SCENE_OFFSET[args.scene][1], 1]
    )
    # record total load time
    total_load_time = time.time() - load_start

    for i in range(300):
        if args.robot:
            action_lo, action_hi = -0.3, 0.3
            result = env.step(
                th.stack(
                    [
                        th.rand(env.robots[i].action_dim) * (action_hi - action_lo)
                        + action_lo
                        for i in range(args.robot)
                    ]
                ).flatten()
            )[4]
        else:
            result = env.step(None)[4]
        results.append(result)

    field = f"{args.scene}" if args.scene else "Empty scene"
    if args.robot:
        field += f", with {args.robot} Fetch"
    output.append(
        {
            "name": field,
            "unit": "time (ms)",
            "value": total_load_time,
            "extra": ["Loading time", "Loading time"],
        }
    )
    results = th.tensor(results)
    for i, title in enumerate(PROFILING_FIELDS):
        unit = "time (ms)" if "time" in title else "GB"
        value = th.mean(results[:, i])
        if title == "FPS":
            value = 1000 / value
            unit = "fps"
        output.append(
            {
                "name": field,
                "unit": unit,
                "value": value.item(),
                "extra": [title, title],
            }
        )

    ret = []
    if os.path.exists(OUTPUT_FP):
        with open(OUTPUT_FP, "r") as f:
            ret = json.load(f)
    ret.append(output)
    os.makedirs(os.path.dirname(OUTPUT_FP), exist_ok=True)
    with open(OUTPUT_FP, "w") as f:
        json.dump(ret, f, indent=4)
    og.shutdown()


if __name__ == "__main__":
    main()
