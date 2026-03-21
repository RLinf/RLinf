# Third-Party: YAM Robot Environment Setup

This directory contains the full `i2rt` and `yam_realtime` packages from the YAM
repository, bundled into RLinf so the robot machine can be set up without cloning
the separate YAM repo.

## Quick Start (Robot Machine)

### 1. Install RLinf with YAM dependencies

The YAM packages (`i2rt` and `yam_realtime`) are included as default
dependencies of RLinf. A standard install picks them up automatically:

```bash
# With uv (recommended)
uv venv && source .venv/bin/activate
UV_TORCH_BACKEND=auto uv sync

# Or with pip
pip install -e .
```

If you only want YAM deps installed individually (e.g. on a lightweight robot
controller that doesn't need the full RLinf stack):

```bash
pip install -e third_party/i2rt
pip install -e third_party/yam_realtime
```

### 2. Verify the installation

```bash
python -c "from yam_realtime.envs.robot_env import RobotEnv; print('OK')"
python -c "from i2rt.robots.robot import Robot, RobotType; print('OK')"
```

### 3. Run the robot server

On the robot machine, start the robot server that RLinf env workers connect to:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --use-follower-servers
```

For testing without real hardware, use dummy mode:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --no-tunnel \
    --dummy
```

### 4. Training (GPU machine)

On the GPU machine (head node), make sure Ray is running and launch training:

```bash
ray start --head
python examples/embodiment/train_embodied_agent.py --config-name <your_yam_config>
```

The training script connects to the robot server via gRPC — the robot machine
and GPU machine don't need to be the same host.

## Architecture

```
GPU Node (RLinf)                    Robot Node
├── Actor Worker                    ├── robot_server.py
├── Rollout Worker                  │   └── YAMEnv (gym.Env)
└── Env Worker ──── gRPC ──────────►│       └── yam_realtime.RobotEnv
                                    │           ├── Robot (left/right arms via i2rt)
                                    │           └── CameraDriver (cameras)
                                    └── i2rt (CAN motor drivers, kinematics)
```

## What's included

### `third_party/i2rt/`
Full i2rt robot control library:
- `robots/robot.py` — `Robot` protocol, `RobotType` enum
- `robots/motor_chain_robot.py` — hardware robot implementation (CAN bus)
- `robots/utils.py` — `JointMapper`, `GripperType`, `GripperForceLimiter`
- `motor_drivers/` — CAN bus interface (`DMChainCanInterface`)
- `robot_models/` — MJCF/XML model files for YAM arms

### `third_party/yam_realtime/`
Full yam_realtime control stack:
- `envs/robot_env.py` — `RobotEnv`: dm_env wrapper over robots + cameras
- `envs/configs/` — YAML config loading and Hydra-like `instantiate()`
- `robots/` — `ROBOT_PROTOCOL_METHODS`, `PrintRobot`, `ConcatenatedRobot`
- `sensors/cameras/` — `CameraDriver` protocol, `CameraNode`, `OpencvCamera`, `DummyCamera`
- `utils/portal_utils.py` — RPC via `portal`: `RemoteServer`, `Client`, `return_futures`
- `utils/launch_utils.py` — `initialize_robots`, `initialize_sensors`
- `agents/` — teleoperation and policy agents
- `scripts/` — CAN reset and launch scripts

## Troubleshooting

**`ImportError: No module named 'portal'`**
Install the portal RPC library: `pip install portal`

**`ImportError: No module named 'dm_env'`**
Install dm-env: `pip install dm-env>=1.6`

**`ModuleNotFoundError: No module named 'can'`**
Install python-can for CAN bus communication: `pip install python-can`
This is only needed on the robot machine with physical hardware.
