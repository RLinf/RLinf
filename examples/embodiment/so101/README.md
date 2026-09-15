# SO-101 inference and DAgger

These examples run an RLinf PI05 checkpoint on SO-101. Choose `local` when the
robot host has a GPU, or `grpc` when a separate GPU host serves the model. Both
backends use the same observation schema, 20-step action chunks, control loop,
and DAgger intervention flow.

## Demonstration collection

Record leader-controlled demonstrations through RLinf's collection pipeline:

```bash
SO101_FOLLOWER_PORT=/dev/ttyACM0 \
SO101_FOLLOWER_ID=rlinf_follower \
SO101_LEADER_PORT=/dev/ttyACM1 \
SO101_LEADER_ID=rlinf_leader \
SO101_CAMERA=/dev/video0 \
examples/embodiment/collect_so101.sh 10
```

The control loop runs at 30 Hz. Finalized episodes use LeRobot format under
`../results/so101-collect/collected_data`, and RLinf trajectories are stored
under `../results/so101-collect/demos`.

## Local inference

Run PI05 in the robot process:

```bash
examples/embodiment/run_so101_inference.sh \
  --inference-backend local \
  --task "Pick up the object and place it in the container." \
  --local-checkpoint /path/to/actor-checkpoint \
  --local-norm-stats /path/to/norm_stats.json \
  --policy-device cuda:0 \
  --serial-port /dev/ttyACM0 --robot-id <robot-id> --camera /dev/video0 \
  --pose-file /path/to/poses.json --allow-hardware-motion
```

The equivalent environment variables are `SO101_INFERENCE_BACKEND`,
`SO101_TASK`, `SO101_SERIAL_PORT`, `SO101_ROBOT_ID`, `SO101_CAMERA`,
`SO101_POSE_FILE`, `SO101_LOCAL_CHECKPOINT`, and `SO101_LOCAL_NORM_STATS`.
The pose JSON contains `joint_order`, `units: "lerobot_native"`,
`standard_position`, and `fold_position`; both positions map the six SO-101
joint names to values recorded for the local rig.

## Remote inference

On the GPU host, create the service configuration and start the server:

```bash
cp examples/embodiment/so101/lerobot_grpc_policy.env.example \
  examples/embodiment/so101/lerobot_grpc_policy.env
# Set the checkpoint, norm-stats, Python, host, and port in the env file.
examples/embodiment/so101/lerobot_grpc_policy_service.sh start
```

Connect the robot host to that endpoint:

```bash
examples/embodiment/run_so101_inference.sh \
  --inference-backend grpc --server-address <inference-host>:50051 \
  --task "Pick up the object and place it in the container." \
  --serial-port /dev/ttyACM0 --robot-id <robot-id> --camera /dev/video0 \
  --pose-file /path/to/poses.json --allow-hardware-motion
```

`SO101_SERVER_ADDRESS` provides the same endpoint without a command-line flag.

## DAgger and SFT

Run DAgger with either inference backend and add the calibrated leader arm. For
example, connect to a GPU server with:

```bash
examples/embodiment/run_so101_dagger.sh \
  --inference-backend grpc --server-address <inference-host>:50051 \
  --task "Pick up the object and place it in the container." \
  --serial-port /dev/ttyACM0 --robot-id <robot-id> --camera /dev/video0 \
  --leader-port /dev/ttyACM1 --leader-id <leader-id> \
  --pose-file /path/to/poses.json --dagger-data-root /path/to/dagger-data \
  --allow-hardware-motion
```

The recorder stores policy, expert, and executed actions in LeRobot format.
Set `SO101_DAGGER_DATA_ROOT` to choose its default output directory.

Set `SO101_DATASET_PATH`, `PI05_BASE_CHECKPOINT`, and
`SO101_NORM_STATS_PATH`, then start SFT:

```bash
examples/embodiment/so101/run_sft.sh
```

For a DAgger training round, set `SO101_DAGGER_DATASET_PATH`,
`SO101_SFT_CHECKPOINT`, `PI05_BASE_CHECKPOINT`, and
`SO101_NORM_STATS_PATH`, then run:

```bash
examples/embodiment/so101/run_dagger_sft.sh
```
