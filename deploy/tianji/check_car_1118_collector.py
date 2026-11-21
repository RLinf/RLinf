"""Read target readiness only, without starting or commanding the collector."""

import os

os.environ["RLINF_TIANJI_STAGE1_ONLY"] = "1"
from preflight_tianji_local import live_status
from rlinf.envs.realworld.tianji.tianji_backend import (
    LoopInfraStage1ObservationReader,
)
from rlinf.envs.realworld.tianji.tianji_contract import contract_from_env
from rlinf.envs.realworld.tianji.tianji_joint import joint_state
from wait_tianji_collector import require_rt_schedule


def main():
    url = os.environ["LOOPINFRA_URL"]
    report = live_status(url, joint=True)
    require_rt_schedule(url)
    if report["policy_stream"] != "loopinfra.policy_reference_stream.v1":
        raise RuntimeError("Collector cannot stream the joint reference window")
    if report["feedback_gap_policy"] != "bounded_pause_100ms_v2":
        raise RuntimeError("Collector lacks the verified feedback-gap policy")
    expected = os.environ.get("TIANJI_CAR_POSE_CONFIG_ID")
    if not expected or report["pose_config_id"] != expected:
        raise RuntimeError(
            f"Verify the local car reset pose and set TIANJI_CAR_POSE_CONFIG_ID "
            f"in site.env. Active collector pose_config_id={report['pose_config_id']!r}"
        )
    reader = LoopInfraStage1ObservationReader(
        url, os.environ.get("LOOPINFRA_TOKEN"), 3.0, contract=contract_from_env()
    )
    observation = reader.observe()
    if set(observation.frames) != {"head", "wrist_A", "wrist_B"}:
        raise RuntimeError(f"Unexpected live camera roles: {sorted(observation.frames)}")
    joint = joint_state(observation.raw, observation.state)
    if joint.shape != (16,):
        raise RuntimeError("Collector joint/gripper observation is not 16-dimensional")
    reset = (observation.status.get("robot_control") or {}).get("reset_pose") or {}
    expected_reset = os.environ.get("TIANJI_CAR_RESET_POSE_ID")
    if not expected_reset or reset.get("id") != expected_reset:
        raise RuntimeError(
            f"Set TIANJI_CAR_RESET_POSE_ID to the approved car reset pose. "
            f"Active collector reset pose ID={reset.get('id')!r}"
        )
    print(
        f"Collector ready, verified joint schedule, pose {expected}, reset {expected_reset}"
    )


if __name__ == "__main__":
    main()
