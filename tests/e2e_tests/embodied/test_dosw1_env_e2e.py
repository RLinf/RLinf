"""E2E tests for the DOSW1 dual-arm environment on real hardware.

These tests connect to a real DOS-W1 robot with real cameras and keyboard.
The only simulated component is the inference service (policy actions).

Run with::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    python -m pytest \
        tests/e2e_tests/embodied/test_dosw1_env_e2e.py \
        -m interactive -v -s \
        --robot-url=<URL> \
        --camera-serials=<SN1>,<SN2>,<SN3>
"""

from __future__ import annotations

import numpy as np
import pytest

from rlinf.envs.realworld.dosw1.dosw1_env import (
    _ACTION_DIM,
    _IMAGE_H,
    _IMAGE_W,
    _NUM_JOINTS,
    ControlMode,
    DOSW1Config,
    DOSW1Env,
)
from rlinf.envs.realworld.dosw1.dosw1_sdk import DOSW1SDKAdapter

_PERTURB_AMP = 0.03
_GRIPPER_AMP = 0.005


class FakeInferenceService:
    """Generate safe sinusoidal actions around the current joint state."""

    def __init__(self, freq: float = 0.5) -> None:
        self._freq = freq

    def predict(self, obs: dict, step: int) -> np.ndarray:
        """Return a small oscillatory action that is safe on hardware."""
        t = step / 30.0
        state = obs["state"]

        left_joint = np.asarray(state["left_joint_positions"], dtype=np.float64)
        left_gripper = float(state["left_gripper"][0])
        right_joint = np.asarray(state["right_joint_positions"], dtype=np.float64)
        right_gripper = float(state["right_gripper"][0])

        phase = 2.0 * np.pi * self._freq * t
        delta = _PERTURB_AMP * np.sin(phase + np.arange(_NUM_JOINTS) * 0.5)
        gripper_delta = _GRIPPER_AMP * (0.5 + 0.5 * np.sin(phase))

        action = np.empty(_ACTION_DIM, dtype=np.float64)
        action[:6] = left_joint + delta
        action[6] = left_gripper + gripper_delta
        action[7:13] = right_joint + delta
        action[13] = right_gripper + gripper_delta
        return action


_MODE_NAMES = {
    ControlMode.MODEL: "MODEL ",
    ControlMode.PAUSE: "PAUSE ",
    ControlMode.TELEOP: "TELEOP",
}


def _fmt_joints(arr: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:+.3f}" for value in arr) + "]"


def _print_step(
    step: int,
    mode: ControlMode,
    obs: dict,
    reward: float,
    *,
    leader_left: np.ndarray | None = None,
    leader_right: np.ndarray | None = None,
) -> None:
    state = obs["state"]
    left_joint = state["left_joint_positions"]
    left_gripper = float(state["left_gripper"][0])
    right_joint = state["right_joint_positions"]
    right_gripper = float(state["right_gripper"][0])
    line = (
        f"  [{step:4d}] {_MODE_NAMES.get(mode, '???   ')}  "
        f"L:{_fmt_joints(left_joint)} g={left_gripper:.4f}  "
        f"R:{_fmt_joints(right_joint)} g={right_gripper:.4f}  "
        f"r={reward:+.4f}"
    )
    if leader_left is not None:
        line += (
            "\n         "
            f"leader L:{_fmt_joints(leader_left[:6])} g={float(leader_left[6]):.4f}"
        )
    if leader_right is not None:
        line += f"  R:{_fmt_joints(leader_right[:6])} g={float(leader_right[6]):.4f}"
    print(line)


def _parse_cli(request: pytest.FixtureRequest) -> tuple[str, list[str] | None]:
    url = request.config.getoption("--robot-url")
    serials_raw = request.config.getoption("--camera-serials")
    serials = [serial.strip() for serial in serials_raw.split(",") if serial.strip()]
    return url, serials or None


def _reset_no_block(env: DOSW1Env) -> tuple[dict, dict]:
    """Reset the env without blocking on FreeTeleop."""
    saved = env.config.enable_human_in_loop
    env.config.enable_human_in_loop = False
    obs, info = env.reset()
    env.config.enable_human_in_loop = saved
    return obs, info


def _make_env(cfg: DOSW1Config) -> DOSW1Env:
    return DOSW1Env(
        config=cfg,
        worker_info=None,
        hardware_info=None,
        env_idx=0,
    )


@pytest.fixture(scope="module")
def shared_env(request: pytest.FixtureRequest):
    """Create one module-scoped real DOS-W1 env for all interactive tests."""
    robot_url, serials = _parse_cli(request)
    cfg = DOSW1Config(
        is_dummy=False,
        robot_url=robot_url,
        camera_serials=serials,
        enable_camera_player=True,
        enable_human_in_loop=True,
        max_num_steps=200,
    )
    env = _make_env(cfg)
    yield env
    env.close()


@pytest.mark.interactive
class TestEpisodeWithFakeInference:
    def test_full_episode(self, shared_env: DOSW1Env) -> None:
        """Run a full episode in MODEL mode with fake policy actions."""
        env = shared_env
        inference = FakeInferenceService()

        print("\n" + "=" * 72)
        print("  TEST: Full episode with FakeInferenceService (automated, no keyboard)")
        print("=" * 72)

        obs, info = _reset_no_block(env)
        del info
        self._check_obs(obs, env)

        step = 0
        truncated = False
        while not truncated:
            action = inference.predict(obs, step)
            obs, reward, terminated, truncated, info = env.step(action)
            del terminated
            mode = ControlMode(info["control_mode"])
            _print_step(step, mode, obs, reward)
            step += 1

            self._check_obs(obs, env)
            assert isinstance(reward, (int, float))
            assert "control_mode" in info

        print(f"\n  Episode done after {step} steps (truncated={truncated}).")
        assert step == env.config.max_num_steps

    @staticmethod
    def _check_obs(obs: dict, env: DOSW1Env) -> None:
        assert "state" in obs and "frames" in obs
        state = obs["state"]
        assert state["left_joint_positions"].shape == (_NUM_JOINTS,)
        assert state["left_gripper"].shape == (1,)
        assert state["right_joint_positions"].shape == (_NUM_JOINTS,)
        assert state["right_gripper"].shape == (1,)
        for name in env._effective_camera_names():
            assert name in obs["frames"]
            assert obs["frames"][name].shape == (_IMAGE_H, _IMAGE_W, 3)


@pytest.mark.interactive
class TestTeleopLeaderSignal:
    def test_leader_follower_tracking(self, shared_env: DOSW1Env) -> None:
        """Verify leader arm signals are readable in TELEOP mode."""
        env = shared_env
        sdk: DOSW1SDKAdapter = env._sdk
        assert sdk is not None, "SDK must be connected for this test"

        print("\n" + "=" * 72)
        print("  TEST: TELEOP leader-arm signal verification")
        print("  Press 'p' to PAUSE, then 't' to enter TELEOP")
        print("  Move leader arms and observe tracking below")
        print("  Press 'p' to go back to PAUSE, 'r' to end episode")
        print("=" * 72)

        obs, _ = _reset_no_block(env)
        inference = FakeInferenceService()

        step = 0
        done = False
        while not done:
            action = inference.predict(obs, step)
            obs, reward, terminated, truncated, info = env.step(action)
            mode = ControlMode(info["control_mode"])

            leader_left = leader_right = None
            if mode == ControlMode.TELEOP:
                leader_left = sdk.get_left_lead_joint()
                leader_right = sdk.get_right_lead_joint()

            _print_step(
                step,
                mode,
                obs,
                reward,
                leader_left=leader_left,
                leader_right=leader_right,
            )
            step += 1
            done = truncated or terminated

        print(f"\n  Ended after {step} steps.")


@pytest.mark.interactive
class TestInteractiveFourModes:
    def test_mode_switching_loop(self, shared_env: DOSW1Env) -> None:
        """Interactive test for all four control modes."""
        env = shared_env
        sdk: DOSW1SDKAdapter = env._sdk
        inference = FakeInferenceService()

        max_episodes = 3
        print("\n" + "=" * 72)
        print("  TEST: Interactive four-mode switching")
        print(f"  Will run up to {max_episodes} episodes.")
        print("  Keyboard: s=start, p=pause, t=teleop, m=model, r=free-teleop")
        print("  Press Ctrl-C to abort early.")
        print("=" * 72)

        for episode in range(max_episodes):
            print(f"\n--- Episode {episode + 1}/{max_episodes} ---")
            print("  FreeTeleop active: move leader arms freely, press 's' to start.")

            env.config.enable_human_in_loop = True
            obs, _ = env.reset()
            step = 0
            done = False
            seen_modes: set[int] = set()

            while not done:
                action = inference.predict(obs, step)
                obs, reward, terminated, truncated, info = env.step(action)
                mode = ControlMode(info["control_mode"])
                seen_modes.add(mode.value)

                leader_left = leader_right = None
                if mode == ControlMode.TELEOP and sdk is not None:
                    leader_left = sdk.get_left_lead_joint()
                    leader_right = sdk.get_right_lead_joint()

                _print_step(
                    step,
                    mode,
                    obs,
                    reward,
                    leader_left=leader_left,
                    leader_right=leader_right,
                )
                step += 1
                done = truncated or terminated

            reason = (
                "truncated (max steps)"
                if step >= env.config.max_num_steps
                else "truncated (FreeTeleop 'r')"
            )
            print(f"\n  Episode {episode + 1} ended: {reason}, {step} steps.")
            print(f"  Modes seen: {sorted(ControlMode(mode).name for mode in seen_modes)}")

        print("\n  All episodes complete.")
