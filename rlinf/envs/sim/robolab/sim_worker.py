# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Isaac Sim process owned by the RLinf RoboLab environment adapter."""

from __future__ import annotations

import pickle
import socket
import struct
import sys
import traceback

from rlinf.utils.logging import get_logger

logger = get_logger()


def _receive(sock: socket.socket) -> dict:
    header = sock.recv(8, socket.MSG_WAITALL)
    if len(header) != 8:
        raise EOFError
    size = struct.unpack("!Q", header)[0]
    parts = []
    while size:
        part = sock.recv(min(size, 1024 * 1024))
        if not part:
            raise EOFError
        parts.append(part)
        size -= len(part)
    return pickle.loads(b"".join(parts))


def _send(sock: socket.socket, value: dict) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def main() -> None:
    """Serve environment commands until the owning adapter disconnects."""
    import cv2  # noqa: F401 -- RoboLab imports OpenCV before Isaac Lab.
    import numpy as np
    import torch
    from isaaclab.app import AppLauncher

    sock = socket.create_connection(("127.0.0.1", int(sys.argv[1])))
    app = None
    env = None
    try:
        message = _receive(sock)
        if message["command"] != "init":
            raise ValueError("Expected RoboLab simulator init command")
        settings = message["settings"]
        app = AppLauncher(
            headless=True,
            enable_cameras=True,
            device=settings.get("device", "cuda:0"),
            renderer=settings.get("renderer", "realtime"),
            kit_args=settings.get("kit_args", ""),
        ).app

        import omni.kit.app
        import omni.timeline
        import robolab.constants
        from openpi_client import image_tools
        from robolab.constants import set_output_dir
        from robolab.core.environments.runtime import create_env, end_episode
        from robolab.core.logging.results import get_current_subtask_info
        from robolab.core.task.instruction import instruction_text
        from robolab.registrations.droid.auto_env_registrations_jointpos import (
            auto_register_droid_envs,
        )

        enable_subtask = bool(settings.get("enable_subtask", True))
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = enable_subtask
        task = settings["task"]
        auto_register_droid_envs(task=task)
        output_dir = settings["output_dir"]
        set_output_dir(output_dir)
        env, env_cfg = create_env(
            task,
            device=settings.get("device", "cuda:0"),
            seed=settings["seed"],
            num_envs=settings["num_envs"],
            instruction_type=settings.get("instruction_type"),
            policy="rlinf",
            renderer=settings.get("renderer", "realtime"),
            rendering_mode=settings.get("rendering_mode"),
        )
        prompt = instruction_text(env_cfg.instruction)
        timeline = omni.timeline.get_timeline_interface()
        kit_app = omni.kit.app.get_app()
        num_envs = settings["num_envs"]
        max_steps = int(settings["max_episode_steps"])
        steps = np.zeros(num_envs, dtype=np.int32)
        finished = np.zeros(num_envs, dtype=bool)
        success = np.zeros(num_envs, dtype=np.float32)
        score = np.zeros(num_envs, dtype=np.float32)
        last_obs = None
        run_index = -1
        episode_open = False

        def finish_episode() -> None:
            nonlocal episode_open
            if episode_open:
                end_episode(env)
                episode_open = False

        def pack_obs(raw):
            images = raw["image_obs"]
            proprio = raw["proprio_obs"]
            main_images = images["over_shoulder_left_camera"].detach().cpu().numpy()
            wrist_images = images["wrist_cam"].detach().cpu().numpy()
            return {
                "main_images": np.stack(
                    [
                        image_tools.resize_with_pad(image, 224, 224)
                        for image in main_images
                    ]
                ),
                "wrist_images": np.stack(
                    [
                        image_tools.resize_with_pad(image, 224, 224)
                        for image in wrist_images
                    ]
                ),
                "states": np.concatenate(
                    [
                        proprio["arm_joint_pos"].detach().cpu().numpy(),
                        proprio["gripper_pos"].detach().cpu().numpy(),
                    ],
                    axis=-1,
                ).astype(np.float32),
                "task_descriptions": [prompt] * num_envs,
                "active_mask": ~finished.copy(),
            }

        _send(sock, {"ready": True})
        while True:
            message = _receive(sock)
            command = message["command"]
            if command == "close":
                finish_episode()
                break
            if command == "reset":
                finish_episode()
                env.reset_eval_state()
                # RoboLab's evaluator resets twice before taking the first action.
                last_obs, _ = env.reset(seed=message.get("seed"))
                last_obs, _ = env.reset()
                run_index += 1
                if env.recorder_manager is not None:
                    env.recorder_manager.set_hdf5_file(f"run_{run_index}.hdf5")
                    for env_id in range(num_envs):
                        env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])
                episode_open = True
                steps[:] = 0
                finished[:] = False
                success[:] = 0
                score[:] = 0
                _send(sock, pack_obs(last_obs))
                continue
            if command != "chunk_step":
                raise ValueError(f"Unknown RoboLab simulator command: {command}")

            actions = (
                np.frombuffer(message["action_bytes"], dtype=np.float32)
                .reshape(message["action_shape"])
                .copy()
            )
            horizon = actions.shape[1]
            rewards = np.zeros((num_envs, horizon), dtype=np.float32)
            terminations = np.zeros((num_envs, horizon), dtype=bool)
            truncations = np.zeros((num_envs, horizon), dtype=bool)
            for i in range(horizon):
                if finished.all():
                    break
                while not timeline.is_playing():
                    kit_app.update()
                active = ~finished
                action = torch.as_tensor(actions[:, i], device=env.device)
                action[~torch.as_tensor(active, device=env.device)] = 0
                last_obs, _, _, _, _ = env.step(action)
                steps[active] += 1
                native_results = env.get_env_results()
                for j, result in enumerate(native_results):
                    if finished[j]:
                        continue
                    native_success = result["success"]
                    if native_success is None and steps[j] < max_steps:
                        continue
                    finished[j] = True
                    success[j] = float(native_success is True)
                    if enable_subtask:
                        subtask_info = get_current_subtask_info(env, env_id=j)
                        if subtask_info is None or subtask_info.get("score") is None:
                            raise RuntimeError(
                                f"Expected a subtask score for {task}, env {j}, "
                                f"but received {subtask_info}"
                            )
                        score[j] = (
                            1.0
                            if native_success is True
                            else float(subtask_info["score"])
                        )
                    rewards[j, i] = success[j]
                    terminations[j, i] = bool(native_success is True)
                    truncations[j, i] = not terminations[j, i]

            if finished.all():
                finish_episode()
            episode = {
                "success_once": success.copy(),
                "return": success.copy(),
                "episode_len": steps.astype(np.float32),
                "reward": success / np.maximum(steps, 1),
            }
            if enable_subtask:
                episode["score"] = score.copy()
            _send(
                sock,
                {
                    "observation": pack_obs(last_obs),
                    "rewards": rewards,
                    "terminations": terminations,
                    "truncations": truncations,
                    "episode": episode,
                },
            )
    except EOFError:
        pass
    except BaseException:
        try:
            _send(sock, {"error": traceback.format_exc()})
        except OSError:
            logger.exception("Failed to send RoboLab simulator error to the adapter")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                logger.exception("Failed to close the RoboLab environment")
        try:
            if app is not None:
                app.close()
        finally:
            sock.close()


if __name__ == "__main__":
    main()
