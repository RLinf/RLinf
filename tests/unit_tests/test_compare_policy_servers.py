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

from __future__ import annotations

import io
import json
import threading
import time
from contextlib import contextmanager

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from openpi_client import msgpack_numpy
from PIL import Image
from websockets.sync.server import serve

from toolkits.lerobot.compare_policy_servers import (
    PolicyServerClient,
    action_1_to_10_jump_metrics,
    chunk_jitter_metrics,
    common_prefix_metrics,
    format_observation_layout,
    load_episode_numeric,
    load_observations,
    load_task_prompts,
    padded_action_chunk,
    randomness_metrics,
    replanning_boundary_metrics,
    scan_episodes,
    select_episodes,
    uniform_frame_indices,
)


def _image_bytes(value: int) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.full((4, 5, 3), value, dtype=np.uint8)).save(
        stream, format="PNG"
    )
    return stream.getvalue()


def _write_dataset(root, *, episodes: int = 3) -> None:
    (root / "meta").mkdir(parents=True)
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [14]},
            **{
                key: {"dtype": "image", "shape": [3, 4, 5]}
                for key in (
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist",
                )
            },
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0, 1]),
                "__index_level_0__": pa.array(["pick", "place"]),
            }
        ),
        root / "meta" / "tasks.parquet",
    )
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    for episode in range(episodes):
        frames = 5
        images = [
            {"bytes": _image_bytes(episode * 10 + frame), "path": None}
            for frame in range(frames)
        ]
        table = pa.table(
            {
                "observation.state": pa.array(
                    [np.full(14, frame, dtype=np.float32) for frame in range(frames)]
                ),
                "action": pa.array(
                    [
                        np.full(14, 100 * episode + frame, dtype=np.float32)
                        for frame in range(frames)
                    ]
                ),
                "observation.images.cam_high": pa.array(images, type=image_type),
                "observation.images.cam_left_wrist": pa.array(images, type=image_type),
                "observation.images.cam_right_wrist": pa.array(images, type=image_type),
                "episode_index": pa.array([episode] * frames),
                "frame_index": pa.array(range(frames)),
                "task_index": pa.array([episode % 2] * frames),
            }
        )
        pq.write_table(
            table, data_dir / f"file-{episode:03d}.parquet", row_group_size=2
        )


def test_dataset_selection_image_decode_padding_and_seed(tmp_path) -> None:
    _write_dataset(tmp_path)
    prompts = load_task_prompts(tmp_path)
    refs = scan_episodes(tmp_path, prompts)

    first = select_episodes(
        refs,
        episodes_per_prompt=1,
        prompt_regex=None,
        episode_ids=None,
        seed=7,
    )
    second = select_episodes(
        refs,
        episodes_per_prompt=1,
        prompt_regex=None,
        episode_ids=None,
        seed=7,
    )
    assert [ref.episode_id for ref in first] == [ref.episode_id for ref in second]
    assert {ref.prompt for ref in first} == {"pick", "place"}

    episode = load_episode_numeric(refs[0])
    observations = load_observations(tmp_path, episode, [1, 4])
    assert observations[1]["prompt"] == "pick"
    assert observations[1]["state"].shape == (14,)
    assert observations[1]["images"]["cam_high"].shape == (4, 5, 3)
    assert int(observations[1]["images"]["cam_high"][0, 0, 0]) == 1
    assert format_observation_layout(observations[1], "chw")["images"][
        "cam_high"
    ].shape == (3, 4, 5)
    assert format_observation_layout(observations[1], "hwc")["images"][
        "cam_high"
    ].shape == (4, 5, 3)
    np.testing.assert_array_equal(
        padded_action_chunk(episode.action, 3, 4)[:, 0], [3, 4, 4, 4]
    )
    np.testing.assert_array_equal(uniform_frame_indices(5, 3), [0, 2, 4])


def test_metric_functions_known_arrays() -> None:
    left = np.asarray([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    right = np.asarray([[1.0, 0.0], [1.0, 2.0]])
    metrics = common_prefix_metrics(left, right, np.asarray([1.0, 2.0]))

    assert metrics["common_horizon"] == 2
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["normalized_mae"] == pytest.approx(1.0)
    np.testing.assert_allclose(metrics["per_joint_mae"], [1.0, 2.0])
    np.testing.assert_allclose(metrics["per_horizon_mae"], [1.5, 1.5])

    randomness = randomness_metrics([left, left + 2])
    assert randomness == {"output_std": 1.0, "repeat_to_repeat_mae": 2.0}

    action_steps = np.arange(20, dtype=np.float64)[:, None]
    endpoint_jump = action_1_to_10_jump_metrics([action_steps, 2 * action_steps])
    assert endpoint_jump["num_chunks"] == 2
    assert endpoint_jump["mae"] == pytest.approx(13.5)
    assert endpoint_jump["rmse"] == pytest.approx(np.sqrt((9**2 + 18**2) / 2))
    assert endpoint_jump["max_abs"] == 18.0
    assert endpoint_jump["mean_l2"] == pytest.approx(13.5)

    jitter = chunk_jitter_metrics(
        np.asarray([[1.0, 1.0], [2.0, 3.0], [4.0, 6.0]]),
        np.zeros(2),
        np.asarray([[1.0, 0.0], [2.0, 2.0], [3.0, 4.0]]),
    )
    assert jitter["first_action_state_jump"] == 1.0
    assert jitter["first_difference_mae"] == 2.0
    assert jitter["second_difference_mae"] == 1.0
    assert jitter["ground_truth_mae"] == pytest.approx(5 / 6)

    boundaries = replanning_boundary_metrics(
        [np.asarray([[0.0], [2.0]]), np.asarray([[5.0], [7.0]])]
    )
    assert boundaries == {"boundary_jump_mae": 3.0, "boundary_jump_max": 3.0}


@contextmanager
def _fake_server(handler):
    with serve(handler, "127.0.0.1", 0) as server:
        port = server.socket.getsockname()[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield port
        finally:
            server.shutdown()
            thread.join(timeout=2)


def _protocol_handler(
    websocket,
    *,
    horizon: int = 3,
    response_kind: str = "normal",
    delay: float = 0.0,
    seen: list[dict] | None = None,
) -> None:
    websocket.send(msgpack_numpy.packb({"model": "fake", "horizon": horizon}))
    request = msgpack_numpy.unpackb(websocket.recv())
    if seen is not None:
        seen.append(request)
    if delay:
        time.sleep(delay)
    if response_kind == "text":
        websocket.send("fake traceback")
        return
    actions = np.ones((horizon, 14), dtype=np.float32)
    if response_kind == "nonfinite":
        actions[0, 0] = np.nan
    websocket.send(
        msgpack_numpy.packb({"actions": actions, "server_timing": {"infer_ms": 1.25}})
    )


def test_policy_client_handshake_request_and_different_horizon() -> None:
    seen: list[dict] = []

    def handler(websocket):
        _protocol_handler(websocket, horizon=5, seen=seen)

    with _fake_server(handler) as port:
        with PolicyServerClient("127.0.0.1", port, 2.0, "fake") as client:
            assert client.metadata == {"model": "fake", "horizon": 5}
            observation = {
                "images": {
                    "cam_high": np.zeros((3, 2, 2), dtype=np.uint8),
                    "cam_left_wrist": np.zeros((3, 2, 2), dtype=np.uint8),
                    "cam_right_wrist": np.zeros((3, 2, 2), dtype=np.uint8),
                },
                "state": np.zeros(14, dtype=np.float32),
                "prompt": "pick",
            }
            result = client.infer(observation)

    assert result.actions.shape == (5, 14)
    assert result.server_infer_ms == 1.25
    assert set(seen[0]) == set(observation)
    assert seen[0]["state"].shape == (14,)


@pytest.mark.parametrize(
    ("response_kind", "match"),
    [("text", "text error"), ("nonfinite", "non-finite")],
)
def test_policy_client_reports_server_errors(response_kind, match) -> None:
    def handler(websocket):
        _protocol_handler(websocket, response_kind=response_kind)

    with _fake_server(handler) as port:
        with PolicyServerClient("127.0.0.1", port, 2.0, "fake") as client:
            with pytest.raises(RuntimeError, match=match):
                client.infer({"state": np.zeros(14)})


def test_policy_client_timeout() -> None:
    def handler(websocket):
        _protocol_handler(websocket, delay=0.2)

    with _fake_server(handler) as port:
        with PolicyServerClient("127.0.0.1", port, 0.05, "fake") as client:
            with pytest.raises(TimeoutError, match="Timed out"):
                client.infer({"state": np.zeros(14)})


def test_policy_client_reports_connection_close() -> None:
    def handler(websocket):
        websocket.send(msgpack_numpy.packb({"model": "fake"}))
        websocket.recv()
        websocket.close()

    with _fake_server(handler) as port:
        with PolicyServerClient("127.0.0.1", port, 2.0, "fake") as client:
            with pytest.raises(RuntimeError, match="Connection to fake"):
                client.infer({"state": np.zeros(14)})
