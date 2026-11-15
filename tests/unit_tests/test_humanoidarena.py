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

"""CPU contracts for the upstream HumanoidArena evaluation bridge."""

import base64
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.data.schema.embodied_types import EnvOutput
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.sim.humanoidarena.bridge import ArenaBridge
from rlinf.envs.sim.humanoidarena.config import validate_humanoidarena_cfg
from rlinf.envs.sim.humanoidarena.runtime import RuntimeProcess, request_json
from rlinf.models import get_model


@pytest.fixture
def config():
    path = (
        Path(__file__).resolve().parents[2]
        / "evaluations/humanoidarena/humanoidarena_wbwam_eval.yaml"
    )
    with initialize_config_dir(config_dir=str(path.parent), version_base="1.1"):
        return compose(config_name=path.stem)


def observation():
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    return rgb, {
        "observation": {
            "state": np.arange(64).tolist(),
            "images": {
                "front": {
                    "shape": list(rgb.shape),
                    "dtype": "uint8",
                    "data_b64": base64.b64encode(rgb.tobytes()).decode(),
                }
            },
        }
    }


def test_bridge_preserves_original_reset_order_and_actions():
    bridge = ArenaBridge(2, 5)
    rgb, payload = observation()
    actions = np.arange(80, dtype=np.float32).reshape(1, 2, 40) / 37 - 0.4
    try:
        request_json(bridge.url + "/reset", {"seed": 123})
        request_json(bridge.url + "/reset", {})
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(request_json, bridge.url + "/infer", payload)
            actual_rgb, state, policy_seed = bridge.observations.get(timeout=5)
            assert policy_seed is None
            np.testing.assert_array_equal(actual_rgb, rgb)
            np.testing.assert_array_equal(state, np.arange(64, dtype=np.float32))
            assert not future.done()
            bridge.send_action(actions)
            np.testing.assert_array_equal(future.result()["action_chunk"], actions[0])
    finally:
        bridge.close()
        bridge.close()


@pytest.mark.parametrize("bad", [np.zeros((1, 2, 39)), np.full((1, 2, 40), np.nan)])
def test_actions_reject_invalid_chunks(bad):
    with pytest.raises(ValueError):
        prepare_actions(bad, "humanoidarena", "wbwam", 2, 40)


def test_actions_preserve_native_values():
    actions = np.linspace(-2, 2, 80, dtype=np.float32).reshape(1, 2, 40)
    result = prepare_actions(actions, "humanoidarena", "wbwam", 2, 40)
    np.testing.assert_array_equal(result, actions)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rollout.collect_prev_infos", True),
        ("env.eval.auto_reset", True),
        ("rollout.model.is_lora", True),
        ("env.eval.chunk_size", 14),
        ("env.eval.video_cfg.save_video", True),
        ("env.eval.total_num_envs", 2),
        ("env.eval.rollout_epoch", 21),
        ("env.eval.max_steps_per_rollout_epoch", 1800),
        ("env.eval.native_post_termination_steps", -1),
    ],
)
def test_config_rejects_unsupported_modes(config, field, value):
    OmegaConf.update(config, field, value)
    with pytest.raises(ValueError):
        validate_humanoidarena_cfg(config, config.rollout.model, True)


def test_config_and_model_registration(config):
    validate_humanoidarena_cfg(config, config.rollout.model, True)
    with pytest.raises(ValueError, match="standalone evaluation"):
        validate_humanoidarena_cfg(config, config.rollout.model, False)
    policy = get_model(config.rollout.model)
    with pytest.raises(NotImplementedError, match="evaluation only"):
        policy.default_forward()
    assert get_env_cls("humanoidarena").__name__ == "HumanoidArenaEnv"
    policy.close()


@pytest.mark.parametrize(
    "task,max_steps,chunk_size,denoise_steps",
    [
        ("kick_football", 2000, 20, 20),
        ("sit_sofa", 2000, 20, 20),
        ("obstacle_navigation", 1800, 14, 40),
        ("hammer", 2000, 14, 40),
        ("box_shelf", 1450, 20, 40),
        ("punch_markers", 900, 20, 40),
        ("open_door", 1800, 20, 40),
    ],
)
def test_task_presets_preserve_native_protocol(
    task, max_steps, chunk_size, denoise_steps
):
    directory = Path(__file__).resolve().parents[2] / "evaluations/humanoidarena"
    with initialize_config_dir(config_dir=str(directory), version_base="1.1"):
        cfg = compose(
            config_name="humanoidarena_wbwam_eval", overrides=[f"task={task}"]
        )
    validate_humanoidarena_cfg(cfg, cfg.rollout.model, True)
    assert cfg.env.eval.max_episode_steps == max_steps
    assert cfg.env.eval.chunk_size == cfg.rollout.model.num_action_chunks == chunk_size
    assert cfg.rollout.model.wbwam.denoise_steps == denoise_steps
    assert cfg.env.eval.prompt == cfg.rollout.model.wbwam.prompt


def test_e2e_config_reuses_evaluation_recipe(monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(repo))
    with initialize_config_dir(
        config_dir=str(repo / "tests/e2e_tests/evaluations"), version_base="1.1"
    ):
        cfg = compose(config_name="humanoidarena_wbwam_eval")
    validate_humanoidarena_cfg(cfg, cfg.rollout.model, True)
    assert cfg.env.eval.max_episode_steps == 40
    assert cfg.env.eval.max_steps_per_rollout_epoch == 40
    assert not cfg.env.eval.native_record_video
    assert cfg.env.eval.native_post_termination_steps == 0
    assert cfg.rollout.model.wbwam.weight_file == "step_015550.pt"


def test_runtime_lease_closes_child(tmp_path):
    script = tmp_path / "child.py"
    marker = tmp_path / "pid"
    script.write_text(
        "import os,time,pathlib,sys\npathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(60)\n"
    )
    process = RuntimeProcess(
        [sys.executable, str(script), str(marker)],
        tmp_path / "child.log",
        cwd=tmp_path,
        env=dict(os.environ),
    )
    try:
        for _ in range(100):
            if marker.exists():
                break
            process.check()
            time.sleep(0.05)
        pid = int(marker.read_text())
        # Closing the pipe also models an owner process disappearing.
        process.process.stdin.close()
        process.process.wait(timeout=10)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        process.close()
        process.close()


@pytest.mark.parametrize("outcome", ["success", "fall", "timeout", "sim_error"])
def test_env_uses_upstream_result(config, tmp_path, outcome):
    repo = tmp_path / "vendor"
    benchmark = repo / "benchmark/humanoidarena"
    benchmark.mkdir(parents=True)
    arena = tmp_path / "arena"
    (arena / "isaaclab_twist2_g1").mkdir(parents=True)
    script = benchmark / "fake_sim.py"
    _, payload = observation()
    script.write_text(
        "import sys,json,urllib.request,pathlib\n"
        "a=sys.argv\n"
        "def arg(k): return a[a.index(k)+1]\n"
        "url=arg('--lerobot_server_url')\n"
        "def post(path,data):\n"
        " req=urllib.request.Request(url+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})\n"
        " return json.load(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=20))\n"
        "specs=json.loads(pathlib.Path(arg('--episode_batch_json')).read_text())['episodes']\n"
        "for spec in specs:\n"
        " post('/reset',{'seed':spec['episode_seed']})\n"
        " post('/reset',{})\n"
        + f" r=post('/infer',{payload!r})\n"
        + " assert len(r['action_chunk'])==20 and len(r['action_chunk'][0])==40\n"
        + f" result={{'success':{outcome == 'success'!r},'failure_reason':{outcome!r},'episode_steps':7}}\n"
        + " pathlib.Path(spec['result_json']).write_text(json.dumps(result))\n"
    )
    (benchmark / "isaac_native_python.sh").write_text(
        'exec "$HA_SIMULATION_PYTHON" "' + str(script) + '" "$@"\n'
    )
    env_cfg = config.env.eval
    env_cfg.wbwam_repo = str(repo)
    env_cfg.arena_root = str(arena)
    env_cfg.simulation_python = sys.executable
    env_cfg.output_dir = str(tmp_path / "results")
    env_cfg.timeout = 20
    env_cfg.rollout_epoch = 2
    env = get_env_cls("humanoidarena")(env_cfg, 1, 0, 1, None)
    try:
        simulator_pid = None
        for repeat in range(2):
            obs, _ = env.reset()
            assert EnvOutput(obs=obs).to_dict()["obs"]["reset_mask"].tolist() == [True]
            if simulator_pid is None:
                simulator_pid = env._process.process.pid
            assert env._process.process.pid == simulator_pid
            if outcome == "sim_error":
                with pytest.raises(RuntimeError, match="Arena episode failed"):
                    env.chunk_step(np.zeros((1, 20, 40)))
                break
            _, rewards, terminated, truncated, infos = env.chunk_step(
                np.zeros((1, 20, 40))
            )
            assert bool(terminated[0, -1]) == (outcome != "timeout")
            assert bool(truncated[0, -1]) == (outcome == "timeout")
            assert rewards.sum().item() == float(outcome == "success")
            assert infos[-1]["episode"]["episode_len"].item() == 7
            assert (env._process is None) == (repeat == 1)
        if outcome != "sim_error":
            with pytest.raises(RuntimeError, match="exceeds configured"):
                env.reset()
    finally:
        env.close()


def test_policy_forwards_observation_and_reset(config, tmp_path):
    repo = tmp_path / "vendor"
    benchmark = repo / "benchmark/humanoidarena"
    benchmark.mkdir(parents=True)
    script = benchmark / "server.py"
    script.write_text("""import json,sys,pathlib
from http.server import BaseHTTPRequestHandler,HTTPServer
a=sys.argv
def arg(k): return a[a.index(k)+1]
out=pathlib.Path(arg('--output'))
health={'state_dim':64,'action_dim':40,'action_source_hz':50,'execute_chunk_steps':20,'num_inference_steps':40,'inference_mode':'idm','action_format':'semantic_v3','instruction_prompt':'Open the door.'}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def reply(self,value):
  b=json.dumps(value).encode();self.send_response(200);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def do_GET(self): self.reply(health)
 def do_POST(self):
  data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  with (out/'requests.jsonl').open('a') as f: f.write(json.dumps([self.path,data])+'\\n')
  self.reply({'status':'ok'} if self.path=='/reset' else {'action_chunk':[[0.25]*38+[-0.2,1.2]]*20})
server=HTTPServer(('127.0.0.1',0),Handler)
(out/'server_health.json').write_text(json.dumps({'port':server.server_port}))
server.serve_forever()
""")
    model_cfg = config.rollout.model
    model_cfg.wbwam.repo_root = str(repo)
    model_cfg.wbwam.python = sys.executable
    model_cfg.wbwam.output_dir = str(tmp_path / "output")
    model_cfg.wbwam.timeout = 20
    policy = get_model(model_cfg)
    rgb, _ = observation()
    obs = {
        "states": np.zeros((1, 64), np.float32),
        "main_images": rgb[None],
        "task_descriptions": ["Open the door."],
        "reset_mask": np.array([True]),
    }
    try:
        actions, metadata = policy.predict_action_batch(
            EnvOutput(obs=obs).to_dict()["obs"]
        )
        assert actions.shape == (1, 20, 40)
        np.testing.assert_allclose(actions[0, 0, -2:], [-0.2, 1.2])
        assert metadata == {}
        obs["reset_mask"] = np.array([False])
        policy.predict_action_batch(EnvOutput(obs=obs).to_dict()["obs"])
        obs["reset_mask"] = np.array([True])
        policy.predict_action_batch(EnvOutput(obs=obs).to_dict()["obs"])
        records = [
            json.loads(line)
            for line in next((tmp_path / "output").glob("*/requests.jsonl"))
            .read_text()
            .splitlines()
        ]
        assert [item[0] for item in records] == [
            "/reset",
            "/infer",
            "/infer",
            "/reset",
            "/infer",
        ]
        assert records[0][1] == {}
        assert records[3][1] == {}
        assert records[1][1]["observation"]["state"] == [0.0] * 64
    finally:
        policy.close()
        policy.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Isaac runtime requires Linux")
def test_runtime_survives_supervisor_kill_without_orphan(tmp_path):
    import signal

    marker = tmp_path / "pid"
    script = tmp_path / "child.py"
    script.write_text(
        "import os,time,pathlib,sys\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    process = RuntimeProcess(
        [sys.executable, str(script), str(marker)],
        tmp_path / "child.log",
        cwd=tmp_path,
        env=dict(os.environ),
    )
    try:
        for _ in range(100):
            if marker.exists():
                break
            process.check()
            time.sleep(0.05)
        pid = int(marker.read_text())
        os.kill(process.process.pid, signal.SIGKILL)
        process.process.wait(timeout=5)
        for _ in range(100):
            status = Path(f"/proc/{pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(0.05)
        else:
            pytest.fail("Runtime stayed alive after its supervisor was killed")
    finally:
        process.close()
