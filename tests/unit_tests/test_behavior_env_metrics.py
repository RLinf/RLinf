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
"""Unit tests for BehaviorEnv._extract_info_done (pure Python, no OmniGibson).

``_extract_info_done`` is the terminal branch of the info->done extraction path.
It must safely handle reset infos (which carry no ``"done"`` key) as well as the
many shapes the ``"done"`` value can take on the step path. Regression coverage
for the PR3 reset-path crash (KeyError: 'done').
"""

import pytest
import torch


def _extract_info_done(info):
    from rlinf.envs.behavior.behavior_env import BehaviorEnv

    return BehaviorEnv._extract_info_done(info)


class TestExtractInfoDone:
    # -- reset-like infos: must return False, never raise ------------------

    def test_empty_info(self):
        assert _extract_info_done({}) is False

    def test_info_without_done_key(self):
        assert _extract_info_done({"reward": {"task_specific": {}}}) is False

    def test_none_info(self):
        assert _extract_info_done(None) is False

    def test_non_dict_info(self):
        assert _extract_info_done("not-a-dict") is False
        assert _extract_info_done(42) is False

    # -- done as a bare bool ----------------------------------------------

    def test_done_true_bool(self):
        assert _extract_info_done({"done": True}) is True

    def test_done_false_bool(self):
        assert _extract_info_done({"done": False}) is False

    # -- done.success short-circuit ---------------------------------------

    def test_done_success_true(self):
        assert _extract_info_done({"done": {"success": True}}) is True

    def test_done_success_false_no_conditions(self):
        assert _extract_info_done({"done": {"success": False}}) is False

    # -- termination_conditions -------------------------------------------

    def test_termination_condition_done_true(self):
        info = {
            "done": {
                "success": False,
                "termination_conditions": {"collision": {"done": True}},
            }
        }
        assert _extract_info_done(info) is True

    def test_termination_conditions_all_false(self):
        info = {
            "done": {
                "success": False,
                "termination_conditions": {
                    "collision": {"done": False},
                    "timeout": {"done": False},
                },
            }
        }
        assert _extract_info_done(info) is False

    def test_mixed_conditions_any_true(self):
        info = {
            "done": {
                "success": False,
                "termination_conditions": {
                    "collision": {"done": False},
                    "goal_reached": {"done": True},
                },
            }
        }
        assert _extract_info_done(info) is True

    # -- malformed / non-dict done: must return False, never raise ---------

    @pytest.mark.parametrize(
        "bad_done",
        [None, "done", 42, 0, 3.14, [True], ("done",), {"unexpected_shape"}],
    )
    def test_non_dict_done_returns_false(self, bad_done):
        assert _extract_info_done({"done": bad_done}) is False

    def test_non_dict_termination_conditions_returns_false(self):
        info = {"done": {"success": False, "termination_conditions": "oops"}}
        assert _extract_info_done(info) is False

    def test_non_dict_termination_condition_values_ignored(self):
        info = {
            "done": {
                "success": False,
                "termination_conditions": {
                    "a": "not-a-dict",
                    "b": 123,
                    "c": None,
                },
            }
        }
        assert _extract_info_done(info) is False


def test_replay_metrics_keep_metadata_rows_aligned(monkeypatch):
    from rlinf.envs.behavior.behavior_env import BehaviorEnv

    monkeypatch.setattr(BehaviorEnv, "device", property(lambda _self: "cpu"))
    env = BehaviorEnv.__new__(BehaviorEnv)
    env.record_metrics = True
    env.success_stage_idx = None
    env.returns = torch.zeros(2)
    env.success_once = torch.zeros(2, dtype=torch.bool)
    env.prev_step_reward = torch.zeros(2)

    infos = [
        {"replay_init": {"source_instance_id": 7}},
        {},
    ]
    result = env._record_metrics(torch.zeros(2), infos)

    assert result["replay_init"]["source_instance_id"].tolist() == [7, 0]
