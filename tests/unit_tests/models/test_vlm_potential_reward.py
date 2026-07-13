from types import SimpleNamespace

import torch

from rlinf.models.embodiment.reward.vlm_reward_model import (
    HistoryVLMRewardModel,
    VLMRewardModel,
)


class _HiddenModel:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.hidden = hidden

    def __call__(self, **_kwargs):
        return SimpleNamespace(hidden_states=[self.hidden])


class _IdentityScalarHead:
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return features.squeeze(-1)


def test_compute_scalar_potential_uses_last_nonpadding_token() -> None:
    model = object.__new__(VLMRewardModel)
    hidden = torch.zeros(2, 4, 1)
    hidden[0, 1, 0] = -2.0
    hidden[1, 3, 0] = 2.0
    model._model = _HiddenModel(hidden)
    model.scalar_head = _IdentityScalarHead()

    potentials = model.compute_scalar_potential(
        {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.long
            ),
        }
    )

    torch.testing.assert_close(potentials, torch.sigmoid(torch.tensor([-2.0, 2.0])))


def _potential_model(
    scale: float = 1.0,
    gamma: float = 1.0,
    ema_alpha: float = 1.0,
    clip: float = 0.0,
):
    model = object.__new__(HistoryVLMRewardModel)
    model.potential_scale = scale
    model.potential_gamma = gamma
    model.potential_ema_alpha = ema_alpha
    model.potential_clip = clip
    model._previous_potentials = None
    return model


def test_potential_difference_is_zero_on_first_and_static_observation() -> None:
    model = _potential_model()
    valid = torch.tensor([True, True])

    first = model.potential_differences(torch.tensor([0.2, 0.8]), valid)
    static = model.potential_differences(torch.tensor([0.2, 0.8]), valid)

    torch.testing.assert_close(first, torch.zeros(2))
    torch.testing.assert_close(static, torch.zeros(2))


def test_potential_difference_is_signed_and_scaled() -> None:
    model = _potential_model(scale=0.5)
    valid = torch.tensor([True, True])
    model.potential_differences(torch.tensor([0.2, 0.8]), valid)

    rewards = model.potential_differences(torch.tensor([0.6, 0.5]), valid)

    torch.testing.assert_close(rewards, torch.tensor([0.2, -0.15]))


def test_done_resets_potential_history() -> None:
    model = _potential_model()
    valid = torch.tensor([True, True])
    model.potential_differences(torch.tensor([0.2, 0.8]), valid)
    terminal = model.potential_differences(
        torch.tensor([0.9, 0.7]), valid, dones=torch.tensor([True, False])
    )
    next_episode = model.potential_differences(torch.tensor([0.1, 0.6]), valid)

    torch.testing.assert_close(terminal, torch.tensor([0.7, -0.1]))
    torch.testing.assert_close(next_episode, torch.tensor([0.0, -0.1]))


def test_potential_difference_applies_ema_and_clip() -> None:
    model = _potential_model(scale=2.0, ema_alpha=0.5, clip=0.25)
    valid = torch.tensor([True, True])
    model.potential_differences(torch.tensor([0.2, 0.8]), valid)

    rewards = model.potential_differences(torch.tensor([0.8, 0.0]), valid)

    torch.testing.assert_close(rewards, torch.tensor([0.25, -0.25]))
    torch.testing.assert_close(model._previous_potentials, torch.tensor([0.5, 0.4]))


def test_model_success_bonus_is_one_shot_and_resets_on_done() -> None:
    model = _potential_model()
    model.success_threshold = 0.8
    model.success_bonus = 1.0
    model.success_confirmation_windows = 1
    model._success_fired = None
    model._success_streak = None
    valid = torch.tensor([True, True])

    first = model.apply_model_success_bonus(
        torch.zeros(2), torch.tensor([0.9, 0.7]), valid
    )
    repeated = model.apply_model_success_bonus(
        torch.zeros(2), torch.tensor([0.95, 0.9]), valid
    )
    terminal = model.apply_model_success_bonus(
        torch.zeros(2),
        torch.tensor([0.95, 0.9]),
        valid,
        dones=torch.tensor([True, False]),
    )
    next_episode = model.apply_model_success_bonus(
        torch.zeros(2), torch.tensor([0.9, 0.9]), valid
    )

    torch.testing.assert_close(first, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(repeated, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(terminal, torch.zeros(2))
    torch.testing.assert_close(next_episode, torch.tensor([1.0, 0.0]))


def test_model_success_bonus_requires_consecutive_confirmations() -> None:
    model = _potential_model()
    model.success_threshold = 0.5
    model.success_bonus = 1.0
    model.success_confirmation_windows = 2
    model._success_fired = None
    model._success_streak = None
    valid = torch.tensor([True])

    first = model.apply_model_success_bonus(torch.zeros(1), torch.tensor([0.9]), valid)
    interrupted = model.apply_model_success_bonus(
        torch.zeros(1), torch.tensor([0.1]), valid
    )
    restart = model.apply_model_success_bonus(
        torch.zeros(1), torch.tensor([0.9]), valid
    )
    confirmed = model.apply_model_success_bonus(
        torch.zeros(1), torch.tensor([0.9]), valid
    )

    torch.testing.assert_close(first, torch.zeros(1))
    torch.testing.assert_close(interrupted, torch.zeros(1))
    torch.testing.assert_close(restart, torch.zeros(1))
    torch.testing.assert_close(confirmed, torch.ones(1))
