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

"""Assignment of envs to env worker ranks.

An env mode config (``env.train`` or ``env.eval``) is homogeneous, one env
config that every env worker rank hosts, or heterogeneous, several named env
configs under ``envs``. Every env worker rank hosts exactly one env config.
Each env config is described by an :class:`EnvSpec`, which every stage reads
its per-env settings from; a homogeneous mode has a single spec.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

from omegaconf import DictConfig, OmegaConf

from rlinf.envs.policy_obs import PolicyObsSpec


def _scope_metrics(name: str | None, metrics: dict[str, Any]) -> dict[str, Any]:
    if name is None:
        return dict(metrics)
    return {f"{name}/{key}": value for key, value in metrics.items()}


@dataclass(frozen=True)
class EnvSpec:
    """One env config of a mode and the env worker ranks that host it.

    Attributes:
        name: Key of the env under ``envs``, or ``None`` in a homogeneous mode.
        cfg: Config of the env.
        ranks: Env worker ranks hosting the env.
    """

    name: str | None
    cfg: DictConfig
    ranks: range

    # Algorithm fields that ``algorithm.envs.<name>`` may set for one env.
    ALGORITHM_FIELDS: ClassVar[tuple[str, ...]] = (
        "loss_weight",
        "reward_scale",
        "gamma",
        "gae_lambda",
        "rewards_lower_bound",
        "rewards_upper_bound",
    )
    # Policy fields an env config may set for the policy acting in it. Rollout
    # passes them to the policy with that env's observations, so envs that set
    # different values never share a forward pass.
    POLICY_FIELDS: ClassVar[tuple[str, ...]] = ("unnorm_key", "openpi_io")

    @property
    def policy_kwargs(self) -> dict[str, Any]:
        """Return the :attr:`POLICY_FIELDS` this env config sets."""
        return {
            field: self.cfg[field]
            for field in self.POLICY_FIELDS
            if self.cfg.get(field) is not None
        }

    @property
    def has_loss_mask(self) -> bool:
        """Whether training ignores the steps after a trajectory's first done."""
        return not self.cfg.auto_reset and not self.cfg.ignore_terminations

    @property
    def reference_length(self) -> int:
        """Env steps that one trajectory's loss is normalized to.

        A masked trajectory is normalized to ``max_episode_steps``; an unmasked
        one counts every step of its rollout epoch.
        """
        if self.has_loss_mask:
            return self.cfg.max_episode_steps
        return self.cfg.max_steps_per_rollout_epoch

    def scoped(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Return ``metrics`` as this env reports them.

        A listed env reports them under ``<name>/``. A homogeneous mode's one env
        is the whole mode, so its metrics keep their keys.
        """
        return _scope_metrics(self.name, metrics)

    def chunk_steps(self, num_action_chunks: int) -> int:
        """Return the number of action chunks in one rollout epoch."""
        return self.cfg.max_steps_per_rollout_epoch // num_action_chunks

    def algorithm_cfg(self, algorithm_cfg: DictConfig) -> DictConfig:
        """Return ``algorithm_cfg`` as it applies to this env's data.

        ``group_size`` comes from the env config, which repeats each reset state
        that many times, and ``algorithm.envs.<name>`` overrides the fields in
        :attr:`ALGORITHM_FIELDS`.

        Args:
            algorithm_cfg: The run's ``algorithm`` config.

        Returns:
            A new config without ``envs``.
        """
        resolved = OmegaConf.to_container(algorithm_cfg, resolve=True)
        overrides = resolved.pop("envs", None) or {}
        resolved["group_size"] = self.cfg.get("group_size", resolved.get("group_size"))
        resolved.update(overrides.get(self.name, {}) if self.name else {})
        return OmegaConf.create(resolved)


@dataclass(frozen=True)
class EnvAssignment:
    """The env that one env worker rank hosts in one mode.

    Attributes:
        name: Key of the env under ``envs``, or ``None`` in a homogeneous mode.
        cfg: Config of the env.
        rank: Index of this worker among the workers hosting the env.
        world_size: Number of workers hosting the env.
    """

    name: str | None
    cfg: DictConfig
    rank: int
    world_size: int

    def scoped(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Return ``metrics`` as this env reports them, as :meth:`EnvSpec.scoped`."""
        return _scope_metrics(self.name, metrics)


class EnvLayout:
    """Splits the env worker ranks of one mode among the envs it lists.

    Each listed env gets a contiguous block of ranks, in listing order. Without
    ``num_workers`` the block is sized by that env's share of ``total_num_envs``.
    With ``num_workers`` on every env, the block is that many workers and the
    placement list is consumed in the same order. Every rank still hosts the
    same number of envs, because the env-rollout channel splits batches evenly
    across ranks. ``python_interpreter_path`` on an env config is the interpreter
    those workers start with; unset means the node interpreter.
    """

    ENVS_KEY: ClassVar[str] = "envs"
    # Per-env loop fields. A heterogeneous mode config carries one of them only
    # when every env agrees, for readers that take one value for the mode.
    LOOP_FIELDS: ClassVar[tuple[str, ...]] = (
        "rollout_epoch",
        "max_steps_per_rollout_epoch",
        "auto_reset",
        "ignore_terminations",
        "max_episode_steps",
    )
    # What a heterogeneous mode config may hold beside its envs. Anything else
    # belongs to an env and would be ignored at the mode level.
    MODE_FIELDS: ClassVar[tuple[str, ...]] = (
        ENVS_KEY,
        "total_num_envs",
        PolicyObsSpec.CFG_KEY,
        *LOOP_FIELDS,
    )

    def __init__(self, mode_cfg: DictConfig, world_size: int, path: str):
        """Assign the env worker ranks of a mode to its envs.

        Args:
            mode_cfg: ``env.train`` or ``env.eval``.
            world_size: Number of env worker ranks.
            path: Config path of ``mode_cfg``, used in error messages.

        Raises:
            ValueError: If a heterogeneous mode cannot be split across the ranks.
        """
        self._path = path
        self._world_size = world_size
        self._mode_fields: dict[str, Any] = {}
        if not self.is_heterogeneous(mode_cfg):
            self._specs = (EnvSpec(None, mode_cfg, range(world_size)),)
            return

        envs = self._check_envs(mode_cfg)
        self._specs = tuple(self._assign_rank_blocks(envs))
        total_num_envs = sum(env_cfg.total_num_envs for env_cfg in envs.values())

        self._mode_fields["total_num_envs"] = total_num_envs
        for field in self.LOOP_FIELDS:
            values = {name: env_cfg.get(field) for name, env_cfg in envs.items()}
            if len(set(values.values())) == 1 and None not in values.values():
                self._mode_fields[field] = next(iter(values.values()))
            elif mode_cfg.get(field) is not None:
                raise ValueError(
                    f"{path}.{field} is {mode_cfg[field]!r}, but the envs under "
                    f"{path}.envs set {field} to {values}. Set it on each env "
                    f"and remove {path}.{field}."
                )
        for field, value in self._mode_fields.items():
            if mode_cfg.get(field) is not None and mode_cfg[field] != value:
                raise ValueError(
                    f"{path}.{field} is {mode_cfg[field]!r}, but the envs under "
                    f"{path}.envs give {value!r}. Remove {path}.{field} or make "
                    "the two agree."
                )

    @classmethod
    def is_heterogeneous(cls, mode_cfg: DictConfig) -> bool:
        """Return whether ``mode_cfg`` lists its envs under ``envs``."""
        return mode_cfg.get(cls.ENVS_KEY) is not None

    @classmethod
    def env_cfgs(cls, mode_cfg: DictConfig | None, path: str) -> dict[str, DictConfig]:
        """Return the env configs of a mode keyed by their config path.

        Args:
            mode_cfg: ``env.train`` or ``env.eval``; ``None`` yields no envs.
            path: Config path of ``mode_cfg``.

        Returns:
            ``{path: mode_cfg}`` for a homogeneous mode, otherwise one
            ``{path}.envs.<name>`` entry per listed env.
        """
        if mode_cfg is None:
            return {}
        if not cls.is_heterogeneous(mode_cfg):
            return {path: mode_cfg}
        return {
            f"{path}.{cls.ENVS_KEY}.{name}": env_cfg
            for name, env_cfg in mode_cfg[cls.ENVS_KEY].items()
        }

    def mode_fields(self) -> dict[str, Any]:
        """Return the fields a heterogeneous mode config takes from its envs.

        These are ``total_num_envs`` and each loop field that all envs agree on.
        The result is empty for a homogeneous mode, whose config holds them
        itself.
        """
        return dict(self._mode_fields)

    @property
    def specs(self) -> tuple[EnvSpec, ...]:
        """Specs of the mode's envs, in listing order."""
        return self._specs

    def reference_lengths(self, env_weighting: str) -> dict[str | None, float]:
        """Return the env steps each env's trajectories are normalized to.

        With ``"sample"`` every env keeps its own
        :attr:`EnvSpec.reference_length`, so a longer trajectory weighs more.
        With ``"trajectory"`` all envs share their trajectory-weighted mean, so
        every trajectory weighs the same. One env gets its own length either way.

        Args:
            env_weighting: ``algorithm.env_weighting``.

        Returns:
            The reference length of each env, keyed by env name.
        """
        lengths = {spec.name: float(spec.reference_length) for spec in self._specs}
        if env_weighting == "sample":
            return lengths
        if env_weighting != "trajectory":
            raise ValueError(
                "Expected algorithm.env_weighting to be 'trajectory' or 'sample', "
                f"got {env_weighting!r}."
            )
        counts = {
            spec.name: spec.cfg.total_num_envs * spec.cfg.rollout_epoch
            for spec in self._specs
        }
        shared = sum(counts[name] * lengths[name] for name in lengths) / sum(
            counts.values()
        )
        return dict.fromkeys(lengths, shared)

    def spec(self, name: str | None) -> EnvSpec:
        """Return the spec of the env called ``name`` (``None`` if homogeneous)."""
        for spec in self._specs:
            if spec.name == name:
                return spec
        raise KeyError(
            f"Expected an env name in {[spec.name for spec in self._specs]} for "
            f"{self._path}, got {name!r}."
        )

    def spec_path(self, spec: EnvSpec) -> str:
        """Return the config path of ``spec``, such as ``env.train.envs.<name>``."""
        if spec.name is None:
            return self._path
        return f"{self._path}.{self.ENVS_KEY}.{spec.name}"

    def python_interpreter_path(self, rank: int) -> str | None:
        """Return the interpreter for the env that ``rank`` hosts.

        ``None`` means the worker starts with the node interpreter.
        """
        raw = self.spec_of_rank(rank).cfg.get("python_interpreter_path")
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    def spec_of_rank(self, rank: int) -> EnvSpec:
        """Return the spec of the env that env worker ``rank`` hosts."""
        for spec in self._specs:
            if rank in spec.ranks:
                return spec
        raise ValueError(
            f"Env worker rank {rank} is outside the {self._world_size} ranks "
            f"{self._path} is laid out on."
        )

    def assign(self, rank: int) -> EnvAssignment:
        """Return the env that env worker ``rank`` hosts."""
        spec = self.spec_of_rank(rank)
        return EnvAssignment(
            name=spec.name,
            cfg=spec.cfg,
            rank=rank - spec.ranks.start,
            world_size=len(spec.ranks),
        )

    def _assign_rank_blocks(self, envs: dict[str, DictConfig]) -> list[EnvSpec]:
        """Return one spec per env, covering ``world_size`` ranks in order."""
        counts = [env_cfg.get("num_workers") for env_cfg in envs.values()]
        if any(count is not None for count in counts):
            widths = self._explicit_worker_counts(envs)
        else:
            widths = self._shared_worker_counts(envs)
        specs = []
        first_rank = 0
        for (name, env_cfg), width in zip(envs.items(), widths, strict=True):
            specs.append(EnvSpec(name, env_cfg, range(first_rank, first_rank + width)))
            first_rank += width
        return specs

    def _explicit_worker_counts(self, envs: dict[str, DictConfig]) -> list[int]:
        """Read ``num_workers`` from every env and check the blocks tile the ranks."""
        path = self._path
        counts: list[int] = []
        hosted: dict[str, int] = {}
        for name, env_cfg in envs.items():
            count = env_cfg.get("num_workers")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError(
                    f"{path}.envs.{name}.num_workers must be a positive int on "
                    f"every env once one env sets it, got {count!r}."
                )
            if env_cfg.total_num_envs % count != 0:
                raise ValueError(
                    f"{path}.envs.{name}.total_num_envs ({env_cfg.total_num_envs}) "
                    f"must be divisible by num_workers ({count})."
                )
            counts.append(count)
            hosted[name] = env_cfg.total_num_envs // count
        if sum(counts) != self._world_size:
            requested = {name: env_cfg.num_workers for name, env_cfg in envs.items()}
            raise ValueError(
                f"{path}.envs request {sum(counts)} workers via num_workers "
                f"({requested}), but env placement has {self._world_size}. "
                "The two must match."
            )
        if len(set(hosted.values())) != 1:
            raise ValueError(
                f"Each env worker hosts the same number of envs, but {path}.envs "
                f"give {hosted}. Scale total_num_envs with num_workers so the "
                "quotient is the same for every env."
            )
        return counts

    def _shared_worker_counts(self, envs: dict[str, DictConfig]) -> list[int]:
        """Size each env's block by its share of ``total_num_envs``."""
        path = self._path
        world_size = self._world_size
        total_num_envs = sum(env_cfg.total_num_envs for env_cfg in envs.values())
        if total_num_envs % world_size != 0:
            raise ValueError(
                f"{path}.envs hold {total_num_envs} envs in total, which must be "
                f"divisible by the {world_size} env workers."
            )
        envs_per_rank = total_num_envs // world_size
        counts = []
        for name, env_cfg in envs.items():
            if env_cfg.total_num_envs % envs_per_rank != 0:
                raise ValueError(
                    f"{path}.envs.{name}.total_num_envs must be a multiple of "
                    f"{envs_per_rank}, the number of envs each of the {world_size} "
                    f"env workers hosts, got {env_cfg.total_num_envs}. Each worker "
                    "hosts one env, so give every env a whole number of workers "
                    "(several workers can share a GPU, e.g. 'env: 0:0-1')."
                )
            counts.append(env_cfg.total_num_envs // envs_per_rank)
        return counts

    def _check_envs(self, mode_cfg: DictConfig) -> dict[str, DictConfig]:
        path = self._path
        stray = [key for key in mode_cfg.keys() if key not in self.MODE_FIELDS]
        if stray:
            raise ValueError(
                f"{path} lists its envs under {path}.envs, so {path}.{stray[0]} "
                f"would apply to none of them; set it on each env under "
                f"{path}.envs instead. Fields {path} may hold beside envs: "
                f"{list(self.MODE_FIELDS)}; got {stray}."
            )
        envs = mode_cfg[self.ENVS_KEY]
        if not isinstance(envs, DictConfig) or len(envs) == 0:
            raise ValueError(
                f"{path}.envs must map env names to env configs, got {envs!r}."
            )
        for name, env_cfg in envs.items():
            if not isinstance(env_cfg, DictConfig) or env_cfg.get("env_type") is None:
                raise ValueError(
                    f"{path}.envs.{name} must be an env config with env_type."
                )
            total_num_envs = env_cfg.get("total_num_envs")
            if not isinstance(total_num_envs, int) or total_num_envs <= 0:
                raise ValueError(
                    f"{path}.envs.{name}.total_num_envs must be a positive int, "
                    f"got {total_num_envs!r}."
                )
        return dict(envs.items())


def _interpreter_coverage(layout: EnvLayout, path: str) -> None:
    """Require every env to set ``python_interpreter_path`` once one of them does."""
    if layout.specs[0].name is None:
        return
    missing = [
        spec.name
        for spec in layout.specs
        if layout.python_interpreter_path(spec.ranks.start) is None
    ]
    present = [
        spec.name
        for spec in layout.specs
        if layout.python_interpreter_path(spec.ranks.start) is not None
    ]
    if present and missing:
        raise ValueError(
            f"{path} sets python_interpreter_path on {present} but not on "
            f"{missing}. Set it on every env, or on none of them."
        )


def resolve_env_worker_interpreter(
    cfg: DictConfig, rank: int, world_size: int
) -> str | None:
    """Return the interpreter that env worker ``rank`` starts with.

    Train and eval share the process, so a rank that runs both must name the
    same interpreter. ``None`` keeps the interpreter of the node.
    """
    runner = cfg.get("runner") or {}
    only_eval = bool(runner.get("only_eval", False))
    eval_on = only_eval or int(runner.get("val_check_interval", -1) or -1) > 0
    chosen: list[str | None] = []
    if not only_eval and (cfg.get("env") or {}).get("train") is not None:
        layout = EnvLayout(cfg.env.train, world_size, "env.train")
        _interpreter_coverage(layout, "env.train")
        chosen.append(layout.python_interpreter_path(rank))
    if eval_on and (cfg.get("env") or {}).get("eval") is not None:
        layout = EnvLayout(cfg.env.eval, world_size, "env.eval")
        _interpreter_coverage(layout, "env.eval")
        chosen.append(layout.python_interpreter_path(rank))
    if any(path is not None for path in chosen) and any(
        path is None for path in chosen
    ):
        raise ValueError(
            f"Env worker {rank} runs train and eval, and only one of them sets "
            f"python_interpreter_path ({chosen}). Set the same path on both, "
            "or on neither."
        )
    distinct = {path for path in chosen if path is not None}
    if len(distinct) > 1:
        raise ValueError(
            f"Env worker {rank} would start with two interpreters {sorted(distinct)}. "
            "Train and eval on the same rank share one process, so set the same "
            "python_interpreter_path on both."
        )
    return next(iter(distinct), None)
