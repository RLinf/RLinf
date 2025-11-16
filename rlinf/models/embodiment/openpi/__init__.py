import difflib
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    TrainConfig,
)
import openpi.models.pi0_config as pi0_config
import openpi.training.weight_loaders as weight_loaders
from rlinf.models.embodiment.openpi.dataconfig.real_dataconfig import RealDataConfig


_CONFIGS = [
    TrainConfig(
        name="pi0_real", 
        model=pi0_config.Pi0Config(), 
        data=RealDataConfig(
            repo_id="physical-intelligence/real",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base"),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=30_000,
    )
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}
def get_openpi_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]