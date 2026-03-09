import inspect
import logging
import os
from copy import deepcopy
from typing import Any, List, Tuple, Union

import cloudpickle
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

# from omegaconf import SCMode
from yam_realtime.envs.configs.instantiate import _convert_target_to_string


def _visit_dict_config(cfg: Union[DictConfig, ListConfig, Any], func: Any) -> None:  # type: ignore
    """
    Apply func recursively to all DictConfig in cfg.

    Args:
        cfg: The configuration object to visit
        func: The function to apply to each DictConfig
    """
    if isinstance(cfg, DictConfig):
        func(cfg)
        for v in cfg.values():
            _visit_dict_config(v, func)
    elif isinstance(cfg, ListConfig):
        for v in cfg:
            _visit_dict_config(v, func)


def _cast_to_config(obj: Any) -> Any:
    """
    Convert dictionaries to DictConfig objects.

    Args:
        obj: The object to convert

    Returns:
        DictConfig if obj is a dict, otherwise obj unchanged
    """
    # if given a dict, return DictConfig instead
    if isinstance(obj, dict):
        return DictConfig(obj, flags={"allow_objects": True})
    return obj


class DictLoader:
    """
    Provide methods to save, load, and overrides an omegaconf config object
    which may contain definition of lazily-constructed objects.
    """

    @staticmethod
    def load(filenames: Union[str, List[str]], keys: Union[None, str, Tuple[str, ...]] = None) -> Any:
        loaded = DictLoader._load(filenames, keys)
        return OmegaConf.to_container(loaded, resolve=True)  # resolve to basic container

    @staticmethod
    def load_rel(filename: str, keys: Union[None, str, Tuple[str, ...]] = None) -> Any:
        """
        Similar to :meth:`load()`, but load path relative to the caller's
        source file.

        This has the same functionality as a relative import, except that this method
        accepts filename as a string, so more characters are allowed in the filename.
        """
        caller_frame = inspect.stack()[1]
        caller_fname = caller_frame[0].f_code.co_filename
        assert caller_fname != "<string>", "load_rel Unable to find caller"
        caller_dir = os.path.dirname(caller_fname)
        filename = os.path.join(caller_dir, filename)
        return DictLoader.load(filename, keys)

    @staticmethod
    def _load(filenames: Union[str, List[str]], keys: Union[None, str, Tuple[str, ...]] = None) -> Any:
        """
        Load a config file or multiple config files.

        Args:
            filenames: A single file path or a list of file paths to load.
                       If a list is provided, later files in the list will override
                       the configurations from earlier files.
            keys: Keys to load and return. If not given, return all keys
                  (whose values are config objects) in a dict.

        Returns:
            An OmegaConf config object or a specific subset based on keys.
        """
        # Validate input types first
        if not isinstance(filenames, (str, list)):
            raise TypeError("filenames must be a string or a list of strings.")

        has_keys = keys is not None

        def _handle_keys(
            cfg: Union[DictConfig, ListConfig], keys: Union[None, str, Tuple[str, ...]], has_keys: bool
        ) -> Union[DictConfig, ListConfig, Any, Tuple]:
            """
            Extract specific keys from a config object.

            Args:
                cfg: The configuration object
                keys: The keys to extract
                has_keys: Whether keys are provided

            Returns:
                Either the full config, a single value, or a tuple of values
            """
            if not has_keys:
                return cfg
            if isinstance(keys, str):
                return _cast_to_config(cfg[keys])  # type: ignore
            assert keys is not None
            return tuple(_cast_to_config(cfg[k]) for k in keys)  # type: ignore

        if isinstance(filenames, str):
            # Single file load behavior
            filename = os.path.expanduser(filenames).replace("/./", "/")  # redundant
            if os.path.splitext(filename)[1] not in [".yaml", ".yml"]:
                raise ValueError(f"Config file {filename} has to be a yaml file.")

            with open(filename, "r") as f:
                obj = yaml.unsafe_load(f)
            #  ret = OmegaConf.create(obj, flags={"allow_objects": True})
            ret = OmegaConf.create(obj)
            return _handle_keys(ret, keys, has_keys)

        elif isinstance(filenames, list):
            # Multiple files load and merge behavior
            if not filenames:
                raise ValueError("No configuration files provided to load.")

            # merged_cfg = OmegaConf.create(flags={"allow_objects": True})
            merged_cfg = OmegaConf.create()

            for filename in filenames:
                cfg = DictLoader.load(filename)
                merged_cfg = OmegaConf.merge(merged_cfg, cfg)

            return _handle_keys(merged_cfg, keys, has_keys)

    @staticmethod
    def save(cfg: Union[DictConfig, ListConfig, Any], filename: str) -> None:
        """
        Save a config object to a yaml file.
        Note that when the config dictionary contains complex objects (e.g. lambda),
        it can't be saved to yaml. In that case we will print an error and
        attempt to save to a pkl file instead.

        Args:
            cfg: an OmegaConf config object
            filename: yaml file name to save the config file
        """
        logger = logging.getLogger(__name__)
        try:
            cfg = deepcopy(cfg)
        except Exception:
            pass
        else:
            # if it's deep-copyable, then...
            def _replace_type_by_name(x: DictConfig) -> None:
                if "_target_" in x and callable(x._target_):
                    try:
                        x._target_ = _convert_target_to_string(x._target_)
                    except AttributeError:
                        pass

            # not necessary, but makes yaml looks nicer
            _visit_dict_config(cfg, _replace_type_by_name)

        save_pkl = False
        try:
            dict = OmegaConf.to_container(
                cfg,
                # Do not resolve interpolation when saving, i.e. do not turn ${a} into
                # actual values when saving.
                resolve=False,
                # Save structures (dataclasses) in a format that can be instantiated later.
                # Without this option, the type information of the dataclass will be erased.
                # structured_config_mode=SCMode.INSTANTIATE,
            )
            dumped = yaml.dump(dict, default_flow_style=None, allow_unicode=True, width=9999)
            with open(filename, "w") as f:
                f.write(dumped)

            try:
                _ = yaml.unsafe_load(dumped)  # test that it is loadable
            except Exception:
                logger.warning(
                    "The config contains objects that cannot serialize to a valid yaml. "
                    f"{filename} is human-readable but cannot be loaded."
                )
                save_pkl = True
        except Exception:
            logger.exception("Unable to serialize the config to yaml. Error:")
            save_pkl = True

        if save_pkl:
            new_filename = filename + ".pkl"
            try:
                # retry by pickle
                with open(new_filename, "wb") as f:
                    cloudpickle.dump(cfg, f)
                logger.warning(f"Config is saved using cloudpickle at {new_filename}.")
            except Exception:
                pass
