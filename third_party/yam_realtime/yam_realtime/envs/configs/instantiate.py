import dataclasses
import logging
import pydoc
from collections import abc
from importlib import import_module
from types import ModuleType
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def _convert_target_to_string(t: Any) -> str:
    """
    Inverse of ``locate()``.

    Args:
        t: any object with ``__module__`` and ``__qualname__``

    Returns:
        str: String representation of the object's module and qualname
    """
    module, qualname = t.__module__, t.__qualname__

    # Compress the path to this object, e.g. ``module.submodule._impl.class``
    # may become ``module.submodule.class``, if the later also resolves to the same
    # object. This simplifies the string, and also is less affected by moving the
    # class implementation.
    module_parts = module.split(".")
    for k in range(1, len(module_parts)):
        prefix = ".".join(module_parts[:k])
        candidate = f"{prefix}.{qualname}"
        try:
            if locate(candidate) is t:
                return candidate
        except ImportError:
            pass
    return f"{module}.{qualname}"


def locate(name: str) -> Any:
    """
    Locate and return an object ``x`` using an input string ``{x.__module__}.{x.__qualname__}``,
    such as "module.submodule.class_name".

    Args:
        name: The dotted path to the object to locate

    Returns:
        Any: The located object

    Raises:
        Exception: If the object cannot be found
    """
    obj = pydoc.locate(name)

    # copy from hydra.utils._locate https://github.com/facebookresearch/hydra/blob/57690d7c4e8b5e88dad07d67278f613a739e6d13/hydra/_internal/utils.py#L614-L665
    def _locate(path: str) -> Any:
        """
        Locate an object by name or dotted path, importing as necessary.
        This is similar to the pydoc function `locate`, except that it checks for
        the module from the given path from back to front.

        Args:
            path: The dotted path to the object

        Returns:
            Any: The located object

        Raises:
            ImportError: If the object cannot be found
        """
        if path == "":
            raise ImportError("Empty path")

        parts = [part for part in path.split(".")]
        for part in parts:
            if not len(part):
                raise ValueError(
                    f"Error loading '{path}': invalid dotstring." + "\nRelative imports are not supported."
                )
        assert len(parts) > 0
        part0 = parts[0]
        try:
            obj = import_module(part0)
        except Exception as exc_import:
            raise ImportError(
                f"Error loading '{path}':\n{exc_import!r}" + f"\nAre you sure that module '{part0}' is installed?"
            ) from exc_import
        for m in range(1, len(parts)):
            part = parts[m]
            try:
                obj = getattr(obj, part)
            except AttributeError as exc_attr:
                parent_dotpath = ".".join(parts[:m])
                if isinstance(obj, ModuleType):
                    mod = ".".join(parts[: m + 1])
                    try:
                        obj = import_module(mod)
                        continue
                    except ModuleNotFoundError as exc_import:
                        raise ImportError(
                            f"Error loading '{path}':\n{exc_import!r}"
                            + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                        ) from exc_import
                    except Exception as exc_import:
                        raise ImportError(f"Error loading '{path}':\n{exc_import!r}") from exc_import
                raise ImportError(
                    f"Error loading '{path}':\n{exc_attr!r}"
                    + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
                ) from exc_attr
        return obj

    # Some cases (e.g. torch.optim.sgd.SGD) not handled correctly
    # by pydoc.locate. Try a private function from hydra.
    if obj is None:
        obj = _locate(name)  # it raises if fails

    return obj


def dump_dataclass(obj: Any) -> dict:
    """
    Dump a dataclass recursively into a dict that can be later instantiated.

    Args:
        obj: a dataclass object

    Returns:
        dict: Dictionary representation of the dataclass
    """
    assert dataclasses.is_dataclass(obj) and not isinstance(obj, type), (
        "dump_dataclass() requires an instance of a dataclass."
    )
    ret = {"_target_": _convert_target_to_string(type(obj))}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            v = dump_dataclass(v)
        if isinstance(v, (list, tuple)):
            v = [dump_dataclass(x) if dataclasses.is_dataclass(x) else x for x in v]
        # The actual value will be handled correctly at runtime
        # Type ignore because the type checker doesn't understand this is safe
        ret[f.name] = v  # type: ignore
    return ret


def instantiate(cfg: Any) -> Any:
    """
    Recursively instantiate objects defined in dictionaries by
    "_target_" and arguments.

    Args:
        cfg: a dict-like object with "_target_" that defines the caller, and
            other keys that define the arguments

    Returns:
        object instantiated by cfg
    """

    if isinstance(cfg, ListConfig):
        lst = [instantiate(x) for x in cfg]
        return ListConfig(lst, flags={"allow_objects": True})
    if isinstance(cfg, list):
        # Specialize for list, because many classes take
        # list[objects] as arguments, such as ResNet, DatasetMapper
        return [instantiate(x) for x in cfg]

    # If input is a DictConfig backed by dataclasses (i.e. omegaconf's structured config),
    # instantiate it to the actual dataclass.
    if isinstance(cfg, DictConfig) and dataclasses.is_dataclass(cfg._metadata.object_type):
        return OmegaConf.to_object(cfg)

    if isinstance(cfg, abc.Mapping):  # and "_target_" in cfg:
        if "_target_" in cfg:
            # conceptually equivalent to hydra.utils.instantiate(cfg) with _convert_=all,
            # but faster: https://github.com/facebookresearch/hydra/issues/1200
            cfg = {k: instantiate(v) for k, v in cfg.items()}
            cls = cfg.pop("_target_")
            cls = instantiate(cls)

            if isinstance(cls, str):
                cls_name = cls
                cls = locate(cls_name)
                assert cls is not None, cls_name
            else:
                try:
                    cls_name = cls.__module__ + "." + cls.__qualname__
                except Exception:
                    # target could be anything, so the above could fail
                    cls_name = str(cls)
            assert callable(cls), f"_target_ {cls} does not define a callable object"
            try:
                return cls(**cfg)
            except TypeError:
                logger = logging.getLogger(__name__)
                logger.error(f"Error when instantiating {cls_name}!")
                raise
        else:
            updated = {k: instantiate(v) for k, v in cfg.items()}
            return updated

    return cfg  # return as-is if don't know what to do
