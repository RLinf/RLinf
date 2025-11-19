import copy
from dataclasses import MISSING, field, fields, is_dataclass
from typing import Any, Dict


def _recursive_asdict_dataclass(data: Any) -> Dict[str, Any]:
    def get_value(field):
        if field.default_factory is not MISSING:
            return field.default_factory()
        elif field.default is not MISSING:
            return field.default
        return MISSING

    result = {}
    for field in fields(data):
        value = getattr(data, field.name, get_value(field))
        if value is MISSING:
            # Skip fields with no default value
            continue
        result[field.name] = recursive_asdict(value)
    return result


def recursive_asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return _recursive_asdict_dataclass(obj)
    elif isinstance(obj, dict):
        return {k: recursive_asdict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(recursive_asdict(v) for v in obj)
    else:
        return obj


def default_field(obj):
    return field(default_factory=lambda: copy.copy(obj))
