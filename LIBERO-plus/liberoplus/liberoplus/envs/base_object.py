import re

OBJECTS_DICT = {}
VISUAL_CHANGE_OBJECTS_DICT = {}


def register_object(target_class):
    """We design the mapping to be case-INsensitive."""
    key = "_".join(re.sub(r"([A-Z0-9])", r" \1", target_class.__name__).split()).lower()
    assert key not in OBJECTS_DICT
    OBJECTS_DICT[key] = target_class
    return target_class


def register_visual_change_object(target_class):
    """We keep track of objects that might have visual changes to optimize the codebase"""
    key = "_".join(re.sub(r"([A-Z0-9])", r" \1", target_class.__name__).split()).lower()
    VISUAL_CHANGE_OBJECTS_DICT[key] = target_class
    return target_class
