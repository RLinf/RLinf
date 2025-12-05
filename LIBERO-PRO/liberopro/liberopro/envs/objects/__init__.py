import re

from liberopro.liberopro.envs.base_object import OBJECTS_DICT, VISUAL_CHANGE_OBJECTS_DICT

from .hope_objects import *
from .google_scanned_objects import *
from .articulated_objects import *
from .turbosquid_objects import *
from .site_object import SiteObject
from .target_zones import *
from .self_designed_object import RedSticker, LiberoMugYellow


def get_object_fn(category_name):
    # print(OBJECTS_DICT.keys())
    try:
        s = OBJECTS_DICT[category_name.lower()]
    except:
        print(category_name)
    return OBJECTS_DICT[category_name.lower()]


def get_object_dict():
    return OBJECTS_DICT
