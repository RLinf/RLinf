import re

from liberopro.liberopro.envs.objects import OBJECTS_DICT
from liberopro.liberopro.utils.bddl_generation_utils import (
    get_affordance_region_kwargs_list_from_fixture_info,
    get_object_dict,
)
from liberopro.liberopro.utils.object_utils import get_affordance_regions

MU_DICT = {}

SCENE_DICT = {}


def register_mu(scene_type="general"):
    def _func(target_class):
        """For reusing initial conditions easily, we register each pre-defined initial conditions in a dictionary."""
        key = "_".join(
            re.sub(r"([A-Z])", r" \1", target_class.__name__).split()
        ).lower()
        MU_DICT[key.lower()] = target_class
        if scene_type not in SCENE_DICT:
            SCENE_DICT[scene_type.lower()] = []
        SCENE_DICT[scene_type.lower()].append(target_class)
        return target_class

    return _func


def get_scene_dict(scene_type=None):
    if scene_type is None:
        return SCENE_DICT
    else:
        return SCENE_DICT[scene_type.lower()]


def get_scene_class(scene_name):
    return MU_DICT[scene_name.lower()]


class InitialSceneTemplates:
    def __init__(
        self, workspace_name="main_table", fixture_num_info={}, object_num_info={}
    ):

        self.workspace_name = workspace_name
        # print(self.workspace_name)

        self.fixture_object_dict = get_object_dict(fixture_num_info)
        self.movable_object_dict = get_object_dict(object_num_info)

        affordances = get_affordance_regions(OBJECTS_DICT)
        affordance_fixture_info_dict = {}
        for fixture_category_name in self.fixture_object_dict.keys():
            if (
                fixture_category_name != self.workspace_name
                and fixture_category_name != "table"
                and fixture_category_name != "living_room_table"
                and fixture_category_name != "study_table"
                and fixture_category_name != "kitchen_table"
                and fixture_category_name != "coffee_table"
                and fixture_category_name != "floor"
            ):
                for fixture_name in self.fixture_object_dict[fixture_category_name]:
                    affordance_fixture_info_dict[fixture_name] = affordances[
                        fixture_category_name
                    ]
        for category_name in self.movable_object_dict.keys():
            if category_name in affordances:
                for object_name in self.movable_object_dict[category_name]:
                    affordance_fixture_info_dict[object_name] = affordances[
                        category_name
                    ]
        # print(affordance_fixture_info_dict)
        self.affordance_region_kwargs_list = (
            get_affordance_region_kwargs_list_from_fixture_info(
                affordance_fixture_info_dict
            )
        )

        self.regions = {}
        self.define_regions()

    @property
    def possible_objects_of_interest(self):
        # objects_of_interest = list(self.fixture_object_dict.keys()) + list(self.movable_object_dict.keys())
        # return objects_of_interest
        objects_of_interest = []
        for category_name in self.fixture_object_dict.keys():
            objects_of_interest += self.fixture_object_dict[category_name]
        for category_name in self.movable_object_dict.keys():
            objects_of_interest += self.movable_object_dict[category_name]
        return objects_of_interest

    @property
    def movable_objects(self):
        return list(self.movable_object_dict)

    def define_regions(self):
        """Override this method to define the layout of a scene."""
        raise NotImplementedError

    def get_region_dict(
        self,
        region_centroid_xy,
        region_name,
        target_name=None,
        region_half_len=0.02,
        yaw_rotation=(0.0, 0.0),
    ):
        """This is a function that creates a default region with rectangular shape."""
        if target_name is None:
            target_name = self.workspace_name
        region_key_value = {
            region_name: {
                "target": target_name,
                "ranges": [
                    (
                        region_centroid_xy[0] - region_half_len,
                        region_centroid_xy[1] - region_half_len,
                        region_centroid_xy[0] + region_half_len,
                        region_centroid_xy[1] + region_half_len,
                    )
                ],
                "yaw_rotation": [yaw_rotation],
            }
        }
        print
        return region_key_value

    @property
    def init_states(self):
        raise NotImplementedError
