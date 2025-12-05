"""
object change after usd2mjcf:
1. body name change to "object"
2. add <site rgba="0 0 0 0" size="0.005" pos="0 0 -0.06" name="bottom_site" />
       <site rgba="0 0 0 0" size="0.005" pos="0 0 0.06" name="top_site" />
       <site rgba="0 0 0 0" size="0.005" pos="0.015 0.015 0" name="horizontal_radius_site" />
"""


import os
import re
import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

from liberoplus.liberoplus.envs.base_object import register_object

import pathlib

from liberoplus.liberoplus.envs.base_object import (
    register_visual_change_object,
    register_object,
)
from liberoplus.liberoplus.utils.mu_utils import register_mu, InitialSceneTemplates
from liberoplus.liberoplus.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info


import re
from liberoplus.liberoplus.envs import objects
from liberoplus.liberoplus.utils.bddl_generation_utils import *
from liberoplus.liberoplus.envs.objects import OBJECTS_DICT
from liberoplus.liberoplus.utils.object_utils import get_affordance_regions

from liberoplus.liberoplus.utils.mu_utils import register_mu, InitialSceneTemplates

from liberoplus.liberoplus.envs import OffScreenRenderEnv
from PIL import Image

import torch
import torchvision

class CustomObjects(MujocoXMLObject):
    def __init__(self, custom_path, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        # make sure custom path is an absolute path
        assert(os.path.isabs(custom_path)), "Custom path must be an absolute path"
        # make sure the custom path is also an xml file
        assert(custom_path.endswith(".xml")), "Custom path must be an xml file"
        super().__init__(
            custom_path,
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.object_properties = {"vis_site_names": {}}

@register_object
class LiberoMug1(CustomObjects):
    def __init__(self,
                 name="libero_mug",
                 obj_name="libero_mug",
                 ):
        super().__init__(
            # custom_path=os.path.abspath(os.path.join(
            #     "./", "custom_assets", "libero_mug", "libero_mug.xml"
            # )),
            custom_path="/path/to/BEHAVIOR-1K/OmniGibson/omnigibson/data/og_dataset/objects/bottle_of_gin/qzgcdx/usd/MJCF/try.xml",
            name=name,
            obj_name=obj_name,
        )

        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class LiberoMugYellow1(CustomObjects):
    def __init__(self,
                 name="libero_mug",
                 obj_name="libero_mug",
                 ):
        super().__init__(
            # custom_path=os.path.abspath(os.path.join(
            #     "./", "custom_assets", "libero_mug_yellow", "libero_mug_yellow.xml"
            # )),
            custom_path="/path/to/BEHAVIOR-1K/OmniGibson/omnigibson/data/og_dataset/objects/bottle_of_gin/qzgcdx/usd/MJCF/try.xml",
            name=name,
            obj_name=obj_name,
        )

        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None


@register_mu(scene_type="kitchen")
class KitchenDemoScene(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
        }

        object_num_info = {
            "libero_mug": 2,
            "libero_mug_yellow": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, -0.30],
                region_name="wooden_cabinet_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.0],
                region_name="libero_mug_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.2, 0.2],
                region_name="libero_mug_init_region_2",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.1, 0.15],
                region_name="libero_mug_yellow_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "libero_mug_1", "kitchen_table_libero_mug_init_region"),
            ("On", "libero_mug_2", "kitchen_table_libero_mug_init_region_2"),
            ("On", "libero_mug_yellow_1", "kitchen_table_libero_mug_yellow_init_region"),
            ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
        ]
        return states
        
scene_name = "kitchen_demo_scene"
language = "libero demo behaviors"
register_task_info(language,
                   scene_name=scene_name,
                   objects_of_interest=[],
                   goal_states=[
                       ("Open", "wooden_cabinet_1_top_region"),
                       ("In", "libero_mug_yellow_1", "wooden_cabinet_1_top_region"),
                       ],
)

YOUR_BDDL_FILE_PATH = "./custom_pddl"
YOUR_SAVE_FILE_NAME = "/path/to/LIBERO-pro/custom_pddl/image.jpg"

bddl_file_names, failures = generate_bddl_from_task_info(folder=YOUR_BDDL_FILE_PATH)
print(bddl_file_names)

print("Encountered some failures: ", failures)

with open(bddl_file_names[0], "r") as f:
    print(f.read())


env_args = {
    "bddl_file_name": bddl_file_names[0],
    "camera_heights": 256,
    "camera_widths": 256
}

env = OffScreenRenderEnv(**env_args)
obs = env.reset()
image = Image.fromarray(obs["agentview_image"][::-1])
image.save(YOUR_SAVE_FILE_NAME)
