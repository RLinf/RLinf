"""This is a standalone file for create a task in libero."""
import numpy as np

from liberopro.liberopro.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from liberopro.liberopro.utils.mu_utils import register_mu, InitialSceneTemplates
from liberopro.liberopro.utils.task_generation_utils import (
    register_task_info,
    get_task_info,
    generate_bddl_from_task_info,
)


@register_mu(scene_type="kitchen")
class KitchenScene1(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
        }

        object_num_info = {
            "akita_black_bowl": 1,
            "plate": 1,
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
                region_name="akita_black_bowl_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.25],
                region_name="plate_init_region",
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
            ("On", "akita_black_bowl_1", "kitchen_table_akita_black_bowl_init_region"),
            ("On", "plate_1", "kitchen_table_plate_init_region"),
            ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
        ]
        return states


def main():
    # kitchen_scene_1
    scene_name = "kitchen_scene1"
    language = "Your Language 1"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["wooden_cabinet_1", "akita_black_bowl_1"],
        goal_states=[
            ("Open", "wooden_cabinet_1_top_region"),
            ("In", "akita_black_bowl_1", "wooden_cabinet_1_top_region"),
        ],
    )

    scene_name = "kitchen_scene1"
    language = "Your Language 2"
    register_task_info(
        language,
        scene_name=scene_name,
        objects_of_interest=["wooden_cabinet_1", "akita_black_bowl_1"],
        goal_states=[
            ("Open", "wooden_cabinet_1_top_region"),
            ("In", "akita_black_bowl_1", "wooden_cabinet_1_bottom_region"),
        ],
    )
    bddl_file_names, failures = generate_bddl_from_task_info()
    print(bddl_file_names)


if __name__ == "__main__":
    main()
