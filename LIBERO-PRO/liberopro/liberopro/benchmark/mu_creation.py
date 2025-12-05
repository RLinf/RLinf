import re
from liberopro.liberopro.envs import objects
from liberopro.liberopro.utils.bddl_generation_utils import *
from liberopro.liberopro.envs.objects import OBJECTS_DICT
from liberopro.liberopro.utils.object_utils import get_affordance_regions

from liberopro.liberopro.utils.mu_utils import register_mu, InitialSceneTemplates


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


@register_mu(scene_type="kitchen")
class KitchenScene2(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
        }

        object_num_info = {
            "akita_black_bowl": 3,
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
                region_centroid_xy=[-0.05, 0.20],
                region_name="akita_black_bowl_middle_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.10, 0.15],
                region_name="akita_black_bowl_front_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.15, 0.05],
                region_name="akita_black_bowl_back_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.0],
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
            (
                "On",
                "akita_black_bowl_1",
                "kitchen_table_akita_black_bowl_front_init_region",
            ),
            (
                "On",
                "akita_black_bowl_2",
                "kitchen_table_akita_black_bowl_middle_init_region",
            ),
            (
                "On",
                "akita_black_bowl_3",
                "kitchen_table_akita_black_bowl_back_init_region",
            ),
            ("On", "plate_1", "kitchen_table_plate_init_region"),
            ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene3(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "flat_stove": 1,
        }

        object_num_info = {"chefmate_8_frypan": 1, "moka_pot": 1}

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, 0.20],
                region_name="flat_stove_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, -0.25],
                region_name="frypan_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.05, 0.0],
                region_name="moka_pot_init_region",
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
            ("On", "flat_stove_1", "kitchen_table_flat_stove_init_region"),
            ("On", "chefmate_8_frypan_1", "kitchen_table_frypan_init_region"),
            ("On", "moka_pot_1", "kitchen_table_moka_pot_init_region"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene4(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "white_cabinet": 1,
            "wine_rack": 1,
        }

        object_num_info = {"akita_black_bowl": 1, "wine_bottle": 1}

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.30],
                region_name="white_cabinet_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.30],
                region_name="wine_rack_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.03, -0.05],
                region_name="akita_black_bowl_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.15, 0.05],
                region_name="wine_bottle_init_region",
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
            ("On", "wine_bottle_1", "kitchen_table_wine_bottle_init_region"),
            ("On", "white_cabinet_1", "kitchen_table_white_cabinet_init_region"),
            ("On", "wine_rack_1", "kitchen_table_wine_rack_init_region"),
            ("Open", "white_cabinet_1_bottom_region"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene5(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "white_cabinet": 1,
        }

        object_num_info = {
            "akita_black_bowl": 1,
            "plate": 1,
            "ketchup": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.30],
                region_name="white_cabinet_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.03, -0.05],
                region_name="akita_black_bowl_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.10],
                region_name="ketchup_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, -0.25],
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
            ("On", "white_cabinet_1", "kitchen_table_white_cabinet_init_region"),
            ("On", "ketchup_1", "kitchen_table_ketchup_init_region"),
            ("Open", "white_cabinet_1_top_region"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene6(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "microwave": 1,
        }

        object_num_info = {
            "porcelain_mug": 1,
            "white_yellow_mug": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.35],
                region_name="microwave_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(0, 0),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.0],
                region_name="white_yellow_mug_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.25],
                region_name="porcelain_mug_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, -0.25],
                region_name="porcelain_mug_front_region",
                target_name=self.workspace_name,
                region_half_len=0.05,
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "porcelain_mug_1", "kitchen_table_porcelain_mug_init_region"),
            ("On", "white_yellow_mug_1", "kitchen_table_white_yellow_mug_init_region"),
            ("On", "microwave_1", "kitchen_table_microwave_init_region"),
            ("Open", "microwave_1"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene7(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "microwave": 1,
        }

        object_num_info = {
            "white_bowl": 1,
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
                region_centroid_xy=[0.0, -0.25],
                region_name="microwave_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.0],
                region_name="plate_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.10],
                region_name="plate_right_region",
                target_name=self.workspace_name,
                region_half_len=0.05,
            )
        )

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "white_bowl_1", "microwave_1_top_side"),
            ("On", "microwave_1", "kitchen_table_microwave_init_region"),
            ("Close", "microwave_1"),
            ("On", "plate_1", "kitchen_table_plate_init_region"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene8(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "flat_stove": 1,
        }

        object_num_info = {"moka_pot": 2}

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, -0.20],
                region_name="flat_stove_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, 0.25],
                region_name="moka_pot_right_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.05, 0.05],
                region_name="moka_pot_left_init_region",
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
            ("On", "flat_stove_1", "kitchen_table_flat_stove_init_region"),
            ("On", "moka_pot_1", "kitchen_table_moka_pot_right_init_region"),
            ("On", "moka_pot_2", "kitchen_table_moka_pot_left_init_region"),
            ("Turnon", "flat_stove_1"),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene9(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "flat_stove": 1,
            "wooden_two_layer_shelf": 1,
        }

        object_num_info = {
            "white_bowl": 1,
            "chefmate_8_frypan": 1,
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, 0.30],
                region_name="flat_stove_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0, -0.25],
                region_name="wooden_two_layer_shelf_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.05, 0.0],
                region_name="frypan_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.15, 0.10],
                region_name="white_bowl_init_region",
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
            ("On", "flat_stove_1", "kitchen_table_flat_stove_init_region"),
            ("On", "chefmate_8_frypan_1", "kitchen_table_frypan_init_region"),
            ("On", "white_bowl_1", "kitchen_table_white_bowl_init_region"),
            (
                "On",
                "wooden_two_layer_shelf_1",
                "kitchen_table_wooden_two_layer_shelf_init_region",
            ),
        ]
        return states


@register_mu(scene_type="kitchen")
class KitchenScene10(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
        }

        object_num_info = {
            "akita_black_bowl": 1,
            "butter": 2,
            "chocolate_pudding": 1,
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
                region_centroid_xy=[-0.10, 0.0],
                region_name="akita_black_bowl_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, 0.20],
                region_name="butter_back_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.20],
                region_name="butter_front_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.05],
                region_name="chocolate_pudding_init_region",
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
            ("On", "butter_1", "kitchen_table_butter_front_init_region"),
            ("On", "butter_2", "kitchen_table_butter_back_init_region"),
            (
                "On",
                "chocolate_pudding_1",
                "kitchen_table_chocolate_pudding_init_region",
            ),
            ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
            ("Open", "wooden_cabinet_1_top_region"),
        ]
        return states


@register_mu(scene_type="living_room")
class LivingRoomScene1(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "living_room_table": 1,
        }

        object_num_info = {
            "alphabet_soup": 1,
            "cream_cheese": 1,
            "tomato_sauce": 1,
            "ketchup": 1,
            "basket": 1,
        }

        super().__init__(
            workspace_name="living_room_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.26],
                region_name="basket_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.05, -0.10],
                region_name="alphabet_soup_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.15, 0.06],
                region_name="cream_cheese_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.10, -0.20],
                region_name="tomato_sauce_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, -0.15],
                region_name="ketchup_init_region",
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
            ("On", "alphabet_soup_1", "living_room_table_alphabet_soup_init_region"),
            ("On", "cream_cheese_1", "living_room_table_cream_cheese_init_region"),
            ("On", "tomato_sauce_1", "living_room_table_tomato_sauce_init_region"),
            ("On", "ketchup_1", "living_room_table_ketchup_init_region"),
            ("On", "basket_1", "living_room_table_basket_init_region"),
        ]
        return states


@register_mu(scene_type="living_room")
class LivingRoomScene2(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "living_room_table": 1,
        }

        object_num_info = {
            "alphabet_soup": 1,
            "cream_cheese": 1,
            "tomato_sauce": 1,
            "ketchup": 1,
            "orange_juice": 1,
            "milk": 1,
            "butter": 1,
            "basket": 1,
        }

        super().__init__(
            workspace_name="living_room_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.26],
                region_name="basket_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.05, -0.10],
                region_name="milk_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.10, -0.20],
                region_name="cream_cheese_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, -0.25],
                region_name="orange_juice_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.1, 0.05],
                region_name="tomato_sauce_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.15],
                region_name="alphabet_soup_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.05, 0.05],
                region_name="butter_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.25, -0.15],
                region_name="ketchup_init_region",
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
            ("On", "alphabet_soup_1", "living_room_table_alphabet_soup_init_region"),
            ("On", "cream_cheese_1", "living_room_table_cream_cheese_init_region"),
            ("On", "tomato_sauce_1", "living_room_table_tomato_sauce_init_region"),
            ("On", "ketchup_1", "living_room_table_ketchup_init_region"),
            ("On", "milk_1", "living_room_table_milk_init_region"),
            ("On", "orange_juice_1", "living_room_table_orange_juice_init_region"),
            ("On", "butter_1", "living_room_table_butter_init_region"),
            ("On", "basket_1", "living_room_table_basket_init_region"),
        ]
        return states


@register_mu(scene_type="living_room")
class LivingRoomScene3(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "living_room_table": 1,
        }

        object_num_info = {
            "alphabet_soup": 1,
            "cream_cheese": 1,
            "tomato_sauce": 1,
            "ketchup": 1,
            "butter": 1,
            "wooden_tray": 1,
        }

        super().__init__(
            workspace_name="living_room_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.26],
                region_name="wooden_tray_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.10, -0.20],
                region_name="cream_cheese_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.1, 0.05],
                region_name="tomato_sauce_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.15],
                region_name="alphabet_soup_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.05, 0.05],
                region_name="butter_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.25, -0.15],
                region_name="ketchup_init_region",
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
            ("On", "alphabet_soup_1", "living_room_table_alphabet_soup_init_region"),
            ("On", "cream_cheese_1", "living_room_table_cream_cheese_init_region"),
            ("On", "tomato_sauce_1", "living_room_table_tomato_sauce_init_region"),
            ("On", "ketchup_1", "living_room_table_ketchup_init_region"),
            ("On", "butter_1", "living_room_table_butter_init_region"),
            ("On", "wooden_tray_1", "living_room_table_wooden_tray_init_region"),
        ]
        return states


@register_mu(scene_type="living_room")
class LivingRoomScene4(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "living_room_table": 1,
        }

        object_num_info = {
            "akita_black_bowl": 2,
            "new_salad_dressing": 1,
            "chocolate_pudding": 1,
            "wooden_tray": 1,
        }

        super().__init__(
            workspace_name="living_room_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.26],
                region_name="wooden_tray_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.10, -0.20],
                region_name="chocolate_pudding_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.1, 0.05],
                region_name="akita_black_bowl_right_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.15],
                region_name="akita_black_bowl_left_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.25, -0.10],
                region_name="salad_dressing_init_region",
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
            (
                "On",
                "chocolate_pudding_1",
                "living_room_table_chocolate_pudding_init_region",
            ),
            (
                "On",
                "akita_black_bowl_1",
                "living_room_table_akita_black_bowl_left_init_region",
            ),
            (
                "On",
                "akita_black_bowl_2",
                "living_room_table_akita_black_bowl_right_init_region",
            ),
            (
                "On",
                "new_salad_dressing_1",
                "living_room_table_salad_dressing_init_region",
            ),
            ("On", "wooden_tray_1", "living_room_table_wooden_tray_init_region"),
        ]
        return states


@register_mu(scene_type="living_room")
class LivingRoomScene5(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "living_room_table": 1,
        }

        object_num_info = {
            "porcelain_mug": 1,
            "red_coffee_mug": 1,
            "white_yellow_mug": 1,
            "plate": 2,
        }

        super().__init__(
            workspace_name="living_room_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, -0.30],
                region_name="plate_left_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.30],
                region_name="plate_right_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )

        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.15],
                region_name="porcelain_mug_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, 0.10],
                region_name="white_yellow_mug_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, 0.0],
                region_name="red_coffee_mug_init_region",
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
            ("On", "plate_1", "living_room_table_plate_left_region"),
            ("On", "plate_2", "living_room_table_plate_right_region"),
            ("On", "red_coffee_mug_1", "living_room_table_red_coffee_mug_init_region"),
            (
                "On",
                "white_yellow_mug_1",
                "living_room_table_white_yellow_mug_init_region",
            ),
            ("On", "porcelain_mug_1", "living_room_table_porcelain_mug_init_region"),
        ]
        return states


@register_mu(scene_type="living_room")
class LivingRoomScene6(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "living_room_table": 1,
        }

        object_num_info = {
            "porcelain_mug": 1,
            "red_coffee_mug": 1,
            "plate": 1,
            "chocolate_pudding": 1,
        }

        super().__init__(
            workspace_name="living_room_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.15, -0.10],
                region_name="plate_left_region",
                target_name=self.workspace_name,
                region_half_len=0.05,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.15, 0.10],
                region_name="plate_right_region",
                target_name=self.workspace_name,
                region_half_len=0.05,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.15, 0.0],
                region_name="plate_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.10, -0.15],
                region_name="porcelain_mug_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, 0.10],
                region_name="chocolate_pudding_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, 0.0],
                region_name="red_coffee_mug_init_region",
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
            ("On", "plate_1", "living_room_table_plate_init_region"),
            ("On", "red_coffee_mug_1", "living_room_table_red_coffee_mug_init_region"),
            (
                "On",
                "chocolate_pudding_1",
                "living_room_table_chocolate_pudding_init_region",
            ),
            ("On", "porcelain_mug_1", "living_room_table_porcelain_mug_init_region"),
        ]
        return states


@register_mu(scene_type="study")
class StudyScene1(InitialSceneTemplates):
    def __init__(self):

        fixture_num_info = {
            "study_table": 1,
            "desk_caddy": 1,
        }

        object_num_info = {
            "black_book": 1,
            "white_yellow_mug": 1,
        }

        super().__init__(
            workspace_name="study_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, -0.14],
                region_name="desk_caddy_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.15],
                region_name="black_book_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
                yaw_rotation=(-np.pi / 2, -np.pi / 4),
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.10, 0.0],
                region_name="white_yellow_mug_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, 0.15],
                region_name="desk_caddy_right_region",
                target_name=self.workspace_name,
                region_half_len=0.05,
                yaw_rotation=(np.pi, np.pi),
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "desk_caddy_1", "study_table_desk_caddy_init_region"),
            ("On", "black_book_1", "study_table_black_book_init_region"),
            ("On", "white_yellow_mug_1", "study_table_white_yellow_mug_init_region"),
        ]
        return states


@register_mu(scene_type="study")
class StudyScene2(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            "study_table": 1,
            "desk_caddy": 1,
        }

        object_num_info = {
            "black_book": 1,
            "red_coffee_mug": 1,
        }

        super().__init__(
            workspace_name="study_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.15],
                region_name="red_coffee_mug_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, -0.05],
                region_name="black_book_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
                yaw_rotation=(-np.pi / 2, -np.pi / 4),
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, -0.14],
                region_name="desk_caddy_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "desk_caddy_1", "study_table_desk_caddy_init_region"),
            ("On", "black_book_1", "study_table_black_book_init_region"),
            ("On", "red_coffee_mug_1", "study_table_red_coffee_mug_init_region"),
        ]
        return states


@register_mu(scene_type="study")
class StudyScene3(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            "study_table": 1,
            "desk_caddy": 1,
        }

        object_num_info = {
            "black_book": 1,
            "red_coffee_mug": 1,
            "porcelain_mug": 1,
        }

        super().__init__(
            workspace_name="study_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.20, 0.15],
                region_name="red_coffee_mug_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.10, 0.15],
                region_name="red_coffee_mug_behind_region",
                target_name=self.workspace_name,
                region_half_len=0.05,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.0],
                region_name="porcelain_mug_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.10, 0.0],
                region_name="black_book_init_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, -0.14],
                region_name="desk_caddy_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, -0.15],
                region_name="desk_caddy_front_left_contain_region",
                target_name=self.workspace_name,
                region_half_len=0.025,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.20, 0.15],
                region_name="desk_caddy_right_region",
                target_name=self.workspace_name,
                region_half_len=0.05,
                yaw_rotation=(np.pi, np.pi),
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            ("On", "desk_caddy_1", "study_table_desk_caddy_init_region"),
            ("On", "black_book_1", "study_table_desk_caddy_front_left_contain_region"),
            ("On", "porcelain_mug_1", "study_table_porcelain_mug_init_region"),
            ("On", "red_coffee_mug_1", "study_table_red_coffee_mug_init_region"),
        ]
        return states


@register_mu(scene_type="study")
class StudyScene4(InitialSceneTemplates):
    def __init__(self):
        fixture_num_info = {
            "study_table": 1,
            "wooden_two_layer_shelf": 1,
        }

        object_num_info = {
            "black_book": 1,
            "yellow_book": 2,
        }

        super().__init__(
            workspace_name="study_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.0],
                region_name="yellow_book_right_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.05, -0.25],
                region_name="yellow_book_left_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.05, -0.15],
                region_name="black_book_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
            )
        )
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.0, 0.28],
                region_name="wooden_two_layer_shelf_init_region",
                target_name=self.workspace_name,
                region_half_len=0.01,
                yaw_rotation=(0, 0),
            )
        )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = [
            (
                "On",
                "wooden_two_layer_shelf_1",
                "study_table_wooden_two_layer_shelf_init_region",
            ),
            ("On", "yellow_book_1", "study_table_yellow_book_right_init_region"),
            ("On", "yellow_book_2", "study_table_yellow_book_left_init_region"),
            ("On", "black_book_1", "study_table_black_book_init_region"),
        ]
        return states
