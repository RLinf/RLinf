import os
import numpy as np
import re

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

from liberopro.liberopro.envs.base_object import (
    register_visual_change_object,
    register_object,
)


class GoogleScannedObject(MujocoXMLObject):
    def __init__(self, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        super().__init__(
            os.path.join(
                str(absolute_path),
                f"assets/stable_scanned_objects/{obj_name}/{obj_name}.xml",
            ),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}


@register_object
class Rack(GoogleScannedObject):
    def __init__(
        self,
        name="simple_rack",
        obj_name="simple_rack",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints=joints)
        self.rotation = (0, 0)
        self.rotation_axis = "x"


@register_object
class WhiteBowl(GoogleScannedObject):
    def __init__(self, name="white_bowl", obj_name="white_bowl"):
        super().__init__(name, obj_name)


@register_object
class AkitaBlackBowl(GoogleScannedObject):
    def __init__(self, name="akita_black_bowl", obj_name="akita_black_bowl"):
        super().__init__(name, obj_name)


@register_object
class Plate(GoogleScannedObject):
    def __init__(self, name="plate", obj_name="plate"):
        super().__init__(name, obj_name)


@register_object
class Basket(GoogleScannedObject):
    def __init__(self, name="basket", obj_name="basket"):
        super().__init__(name, obj_name)


@register_object
class Chefmate8Frypan(GoogleScannedObject):
    def __init__(self, name="chefmate_8_frypan", obj_name="chefmate_8_frypan"):
        super().__init__(name, obj_name)


@register_object
class GlazedRimPorcelainRamekin(GoogleScannedObject):
    def __init__(
        self,
        name="glazed_rim_porcelain_ramekin",
        obj_name="glazed_rim_porcelain_ramekin",
    ):
        super().__init__(name, obj_name)


@register_object
class BiggerAkitaBlackBowl(GoogleScannedObject):
    def __init__(self, name="bigger_akita_black_bowl", obj_name="bigger_akita_black_bowl"):
        super().__init__(name, obj_name)

@register_object
class BlackBowl(GoogleScannedObject):
    def __init__(self, name="black_bowl", obj_name="black_bowl"):
        super().__init__(name, obj_name)

@register_object
class RedAkitaBlackBowl(GoogleScannedObject):
    def __init__(self, name="red_akita_black_bowl", obj_name="red_akita_black_bowl"):
        super().__init__(name, obj_name)

@register_object
class RedBasket(GoogleScannedObject):
    def __init__(self, name="red_basket", obj_name="red_basket"):
        super().__init__(name, obj_name)

@register_object
class RedRamekin(GoogleScannedObject):
    def __init__(
        self,
        name="red_ramekin",
        obj_name="red_ramekin",
    ):
        super().__init__(name, obj_name)

@register_object
class YellowBowl(GoogleScannedObject):
    def __init__(self, name="yellow_bowl", obj_name="yellow_bowl"):
        super().__init__(name, obj_name)

@register_object
class YellowPlate(GoogleScannedObject):
    def __init__(self, name="yellow_plate", obj_name="yellow_plate"):
        super().__init__(name, obj_name)


@register_object
class RedSticker(CustomObjects):
    def __init__(self,
                 name="red_sticker",
                 obj_name="red_sticker",
                 ):
        super().__init__(
            name=name,
            obj_name=obj_name,
        )

        # 设置圆柱体100%直立状态
        self.rotation = {
            "x": (0.0, 0.0),  # x轴不旋转，保持直立
            "y": (0.0, 0.0),  # y轴不旋转，保持直立
            "z": (0.0, 0.0),  # z轴也不旋转，完全固定朝向
        }
        self.rotation_axis = "z"  # 设置主旋转轴为z轴

        # 确保100%直立，不设置任何强制旋转的init_quat