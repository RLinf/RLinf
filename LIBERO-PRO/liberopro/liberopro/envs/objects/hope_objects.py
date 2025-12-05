import os
import re
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string

import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

from liberopro.liberopro.envs.base_object import register_object


class HopeBaseObject(MujocoXMLObject):
    def __init__(self, name, obj_name):
        super().__init__(
            os.path.join(
                str(absolute_path),
                f"assets/stable_hope_objects/{obj_name}/{obj_name}.xml",
            ),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
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
class AlphabetSoup(HopeBaseObject):
    def __init__(self, name="alphabet_soup", obj_name="alphabet_soup"):
        super().__init__(name, obj_name)
        self.rotation_axis = "z"
        
@register_object
class BiggerAlphabetSoup(HopeBaseObject):
    def __init__(self, name="bigger_alphabet_soup", obj_name="bigger_alphabet_soup"):
        super().__init__(name, obj_name)
        self.rotation_axis = "z"        

@register_object
class RedAlphabetSoup(HopeBaseObject):
    def __init__(self, name="red_alphabet_soup", obj_name="red_alphabet_soup"):
        super().__init__(name, obj_name)
        self.rotation_axis = "z"

@register_object
class BbqSauce(HopeBaseObject):
    def __init__(self, name="bbq_sauce", obj_name="bbq_sauce"):
        super().__init__(name, obj_name)


@register_object
class GreenBbqSauce(HopeBaseObject):
    def __init__(self, name="green_bbq_sauce", obj_name="green_bbq_sauce"):
        super().__init__(name, obj_name)

@register_object
class Butter(HopeBaseObject):
    def __init__(self, name="butter", obj_name="butter"):
        super().__init__(name, obj_name)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"


@register_object
class GreenButter(HopeBaseObject):
    def __init__(self, name="green_butter", obj_name="green_butter"):
        super().__init__(name, obj_name)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"

@register_object
class Cherries(HopeBaseObject):
    def __init__(self, name="cherries", obj_name="cherries"):
        super().__init__(name, obj_name)


@register_object
class ChocolatePudding(HopeBaseObject):
    def __init__(self, name="chocolate_pudding", obj_name="chocolate_pudding"):
        super().__init__(name, obj_name)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"


@register_object
class GreenChocolatePudding(HopeBaseObject):
    def __init__(self, name="green_chocolate_pudding", obj_name="green_chocolate_pudding"):
        super().__init__(name, obj_name)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"


@register_object
class Cookies(HopeBaseObject):
    def __init__(self, name="cookies", obj_name="cookies"):
        super().__init__(name, obj_name)

@register_object
class YellowCookies(HopeBaseObject):
    def __init__(self, name="yellow_cookies", obj_name="yellow_cookies"):
        super().__init__(name, obj_name)

@register_object
class Corn(HopeBaseObject):
    def __init__(self, name="corn", obj_name="corn"):
        super().__init__(name, obj_name)


@register_object
class CreamCheese(HopeBaseObject):
    def __init__(self, name="cream_cheese", obj_name="cream_cheese"):
        super().__init__(name, obj_name)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"


@register_object
class RedCreamCheese(HopeBaseObject):
    def __init__(self, name="red_cream_cheese", obj_name="red_cream_cheese"):
        super().__init__(name, obj_name)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"


@register_object
class Ketchup(HopeBaseObject):
    def __init__(self, name="ketchup", obj_name="ketchup"):
        super().__init__(name, obj_name)
        self.rotation = {
            "x": (np.pi / 2, np.pi / 2),
            "z": (np.pi / 2, np.pi / 2),
        }
        self.rotation_axis = None

@register_object
class GreenKetchup(HopeBaseObject):
    def __init__(self, name="green_ketchup", obj_name="green_ketchup"):
        super().__init__(name, obj_name)
        self.rotation = {
            "x": (np.pi / 2, np.pi / 2),
            "z": (np.pi / 2, np.pi / 2),
        }
        self.rotation_axis = None


@register_object
class MacaroniAndCheese(HopeBaseObject):
    def __init__(self, name="macaroni_and_cheese", obj_name="macaroni_and_cheese"):
        super().__init__(name, obj_name)
        self.rotation = {
            "x": (np.pi / 2, np.pi / 2),
            "z": (np.pi / 2, np.pi / 2),
        }
        self.rotation_axis = None


@register_object
class Mayo(HopeBaseObject):
    def __init__(self, name="mayo", obj_name="mayo"):
        super().__init__(name, obj_name)
        self.rotation = {
            "x": (np.pi / 2, np.pi / 2),
            "z": (np.pi / 2, np.pi / 2),
        }
        self.rotation_axis = None


@register_object
class Milk(HopeBaseObject):
    def __init__(self, name="milk", obj_name="milk"):
        super().__init__(name, obj_name)
        self.rotation = {
            "x": (np.pi / 2, np.pi / 2),
            "z": (np.pi / 2, np.pi / 2),
        }


# class Mushrooms(HopeBaseObject):
#     def __init__(self,
#                  name="mushrooms",
#                  obj_name="mushrooms"):
#         super().__init__(name, obj_name)

# class Mustard(HopeBaseObject):
#     def __init__(self,
#                  name="mustard",
#                  obj_name="mustard"):
#         super().__init__(name, obj_name)
#         self.rotation={
#             "x": (np.pi / 2, np.pi/2),
#             "z": (np.pi / 2, np.pi/2),
#         }
#         self.rotation_axis= None


@register_object
class OrangeJuice(HopeBaseObject):
    def __init__(self, name="orange_juice", obj_name="orange_juice"):
        super().__init__(name, obj_name)
        self.rotation = {
            "x": (np.pi / 2, np.pi / 2),
            "z": (np.pi / 2, np.pi / 2),
        }

@register_object
class RedOrangeJuice(HopeBaseObject):
    def __init__(self, name="red_orange_juice", obj_name="red_orange_juice"):
        super().__init__(name, obj_name)
        self.rotation = {
            "x": (np.pi / 2, np.pi / 2),
            "z": (np.pi / 2, np.pi / 2),
        }

# class Parmesan(HopeBaseObject):
#     def __init__(self,
#                  name="parmesan",
#                  obj_name="parmesan"):
#         super().__init__(name, obj_name)

# class Peaches(HopeBaseObject):
#     def __init__(self,
#                  name="peaches",
#                  obj_name="peaches"):
#         super().__init__(name, obj_name)

# class PeasAndCarrots(HopeBaseObject):
#     def __init__(self,
#                  name="peas_and_carrots",
#                  obj_name="peas_and_carrots"):
#         super().__init__(name, obj_name)

# class Pineapple(HopeBaseObject):
#     def __init__(self,
#                  name="pineapple",
#                  obj_name="pineapple"):
#         super().__init__(name, obj_name)


@register_object
class Popcorn(HopeBaseObject):
    def __init__(self, name="popcorn", obj_name="popcorn"):
        super().__init__(name, obj_name)
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"


# class Raisins(HopeBaseObject):
#     def __init__(self,
#                  name="raisins",
#                  obj_name="raisins"):
#         super().__init__(name, obj_name)


@register_object
class SaladDressing(HopeBaseObject):
    def __init__(self, name="salad_dressing", obj_name="salad_dressing"):
        super().__init__(name, obj_name)
        self.rotation = {
            "x": (np.pi / 2, np.pi / 2),
            "z": (np.pi / 2, np.pi / 2),
        }
        self.rotation_axis = None


@register_object
class NewSaladDressing(HopeBaseObject):
    def __init__(self, name="new_salad_dressing", obj_name="new_salad_dressing"):
        super().__init__(name, obj_name)
        self.rotation = {
            "x": (np.pi / 2, np.pi / 2),
            "z": (np.pi / 2, np.pi / 2),
        }
        self.rotation_axis = None


@register_object
class RedSaladDressing(HopeBaseObject):
    def __init__(self, name="red_salad_dressing", obj_name="red_salad_dressing"):
        super().__init__(name, obj_name)
        self.rotation = {
            "x": (np.pi / 2, np.pi / 2),
            "z": (np.pi / 2, np.pi / 2),
        }
        self.rotation_axis = None


@register_object
class TomatoSauce(HopeBaseObject):
    def __init__(self, name="tomato_sauce", obj_name="tomato_sauce"):
        super().__init__(name, obj_name)
        self.rotation_axis = "z"


@register_object
class YellowTomatoSauce(HopeBaseObject):
    def __init__(self, name="yellow_tomato_sauce", obj_name="yellow_tomato_sauce"):
        super().__init__(name, obj_name)
        self.rotation_axis = "z"


# class Tuna(HopeBaseObject):
#     def __init__(self,
#                  name="tuna",
#                  obj_name="tuna"):
#         super().__init__(name, obj_name)

# class Yogurt(HopeBaseObject):
#     def __init__(self,
#                  name="yogurt",
#                  obj_name="yogurt"):
#         super().__init__(name, obj_name)
@register_object
class BlueKetchup(HopeBaseObject):
    def __init__(self, name="blue_ketchup", obj_name="blue_ketchup"):
        super().__init__(name, obj_name)
        self.rotation_axis = "z"
        
@register_object
class BiggerMilk(HopeBaseObject):
    def __init__(self, name="bigger_milk", obj_name="bigger_milk"):
        super().__init__(name, obj_name)
        self.rotation_axis = "z"


@register_object
class YellowMilk(HopeBaseObject):
    def __init__(self, name="yellow_milk", obj_name="yellow_milk"):
        super().__init__(name, obj_name)
        self.rotation_axis = "z"

@register_object
class RedAlphabetSoup(HopeBaseObject):
    def __init__(self, name="red_alphabet_soup", obj_name="red_alphabet_soup"):
        super().__init__(name, obj_name)
        self.rotation_axis = "z"
@register_object
class RedAlphabetSoup(HopeBaseObject):
    def __init__(self, name="red_alphabet_soup", obj_name="red_alphabet_soup"):
        super().__init__(name, obj_name)
        self.rotation_axis = "z"
@register_object
class RedAlphabetSoup(HopeBaseObject):
    def __init__(self, name="red_alphabet_soup", obj_name="red_alphabet_soup"):
        super().__init__(name, obj_name)
        self.rotation_axis = "z"
@register_object
class RedAlphabetSoup(HopeBaseObject):
    def __init__(self, name="red_alphabet_soup", obj_name="red_alphabet_soup"):
        super().__init__(name, obj_name)
        self.rotation_axis = "z"

