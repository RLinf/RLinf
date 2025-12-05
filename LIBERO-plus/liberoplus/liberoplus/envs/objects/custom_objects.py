import os
import re
import numpy as np
import copy
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string
 
import pathlib
 
# 使用pathlib计算项目根目录的绝对路径
absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()
 
from liberoplus.liberoplus.envs.base_object import register_object
 
def hack_robosuite():
    """
    Hack 'robosuite.models.objects.objects.MujocoXMLObject._get_object_subtree' function for custom objects supports.
    """
    from robosuite.models.objects.objects import (
        MujocoXMLObject, GEOMTYPE2GROUP, array_to_string, OBJECT_COLLISION_COLOR,
        ET, new_joint
    )

    def _get_object_subtree(self):
        # Parse object

        # # ===== Original Code =====================================
        # obj = copy.deepcopy(self.worldbody.find("./body/body[@name='object']"))
        # # =========================================================

        # ===== Modified Code =====================================
        obj = self.worldbody.find("./body/body[@name='object']")
        if obj is None:
            obj = self.worldbody.find("./body[@name='object']")
        if obj is None:
            raise ValueError(
                "Could not find object body with name='object' at either "
                "./body[@name='object'] or ./body/body[@name='object']. "
                "Please check your XML structure."
            )
        obj = copy.deepcopy(obj)
        # =========================================================

        # Rename this top level object body (will have self.naming_prefix added later)
        obj.attrib["name"] = "main"
        # Get all geom_pairs in this tree
        geom_pairs = self._get_geoms(obj)

        # Define a temp function so we don't duplicate so much code
        obj_type = self.obj_type

        def _should_keep(el):
            return int(el.get("group")) in GEOMTYPE2GROUP[obj_type]

        # Loop through each of these pairs and modify them according to @elements arg
        for i, (parent, element) in enumerate(geom_pairs):
            # Delete non-relevant geoms and rename remaining ones
            if not _should_keep(element):
                parent.remove(element)
            else:
                g_name = element.get("name")
                g_name = g_name if g_name is not None else f"g{i}"
                element.set("name", g_name)
                # Also optionally duplicate collision geoms if requested (and this is a collision geom)
                if self.duplicate_collision_geoms and element.get("group") in {None, "0"}:
                    parent.append(self._duplicate_visual_from_collision(element))
                    # Also manually set the visual appearances to the original collision model
                    element.set("rgba", array_to_string(OBJECT_COLLISION_COLOR))
                    if element.get("material") is not None:
                        del element.attrib["material"]
        # add joint(s)
        for joint_spec in self.joint_specs:
            obj.append(new_joint(**joint_spec))
        # Lastly, add a site for this object
        template = self.get_site_attrib_template()
        template["rgba"] = "1 0 0 0"
        template["name"] = "default_site"
        obj.append(ET.Element("site", attrib=template))

        return obj

    MujocoXMLObject._get_object_subtree = _get_object_subtree


hack_robosuite()
 
class CustomObjects(MujocoXMLObject):
    def __init__(self, custom_path, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        # 确保custom_path是一个绝对路径
        assert (os.path.isabs(custom_path)), "Custom path must be an absolute path"
        # 确保custom_path指向一个xml文件
        assert (custom_path.endswith(".xml")), "Custom path must be an xml file"
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
class AlarmClock_1(CustomObjects):
    def __init__(self,
                 name="alarm_clock__1",
                 obj_name="alarm_clock__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/alarm_clock/cvknrh/usd/MJCF/cvknrh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class AlarmClock_2(CustomObjects):
    def __init__(self,
                 name="alarm_clock__2",
                 obj_name="alarm_clock__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/alarm_clock/trwyaq/usd/MJCF/trwyaq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class AlarmClock_3(CustomObjects):
    def __init__(self,
                 name="alarm_clock__3",
                 obj_name="alarm_clock__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/alarm_clock/vqwovi/usd/MJCF/vqwovi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class AllenWrench(CustomObjects):
    def __init__(self,
                 name="allen_wrench",
                 obj_name="allen_wrench",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/allen_wrench/neqlcn/usd/MJCF/neqlcn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple_1(CustomObjects):
    def __init__(self,
                 name="apple__1",
                 obj_name="apple__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple/agveuv/usd/MJCF/agveuv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple_2(CustomObjects):
    def __init__(self,
                 name="apple__2",
                 obj_name="apple__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple/bwteqh/usd/MJCF/bwteqh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple_3(CustomObjects):
    def __init__(self,
                 name="apple__3",
                 obj_name="apple__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple/dfgurb/usd/MJCF/dfgurb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple_4(CustomObjects):
    def __init__(self,
                 name="apple__4",
                 obj_name="apple__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple/hwrflj/usd/MJCF/hwrflj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple_5(CustomObjects):
    def __init__(self,
                 name="apple__5",
                 obj_name="apple__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple/netbsb/usd/MJCF/netbsb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple_6(CustomObjects):
    def __init__(self,
                 name="apple__6",
                 obj_name="apple__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple/obixxh/usd/MJCF/obixxh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple_7(CustomObjects):
    def __init__(self,
                 name="apple__7",
                 obj_name="apple__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple/omzprq/usd/MJCF/omzprq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple_8(CustomObjects):
    def __init__(self,
                 name="apple__8",
                 obj_name="apple__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple/qrqzvs/usd/MJCF/qrqzvs.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple_9(CustomObjects):
    def __init__(self,
                 name="apple__9",
                 obj_name="apple__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple/rizrsp/usd/MJCF/rizrsp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple_10(CustomObjects):
    def __init__(self,
                 name="apple__10",
                 obj_name="apple__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple/ymhxqk/usd/MJCF/ymhxqk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple_11(CustomObjects):
    def __init__(self,
                 name="apple__11",
                 obj_name="apple__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple/yyuiva/usd/MJCF/yyuiva.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple_12(CustomObjects):
    def __init__(self,
                 name="apple__12",
                 obj_name="apple__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple/zlxfxt/usd/MJCF/zlxfxt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apple_13(CustomObjects):
    def __init__(self,
                 name="apple__13",
                 obj_name="apple__13",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple/zutnsf/usd/MJCF/zutnsf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ApplePie_1(CustomObjects):
    def __init__(self,
                 name="apple_pie__1",
                 obj_name="apple_pie__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple_pie/ejrgdj/usd/MJCF/ejrgdj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ApplePie_2(CustomObjects):
    def __init__(self,
                 name="apple_pie__2",
                 obj_name="apple_pie__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apple_pie/rpdhbr/usd/MJCF/rpdhbr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Apricot(CustomObjects):
    def __init__(self,
                 name="apricot",
                 obj_name="apricot",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/apricot/qmwmwm/usd/MJCF/qmwmwm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Artichoke(CustomObjects):
    def __init__(self,
                 name="artichoke",
                 obj_name="artichoke",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/artichoke/tfclmg/usd/MJCF/tfclmg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Ashtray_1(CustomObjects):
    def __init__(self,
                 name="ashtray__1",
                 obj_name="ashtray__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/ashtray/dhkkfo/usd/MJCF/dhkkfo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Ashtray_2(CustomObjects):
    def __init__(self,
                 name="ashtray__2",
                 obj_name="ashtray__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/ashtray/nfuxzd/usd/MJCF/nfuxzd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Asparagus_1(CustomObjects):
    def __init__(self,
                 name="asparagus__1",
                 obj_name="asparagus__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/asparagus/eodozo/usd/MJCF/eodozo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Asparagus_2(CustomObjects):
    def __init__(self,
                 name="asparagus__2",
                 obj_name="asparagus__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/asparagus/npggjn/usd/MJCF/npggjn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Asparagus_3(CustomObjects):
    def __init__(self,
                 name="asparagus__3",
                 obj_name="asparagus__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/asparagus/xguktb/usd/MJCF/xguktb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Avocado(CustomObjects):
    def __init__(self,
                 name="avocado",
                 obj_name="avocado",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/avocado/arswzs/usd/MJCF/arswzs.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfBreadcrumbs(CustomObjects):
    def __init__(self,
                 name="bag_of_breadcrumbs",
                 obj_name="bag_of_breadcrumbs",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_breadcrumbs/nvhvxe/usd/MJCF/nvhvxe.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfChips_1(CustomObjects):
    def __init__(self,
                 name="bag_of_chips__1",
                 obj_name="bag_of_chips__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_chips/bryahw/usd/MJCF/bryahw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfChips_2(CustomObjects):
    def __init__(self,
                 name="bag_of_chips__2",
                 obj_name="bag_of_chips__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_chips/dwkdko/usd/MJCF/dwkdko.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfChips_3(CustomObjects):
    def __init__(self,
                 name="bag_of_chips__3",
                 obj_name="bag_of_chips__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_chips/ennnjj/usd/MJCF/ennnjj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfChips_4(CustomObjects):
    def __init__(self,
                 name="bag_of_chips__4",
                 obj_name="bag_of_chips__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_chips/jphwer/usd/MJCF/jphwer.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfChips_5(CustomObjects):
    def __init__(self,
                 name="bag_of_chips__5",
                 obj_name="bag_of_chips__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_chips/qstxpj/usd/MJCF/qstxpj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfChips_6(CustomObjects):
    def __init__(self,
                 name="bag_of_chips__6",
                 obj_name="bag_of_chips__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_chips/uevvib/usd/MJCF/uevvib.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfChips_7(CustomObjects):
    def __init__(self,
                 name="bag_of_chips__7",
                 obj_name="bag_of_chips__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_chips/wnuoym/usd/MJCF/wnuoym.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfCookies_1(CustomObjects):
    def __init__(self,
                 name="bag_of_cookies__1",
                 obj_name="bag_of_cookies__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_cookies/bbyvsc/usd/MJCF/bbyvsc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfCookies_2(CustomObjects):
    def __init__(self,
                 name="bag_of_cookies__2",
                 obj_name="bag_of_cookies__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_cookies/ikivgk/usd/MJCF/ikivgk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfCookies_3(CustomObjects):
    def __init__(self,
                 name="bag_of_cookies__3",
                 obj_name="bag_of_cookies__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_cookies/ksjtde/usd/MJCF/ksjtde.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfCookies_4(CustomObjects):
    def __init__(self,
                 name="bag_of_cookies__4",
                 obj_name="bag_of_cookies__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_cookies/vafomx/usd/MJCF/vafomx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfFlour_1(CustomObjects):
    def __init__(self,
                 name="bag_of_flour__1",
                 obj_name="bag_of_flour__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_flour/fntqmd/usd/MJCF/fntqmd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfFlour_2(CustomObjects):
    def __init__(self,
                 name="bag_of_flour__2",
                 obj_name="bag_of_flour__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_flour/rlejxx/usd/MJCF/rlejxx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfJerky(CustomObjects):
    def __init__(self,
                 name="bag_of_jerky",
                 obj_name="bag_of_jerky",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_jerky/wblype/usd/MJCF/wblype.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfPopcorn_1(CustomObjects):
    def __init__(self,
                 name="bag_of_popcorn__1",
                 obj_name="bag_of_popcorn__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_popcorn/dmubtt/usd/MJCF/dmubtt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfPopcorn_2(CustomObjects):
    def __init__(self,
                 name="bag_of_popcorn__2",
                 obj_name="bag_of_popcorn__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_popcorn/ebygfp/usd/MJCF/ebygfp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfRice_1(CustomObjects):
    def __init__(self,
                 name="bag_of_rice__1",
                 obj_name="bag_of_rice__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_rice/eddcjz/usd/MJCF/eddcjz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfRice_2(CustomObjects):
    def __init__(self,
                 name="bag_of_rice__2",
                 obj_name="bag_of_rice__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_rice/feerye/usd/MJCF/feerye.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfRice_3(CustomObjects):
    def __init__(self,
                 name="bag_of_rice__3",
                 obj_name="bag_of_rice__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_rice/jzjqjb/usd/MJCF/jzjqjb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfRice_4(CustomObjects):
    def __init__(self,
                 name="bag_of_rice__4",
                 obj_name="bag_of_rice__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_rice/xcokwx/usd/MJCF/xcokwx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfShiitake(CustomObjects):
    def __init__(self,
                 name="bag_of_shiitake",
                 obj_name="bag_of_shiitake",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_shiitake/jazecy/usd/MJCF/jazecy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfSnacks(CustomObjects):
    def __init__(self,
                 name="bag_of_snacks",
                 obj_name="bag_of_snacks",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_snacks/pkwgid/usd/MJCF/pkwgid.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfStarch(CustomObjects):
    def __init__(self,
                 name="bag_of_starch",
                 obj_name="bag_of_starch",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_starch/npegpl/usd/MJCF/npegpl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfTea(CustomObjects):
    def __init__(self,
                 name="bag_of_tea",
                 obj_name="bag_of_tea",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_tea/jjweyi/usd/MJCF/jjweyi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BagOfYeast(CustomObjects):
    def __init__(self,
                 name="bag_of_yeast",
                 obj_name="bag_of_yeast",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bag_of_yeast/ibvtik/usd/MJCF/ibvtik.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bagel(CustomObjects):
    def __init__(self,
                 name="bagel",
                 obj_name="bagel",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bagel/zlxkry/usd/MJCF/zlxkry.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Baguette_1(CustomObjects):
    def __init__(self,
                 name="baguette__1",
                 obj_name="baguette__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/baguette/pjzkeh/usd/MJCF/pjzkeh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Baguette_2(CustomObjects):
    def __init__(self,
                 name="baguette__2",
                 obj_name="baguette__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/baguette/xhqnuc/usd/MJCF/xhqnuc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Baguette_3(CustomObjects):
    def __init__(self,
                 name="baguette__3",
                 obj_name="baguette__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/baguette/xydhpd/usd/MJCF/xydhpd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Baguette_4(CustomObjects):
    def __init__(self,
                 name="baguette__4",
                 obj_name="baguette__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/baguette/ypbyek/usd/MJCF/ypbyek.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BakingPowderJar(CustomObjects):
    def __init__(self,
                 name="baking_powder_jar",
                 obj_name="baking_powder_jar",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/baking_powder_jar/lgopij/usd/MJCF/lgopij.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Banana_1(CustomObjects):
    def __init__(self,
                 name="banana__1",
                 obj_name="banana__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/banana/verqwv/usd/MJCF/verqwv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Banana_2(CustomObjects):
    def __init__(self,
                 name="banana__2",
                 obj_name="banana__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/banana/vvyyyv/usd/MJCF/vvyyyv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Banana_3(CustomObjects):
    def __init__(self,
                 name="banana__3",
                 obj_name="banana__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/banana/wmglhc/usd/MJCF/wmglhc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Banana_4(CustomObjects):
    def __init__(self,
                 name="banana__4",
                 obj_name="banana__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/banana/znakxm/usd/MJCF/znakxm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BananaBread(CustomObjects):
    def __init__(self,
                 name="banana_bread",
                 obj_name="banana_bread",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/banana_bread/outrja/usd/MJCF/outrja.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bandage(CustomObjects):
    def __init__(self,
                 name="bandage",
                 obj_name="bandage",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bandage/riftxh/usd/MJCF/riftxh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BarSoap_1(CustomObjects):
    def __init__(self,
                 name="bar_soap__1",
                 obj_name="bar_soap__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bar_soap/feqemg/usd/MJCF/feqemg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BarSoap_2(CustomObjects):
    def __init__(self,
                 name="bar_soap__2",
                 obj_name="bar_soap__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bar_soap/lyigsj/usd/MJCF/lyigsj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BarSoap_3(CustomObjects):
    def __init__(self,
                 name="bar_soap__3",
                 obj_name="bar_soap__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bar_soap/ofargb/usd/MJCF/ofargb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BarSoap_4(CustomObjects):
    def __init__(self,
                 name="bar_soap__4",
                 obj_name="bar_soap__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bar_soap/ozifwa/usd/MJCF/ozifwa.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BarSoap_5(CustomObjects):
    def __init__(self,
                 name="bar_soap__5",
                 obj_name="bar_soap__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bar_soap/utgixp/usd/MJCF/utgixp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BasilJar(CustomObjects):
    def __init__(self,
                 name="basil_jar",
                 obj_name="basil_jar",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/basil_jar/swytaw/usd/MJCF/swytaw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Battery(CustomObjects):
    def __init__(self,
                 name="battery",
                 obj_name="battery",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/battery/dcjyzg/usd/MJCF/dcjyzg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Beaker_1(CustomObjects):
    def __init__(self,
                 name="beaker__1",
                 obj_name="beaker__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beaker/dtjmai/usd/MJCF/dtjmai.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Beaker_2(CustomObjects):
    def __init__(self,
                 name="beaker__2",
                 obj_name="beaker__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beaker/effbnc/usd/MJCF/effbnc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Beaker_3(CustomObjects):
    def __init__(self,
                 name="beaker__3",
                 obj_name="beaker__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beaker/exzsal/usd/MJCF/exzsal.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Beaker_4(CustomObjects):
    def __init__(self,
                 name="beaker__4",
                 obj_name="beaker__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beaker/fxrsyi/usd/MJCF/fxrsyi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Beaker_5(CustomObjects):
    def __init__(self,
                 name="beaker__5",
                 obj_name="beaker__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beaker/fyrkzs/usd/MJCF/fyrkzs.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Beaker_6(CustomObjects):
    def __init__(self,
                 name="beaker__6",
                 obj_name="beaker__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beaker/jdijek/usd/MJCF/jdijek.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Beaker_7(CustomObjects):
    def __init__(self,
                 name="beaker__7",
                 obj_name="beaker__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beaker/rhohgs/usd/MJCF/rhohgs.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Beaker_8(CustomObjects):
    def __init__(self,
                 name="beaker__8",
                 obj_name="beaker__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beaker/sfvswx/usd/MJCF/sfvswx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Beaker_9(CustomObjects):
    def __init__(self,
                 name="beaker__9",
                 obj_name="beaker__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beaker/sstojv/usd/MJCF/sstojv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Beaker_10(CustomObjects):
    def __init__(self,
                 name="beaker__10",
                 obj_name="beaker__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beaker/uobdoq/usd/MJCF/uobdoq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Beaker_11(CustomObjects):
    def __init__(self,
                 name="beaker__11",
                 obj_name="beaker__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beaker/uzgibd/usd/MJCF/uzgibd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Beaker_12(CustomObjects):
    def __init__(self,
                 name="beaker__12",
                 obj_name="beaker__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beaker/zycgen/usd/MJCF/zycgen.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeanCurd(CustomObjects):
    def __init__(self,
                 name="bean_curd",
                 obj_name="bean_curd",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bean_curd/hekigc/usd/MJCF/hekigc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeefBrothCarton(CustomObjects):
    def __init__(self,
                 name="beef_broth_carton",
                 obj_name="beef_broth_carton",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beef_broth_carton/ecqxgd/usd/MJCF/ecqxgd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeefsteakTomato_1(CustomObjects):
    def __init__(self,
                 name="beefsteak_tomato__1",
                 obj_name="beefsteak_tomato__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beefsteak_tomato/altlfz/usd/MJCF/altlfz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeefsteakTomato_2(CustomObjects):
    def __init__(self,
                 name="beefsteak_tomato__2",
                 obj_name="beefsteak_tomato__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beefsteak_tomato/eevvzv/usd/MJCF/eevvzv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeefsteakTomato_3(CustomObjects):
    def __init__(self,
                 name="beefsteak_tomato__3",
                 obj_name="beefsteak_tomato__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beefsteak_tomato/ogpans/usd/MJCF/ogpans.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeefsteakTomato_4(CustomObjects):
    def __init__(self,
                 name="beefsteak_tomato__4",
                 obj_name="beefsteak_tomato__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beefsteak_tomato/pnrdxh/usd/MJCF/pnrdxh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeeswaxCandle_1(CustomObjects):
    def __init__(self,
                 name="beeswax_candle__1",
                 obj_name="beeswax_candle__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beeswax_candle/aiuhyv/usd/MJCF/aiuhyv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeeswaxCandle_2(CustomObjects):
    def __init__(self,
                 name="beeswax_candle__2",
                 obj_name="beeswax_candle__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beeswax_candle/kxange/usd/MJCF/kxange.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeeswaxCandle_3(CustomObjects):
    def __init__(self,
                 name="beeswax_candle__3",
                 obj_name="beeswax_candle__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beeswax_candle/nhdnje/usd/MJCF/nhdnje.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeeswaxCandle_4(CustomObjects):
    def __init__(self,
                 name="beeswax_candle__4",
                 obj_name="beeswax_candle__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beeswax_candle/nxewyk/usd/MJCF/nxewyk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeeswaxCandle_5(CustomObjects):
    def __init__(self,
                 name="beeswax_candle__5",
                 obj_name="beeswax_candle__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beeswax_candle/oimgmh/usd/MJCF/oimgmh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeeswaxCandle_6(CustomObjects):
    def __init__(self,
                 name="beeswax_candle__6",
                 obj_name="beeswax_candle__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beeswax_candle/ouzkdj/usd/MJCF/ouzkdj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeeswaxCandle_7(CustomObjects):
    def __init__(self,
                 name="beeswax_candle__7",
                 obj_name="beeswax_candle__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beeswax_candle/pfewit/usd/MJCF/pfewit.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BeeswaxCandle_8(CustomObjects):
    def __init__(self,
                 name="beeswax_candle__8",
                 obj_name="beeswax_candle__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beeswax_candle/rptogj/usd/MJCF/rptogj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Beet(CustomObjects):
    def __init__(self,
                 name="beet",
                 obj_name="beet",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/beet/wantjv/usd/MJCF/wantjv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bell(CustomObjects):
    def __init__(self,
                 name="bell",
                 obj_name="bell",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bell/oshurh/usd/MJCF/oshurh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BellPepper_1(CustomObjects):
    def __init__(self,
                 name="bell_pepper__1",
                 obj_name="bell_pepper__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bell_pepper/ggurxn/usd/MJCF/ggurxn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BellPepper_2(CustomObjects):
    def __init__(self,
                 name="bell_pepper__2",
                 obj_name="bell_pepper__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bell_pepper/ihctxa/usd/MJCF/ihctxa.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BellPepper_3(CustomObjects):
    def __init__(self,
                 name="bell_pepper__3",
                 obj_name="bell_pepper__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bell_pepper/ukkycp/usd/MJCF/ukkycp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BellPepper_4(CustomObjects):
    def __init__(self,
                 name="bell_pepper__4",
                 obj_name="bell_pepper__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bell_pepper/uqcenz/usd/MJCF/uqcenz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BellPepper_5(CustomObjects):
    def __init__(self,
                 name="bell_pepper__5",
                 obj_name="bell_pepper__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bell_pepper/wszvwc/usd/MJCF/wszvwc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BirdFeedBag(CustomObjects):
    def __init__(self,
                 name="bird_feed_bag",
                 obj_name="bird_feed_bag",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bird_feed_bag/dpxnlc/usd/MJCF/dpxnlc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Biscuit(CustomObjects):
    def __init__(self,
                 name="biscuit",
                 obj_name="biscuit",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/biscuit/ukcwqw/usd/MJCF/ukcwqw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BlackboardEraser(CustomObjects):
    def __init__(self,
                 name="blackboard_eraser",
                 obj_name="blackboard_eraser",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/blackboard_eraser/oynrtw/usd/MJCF/oynrtw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BobbyPin(CustomObjects):
    def __init__(self,
                 name="bobby_pin",
                 obj_name="bobby_pin",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bobby_pin/zphpcz/usd/MJCF/zphpcz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BokChoy_1(CustomObjects):
    def __init__(self,
                 name="bok_choy__1",
                 obj_name="bok_choy__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bok_choy/bbvcji/usd/MJCF/bbvcji.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BokChoy_2(CustomObjects):
    def __init__(self,
                 name="bok_choy__2",
                 obj_name="bok_choy__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bok_choy/jpkewd/usd/MJCF/jpkewd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bookend_1(CustomObjects):
    def __init__(self,
                 name="bookend__1",
                 obj_name="bookend__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bookend/detqsw/usd/MJCF/detqsw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bookend_2(CustomObjects):
    def __init__(self,
                 name="bookend__2",
                 obj_name="bookend__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bookend/eyddem/usd/MJCF/eyddem.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bookend_3(CustomObjects):
    def __init__(self,
                 name="bookend__3",
                 obj_name="bookend__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bookend/hlpgle/usd/MJCF/hlpgle.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bookend_4(CustomObjects):
    def __init__(self,
                 name="bookend__4",
                 obj_name="bookend__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bookend/huxmnl/usd/MJCF/huxmnl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bookend_5(CustomObjects):
    def __init__(self,
                 name="bookend__5",
                 obj_name="bookend__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bookend/jmodol/usd/MJCF/jmodol.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bookend_6(CustomObjects):
    def __init__(self,
                 name="bookend__6",
                 obj_name="bookend__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bookend/oxfecv/usd/MJCF/oxfecv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bookend_7(CustomObjects):
    def __init__(self,
                 name="bookend__7",
                 obj_name="bookend__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bookend/rwvcah/usd/MJCF/rwvcah.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bookend_8(CustomObjects):
    def __init__(self,
                 name="bookend__8",
                 obj_name="bookend__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bookend/sgiryo/usd/MJCF/sgiryo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bookend_9(CustomObjects):
    def __init__(self,
                 name="bookend__9",
                 obj_name="bookend__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bookend/xqyxrq/usd/MJCF/xqyxrq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bookend_10(CustomObjects):
    def __init__(self,
                 name="bookend__10",
                 obj_name="bookend__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bookend/ygynyq/usd/MJCF/ygynyq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfAlcohol(CustomObjects):
    def __init__(self,
                 name="bottle_of_alcohol",
                 obj_name="bottle_of_alcohol",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_alcohol/qvhrjh/usd/MJCF/qvhrjh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfAlfredoSauce(CustomObjects):
    def __init__(self,
                 name="bottle_of_alfredo_sauce",
                 obj_name="bottle_of_alfredo_sauce",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_alfredo_sauce/xwzqjr/usd/MJCF/xwzqjr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfAlmondOil(CustomObjects):
    def __init__(self,
                 name="bottle_of_almond_oil",
                 obj_name="bottle_of_almond_oil",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_almond_oil/nlokfa/usd/MJCF/nlokfa.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfAntihistamines(CustomObjects):
    def __init__(self,
                 name="bottle_of_antihistamines",
                 obj_name="bottle_of_antihistamines",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_antihistamines/agavwx/usd/MJCF/agavwx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfAppleCider(CustomObjects):
    def __init__(self,
                 name="bottle_of_apple_cider",
                 obj_name="bottle_of_apple_cider",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_apple_cider/frekrp/usd/MJCF/frekrp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfAppleJuice(CustomObjects):
    def __init__(self,
                 name="bottle_of_apple_juice",
                 obj_name="bottle_of_apple_juice",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_apple_juice/xvrbdy/usd/MJCF/xvrbdy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfAspirin(CustomObjects):
    def __init__(self,
                 name="bottle_of_aspirin",
                 obj_name="bottle_of_aspirin",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_aspirin/psvktc/usd/MJCF/psvktc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBabyOil(CustomObjects):
    def __init__(self,
                 name="bottle_of_baby_oil",
                 obj_name="bottle_of_baby_oil",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_baby_oil/xpdlrr/usd/MJCF/xpdlrr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBarbecueSauce_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_barbecue_sauce__1",
                 obj_name="bottle_of_barbecue_sauce__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_barbecue_sauce/ikbsox/usd/MJCF/ikbsox.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBarbecueSauce_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_barbecue_sauce__2",
                 obj_name="bottle_of_barbecue_sauce__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_barbecue_sauce/nkqvex/usd/MJCF/nkqvex.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBarbecueSauce_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_barbecue_sauce__3",
                 obj_name="bottle_of_barbecue_sauce__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_barbecue_sauce/rzevkb/usd/MJCF/rzevkb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__1",
                 obj_name="bottle_of_beer__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/bwaboq/usd/MJCF/bwaboq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__2",
                 obj_name="bottle_of_beer__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/crfcwo/usd/MJCF/crfcwo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__3",
                 obj_name="bottle_of_beer__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/dcwvkg/usd/MJCF/dcwvkg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_4(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__4",
                 obj_name="bottle_of_beer__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/dqfsgv/usd/MJCF/dqfsgv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_5(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__5",
                 obj_name="bottle_of_beer__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/eicnxj/usd/MJCF/eicnxj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_6(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__6",
                 obj_name="bottle_of_beer__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/fcnrqt/usd/MJCF/fcnrqt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_7(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__7",
                 obj_name="bottle_of_beer__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/fgzjnb/usd/MJCF/fgzjnb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_8(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__8",
                 obj_name="bottle_of_beer__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/gxxbhh/usd/MJCF/gxxbhh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_9(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__9",
                 obj_name="bottle_of_beer__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/hauvsg/usd/MJCF/hauvsg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_10(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__10",
                 obj_name="bottle_of_beer__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/ikgezm/usd/MJCF/ikgezm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_11(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__11",
                 obj_name="bottle_of_beer__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/jssuog/usd/MJCF/jssuog.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_12(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__12",
                 obj_name="bottle_of_beer__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/jtgyoo/usd/MJCF/jtgyoo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_13(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__13",
                 obj_name="bottle_of_beer__13",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/jxhtdl/usd/MJCF/jxhtdl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_14(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__14",
                 obj_name="bottle_of_beer__14",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/kqskmv/usd/MJCF/kqskmv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_15(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__15",
                 obj_name="bottle_of_beer__15",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/meqliv/usd/MJCF/meqliv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_16(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__16",
                 obj_name="bottle_of_beer__16",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/mhzpkh/usd/MJCF/mhzpkh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_17(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__17",
                 obj_name="bottle_of_beer__17",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/miiijl/usd/MJCF/miiijl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_18(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__18",
                 obj_name="bottle_of_beer__18",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/mljzrl/usd/MJCF/mljzrl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_19(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__19",
                 obj_name="bottle_of_beer__19",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/nfzzqc/usd/MJCF/nfzzqc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_20(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__20",
                 obj_name="bottle_of_beer__20",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/noxtlc/usd/MJCF/noxtlc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_21(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__21",
                 obj_name="bottle_of_beer__21",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/phdimo/usd/MJCF/phdimo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_22(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__22",
                 obj_name="bottle_of_beer__22",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/qepxvl/usd/MJCF/qepxvl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_23(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__23",
                 obj_name="bottle_of_beer__23",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/rbpakt/usd/MJCF/rbpakt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_24(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__24",
                 obj_name="bottle_of_beer__24",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/rdnopv/usd/MJCF/rdnopv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_25(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__25",
                 obj_name="bottle_of_beer__25",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/rjwdae/usd/MJCF/rjwdae.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_26(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__26",
                 obj_name="bottle_of_beer__26",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/saslsh/usd/MJCF/saslsh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_27(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__27",
                 obj_name="bottle_of_beer__27",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/ukbhdj/usd/MJCF/ukbhdj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_28(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__28",
                 obj_name="bottle_of_beer__28",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/vhscym/usd/MJCF/vhscym.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_29(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__29",
                 obj_name="bottle_of_beer__29",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/xpqnfz/usd/MJCF/xpqnfz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_30(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__30",
                 obj_name="bottle_of_beer__30",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/zbsxro/usd/MJCF/zbsxro.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBeer_31(CustomObjects):
    def __init__(self,
                 name="bottle_of_beer__31",
                 obj_name="bottle_of_beer__31",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_beer/zlmwyn/usd/MJCF/zlmwyn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBlackPepper_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_black_pepper__1",
                 obj_name="bottle_of_black_pepper__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_black_pepper/ejtiig/usd/MJCF/ejtiig.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBlackPepper_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_black_pepper__2",
                 obj_name="bottle_of_black_pepper__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_black_pepper/honise/usd/MJCF/honise.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBlackPepper_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_black_pepper__3",
                 obj_name="bottle_of_black_pepper__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_black_pepper/ydzzrv/usd/MJCF/ydzzrv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBlackPepper_4(CustomObjects):
    def __init__(self,
                 name="bottle_of_black_pepper__4",
                 obj_name="bottle_of_black_pepper__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_black_pepper/zybfok/usd/MJCF/zybfok.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBleachAgent(CustomObjects):
    def __init__(self,
                 name="bottle_of_bleach_agent",
                 obj_name="bottle_of_bleach_agent",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_bleach_agent/lfjumk/usd/MJCF/lfjumk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfBugRepellent(CustomObjects):
    def __init__(self,
                 name="bottle_of_bug_repellent",
                 obj_name="bottle_of_bug_repellent",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_bug_repellent/qqztry/usd/MJCF/qqztry.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCarrotJuice(CustomObjects):
    def __init__(self,
                 name="bottle_of_carrot_juice",
                 obj_name="bottle_of_carrot_juice",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_carrot_juice/jkuhio/usd/MJCF/jkuhio.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCatsup_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_catsup__1",
                 obj_name="bottle_of_catsup__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_catsup/ahoiqe/usd/MJCF/ahoiqe.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCatsup_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_catsup__2",
                 obj_name="bottle_of_catsup__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_catsup/bcqfxb/usd/MJCF/bcqfxb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCatsup_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_catsup__3",
                 obj_name="bottle_of_catsup__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_catsup/dmyfdf/usd/MJCF/dmyfdf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCatsup_4(CustomObjects):
    def __init__(self,
                 name="bottle_of_catsup__4",
                 obj_name="bottle_of_catsup__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_catsup/hvxkso/usd/MJCF/hvxkso.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCatsup_5(CustomObjects):
    def __init__(self,
                 name="bottle_of_catsup__5",
                 obj_name="bottle_of_catsup__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_catsup/ialodu/usd/MJCF/ialodu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfChiliPepper(CustomObjects):
    def __init__(self,
                 name="bottle_of_chili_pepper",
                 obj_name="bottle_of_chili_pepper",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_chili_pepper/hjalqq/usd/MJCF/hjalqq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfChocolateSauce(CustomObjects):
    def __init__(self,
                 name="bottle_of_chocolate_sauce",
                 obj_name="bottle_of_chocolate_sauce",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_chocolate_sauce/tsyldw/usd/MJCF/tsyldw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCleaner_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_cleaner__1",
                 obj_name="bottle_of_cleaner__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_cleaner/svzbeq/usd/MJCF/svzbeq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCleaner_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_cleaner__2",
                 obj_name="bottle_of_cleaner__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_cleaner/ykzonz/usd/MJCF/ykzonz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCocoa(CustomObjects):
    def __init__(self,
                 name="bottle_of_cocoa",
                 obj_name="bottle_of_cocoa",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_cocoa/ganhpw/usd/MJCF/ganhpw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCoconutMilk(CustomObjects):
    def __init__(self,
                 name="bottle_of_coconut_milk",
                 obj_name="bottle_of_coconut_milk",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_coconut_milk/idenxg/usd/MJCF/idenxg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCoconutOil(CustomObjects):
    def __init__(self,
                 name="bottle_of_coconut_oil",
                 obj_name="bottle_of_coconut_oil",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_coconut_oil/rrwzkq/usd/MJCF/rrwzkq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCoconutWater(CustomObjects):
    def __init__(self,
                 name="bottle_of_coconut_water",
                 obj_name="bottle_of_coconut_water",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_coconut_water/lsixio/usd/MJCF/lsixio.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCoffee(CustomObjects):
    def __init__(self,
                 name="bottle_of_coffee",
                 obj_name="bottle_of_coffee",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_coffee/zywanc/usd/MJCF/zywanc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCoke(CustomObjects):
    def __init__(self,
                 name="bottle_of_coke",
                 obj_name="bottle_of_coke",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_coke/bmtvvb/usd/MJCF/bmtvvb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfColdCream(CustomObjects):
    def __init__(self,
                 name="bottle_of_cold_cream",
                 obj_name="bottle_of_cold_cream",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_cold_cream/lyzvuk/usd/MJCF/lyzvuk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCologne(CustomObjects):
    def __init__(self,
                 name="bottle_of_cologne",
                 obj_name="bottle_of_cologne",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_cologne/lyipur/usd/MJCF/lyipur.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfConditioner(CustomObjects):
    def __init__(self,
                 name="bottle_of_conditioner",
                 obj_name="bottle_of_conditioner",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_conditioner/teafxb/usd/MJCF/teafxb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCookingOil(CustomObjects):
    def __init__(self,
                 name="bottle_of_cooking_oil",
                 obj_name="bottle_of_cooking_oil",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_cooking_oil/ywrkyg/usd/MJCF/ywrkyg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfCranberryJuice(CustomObjects):
    def __init__(self,
                 name="bottle_of_cranberry_juice",
                 obj_name="bottle_of_cranberry_juice",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_cranberry_juice/heoxnw/usd/MJCF/heoxnw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfDetergent_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_detergent__1",
                 obj_name="bottle_of_detergent__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_detergent/gkpmii/usd/MJCF/gkpmii.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfDetergent_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_detergent__2",
                 obj_name="bottle_of_detergent__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_detergent/qjkmhq/usd/MJCF/qjkmhq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfDishSoap(CustomObjects):
    def __init__(self,
                 name="bottle_of_dish_soap",
                 obj_name="bottle_of_dish_soap",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_dish_soap/bnmixt/usd/MJCF/bnmixt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfDisinfectant_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_disinfectant__1",
                 obj_name="bottle_of_disinfectant__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_disinfectant/faedff/usd/MJCF/faedff.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfDisinfectant_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_disinfectant__2",
                 obj_name="bottle_of_disinfectant__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_disinfectant/ucqzck/usd/MJCF/ucqzck.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfEssentialOil_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_essential_oil__1",
                 obj_name="bottle_of_essential_oil__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_essential_oil/eyyhld/usd/MJCF/eyyhld.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfEssentialOil_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_essential_oil__2",
                 obj_name="bottle_of_essential_oil__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_essential_oil/wansva/usd/MJCF/wansva.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfEssentialOil_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_essential_oil__3",
                 obj_name="bottle_of_essential_oil__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_essential_oil/xhoipk/usd/MJCF/xhoipk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfEssentialOil_4(CustomObjects):
    def __init__(self,
                 name="bottle_of_essential_oil__4",
                 obj_name="bottle_of_essential_oil__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_essential_oil/xvqshn/usd/MJCF/xvqshn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfEssentialOil_5(CustomObjects):
    def __init__(self,
                 name="bottle_of_essential_oil__5",
                 obj_name="bottle_of_essential_oil__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_essential_oil/yjxvpg/usd/MJCF/yjxvpg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfFabricSoftener(CustomObjects):
    def __init__(self,
                 name="bottle_of_fabric_softener",
                 obj_name="bottle_of_fabric_softener",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_fabric_softener/rmrnev/usd/MJCF/rmrnev.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfFaceCream(CustomObjects):
    def __init__(self,
                 name="bottle_of_face_cream",
                 obj_name="bottle_of_face_cream",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_face_cream/dztaed/usd/MJCF/dztaed.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfFennel(CustomObjects):
    def __init__(self,
                 name="bottle_of_fennel",
                 obj_name="bottle_of_fennel",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_fennel/ihlkfu/usd/MJCF/ihlkfu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfFrosting(CustomObjects):
    def __init__(self,
                 name="bottle_of_frosting",
                 obj_name="bottle_of_frosting",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_frosting/eqdsmn/usd/MJCF/eqdsmn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfFruitPunch_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_fruit_punch__1",
                 obj_name="bottle_of_fruit_punch__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_fruit_punch/azcigi/usd/MJCF/azcigi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfFruitPunch_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_fruit_punch__2",
                 obj_name="bottle_of_fruit_punch__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_fruit_punch/ykfnwi/usd/MJCF/ykfnwi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfGarlicSauce(CustomObjects):
    def __init__(self,
                 name="bottle_of_garlic_sauce",
                 obj_name="bottle_of_garlic_sauce",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_garlic_sauce/ucnmax/usd/MJCF/ucnmax.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfGin(CustomObjects):
    def __init__(self,
                 name="bottle_of_gin",
                 obj_name="bottle_of_gin",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_gin/qzgcdx/usd/MJCF/qzgcdx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfGinger(CustomObjects):
    def __init__(self,
                 name="bottle_of_ginger",
                 obj_name="bottle_of_ginger",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_ginger/drqhzo/usd/MJCF/drqhzo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfGingerBeer(CustomObjects):
    def __init__(self,
                 name="bottle_of_ginger_beer",
                 obj_name="bottle_of_ginger_beer",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_ginger_beer/zkocwb/usd/MJCF/zkocwb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfGlassCleaner(CustomObjects):
    def __init__(self,
                 name="bottle_of_glass_cleaner",
                 obj_name="bottle_of_glass_cleaner",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_glass_cleaner/yukmlw/usd/MJCF/yukmlw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfGlue(CustomObjects):
    def __init__(self,
                 name="bottle_of_glue",
                 obj_name="bottle_of_glue",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_glue/evtytd/usd/MJCF/evtytd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfGroundCloves(CustomObjects):
    def __init__(self,
                 name="bottle_of_ground_cloves",
                 obj_name="bottle_of_ground_cloves",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_ground_cloves/vzamzb/usd/MJCF/vzamzb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfGroundMace(CustomObjects):
    def __init__(self,
                 name="bottle_of_ground_mace",
                 obj_name="bottle_of_ground_mace",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_ground_mace/lgpxro/usd/MJCF/lgpxro.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfGroundNutmeg(CustomObjects):
    def __init__(self,
                 name="bottle_of_ground_nutmeg",
                 obj_name="bottle_of_ground_nutmeg",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_ground_nutmeg/qebruq/usd/MJCF/qebruq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfHotSauce(CustomObjects):
    def __init__(self,
                 name="bottle_of_hot_sauce",
                 obj_name="bottle_of_hot_sauce",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_hot_sauce/zqhkzh/usd/MJCF/zqhkzh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfLavenderOil(CustomObjects):
    def __init__(self,
                 name="bottle_of_lavender_oil",
                 obj_name="bottle_of_lavender_oil",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_lavender_oil/csalbx/usd/MJCF/csalbx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfLemonJuice(CustomObjects):
    def __init__(self,
                 name="bottle_of_lemon_juice",
                 obj_name="bottle_of_lemon_juice",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_lemon_juice/vsjter/usd/MJCF/vsjter.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfLemonSauce(CustomObjects):
    def __init__(self,
                 name="bottle_of_lemon_sauce",
                 obj_name="bottle_of_lemon_sauce",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_lemon_sauce/iyijeb/usd/MJCF/iyijeb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfLemonade(CustomObjects):
    def __init__(self,
                 name="bottle_of_lemonade",
                 obj_name="bottle_of_lemonade",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_lemonade/hqobwj/usd/MJCF/hqobwj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfLimeJuice(CustomObjects):
    def __init__(self,
                 name="bottle_of_lime_juice",
                 obj_name="bottle_of_lime_juice",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_lime_juice/ouuhaa/usd/MJCF/ouuhaa.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfLiquidSoap(CustomObjects):
    def __init__(self,
                 name="bottle_of_liquid_soap",
                 obj_name="bottle_of_liquid_soap",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_liquid_soap/bhquvg/usd/MJCF/bhquvg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfLotion(CustomObjects):
    def __init__(self,
                 name="bottle_of_lotion",
                 obj_name="bottle_of_lotion",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_lotion/tkryrh/usd/MJCF/tkryrh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfLubricant(CustomObjects):
    def __init__(self,
                 name="bottle_of_lubricant",
                 obj_name="bottle_of_lubricant",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_lubricant/bjfgim/usd/MJCF/bjfgim.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMapleSyrup(CustomObjects):
    def __init__(self,
                 name="bottle_of_maple_syrup",
                 obj_name="bottle_of_maple_syrup",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_maple_syrup/qfgewi/usd/MJCF/qfgewi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMayonnaise(CustomObjects):
    def __init__(self,
                 name="bottle_of_mayonnaise",
                 obj_name="bottle_of_mayonnaise",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_mayonnaise/cirpak/usd/MJCF/cirpak.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMedicine_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_medicine__1",
                 obj_name="bottle_of_medicine__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_medicine/egondf/usd/MJCF/egondf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMedicine_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_medicine__2",
                 obj_name="bottle_of_medicine__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_medicine/fmfwng/usd/MJCF/fmfwng.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMedicine_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_medicine__3",
                 obj_name="bottle_of_medicine__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_medicine/hvocpc/usd/MJCF/hvocpc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMedicine_4(CustomObjects):
    def __init__(self,
                 name="bottle_of_medicine__4",
                 obj_name="bottle_of_medicine__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_medicine/kasbsy/usd/MJCF/kasbsy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMedicine_5(CustomObjects):
    def __init__(self,
                 name="bottle_of_medicine__5",
                 obj_name="bottle_of_medicine__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_medicine/kqkwoq/usd/MJCF/kqkwoq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMedicine_6(CustomObjects):
    def __init__(self,
                 name="bottle_of_medicine__6",
                 obj_name="bottle_of_medicine__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_medicine/qqsukh/usd/MJCF/qqsukh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMedicine_7(CustomObjects):
    def __init__(self,
                 name="bottle_of_medicine__7",
                 obj_name="bottle_of_medicine__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_medicine/syfpak/usd/MJCF/syfpak.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMedicine_8(CustomObjects):
    def __init__(self,
                 name="bottle_of_medicine__8",
                 obj_name="bottle_of_medicine__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_medicine/zfbnjh/usd/MJCF/zfbnjh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMilk_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_milk__1",
                 obj_name="bottle_of_milk__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_milk/czblzn/usd/MJCF/czblzn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMilk_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_milk__2",
                 obj_name="bottle_of_milk__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_milk/ddvsgl/usd/MJCF/ddvsgl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMilk_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_milk__3",
                 obj_name="bottle_of_milk__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_milk/debqen/usd/MJCF/debqen.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMilk_4(CustomObjects):
    def __init__(self,
                 name="bottle_of_milk__4",
                 obj_name="bottle_of_milk__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_milk/gchrwu/usd/MJCF/gchrwu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMilk_5(CustomObjects):
    def __init__(self,
                 name="bottle_of_milk__5",
                 obj_name="bottle_of_milk__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_milk/mrejrs/usd/MJCF/mrejrs.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMilk_6(CustomObjects):
    def __init__(self,
                 name="bottle_of_milk__6",
                 obj_name="bottle_of_milk__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_milk/qsnnbp/usd/MJCF/qsnnbp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMilk_7(CustomObjects):
    def __init__(self,
                 name="bottle_of_milk__7",
                 obj_name="bottle_of_milk__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_milk/utfsmp/usd/MJCF/utfsmp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMilk_8(CustomObjects):
    def __init__(self,
                 name="bottle_of_milk__8",
                 obj_name="bottle_of_milk__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_milk/uuvmfm/usd/MJCF/uuvmfm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMilkshake(CustomObjects):
    def __init__(self,
                 name="bottle_of_milkshake",
                 obj_name="bottle_of_milkshake",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_milkshake/naxqya/usd/MJCF/naxqya.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMolasses(CustomObjects):
    def __init__(self,
                 name="bottle_of_molasses",
                 obj_name="bottle_of_molasses",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_molasses/jvsjop/usd/MJCF/jvsjop.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMushroomSauce(CustomObjects):
    def __init__(self,
                 name="bottle_of_mushroom_sauce",
                 obj_name="bottle_of_mushroom_sauce",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_mushroom_sauce/xamfxi/usd/MJCF/xamfxi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMustard_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_mustard__1",
                 obj_name="bottle_of_mustard__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_mustard/qbbqat/usd/MJCF/qbbqat.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMustard_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_mustard__2",
                 obj_name="bottle_of_mustard__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_mustard/sjasxe/usd/MJCF/sjasxe.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfMustardSeeds(CustomObjects):
    def __init__(self,
                 name="bottle_of_mustard_seeds",
                 obj_name="bottle_of_mustard_seeds",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_mustard_seeds/grryaf/usd/MJCF/grryaf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfOil_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_oil__1",
                 obj_name="bottle_of_oil__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_oil/kzvkyp/usd/MJCF/kzvkyp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfOil_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_oil__2",
                 obj_name="bottle_of_oil__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_oil/ueanhf/usd/MJCF/ueanhf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfOliveOil_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_olive_oil__1",
                 obj_name="bottle_of_olive_oil__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_olive_oil/ajswsh/usd/MJCF/ajswsh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfOliveOil_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_olive_oil__2",
                 obj_name="bottle_of_olive_oil__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_olive_oil/cqycjk/usd/MJCF/cqycjk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfOliveOil_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_olive_oil__3",
                 obj_name="bottle_of_olive_oil__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_olive_oil/jocrsz/usd/MJCF/jocrsz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfOliveOil_4(CustomObjects):
    def __init__(self,
                 name="bottle_of_olive_oil__4",
                 obj_name="bottle_of_olive_oil__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_olive_oil/ksxqkk/usd/MJCF/ksxqkk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfOliveOil_5(CustomObjects):
    def __init__(self,
                 name="bottle_of_olive_oil__5",
                 obj_name="bottle_of_olive_oil__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_olive_oil/luikop/usd/MJCF/luikop.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfOliveOil_6(CustomObjects):
    def __init__(self,
                 name="bottle_of_olive_oil__6",
                 obj_name="bottle_of_olive_oil__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_olive_oil/lvsfgc/usd/MJCF/lvsfgc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfOliveOil_7(CustomObjects):
    def __init__(self,
                 name="bottle_of_olive_oil__7",
                 obj_name="bottle_of_olive_oil__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_olive_oil/wztvie/usd/MJCF/wztvie.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfOnionPowder(CustomObjects):
    def __init__(self,
                 name="bottle_of_onion_powder",
                 obj_name="bottle_of_onion_powder",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_onion_powder/xruqod/usd/MJCF/xruqod.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfOrangeJuice_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_orange_juice__1",
                 obj_name="bottle_of_orange_juice__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_orange_juice/edltwh/usd/MJCF/edltwh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfOrangeJuice_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_orange_juice__2",
                 obj_name="bottle_of_orange_juice__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_orange_juice/jcvqmb/usd/MJCF/jcvqmb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfOrangeJuice_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_orange_juice__3",
                 obj_name="bottle_of_orange_juice__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_orange_juice/rtqqor/usd/MJCF/rtqqor.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPaint(CustomObjects):
    def __init__(self,
                 name="bottle_of_paint",
                 obj_name="bottle_of_paint",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_paint/volzrj/usd/MJCF/volzrj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPapayaJuice_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_papaya_juice__1",
                 obj_name="bottle_of_papaya_juice__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_papaya_juice/nmfmxy/usd/MJCF/nmfmxy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPapayaJuice_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_papaya_juice__2",
                 obj_name="bottle_of_papaya_juice__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_papaya_juice/tcauim/usd/MJCF/tcauim.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPaprika(CustomObjects):
    def __init__(self,
                 name="bottle_of_paprika",
                 obj_name="bottle_of_paprika",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_paprika/nrdgrp/usd/MJCF/nrdgrp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPeanutButter(CustomObjects):
    def __init__(self,
                 name="bottle_of_peanut_butter",
                 obj_name="bottle_of_peanut_butter",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_peanut_butter/edcvwr/usd/MJCF/edcvwr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPerfume(CustomObjects):
    def __init__(self,
                 name="bottle_of_perfume",
                 obj_name="bottle_of_perfume",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_perfume/ipurzb/usd/MJCF/ipurzb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPesto(CustomObjects):
    def __init__(self,
                 name="bottle_of_pesto",
                 obj_name="bottle_of_pesto",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pesto/hyeetr/usd/MJCF/hyeetr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPizzaSauce(CustomObjects):
    def __init__(self,
                 name="bottle_of_pizza_sauce",
                 obj_name="bottle_of_pizza_sauce",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pizza_sauce/clttao/usd/MJCF/clttao.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__1",
                 obj_name="bottle_of_pop__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/bqretx/usd/MJCF/bqretx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__2",
                 obj_name="bottle_of_pop__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/cmqubs/usd/MJCF/cmqubs.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__3",
                 obj_name="bottle_of_pop__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/dlqmit/usd/MJCF/dlqmit.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_4(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__4",
                 obj_name="bottle_of_pop__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/drqbiy/usd/MJCF/drqbiy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_5(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__5",
                 obj_name="bottle_of_pop__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/exapzb/usd/MJCF/exapzb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_6(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__6",
                 obj_name="bottle_of_pop__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/eyazzi/usd/MJCF/eyazzi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_7(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__7",
                 obj_name="bottle_of_pop__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/ghxeqz/usd/MJCF/ghxeqz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_8(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__8",
                 obj_name="bottle_of_pop__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/gkwdyt/usd/MJCF/gkwdyt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_9(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__9",
                 obj_name="bottle_of_pop__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/haoywb/usd/MJCF/haoywb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_10(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__10",
                 obj_name="bottle_of_pop__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/hjrwqb/usd/MJCF/hjrwqb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_11(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__11",
                 obj_name="bottle_of_pop__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/iynssk/usd/MJCF/iynssk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_12(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__12",
                 obj_name="bottle_of_pop__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/jnoksl/usd/MJCF/jnoksl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_13(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__13",
                 obj_name="bottle_of_pop__13",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/kviogj/usd/MJCF/kviogj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_14(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__14",
                 obj_name="bottle_of_pop__14",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/mdznsn/usd/MJCF/mdznsn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_15(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__15",
                 obj_name="bottle_of_pop__15",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/mhzttm/usd/MJCF/mhzttm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_16(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__16",
                 obj_name="bottle_of_pop__16",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/msmlud/usd/MJCF/msmlud.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_17(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__17",
                 obj_name="bottle_of_pop__17",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/nepfjl/usd/MJCF/nepfjl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_18(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__18",
                 obj_name="bottle_of_pop__18",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/nyupqh/usd/MJCF/nyupqh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_19(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__19",
                 obj_name="bottle_of_pop__19",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/oeipjl/usd/MJCF/oeipjl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_20(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__20",
                 obj_name="bottle_of_pop__20",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/oghrnk/usd/MJCF/oghrnk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_21(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__21",
                 obj_name="bottle_of_pop__21",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/pfjzkn/usd/MJCF/pfjzkn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_22(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__22",
                 obj_name="bottle_of_pop__22",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/pvabxf/usd/MJCF/pvabxf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_23(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__23",
                 obj_name="bottle_of_pop__23",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/pyrics/usd/MJCF/pyrics.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_24(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__24",
                 obj_name="bottle_of_pop__24",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/rhclfg/usd/MJCF/rhclfg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_25(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__25",
                 obj_name="bottle_of_pop__25",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/ribekf/usd/MJCF/ribekf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_26(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__26",
                 obj_name="bottle_of_pop__26",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/sevoto/usd/MJCF/sevoto.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_27(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__27",
                 obj_name="bottle_of_pop__27",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/tfvmik/usd/MJCF/tfvmik.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_28(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__28",
                 obj_name="bottle_of_pop__28",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/twtsry/usd/MJCF/twtsry.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_29(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__29",
                 obj_name="bottle_of_pop__29",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/ubazru/usd/MJCF/ubazru.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_30(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__30",
                 obj_name="bottle_of_pop__30",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/uwdeok/usd/MJCF/uwdeok.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_31(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__31",
                 obj_name="bottle_of_pop__31",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/uwmchl/usd/MJCF/uwmchl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_32(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__32",
                 obj_name="bottle_of_pop__32",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/vfjhav/usd/MJCF/vfjhav.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_33(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__33",
                 obj_name="bottle_of_pop__33",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/wmqhul/usd/MJCF/wmqhul.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_34(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__34",
                 obj_name="bottle_of_pop__34",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/wrgmdt/usd/MJCF/wrgmdt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_35(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__35",
                 obj_name="bottle_of_pop__35",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/wuuoes/usd/MJCF/wuuoes.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_36(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__36",
                 obj_name="bottle_of_pop__36",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/xoldze/usd/MJCF/xoldze.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_37(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__37",
                 obj_name="bottle_of_pop__37",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/xvxcvv/usd/MJCF/xvxcvv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_38(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__38",
                 obj_name="bottle_of_pop__38",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/zjbqvb/usd/MJCF/zjbqvb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_39(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__39",
                 obj_name="bottle_of_pop__39",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/zsisxf/usd/MJCF/zsisxf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPop_40(CustomObjects):
    def __init__(self,
                 name="bottle_of_pop__40",
                 obj_name="bottle_of_pop__40",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pop/zxambx/usd/MJCF/zxambx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPoppySeeds(CustomObjects):
    def __init__(self,
                 name="bottle_of_poppy_seeds",
                 obj_name="bottle_of_poppy_seeds",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_poppy_seeds/xdtrgi/usd/MJCF/xdtrgi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPowder(CustomObjects):
    def __init__(self,
                 name="bottle_of_powder",
                 obj_name="bottle_of_powder",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_powder/vfabau/usd/MJCF/vfabau.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfProteinPowder_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_protein_powder__1",
                 obj_name="bottle_of_protein_powder__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_protein_powder/rbvrrp/usd/MJCF/rbvrrp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfProteinPowder_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_protein_powder__2",
                 obj_name="bottle_of_protein_powder__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_protein_powder/ysgesq/usd/MJCF/ysgesq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfPumpkinPieSpice(CustomObjects):
    def __init__(self,
                 name="bottle_of_pumpkin_pie_spice",
                 obj_name="bottle_of_pumpkin_pie_spice",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_pumpkin_pie_spice/oindrv/usd/MJCF/oindrv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSage(CustomObjects):
    def __init__(self,
                 name="bottle_of_sage",
                 obj_name="bottle_of_sage",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sage/bterim/usd/MJCF/bterim.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSake_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_sake__1",
                 obj_name="bottle_of_sake__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sake/hbpenm/usd/MJCF/hbpenm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSake_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_sake__2",
                 obj_name="bottle_of_sake__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sake/luadgb/usd/MJCF/luadgb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSake_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_sake__3",
                 obj_name="bottle_of_sake__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sake/mvvomd/usd/MJCF/mvvomd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSake_4(CustomObjects):
    def __init__(self,
                 name="bottle_of_sake__4",
                 obj_name="bottle_of_sake__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sake/pwtebq/usd/MJCF/pwtebq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSake_5(CustomObjects):
    def __init__(self,
                 name="bottle_of_sake__5",
                 obj_name="bottle_of_sake__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sake/rctijo/usd/MJCF/rctijo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSake_6(CustomObjects):
    def __init__(self,
                 name="bottle_of_sake__6",
                 obj_name="bottle_of_sake__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sake/swlykk/usd/MJCF/swlykk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSake_7(CustomObjects):
    def __init__(self,
                 name="bottle_of_sake__7",
                 obj_name="bottle_of_sake__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sake/utwglw/usd/MJCF/utwglw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSake_8(CustomObjects):
    def __init__(self,
                 name="bottle_of_sake__8",
                 obj_name="bottle_of_sake__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sake/vfxfuj/usd/MJCF/vfxfuj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSake_9(CustomObjects):
    def __init__(self,
                 name="bottle_of_sake__9",
                 obj_name="bottle_of_sake__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sake/zrkfim/usd/MJCF/zrkfim.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSalsa_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_salsa__1",
                 obj_name="bottle_of_salsa__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_salsa/dvxotd/usd/MJCF/dvxotd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSalsa_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_salsa__2",
                 obj_name="bottle_of_salsa__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_salsa/mavope/usd/MJCF/mavope.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSalsa_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_salsa__3",
                 obj_name="bottle_of_salsa__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_salsa/nafwlf/usd/MJCF/nafwlf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSeasoning(CustomObjects):
    def __init__(self,
                 name="bottle_of_seasoning",
                 obj_name="bottle_of_seasoning",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_seasoning/vqwqqv/usd/MJCF/vqwqqv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSesameOil(CustomObjects):
    def __init__(self,
                 name="bottle_of_sesame_oil",
                 obj_name="bottle_of_sesame_oil",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sesame_oil/jipawg/usd/MJCF/jipawg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSesameSeeds(CustomObjects):
    def __init__(self,
                 name="bottle_of_sesame_seeds",
                 obj_name="bottle_of_sesame_seeds",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sesame_seeds/lzndie/usd/MJCF/lzndie.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfShampoo_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_shampoo__1",
                 obj_name="bottle_of_shampoo__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_shampoo/dvrzmy/usd/MJCF/dvrzmy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfShampoo_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_shampoo__2",
                 obj_name="bottle_of_shampoo__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_shampoo/hlkpwd/usd/MJCF/hlkpwd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfShampoo_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_shampoo__3",
                 obj_name="bottle_of_shampoo__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_shampoo/lvvkhx/usd/MJCF/lvvkhx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfShampoo_4(CustomObjects):
    def __init__(self,
                 name="bottle_of_shampoo__4",
                 obj_name="bottle_of_shampoo__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_shampoo/stjjjm/usd/MJCF/stjjjm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSkinCream(CustomObjects):
    def __init__(self,
                 name="bottle_of_skin_cream",
                 obj_name="bottle_of_skin_cream",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_skin_cream/ynwxtx/usd/MJCF/ynwxtx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSoda(CustomObjects):
    def __init__(self,
                 name="bottle_of_soda",
                 obj_name="bottle_of_soda",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_soda/eqyqlx/usd/MJCF/eqyqlx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSolvent(CustomObjects):
    def __init__(self,
                 name="bottle_of_solvent",
                 obj_name="bottle_of_solvent",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_solvent/gsafbo/usd/MJCF/gsafbo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSoup_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_soup__1",
                 obj_name="bottle_of_soup__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_soup/maycxf/usd/MJCF/maycxf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSoup_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_soup__2",
                 obj_name="bottle_of_soup__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_soup/xctslq/usd/MJCF/xctslq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSourCream(CustomObjects):
    def __init__(self,
                 name="bottle_of_sour_cream",
                 obj_name="bottle_of_sour_cream",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sour_cream/pmawft/usd/MJCF/pmawft.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSoyMilk_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_soy_milk__1",
                 obj_name="bottle_of_soy_milk__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_soy_milk/lyxwhd/usd/MJCF/lyxwhd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSoyMilk_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_soy_milk__2",
                 obj_name="bottle_of_soy_milk__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_soy_milk/mcjlhs/usd/MJCF/mcjlhs.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSoySauce(CustomObjects):
    def __init__(self,
                 name="bottle_of_soy_sauce",
                 obj_name="bottle_of_soy_sauce",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_soy_sauce/afxisg/usd/MJCF/afxisg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSriracha(CustomObjects):
    def __init__(self,
                 name="bottle_of_sriracha",
                 obj_name="bottle_of_sriracha",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sriracha/gnklax/usd/MJCF/gnklax.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfStrawberryJuice(CustomObjects):
    def __init__(self,
                 name="bottle_of_strawberry_juice",
                 obj_name="bottle_of_strawberry_juice",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_strawberry_juice/mlnuza/usd/MJCF/mlnuza.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSunscreen(CustomObjects):
    def __init__(self,
                 name="bottle_of_sunscreen",
                 obj_name="bottle_of_sunscreen",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_sunscreen/prlrwi/usd/MJCF/prlrwi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSupplements_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_supplements__1",
                 obj_name="bottle_of_supplements__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_supplements/kgreql/usd/MJCF/kgreql.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfSupplements_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_supplements__2",
                 obj_name="bottle_of_supplements__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_supplements/oqakev/usd/MJCF/oqakev.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfTea_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_tea__1",
                 obj_name="bottle_of_tea__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_tea/iladfg/usd/MJCF/iladfg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfTea_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_tea__2",
                 obj_name="bottle_of_tea__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_tea/yatmrs/usd/MJCF/yatmrs.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfTeaLeaves(CustomObjects):
    def __init__(self,
                 name="bottle_of_tea_leaves",
                 obj_name="bottle_of_tea_leaves",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_tea_leaves/pfbtus/usd/MJCF/pfbtus.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfTomatoJuice(CustomObjects):
    def __init__(self,
                 name="bottle_of_tomato_juice",
                 obj_name="bottle_of_tomato_juice",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_tomato_juice/csumos/usd/MJCF/csumos.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfTomatoPaste_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_tomato_paste__1",
                 obj_name="bottle_of_tomato_paste__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_tomato_paste/kkqtjv/usd/MJCF/kkqtjv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfTomatoPaste_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_tomato_paste__2",
                 obj_name="bottle_of_tomato_paste__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_tomato_paste/pnshkj/usd/MJCF/pnshkj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfTomatoPaste_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_tomato_paste__3",
                 obj_name="bottle_of_tomato_paste__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_tomato_paste/toeelk/usd/MJCF/toeelk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfTonic_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_tonic__1",
                 obj_name="bottle_of_tonic__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_tonic/hpddkk/usd/MJCF/hpddkk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfTonic_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_tonic__2",
                 obj_name="bottle_of_tonic__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_tonic/zblovn/usd/MJCF/zblovn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfVinegar(CustomObjects):
    def __init__(self,
                 name="bottle_of_vinegar",
                 obj_name="bottle_of_vinegar",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_vinegar/snzyfk/usd/MJCF/snzyfk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_1(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__1",
                 obj_name="bottle_of_water__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/atvnqy/usd/MJCF/atvnqy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_2(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__2",
                 obj_name="bottle_of_water__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/cytqio/usd/MJCF/cytqio.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_3(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__3",
                 obj_name="bottle_of_water__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/cyzaue/usd/MJCF/cyzaue.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_4(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__4",
                 obj_name="bottle_of_water__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/eeaimz/usd/MJCF/eeaimz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_5(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__5",
                 obj_name="bottle_of_water__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/emquat/usd/MJCF/emquat.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_6(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__6",
                 obj_name="bottle_of_water__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/hrzznl/usd/MJCF/hrzznl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_7(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__7",
                 obj_name="bottle_of_water__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/jmwngr/usd/MJCF/jmwngr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_8(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__8",
                 obj_name="bottle_of_water__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/lkpaas/usd/MJCF/lkpaas.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_9(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__9",
                 obj_name="bottle_of_water__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/lojipo/usd/MJCF/lojipo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_10(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__10",
                 obj_name="bottle_of_water__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/migvlt/usd/MJCF/migvlt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_11(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__11",
                 obj_name="bottle_of_water__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/qaceen/usd/MJCF/qaceen.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_12(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__12",
                 obj_name="bottle_of_water__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/rmtdxh/usd/MJCF/rmtdxh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_13(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__13",
                 obj_name="bottle_of_water__13",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/rrkhva/usd/MJCF/rrkhva.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_14(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__14",
                 obj_name="bottle_of_water__14",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/suytyi/usd/MJCF/suytyi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BottleOfWater_15(CustomObjects):
    def __init__(self,
                 name="bottle_of_water__15",
                 obj_name="bottle_of_water__15",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bottle_of_water/sxffiv/usd/MJCF/sxffiv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BouillonCube(CustomObjects):
    def __init__(self,
                 name="bouillon_cube",
                 obj_name="bouillon_cube",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bouillon_cube/ctzwzz/usd/MJCF/ctzwzz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_1(CustomObjects):
    def __init__(self,
                 name="bowl__1",
                 obj_name="bowl__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/adciys/usd/MJCF/adciys.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_2(CustomObjects):
    def __init__(self,
                 name="bowl__2",
                 obj_name="bowl__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/ajzltc/usd/MJCF/ajzltc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_3(CustomObjects):
    def __init__(self,
                 name="bowl__3",
                 obj_name="bowl__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/aspeds/usd/MJCF/aspeds.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_4(CustomObjects):
    def __init__(self,
                 name="bowl__4",
                 obj_name="bowl__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/belcml/usd/MJCF/belcml.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_5(CustomObjects):
    def __init__(self,
                 name="bowl__5",
                 obj_name="bowl__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/bexgtn/usd/MJCF/bexgtn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_6(CustomObjects):
    def __init__(self,
                 name="bowl__6",
                 obj_name="bowl__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/byzaxy/usd/MJCF/byzaxy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_7(CustomObjects):
    def __init__(self,
                 name="bowl__7",
                 obj_name="bowl__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/ckxwea/usd/MJCF/ckxwea.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_8(CustomObjects):
    def __init__(self,
                 name="bowl__8",
                 obj_name="bowl__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/cypjlv/usd/MJCF/cypjlv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_9(CustomObjects):
    def __init__(self,
                 name="bowl__9",
                 obj_name="bowl__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/dalyim/usd/MJCF/dalyim.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_10(CustomObjects):
    def __init__(self,
                 name="bowl__10",
                 obj_name="bowl__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/eawgwj/usd/MJCF/eawgwj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_11(CustomObjects):
    def __init__(self,
                 name="bowl__11",
                 obj_name="bowl__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/eipwho/usd/MJCF/eipwho.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_12(CustomObjects):
    def __init__(self,
                 name="bowl__12",
                 obj_name="bowl__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/fedafr/usd/MJCF/fedafr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_13(CustomObjects):
    def __init__(self,
                 name="bowl__13",
                 obj_name="bowl__13",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/feuaak/usd/MJCF/feuaak.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_14(CustomObjects):
    def __init__(self,
                 name="bowl__14",
                 obj_name="bowl__14",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/fiarri/usd/MJCF/fiarri.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_15(CustomObjects):
    def __init__(self,
                 name="bowl__15",
                 obj_name="bowl__15",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/fwdfeg/usd/MJCF/fwdfeg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_16(CustomObjects):
    def __init__(self,
                 name="bowl__16",
                 obj_name="bowl__16",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/haewxp/usd/MJCF/haewxp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_17(CustomObjects):
    def __init__(self,
                 name="bowl__17",
                 obj_name="bowl__17",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/hitnkv/usd/MJCF/hitnkv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_18(CustomObjects):
    def __init__(self,
                 name="bowl__18",
                 obj_name="bowl__18",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/hpqjug/usd/MJCF/hpqjug.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_19(CustomObjects):
    def __init__(self,
                 name="bowl__19",
                 obj_name="bowl__19",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/hynhgz/usd/MJCF/hynhgz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_20(CustomObjects):
    def __init__(self,
                 name="bowl__20",
                 obj_name="bowl__20",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/jblalf/usd/MJCF/jblalf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_21(CustomObjects):
    def __init__(self,
                 name="bowl__21",
                 obj_name="bowl__21",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/jfvjep/usd/MJCF/jfvjep.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_22(CustomObjects):
    def __init__(self,
                 name="bowl__22",
                 obj_name="bowl__22",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/jhtxxh/usd/MJCF/jhtxxh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_23(CustomObjects):
    def __init__(self,
                 name="bowl__23",
                 obj_name="bowl__23",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/jpvcjv/usd/MJCF/jpvcjv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_24(CustomObjects):
    def __init__(self,
                 name="bowl__24",
                 obj_name="bowl__24",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/kasebx/usd/MJCF/kasebx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_25(CustomObjects):
    def __init__(self,
                 name="bowl__25",
                 obj_name="bowl__25",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/kdkrov/usd/MJCF/kdkrov.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_26(CustomObjects):
    def __init__(self,
                 name="bowl__26",
                 obj_name="bowl__26",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/kthvrl/usd/MJCF/kthvrl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_27(CustomObjects):
    def __init__(self,
                 name="bowl__27",
                 obj_name="bowl__27",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/lgaxzt/usd/MJCF/lgaxzt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_28(CustomObjects):
    def __init__(self,
                 name="bowl__28",
                 obj_name="bowl__28",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/mspdar/usd/MJCF/mspdar.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_29(CustomObjects):
    def __init__(self,
                 name="bowl__29",
                 obj_name="bowl__29",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/nkkhbn/usd/MJCF/nkkhbn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_30(CustomObjects):
    def __init__(self,
                 name="bowl__30",
                 obj_name="bowl__30",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/nmhxfz/usd/MJCF/nmhxfz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_31(CustomObjects):
    def __init__(self,
                 name="bowl__31",
                 obj_name="bowl__31",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/npuuir/usd/MJCF/npuuir.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_32(CustomObjects):
    def __init__(self,
                 name="bowl__32",
                 obj_name="bowl__32",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/oyidja/usd/MJCF/oyidja.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_33(CustomObjects):
    def __init__(self,
                 name="bowl__33",
                 obj_name="bowl__33",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/pihjqa/usd/MJCF/pihjqa.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_34(CustomObjects):
    def __init__(self,
                 name="bowl__34",
                 obj_name="bowl__34",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/pyilfa/usd/MJCF/pyilfa.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_35(CustomObjects):
    def __init__(self,
                 name="bowl__35",
                 obj_name="bowl__35",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/qzodht/usd/MJCF/qzodht.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_36(CustomObjects):
    def __init__(self,
                 name="bowl__36",
                 obj_name="bowl__36",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/rbnyxi/usd/MJCF/rbnyxi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_37(CustomObjects):
    def __init__(self,
                 name="bowl__37",
                 obj_name="bowl__37",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/rlwpcd/usd/MJCF/rlwpcd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_38(CustomObjects):
    def __init__(self,
                 name="bowl__38",
                 obj_name="bowl__38",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/sqqahm/usd/MJCF/sqqahm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_39(CustomObjects):
    def __init__(self,
                 name="bowl__39",
                 obj_name="bowl__39",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/szgdpc/usd/MJCF/szgdpc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_40(CustomObjects):
    def __init__(self,
                 name="bowl__40",
                 obj_name="bowl__40",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/tvtive/usd/MJCF/tvtive.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_41(CustomObjects):
    def __init__(self,
                 name="bowl__41",
                 obj_name="bowl__41",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/tyczoo/usd/MJCF/tyczoo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_42(CustomObjects):
    def __init__(self,
                 name="bowl__42",
                 obj_name="bowl__42",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/vccsrl/usd/MJCF/vccsrl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_43(CustomObjects):
    def __init__(self,
                 name="bowl__43",
                 obj_name="bowl__43",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/wryghu/usd/MJCF/wryghu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_44(CustomObjects):
    def __init__(self,
                 name="bowl__44",
                 obj_name="bowl__44",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/wtepsx/usd/MJCF/wtepsx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_45(CustomObjects):
    def __init__(self,
                 name="bowl__45",
                 obj_name="bowl__45",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/xplzbo/usd/MJCF/xplzbo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_46(CustomObjects):
    def __init__(self,
                 name="bowl__46",
                 obj_name="bowl__46",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/xpnlup/usd/MJCF/xpnlup.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bowl_47(CustomObjects):
    def __init__(self,
                 name="bowl__47",
                 obj_name="bowl__47",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bowl/ycbbwl/usd/MJCF/ycbbwl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfAlmondMilk(CustomObjects):
    def __init__(self,
                 name="box_of_almond_milk",
                 obj_name="box_of_almond_milk",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_almond_milk/oiiqwq/usd/MJCF/oiiqwq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfAluminiumFoil(CustomObjects):
    def __init__(self,
                 name="box_of_aluminium_foil",
                 obj_name="box_of_aluminium_foil",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_aluminium_foil/lhwgty/usd/MJCF/lhwgty.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfAppleJuice_1(CustomObjects):
    def __init__(self,
                 name="box_of_apple_juice__1",
                 obj_name="box_of_apple_juice__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_apple_juice/pttzdw/usd/MJCF/pttzdw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfAppleJuice_2(CustomObjects):
    def __init__(self,
                 name="box_of_apple_juice__2",
                 obj_name="box_of_apple_juice__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_apple_juice/zjzgjy/usd/MJCF/zjzgjy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfBakingMix_1(CustomObjects):
    def __init__(self,
                 name="box_of_baking_mix__1",
                 obj_name="box_of_baking_mix__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_baking_mix/fmewsz/usd/MJCF/fmewsz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfBakingMix_2(CustomObjects):
    def __init__(self,
                 name="box_of_baking_mix__2",
                 obj_name="box_of_baking_mix__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_baking_mix/fnbgwd/usd/MJCF/fnbgwd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfBakingMix_3(CustomObjects):
    def __init__(self,
                 name="box_of_baking_mix__3",
                 obj_name="box_of_baking_mix__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_baking_mix/lgusnr/usd/MJCF/lgusnr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfBakingMix_4(CustomObjects):
    def __init__(self,
                 name="box_of_baking_mix__4",
                 obj_name="box_of_baking_mix__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_baking_mix/ltccey/usd/MJCF/ltccey.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfBakingMix_5(CustomObjects):
    def __init__(self,
                 name="box_of_baking_mix__5",
                 obj_name="box_of_baking_mix__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_baking_mix/luwlfd/usd/MJCF/luwlfd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfBakingMix_6(CustomObjects):
    def __init__(self,
                 name="box_of_baking_mix__6",
                 obj_name="box_of_baking_mix__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_baking_mix/qkvqkm/usd/MJCF/qkvqkm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfBakingMix_7(CustomObjects):
    def __init__(self,
                 name="box_of_baking_mix__7",
                 obj_name="box_of_baking_mix__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_baking_mix/uhvkxj/usd/MJCF/uhvkxj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfBakingMix_8(CustomObjects):
    def __init__(self,
                 name="box_of_baking_mix__8",
                 obj_name="box_of_baking_mix__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_baking_mix/xcxrek/usd/MJCF/xcxrek.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfBakingPowder_1(CustomObjects):
    def __init__(self,
                 name="box_of_baking_powder__1",
                 obj_name="box_of_baking_powder__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_baking_powder/vzgrlv/usd/MJCF/vzgrlv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfBakingPowder_2(CustomObjects):
    def __init__(self,
                 name="box_of_baking_powder__2",
                 obj_name="box_of_baking_powder__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_baking_powder/zevydc/usd/MJCF/zevydc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfBakingSoda(CustomObjects):
    def __init__(self,
                 name="box_of_baking_soda",
                 obj_name="box_of_baking_soda",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_baking_soda/pskrgy/usd/MJCF/pskrgy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfBarley(CustomObjects):
    def __init__(self,
                 name="box_of_barley",
                 obj_name="box_of_barley",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_barley/lxpwnw/usd/MJCF/lxpwnw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfBrownSugar(CustomObjects):
    def __init__(self,
                 name="box_of_brown_sugar",
                 obj_name="box_of_brown_sugar",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_brown_sugar/kqyevo/usd/MJCF/kqyevo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfButter_1(CustomObjects):
    def __init__(self,
                 name="box_of_butter__1",
                 obj_name="box_of_butter__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_butter/mfjuil/usd/MJCF/mfjuil.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfButter_2(CustomObjects):
    def __init__(self,
                 name="box_of_butter__2",
                 obj_name="box_of_butter__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_butter/oixapu/usd/MJCF/oixapu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCandy_1(CustomObjects):
    def __init__(self,
                 name="box_of_candy__1",
                 obj_name="box_of_candy__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_candy/gvreky/usd/MJCF/gvreky.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCandy_2(CustomObjects):
    def __init__(self,
                 name="box_of_candy__2",
                 obj_name="box_of_candy__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_candy/jodwfv/usd/MJCF/jodwfv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCaneSugar_1(CustomObjects):
    def __init__(self,
                 name="box_of_cane_sugar__1",
                 obj_name="box_of_cane_sugar__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cane_sugar/cqlofx/usd/MJCF/cqlofx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCaneSugar_2(CustomObjects):
    def __init__(self,
                 name="box_of_cane_sugar__2",
                 obj_name="box_of_cane_sugar__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cane_sugar/gpbobo/usd/MJCF/gpbobo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCaneSugar_3(CustomObjects):
    def __init__(self,
                 name="box_of_cane_sugar__3",
                 obj_name="box_of_cane_sugar__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cane_sugar/pozpqi/usd/MJCF/pozpqi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCaneSugar_4(CustomObjects):
    def __init__(self,
                 name="box_of_cane_sugar__4",
                 obj_name="box_of_cane_sugar__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cane_sugar/qvjiti/usd/MJCF/qvjiti.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCaneSugar_5(CustomObjects):
    def __init__(self,
                 name="box_of_cane_sugar__5",
                 obj_name="box_of_cane_sugar__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cane_sugar/rvsivw/usd/MJCF/rvsivw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_1(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__1",
                 obj_name="box_of_cereal__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/dkhgxn/usd/MJCF/dkhgxn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_2(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__2",
                 obj_name="box_of_cereal__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/hcwbzw/usd/MJCF/hcwbzw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_3(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__3",
                 obj_name="box_of_cereal__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/hosfhj/usd/MJCF/hosfhj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_4(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__4",
                 obj_name="box_of_cereal__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/ihedod/usd/MJCF/ihedod.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_5(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__5",
                 obj_name="box_of_cereal__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/iucshm/usd/MJCF/iucshm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_6(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__6",
                 obj_name="box_of_cereal__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/pfwgkm/usd/MJCF/pfwgkm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_7(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__7",
                 obj_name="box_of_cereal__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/qirvmd/usd/MJCF/qirvmd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_8(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__8",
                 obj_name="box_of_cereal__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/sgirte/usd/MJCF/sgirte.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_9(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__9",
                 obj_name="box_of_cereal__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/surzft/usd/MJCF/surzft.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_10(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__10",
                 obj_name="box_of_cereal__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/tiykku/usd/MJCF/tiykku.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_11(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__11",
                 obj_name="box_of_cereal__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/wdkcbu/usd/MJCF/wdkcbu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_12(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__12",
                 obj_name="box_of_cereal__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/wrlalk/usd/MJCF/wrlalk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_13(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__13",
                 obj_name="box_of_cereal__13",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/ykuyqb/usd/MJCF/ykuyqb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCereal_14(CustomObjects):
    def __init__(self,
                 name="box_of_cereal__14",
                 obj_name="box_of_cereal__14",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cereal/yorray/usd/MJCF/yorray.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfChocolates(CustomObjects):
    def __init__(self,
                 name="box_of_chocolates",
                 obj_name="box_of_chocolates",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_chocolates/bdvkbh/usd/MJCF/bdvkbh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCoconutMilk(CustomObjects):
    def __init__(self,
                 name="box_of_coconut_milk",
                 obj_name="box_of_coconut_milk",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_coconut_milk/dagbyl/usd/MJCF/dagbyl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCoffee_1(CustomObjects):
    def __init__(self,
                 name="box_of_coffee__1",
                 obj_name="box_of_coffee__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_coffee/cjtadw/usd/MJCF/cjtadw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCoffee_2(CustomObjects):
    def __init__(self,
                 name="box_of_coffee__2",
                 obj_name="box_of_coffee__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_coffee/mreyla/usd/MJCF/mreyla.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_1(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__1",
                 obj_name="box_of_cookies__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/apjokz/usd/MJCF/apjokz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_2(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__2",
                 obj_name="box_of_cookies__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/dlhqft/usd/MJCF/dlhqft.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_3(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__3",
                 obj_name="box_of_cookies__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/fsdjun/usd/MJCF/fsdjun.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_4(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__4",
                 obj_name="box_of_cookies__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/fzlaol/usd/MJCF/fzlaol.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_5(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__5",
                 obj_name="box_of_cookies__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/hdgsxz/usd/MJCF/hdgsxz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_6(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__6",
                 obj_name="box_of_cookies__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/jtalxs/usd/MJCF/jtalxs.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_7(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__7",
                 obj_name="box_of_cookies__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/levwda/usd/MJCF/levwda.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_8(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__8",
                 obj_name="box_of_cookies__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/lwhbid/usd/MJCF/lwhbid.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_9(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__9",
                 obj_name="box_of_cookies__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/nirfva/usd/MJCF/nirfva.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_10(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__10",
                 obj_name="box_of_cookies__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/nnfisz/usd/MJCF/nnfisz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_11(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__11",
                 obj_name="box_of_cookies__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/ohhshb/usd/MJCF/ohhshb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_12(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__12",
                 obj_name="box_of_cookies__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/qkzrdd/usd/MJCF/qkzrdd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_13(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__13",
                 obj_name="box_of_cookies__13",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/sjcnoq/usd/MJCF/sjcnoq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCookies_14(CustomObjects):
    def __init__(self,
                 name="box_of_cookies__14",
                 obj_name="box_of_cookies__14",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cookies/swilxz/usd/MJCF/swilxz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCornFlakes(CustomObjects):
    def __init__(self,
                 name="box_of_corn_flakes",
                 obj_name="box_of_corn_flakes",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_corn_flakes/zbwzkq/usd/MJCF/zbwzkq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCrackers(CustomObjects):
    def __init__(self,
                 name="box_of_crackers",
                 obj_name="box_of_crackers",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_crackers/cmdigf/usd/MJCF/cmdigf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfCream(CustomObjects):
    def __init__(self,
                 name="box_of_cream",
                 obj_name="box_of_cream",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_cream/njrpai/usd/MJCF/njrpai.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfFlour(CustomObjects):
    def __init__(self,
                 name="box_of_flour",
                 obj_name="box_of_flour",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_flour/ylezpk/usd/MJCF/ylezpk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfFruit_1(CustomObjects):
    def __init__(self,
                 name="box_of_fruit__1",
                 obj_name="box_of_fruit__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_fruit/jsgbcz/usd/MJCF/jsgbcz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfFruit_2(CustomObjects):
    def __init__(self,
                 name="box_of_fruit__2",
                 obj_name="box_of_fruit__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_fruit/jzfgwc/usd/MJCF/jzfgwc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfGranolaBars_1(CustomObjects):
    def __init__(self,
                 name="box_of_granola_bars__1",
                 obj_name="box_of_granola_bars__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_granola_bars/awombj/usd/MJCF/awombj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfGranolaBars_2(CustomObjects):
    def __init__(self,
                 name="box_of_granola_bars__2",
                 obj_name="box_of_granola_bars__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_granola_bars/bqeeki/usd/MJCF/bqeeki.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfGranolaBars_3(CustomObjects):
    def __init__(self,
                 name="box_of_granola_bars__3",
                 obj_name="box_of_granola_bars__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_granola_bars/tqwhvz/usd/MJCF/tqwhvz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfLemons(CustomObjects):
    def __init__(self,
                 name="box_of_lemons",
                 obj_name="box_of_lemons",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_lemons/suhxjl/usd/MJCF/suhxjl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfMilk(CustomObjects):
    def __init__(self,
                 name="box_of_milk",
                 obj_name="box_of_milk",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_milk/ahmgjv/usd/MJCF/ahmgjv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfOatmeal_1(CustomObjects):
    def __init__(self,
                 name="box_of_oatmeal__1",
                 obj_name="box_of_oatmeal__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_oatmeal/jtqeef/usd/MJCF/jtqeef.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfOatmeal_2(CustomObjects):
    def __init__(self,
                 name="box_of_oatmeal__2",
                 obj_name="box_of_oatmeal__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_oatmeal/rabeel/usd/MJCF/rabeel.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfOatmeal_3(CustomObjects):
    def __init__(self,
                 name="box_of_oatmeal__3",
                 obj_name="box_of_oatmeal__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_oatmeal/zkggxm/usd/MJCF/zkggxm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfRaspberries(CustomObjects):
    def __init__(self,
                 name="box_of_raspberries",
                 obj_name="box_of_raspberries",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_raspberries/rxvopf/usd/MJCF/rxvopf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfRice_1(CustomObjects):
    def __init__(self,
                 name="box_of_rice__1",
                 obj_name="box_of_rice__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_rice/fvulow/usd/MJCF/fvulow.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfRice_2(CustomObjects):
    def __init__(self,
                 name="box_of_rice__2",
                 obj_name="box_of_rice__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_rice/pstqvm/usd/MJCF/pstqvm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfRice_3(CustomObjects):
    def __init__(self,
                 name="box_of_rice__3",
                 obj_name="box_of_rice__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_rice/yyheuw/usd/MJCF/yyheuw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSake_1(CustomObjects):
    def __init__(self,
                 name="box_of_sake__1",
                 obj_name="box_of_sake__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sake/abpqcs/usd/MJCF/abpqcs.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSake_2(CustomObjects):
    def __init__(self,
                 name="box_of_sake__2",
                 obj_name="box_of_sake__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sake/azjfky/usd/MJCF/azjfky.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSake_3(CustomObjects):
    def __init__(self,
                 name="box_of_sake__3",
                 obj_name="box_of_sake__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sake/hxrhoc/usd/MJCF/hxrhoc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSake_4(CustomObjects):
    def __init__(self,
                 name="box_of_sake__4",
                 obj_name="box_of_sake__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sake/iohsjz/usd/MJCF/iohsjz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSake_5(CustomObjects):
    def __init__(self,
                 name="box_of_sake__5",
                 obj_name="box_of_sake__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sake/ojvfmv/usd/MJCF/ojvfmv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSake_6(CustomObjects):
    def __init__(self,
                 name="box_of_sake__6",
                 obj_name="box_of_sake__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sake/qfrurx/usd/MJCF/qfrurx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSake_7(CustomObjects):
    def __init__(self,
                 name="box_of_sake__7",
                 obj_name="box_of_sake__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sake/rczpdw/usd/MJCF/rczpdw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSake_8(CustomObjects):
    def __init__(self,
                 name="box_of_sake__8",
                 obj_name="box_of_sake__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sake/tuzwhy/usd/MJCF/tuzwhy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSanitaryNapkins_1(CustomObjects):
    def __init__(self,
                 name="box_of_sanitary_napkins__1",
                 obj_name="box_of_sanitary_napkins__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sanitary_napkins/acaivd/usd/MJCF/acaivd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSanitaryNapkins_2(CustomObjects):
    def __init__(self,
                 name="box_of_sanitary_napkins__2",
                 obj_name="box_of_sanitary_napkins__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sanitary_napkins/arqcyu/usd/MJCF/arqcyu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSanitaryNapkins_3(CustomObjects):
    def __init__(self,
                 name="box_of_sanitary_napkins__3",
                 obj_name="box_of_sanitary_napkins__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sanitary_napkins/cmnpuv/usd/MJCF/cmnpuv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSanitaryNapkins_4(CustomObjects):
    def __init__(self,
                 name="box_of_sanitary_napkins__4",
                 obj_name="box_of_sanitary_napkins__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sanitary_napkins/humkqw/usd/MJCF/humkqw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSanitaryNapkins_5(CustomObjects):
    def __init__(self,
                 name="box_of_sanitary_napkins__5",
                 obj_name="box_of_sanitary_napkins__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sanitary_napkins/mxdefq/usd/MJCF/mxdefq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSanitaryNapkins_6(CustomObjects):
    def __init__(self,
                 name="box_of_sanitary_napkins__6",
                 obj_name="box_of_sanitary_napkins__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sanitary_napkins/skltgp/usd/MJCF/skltgp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfSanitaryNapkins_7(CustomObjects):
    def __init__(self,
                 name="box_of_sanitary_napkins__7",
                 obj_name="box_of_sanitary_napkins__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_sanitary_napkins/tkrzkb/usd/MJCF/tkrzkb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfShampoo_1(CustomObjects):
    def __init__(self,
                 name="box_of_shampoo__1",
                 obj_name="box_of_shampoo__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_shampoo/bclpiq/usd/MJCF/bclpiq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfShampoo_2(CustomObjects):
    def __init__(self,
                 name="box_of_shampoo__2",
                 obj_name="box_of_shampoo__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_shampoo/jccyom/usd/MJCF/jccyom.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfShampoo_3(CustomObjects):
    def __init__(self,
                 name="box_of_shampoo__3",
                 obj_name="box_of_shampoo__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_shampoo/nijidu/usd/MJCF/nijidu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfShampoo_4(CustomObjects):
    def __init__(self,
                 name="box_of_shampoo__4",
                 obj_name="box_of_shampoo__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_shampoo/tzghev/usd/MJCF/tzghev.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfTakeout(CustomObjects):
    def __init__(self,
                 name="box_of_takeout",
                 obj_name="box_of_takeout",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_takeout/bvcopi/usd/MJCF/bvcopi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfTissues_1(CustomObjects):
    def __init__(self,
                 name="box_of_tissues__1",
                 obj_name="box_of_tissues__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_tissues/ntbrtz/usd/MJCF/ntbrtz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfTissues_2(CustomObjects):
    def __init__(self,
                 name="box_of_tissues__2",
                 obj_name="box_of_tissues__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_tissues/uglbjc/usd/MJCF/uglbjc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfTissues_3(CustomObjects):
    def __init__(self,
                 name="box_of_tissues__3",
                 obj_name="box_of_tissues__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_tissues/xwstls/usd/MJCF/xwstls.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfTissues_4(CustomObjects):
    def __init__(self,
                 name="box_of_tissues__4",
                 obj_name="box_of_tissues__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_tissues/zutrxn/usd/MJCF/zutrxn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfVegetableJuice_1(CustomObjects):
    def __init__(self,
                 name="box_of_vegetable_juice__1",
                 obj_name="box_of_vegetable_juice__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_vegetable_juice/jsnnlv/usd/MJCF/jsnnlv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfVegetableJuice_2(CustomObjects):
    def __init__(self,
                 name="box_of_vegetable_juice__2",
                 obj_name="box_of_vegetable_juice__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_vegetable_juice/qgjdbn/usd/MJCF/qgjdbn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfWagashi(CustomObjects):
    def __init__(self,
                 name="box_of_wagashi",
                 obj_name="box_of_wagashi",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_wagashi/mlmrwy/usd/MJCF/mlmrwy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfYogurt_1(CustomObjects):
    def __init__(self,
                 name="box_of_yogurt__1",
                 obj_name="box_of_yogurt__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_yogurt/jkemqc/usd/MJCF/jkemqc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfYogurt_2(CustomObjects):
    def __init__(self,
                 name="box_of_yogurt__2",
                 obj_name="box_of_yogurt__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_yogurt/jtjcrx/usd/MJCF/jtjcrx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfYogurt_3(CustomObjects):
    def __init__(self,
                 name="box_of_yogurt__3",
                 obj_name="box_of_yogurt__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_yogurt/thxfjq/usd/MJCF/thxfjq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxOfYogurt_4(CustomObjects):
    def __init__(self,
                 name="box_of_yogurt__4",
                 obj_name="box_of_yogurt__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/box_of_yogurt/znhjgm/usd/MJCF/znhjgm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxedCake(CustomObjects):
    def __init__(self,
                 name="boxed_cake",
                 obj_name="boxed_cake",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/boxed_cake/yjazet/usd/MJCF/yjazet.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxedCpuBoard(CustomObjects):
    def __init__(self,
                 name="boxed_cpu_board",
                 obj_name="boxed_cpu_board",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/boxed_cpu_board/xlqvil/usd/MJCF/xlqvil.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxedInkCartridge_1(CustomObjects):
    def __init__(self,
                 name="boxed_ink_cartridge__1",
                 obj_name="boxed_ink_cartridge__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/boxed_ink_cartridge/ltblou/usd/MJCF/ltblou.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxedInkCartridge_2(CustomObjects):
    def __init__(self,
                 name="boxed_ink_cartridge__2",
                 obj_name="boxed_ink_cartridge__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/boxed_ink_cartridge/oepuip/usd/MJCF/oepuip.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxedInkCartridge_3(CustomObjects):
    def __init__(self,
                 name="boxed_ink_cartridge__3",
                 obj_name="boxed_ink_cartridge__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/boxed_ink_cartridge/oinelo/usd/MJCF/oinelo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BoxedRouter(CustomObjects):
    def __init__(self,
                 name="boxed_router",
                 obj_name="boxed_router",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/boxed_router/dsgtpk/usd/MJCF/dsgtpk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bratwurst_1(CustomObjects):
    def __init__(self,
                 name="bratwurst__1",
                 obj_name="bratwurst__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bratwurst/pqfrrn/usd/MJCF/pqfrrn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Bratwurst_2(CustomObjects):
    def __init__(self,
                 name="bratwurst__2",
                 obj_name="bratwurst__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bratwurst/wuyflp/usd/MJCF/wuyflp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BreadSlice_1(CustomObjects):
    def __init__(self,
                 name="bread_slice__1",
                 obj_name="bread_slice__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bread_slice/pfggnm/usd/MJCF/pfggnm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BreadSlice_2(CustomObjects):
    def __init__(self,
                 name="bread_slice__2",
                 obj_name="bread_slice__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bread_slice/pgzsxe/usd/MJCF/pgzsxe.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BreadSlice_3(CustomObjects):
    def __init__(self,
                 name="bread_slice__3",
                 obj_name="bread_slice__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bread_slice/yedlgq/usd/MJCF/yedlgq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BreadSlice_4(CustomObjects):
    def __init__(self,
                 name="bread_slice__4",
                 obj_name="bread_slice__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bread_slice/yremdf/usd/MJCF/yremdf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Broccoli(CustomObjects):
    def __init__(self,
                 name="broccoli",
                 obj_name="broccoli",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/broccoli/wsxavx/usd/MJCF/wsxavx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BroccoliRabe(CustomObjects):
    def __init__(self,
                 name="broccoli_rabe",
                 obj_name="broccoli_rabe",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/broccoli_rabe/ushqbz/usd/MJCF/ushqbz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Broccolini(CustomObjects):
    def __init__(self,
                 name="broccolini",
                 obj_name="broccolini",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/broccolini/rlsytp/usd/MJCF/rlsytp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BrownSugarSack(CustomObjects):
    def __init__(self,
                 name="brown_sugar_sack",
                 obj_name="brown_sugar_sack",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/brown_sugar_sack/uftzyo/usd/MJCF/uftzyo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Brownie(CustomObjects):
    def __init__(self,
                 name="brownie",
                 obj_name="brownie",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/brownie/vqgqja/usd/MJCF/vqgqja.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BrusselsSprouts_1(CustomObjects):
    def __init__(self,
                 name="brussels_sprouts__1",
                 obj_name="brussels_sprouts__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/brussels_sprouts/hkwyzk/usd/MJCF/hkwyzk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BrusselsSprouts_2(CustomObjects):
    def __init__(self,
                 name="brussels_sprouts__2",
                 obj_name="brussels_sprouts__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/brussels_sprouts/mbkrxe/usd/MJCF/mbkrxe.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BrusselsSprouts_3(CustomObjects):
    def __init__(self,
                 name="brussels_sprouts__3",
                 obj_name="brussels_sprouts__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/brussels_sprouts/siodbb/usd/MJCF/siodbb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BrusselsSprouts_4(CustomObjects):
    def __init__(self,
                 name="brussels_sprouts__4",
                 obj_name="brussels_sprouts__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/brussels_sprouts/vdamtq/usd/MJCF/vdamtq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BulldogClip(CustomObjects):
    def __init__(self,
                 name="bulldog_clip",
                 obj_name="bulldog_clip",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bulldog_clip/cqxnkn/usd/MJCF/cqxnkn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BunchOfBananas(CustomObjects):
    def __init__(self,
                 name="bunch_of_bananas",
                 obj_name="bunch_of_bananas",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/bunch_of_bananas/mxnwwk/usd/MJCF/mxnwwk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Buret(CustomObjects):
    def __init__(self,
                 name="buret",
                 obj_name="buret",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/buret/naqnom/usd/MJCF/naqnom.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BuretClamp_1(CustomObjects):
    def __init__(self,
                 name="buret_clamp__1",
                 obj_name="buret_clamp__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/buret_clamp/gknfxt/usd/MJCF/gknfxt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BuretClamp_2(CustomObjects):
    def __init__(self,
                 name="buret_clamp__2",
                 obj_name="buret_clamp__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/buret_clamp/jkzyfr/usd/MJCF/jkzyfr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BuretClamp_3(CustomObjects):
    def __init__(self,
                 name="buret_clamp__3",
                 obj_name="buret_clamp__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/buret_clamp/pflmbv/usd/MJCF/pflmbv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ButterCookie(CustomObjects):
    def __init__(self,
                 name="butter_cookie",
                 obj_name="butter_cookie",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/butter_cookie/kukrla/usd/MJCF/kukrla.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ButterPackage(CustomObjects):
    def __init__(self,
                 name="butter_package",
                 obj_name="butter_package",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/butter_package/qixpto/usd/MJCF/qixpto.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cactus(CustomObjects):
    def __init__(self,
                 name="cactus",
                 obj_name="cactus",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cactus/imitmg/usd/MJCF/imitmg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Calendar(CustomObjects):
    def __init__(self,
                 name="calendar",
                 obj_name="calendar",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/calendar/ydunsu/usd/MJCF/ydunsu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Caliper(CustomObjects):
    def __init__(self,
                 name="caliper",
                 obj_name="caliper",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/caliper/sngjmj/usd/MJCF/sngjmj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Can_1(CustomObjects):
    def __init__(self,
                 name="can__1",
                 obj_name="can__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can/damllm/usd/MJCF/damllm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Can_2(CustomObjects):
    def __init__(self,
                 name="can__2",
                 obj_name="can__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can/dhqrwr/usd/MJCF/dhqrwr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Can_3(CustomObjects):
    def __init__(self,
                 name="can__3",
                 obj_name="can__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can/moeqjz/usd/MJCF/moeqjz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfBakingMix_1(CustomObjects):
    def __init__(self,
                 name="can_of_baking_mix__1",
                 obj_name="can_of_baking_mix__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_baking_mix/blrqqz/usd/MJCF/blrqqz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfBakingMix_2(CustomObjects):
    def __init__(self,
                 name="can_of_baking_mix__2",
                 obj_name="can_of_baking_mix__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_baking_mix/fpbxfp/usd/MJCF/fpbxfp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfBakingMix_3(CustomObjects):
    def __init__(self,
                 name="can_of_baking_mix__3",
                 obj_name="can_of_baking_mix__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_baking_mix/ohtiap/usd/MJCF/ohtiap.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfBakingMix_4(CustomObjects):
    def __init__(self,
                 name="can_of_baking_mix__4",
                 obj_name="can_of_baking_mix__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_baking_mix/orwvfx/usd/MJCF/orwvfx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfBakingMix_5(CustomObjects):
    def __init__(self,
                 name="can_of_baking_mix__5",
                 obj_name="can_of_baking_mix__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_baking_mix/rxorlp/usd/MJCF/rxorlp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfBakingMix_6(CustomObjects):
    def __init__(self,
                 name="can_of_baking_mix__6",
                 obj_name="can_of_baking_mix__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_baking_mix/xefopo/usd/MJCF/xefopo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfBayLeaves(CustomObjects):
    def __init__(self,
                 name="can_of_bay_leaves",
                 obj_name="can_of_bay_leaves",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_bay_leaves/ppwvjf/usd/MJCF/ppwvjf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None


@register_object
class CanOfBeans_1(CustomObjects):
    def __init__(self,
                 name="can_of_beans__1",
                 obj_name="can_of_beans__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_beans/kclbuu/usd/MJCF/kclbuu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfBeans_2(CustomObjects):
    def __init__(self,
                 name="can_of_beans__2",
                 obj_name="can_of_beans__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_beans/ojqgjz/usd/MJCF/ojqgjz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfCatFood(CustomObjects):
    def __init__(self,
                 name="can_of_cat_food",
                 obj_name="can_of_cat_food",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_cat_food/omiuox/usd/MJCF/omiuox.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfCoffee_1(CustomObjects):
    def __init__(self,
                 name="can_of_coffee__1",
                 obj_name="can_of_coffee__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_coffee/poteji/usd/MJCF/poteji.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfCoffee_2(CustomObjects):
    def __init__(self,
                 name="can_of_coffee__2",
                 obj_name="can_of_coffee__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_coffee/zubwua/usd/MJCF/zubwua.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfCorn_1(CustomObjects):
    def __init__(self,
                 name="can_of_corn__1",
                 obj_name="can_of_corn__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_corn/alphlq/usd/MJCF/alphlq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfCorn_2(CustomObjects):
    def __init__(self,
                 name="can_of_corn__2",
                 obj_name="can_of_corn__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_corn/kwwlfn/usd/MJCF/kwwlfn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfCorn_3(CustomObjects):
    def __init__(self,
                 name="can_of_corn__3",
                 obj_name="can_of_corn__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_corn/pddtfk/usd/MJCF/pddtfk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfDogFood(CustomObjects):
    def __init__(self,
                 name="can_of_dog_food",
                 obj_name="can_of_dog_food",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_dog_food/rgwfxq/usd/MJCF/rgwfxq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfIcetea(CustomObjects):
    def __init__(self,
                 name="can_of_icetea",
                 obj_name="can_of_icetea",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_icetea/ifrjsc/usd/MJCF/ifrjsc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfOatmeal(CustomObjects):
    def __init__(self,
                 name="can_of_oatmeal",
                 obj_name="can_of_oatmeal",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_oatmeal/qyukhm/usd/MJCF/qyukhm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSardines(CustomObjects):
    def __init__(self,
                 name="can_of_sardines",
                 obj_name="can_of_sardines",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_sardines/dpgmry/usd/MJCF/dpgmry.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_1(CustomObjects):
    def __init__(self,
                 name="can_of_soda__1",
                 obj_name="can_of_soda__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/bfrzvk/usd/MJCF/bfrzvk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_2(CustomObjects):
    def __init__(self,
                 name="can_of_soda__2",
                 obj_name="can_of_soda__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/chwjfu/usd/MJCF/chwjfu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_3(CustomObjects):
    def __init__(self,
                 name="can_of_soda__3",
                 obj_name="can_of_soda__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/evcxlr/usd/MJCF/evcxlr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_4(CustomObjects):
    def __init__(self,
                 name="can_of_soda__4",
                 obj_name="can_of_soda__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/frewxk/usd/MJCF/frewxk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_5(CustomObjects):
    def __init__(self,
                 name="can_of_soda__5",
                 obj_name="can_of_soda__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/iloapr/usd/MJCF/iloapr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_6(CustomObjects):
    def __init__(self,
                 name="can_of_soda__6",
                 obj_name="can_of_soda__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/itolcg/usd/MJCF/itolcg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_7(CustomObjects):
    def __init__(self,
                 name="can_of_soda__7",
                 obj_name="can_of_soda__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/ixrfxv/usd/MJCF/ixrfxv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_8(CustomObjects):
    def __init__(self,
                 name="can_of_soda__8",
                 obj_name="can_of_soda__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/lugwcz/usd/MJCF/lugwcz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_9(CustomObjects):
    def __init__(self,
                 name="can_of_soda__9",
                 obj_name="can_of_soda__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/mrrozu/usd/MJCF/mrrozu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_10(CustomObjects):
    def __init__(self,
                 name="can_of_soda__10",
                 obj_name="can_of_soda__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/opivig/usd/MJCF/opivig.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_11(CustomObjects):
    def __init__(self,
                 name="can_of_soda__11",
                 obj_name="can_of_soda__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/ttxyui/usd/MJCF/ttxyui.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_12(CustomObjects):
    def __init__(self,
                 name="can_of_soda__12",
                 obj_name="can_of_soda__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/uzbpnw/usd/MJCF/uzbpnw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_13(CustomObjects):
    def __init__(self,
                 name="can_of_soda__13",
                 obj_name="can_of_soda__13",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/vszbvb/usd/MJCF/vszbvb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_14(CustomObjects):
    def __init__(self,
                 name="can_of_soda__14",
                 obj_name="can_of_soda__14",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/wbrrad/usd/MJCF/wbrrad.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_15(CustomObjects):
    def __init__(self,
                 name="can_of_soda__15",
                 obj_name="can_of_soda__15",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/xlyult/usd/MJCF/xlyult.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfSoda_16(CustomObjects):
    def __init__(self,
                 name="can_of_soda__16",
                 obj_name="can_of_soda__16",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_soda/xmjfcg/usd/MJCF/xmjfcg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfTomatoPaste(CustomObjects):
    def __init__(self,
                 name="can_of_tomato_paste",
                 obj_name="can_of_tomato_paste",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_tomato_paste/sqqdzb/usd/MJCF/sqqdzb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CanOfTomatoes(CustomObjects):
    def __init__(self,
                 name="can_of_tomatoes",
                 obj_name="can_of_tomatoes",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/can_of_tomatoes/ckdouu/usd/MJCF/ckdouu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CandleHolder_1(CustomObjects):
    def __init__(self,
                 name="candle_holder__1",
                 obj_name="candle_holder__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/candle_holder/nygnlp/usd/MJCF/nygnlp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CandleHolder_2(CustomObjects):
    def __init__(self,
                 name="candle_holder__2",
                 obj_name="candle_holder__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/candle_holder/svqdrl/usd/MJCF/svqdrl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CandleHolder_3(CustomObjects):
    def __init__(self,
                 name="candle_holder__3",
                 obj_name="candle_holder__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/candle_holder/szulaa/usd/MJCF/szulaa.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CandleHolder_4(CustomObjects):
    def __init__(self,
                 name="candle_holder__4",
                 obj_name="candle_holder__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/candle_holder/tnlkzg/usd/MJCF/tnlkzg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CandleHolder_5(CustomObjects):
    def __init__(self,
                 name="candle_holder__5",
                 obj_name="candle_holder__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/candle_holder/wiufnv/usd/MJCF/wiufnv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Canister_1(CustomObjects):
    def __init__(self,
                 name="canister__1",
                 obj_name="canister__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/canister/gqwnfv/usd/MJCF/gqwnfv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Canister_2(CustomObjects):
    def __init__(self,
                 name="canister__2",
                 obj_name="canister__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/canister/xcppkc/usd/MJCF/xcppkc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CannedFood_1(CustomObjects):
    def __init__(self,
                 name="canned_food__1",
                 obj_name="canned_food__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/canned_food/acgdtc/usd/MJCF/acgdtc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CannedFood_2(CustomObjects):
    def __init__(self,
                 name="canned_food__2",
                 obj_name="canned_food__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/canned_food/byakrm/usd/MJCF/byakrm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CannedFood_3(CustomObjects):
    def __init__(self,
                 name="canned_food__3",
                 obj_name="canned_food__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/canned_food/cbmndg/usd/MJCF/cbmndg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CannedFood_4(CustomObjects):
    def __init__(self,
                 name="canned_food__4",
                 obj_name="canned_food__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/canned_food/foetdd/usd/MJCF/foetdd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CannedFood_5(CustomObjects):
    def __init__(self,
                 name="canned_food__5",
                 obj_name="canned_food__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/canned_food/pkopdw/usd/MJCF/pkopdw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CannedFood_6(CustomObjects):
    def __init__(self,
                 name="canned_food__6",
                 obj_name="canned_food__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/canned_food/qhgdys/usd/MJCF/qhgdys.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CannedFood_7(CustomObjects):
    def __init__(self,
                 name="canned_food__7",
                 obj_name="canned_food__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/canned_food/ycbspm/usd/MJCF/ycbspm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CannedFood_8(CustomObjects):
    def __init__(self,
                 name="canned_food__8",
                 obj_name="canned_food__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/canned_food/ycodks/usd/MJCF/ycodks.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CannedFood_9(CustomObjects):
    def __init__(self,
                 name="canned_food__9",
                 obj_name="canned_food__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/canned_food/zfmfje/usd/MJCF/zfmfje.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Canteen_1(CustomObjects):
    def __init__(self,
                 name="canteen__1",
                 obj_name="canteen__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/canteen/ouhqnw/usd/MJCF/ouhqnw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Canteen_2(CustomObjects):
    def __init__(self,
                 name="canteen__2",
                 obj_name="canteen__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/canteen/ttxunv/usd/MJCF/ttxunv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_1(CustomObjects):
    def __init__(self,
                 name="cap__1",
                 obj_name="cap__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/actmgl/usd/MJCF/actmgl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_2(CustomObjects):
    def __init__(self,
                 name="cap__2",
                 obj_name="cap__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/amryfj/usd/MJCF/amryfj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_3(CustomObjects):
    def __init__(self,
                 name="cap__3",
                 obj_name="cap__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/arskxc/usd/MJCF/arskxc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_4(CustomObjects):
    def __init__(self,
                 name="cap__4",
                 obj_name="cap__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/ceizxn/usd/MJCF/ceizxn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_5(CustomObjects):
    def __init__(self,
                 name="cap__5",
                 obj_name="cap__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/ciopwh/usd/MJCF/ciopwh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_6(CustomObjects):
    def __init__(self,
                 name="cap__6",
                 obj_name="cap__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/clhquh/usd/MJCF/clhquh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_7(CustomObjects):
    def __init__(self,
                 name="cap__7",
                 obj_name="cap__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/dduopd/usd/MJCF/dduopd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_8(CustomObjects):
    def __init__(self,
                 name="cap__8",
                 obj_name="cap__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/dmzavi/usd/MJCF/dmzavi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_9(CustomObjects):
    def __init__(self,
                 name="cap__9",
                 obj_name="cap__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/dwpcld/usd/MJCF/dwpcld.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_10(CustomObjects):
    def __init__(self,
                 name="cap__10",
                 obj_name="cap__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/eukcfr/usd/MJCF/eukcfr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_11(CustomObjects):
    def __init__(self,
                 name="cap__11",
                 obj_name="cap__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/fomiem/usd/MJCF/fomiem.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_12(CustomObjects):
    def __init__(self,
                 name="cap__12",
                 obj_name="cap__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/ghwjwe/usd/MJCF/ghwjwe.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_13(CustomObjects):
    def __init__(self,
                 name="cap__13",
                 obj_name="cap__13",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/gmdwwe/usd/MJCF/gmdwwe.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_14(CustomObjects):
    def __init__(self,
                 name="cap__14",
                 obj_name="cap__14",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/gxkbcd/usd/MJCF/gxkbcd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_15(CustomObjects):
    def __init__(self,
                 name="cap__15",
                 obj_name="cap__15",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/hbafeb/usd/MJCF/hbafeb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_16(CustomObjects):
    def __init__(self,
                 name="cap__16",
                 obj_name="cap__16",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/hkvuxj/usd/MJCF/hkvuxj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_17(CustomObjects):
    def __init__(self,
                 name="cap__17",
                 obj_name="cap__17",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/iizvmn/usd/MJCF/iizvmn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_18(CustomObjects):
    def __init__(self,
                 name="cap__18",
                 obj_name="cap__18",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/iqeyba/usd/MJCF/iqeyba.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_19(CustomObjects):
    def __init__(self,
                 name="cap__19",
                 obj_name="cap__19",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/jybxvq/usd/MJCF/jybxvq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_20(CustomObjects):
    def __init__(self,
                 name="cap__20",
                 obj_name="cap__20",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/kaolpg/usd/MJCF/kaolpg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_21(CustomObjects):
    def __init__(self,
                 name="cap__21",
                 obj_name="cap__21",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/mgirzi/usd/MJCF/mgirzi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_22(CustomObjects):
    def __init__(self,
                 name="cap__22",
                 obj_name="cap__22",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/ngionj/usd/MJCF/ngionj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_23(CustomObjects):
    def __init__(self,
                 name="cap__23",
                 obj_name="cap__23",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/nkjxbc/usd/MJCF/nkjxbc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_24(CustomObjects):
    def __init__(self,
                 name="cap__24",
                 obj_name="cap__24",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/owcsun/usd/MJCF/owcsun.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_25(CustomObjects):
    def __init__(self,
                 name="cap__25",
                 obj_name="cap__25",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/oyofsv/usd/MJCF/oyofsv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_26(CustomObjects):
    def __init__(self,
                 name="cap__26",
                 obj_name="cap__26",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/pwsngg/usd/MJCF/pwsngg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_27(CustomObjects):
    def __init__(self,
                 name="cap__27",
                 obj_name="cap__27",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/qcfsnv/usd/MJCF/qcfsnv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_28(CustomObjects):
    def __init__(self,
                 name="cap__28",
                 obj_name="cap__28",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/qscujv/usd/MJCF/qscujv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_29(CustomObjects):
    def __init__(self,
                 name="cap__29",
                 obj_name="cap__29",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/qwrndi/usd/MJCF/qwrndi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_30(CustomObjects):
    def __init__(self,
                 name="cap__30",
                 obj_name="cap__30",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/rgtedj/usd/MJCF/rgtedj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_31(CustomObjects):
    def __init__(self,
                 name="cap__31",
                 obj_name="cap__31",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/rhoycw/usd/MJCF/rhoycw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_32(CustomObjects):
    def __init__(self,
                 name="cap__32",
                 obj_name="cap__32",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/sjsles/usd/MJCF/sjsles.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_33(CustomObjects):
    def __init__(self,
                 name="cap__33",
                 obj_name="cap__33",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/tkwpyr/usd/MJCF/tkwpyr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_34(CustomObjects):
    def __init__(self,
                 name="cap__34",
                 obj_name="cap__34",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/tpknvf/usd/MJCF/tpknvf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_35(CustomObjects):
    def __init__(self,
                 name="cap__35",
                 obj_name="cap__35",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/uxpeaz/usd/MJCF/uxpeaz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_36(CustomObjects):
    def __init__(self,
                 name="cap__36",
                 obj_name="cap__36",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/vnpjfn/usd/MJCF/vnpjfn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_37(CustomObjects):
    def __init__(self,
                 name="cap__37",
                 obj_name="cap__37",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/vsvwig/usd/MJCF/vsvwig.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_38(CustomObjects):
    def __init__(self,
                 name="cap__38",
                 obj_name="cap__38",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/wwbayp/usd/MJCF/wwbayp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_39(CustomObjects):
    def __init__(self,
                 name="cap__39",
                 obj_name="cap__39",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/xeswtq/usd/MJCF/xeswtq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_40(CustomObjects):
    def __init__(self,
                 name="cap__40",
                 obj_name="cap__40",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/xsxeij/usd/MJCF/xsxeij.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_41(CustomObjects):
    def __init__(self,
                 name="cap__41",
                 obj_name="cap__41",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/xxqyrt/usd/MJCF/xxqyrt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_42(CustomObjects):
    def __init__(self,
                 name="cap__42",
                 obj_name="cap__42",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/ygsmgm/usd/MJCF/ygsmgm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_43(CustomObjects):
    def __init__(self,
                 name="cap__43",
                 obj_name="cap__43",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/yivige/usd/MJCF/yivige.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_44(CustomObjects):
    def __init__(self,
                 name="cap__44",
                 obj_name="cap__44",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/yqvild/usd/MJCF/yqvild.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_45(CustomObjects):
    def __init__(self,
                 name="cap__45",
                 obj_name="cap__45",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/ytjxqn/usd/MJCF/ytjxqn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_46(CustomObjects):
    def __init__(self,
                 name="cap__46",
                 obj_name="cap__46",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/zggjif/usd/MJCF/zggjif.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cap_47(CustomObjects):
    def __init__(self,
                 name="cap__47",
                 obj_name="cap__47",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cap/zknitk/usd/MJCF/zknitk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Carafe_1(CustomObjects):
    def __init__(self,
                 name="carafe__1",
                 obj_name="carafe__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carafe/hdbsog/usd/MJCF/hdbsog.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Carafe_2(CustomObjects):
    def __init__(self,
                 name="carafe__2",
                 obj_name="carafe__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carafe/mdtkkv/usd/MJCF/mdtkkv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Carafe_3(CustomObjects):
    def __init__(self,
                 name="carafe__3",
                 obj_name="carafe__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carafe/ocjcgp/usd/MJCF/ocjcgp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cardstock(CustomObjects):
    def __init__(self,
                 name="cardstock",
                 obj_name="cardstock",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cardstock/bihwte/usd/MJCF/bihwte.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Carrot_1(CustomObjects):
    def __init__(self,
                 name="carrot__1",
                 obj_name="carrot__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carrot/aucrah/usd/MJCF/aucrah.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Carrot_2(CustomObjects):
    def __init__(self,
                 name="carrot__2",
                 obj_name="carrot__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carrot/nktmff/usd/MJCF/nktmff.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Carrot_3(CustomObjects):
    def __init__(self,
                 name="carrot__3",
                 obj_name="carrot__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carrot/qhmmmx/usd/MJCF/qhmmmx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Carton_1(CustomObjects):
    def __init__(self,
                 name="carton__1",
                 obj_name="carton__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton/causya/usd/MJCF/causya.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Carton_2(CustomObjects):
    def __init__(self,
                 name="carton__2",
                 obj_name="carton__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton/cdmmwy/usd/MJCF/cdmmwy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Carton_3(CustomObjects):
    def __init__(self,
                 name="carton__3",
                 obj_name="carton__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton/hhlmbi/usd/MJCF/hhlmbi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Carton_4(CustomObjects):
    def __init__(self,
                 name="carton__4",
                 obj_name="carton__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton/libote/usd/MJCF/libote.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Carton_5(CustomObjects):
    def __init__(self,
                 name="carton__5",
                 obj_name="carton__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton/msfzpz/usd/MJCF/msfzpz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Carton_6(CustomObjects):
    def __init__(self,
                 name="carton__6",
                 obj_name="carton__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton/sxlklf/usd/MJCF/sxlklf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Carton_7(CustomObjects):
    def __init__(self,
                 name="carton__7",
                 obj_name="carton__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton/ylrxhe/usd/MJCF/ylrxhe.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfEggs_1(CustomObjects):
    def __init__(self,
                 name="carton_of_eggs__1",
                 obj_name="carton_of_eggs__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_eggs/mimzbz/usd/MJCF/mimzbz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfEggs_2(CustomObjects):
    def __init__(self,
                 name="carton_of_eggs__2",
                 obj_name="carton_of_eggs__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_eggs/rixhgu/usd/MJCF/rixhgu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfEggs_3(CustomObjects):
    def __init__(self,
                 name="carton_of_eggs__3",
                 obj_name="carton_of_eggs__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_eggs/stxfxb/usd/MJCF/stxfxb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfEggs_4(CustomObjects):
    def __init__(self,
                 name="carton_of_eggs__4",
                 obj_name="carton_of_eggs__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_eggs/tacdgl/usd/MJCF/tacdgl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfMilk_1(CustomObjects):
    def __init__(self,
                 name="carton_of_milk__1",
                 obj_name="carton_of_milk__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_milk/atyqub/usd/MJCF/atyqub.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfMilk_2(CustomObjects):
    def __init__(self,
                 name="carton_of_milk__2",
                 obj_name="carton_of_milk__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_milk/kklgxk/usd/MJCF/kklgxk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfMilk_3(CustomObjects):
    def __init__(self,
                 name="carton_of_milk__3",
                 obj_name="carton_of_milk__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_milk/kszoro/usd/MJCF/kszoro.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfMilk_4(CustomObjects):
    def __init__(self,
                 name="carton_of_milk__4",
                 obj_name="carton_of_milk__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_milk/orikxq/usd/MJCF/orikxq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfMilk_5(CustomObjects):
    def __init__(self,
                 name="carton_of_milk__5",
                 obj_name="carton_of_milk__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_milk/vkttfb/usd/MJCF/vkttfb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfMilk_6(CustomObjects):
    def __init__(self,
                 name="carton_of_milk__6",
                 obj_name="carton_of_milk__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_milk/xmugpm/usd/MJCF/xmugpm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfMilk_7(CustomObjects):
    def __init__(self,
                 name="carton_of_milk__7",
                 obj_name="carton_of_milk__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_milk/znqqft/usd/MJCF/znqqft.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfOrangeJuice_1(CustomObjects):
    def __init__(self,
                 name="carton_of_orange_juice__1",
                 obj_name="carton_of_orange_juice__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_orange_juice/brryuo/usd/MJCF/brryuo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfOrangeJuice_2(CustomObjects):
    def __init__(self,
                 name="carton_of_orange_juice__2",
                 obj_name="carton_of_orange_juice__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_orange_juice/gpbmnk/usd/MJCF/gpbmnk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfPineappleJuice(CustomObjects):
    def __init__(self,
                 name="carton_of_pineapple_juice",
                 obj_name="carton_of_pineapple_juice",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_pineapple_juice/vzueyg/usd/MJCF/vzueyg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CartonOfSoyMilk(CustomObjects):
    def __init__(self,
                 name="carton_of_soy_milk",
                 obj_name="carton_of_soy_milk",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/carton_of_soy_milk/orgyvw/usd/MJCF/orgyvw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CaseOfEyeshadow(CustomObjects):
    def __init__(self,
                 name="case_of_eyeshadow",
                 obj_name="case_of_eyeshadow",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/case_of_eyeshadow/zgervc/usd/MJCF/zgervc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CatFoodTin(CustomObjects):
    def __init__(self,
                 name="cat_food_tin",
                 obj_name="cat_food_tin",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cat_food_tin/rclizj/usd/MJCF/rclizj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Catalog_1(CustomObjects):
    def __init__(self,
                 name="catalog__1",
                 obj_name="catalog__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/catalog/aygcnt/usd/MJCF/aygcnt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Catalog_2(CustomObjects):
    def __init__(self,
                 name="catalog__2",
                 obj_name="catalog__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/catalog/deirql/usd/MJCF/deirql.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Catalog_3(CustomObjects):
    def __init__(self,
                 name="catalog__3",
                 obj_name="catalog__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/catalog/uilzqm/usd/MJCF/uilzqm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Catalog_4(CustomObjects):
    def __init__(self,
                 name="catalog__4",
                 obj_name="catalog__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/catalog/zmidof/usd/MJCF/zmidof.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cd(CustomObjects):
    def __init__(self,
                 name="cd",
                 obj_name="cd",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cd/xkfnrj/usd/MJCF/xkfnrj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CelluloseTape_1(CustomObjects):
    def __init__(self,
                 name="cellulose_tape__1",
                 obj_name="cellulose_tape__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cellulose_tape/gchdhk/usd/MJCF/gchdhk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CelluloseTape_2(CustomObjects):
    def __init__(self,
                 name="cellulose_tape__2",
                 obj_name="cellulose_tape__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cellulose_tape/kavsnx/usd/MJCF/kavsnx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CelluloseTape_3(CustomObjects):
    def __init__(self,
                 name="cellulose_tape__3",
                 obj_name="cellulose_tape__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cellulose_tape/sklkyc/usd/MJCF/sklkyc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CelluloseTapeDispenser_1(CustomObjects):
    def __init__(self,
                 name="cellulose_tape_dispenser__1",
                 obj_name="cellulose_tape_dispenser__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cellulose_tape_dispenser/budhaz/usd/MJCF/budhaz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CelluloseTapeDispenser_2(CustomObjects):
    def __init__(self,
                 name="cellulose_tape_dispenser__2",
                 obj_name="cellulose_tape_dispenser__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cellulose_tape_dispenser/fetnry/usd/MJCF/fetnry.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CelluloseTapeDispenser_3(CustomObjects):
    def __init__(self,
                 name="cellulose_tape_dispenser__3",
                 obj_name="cellulose_tape_dispenser__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cellulose_tape_dispenser/yyekns/usd/MJCF/yyekns.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Charger(CustomObjects):
    def __init__(self,
                 name="charger",
                 obj_name="charger",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/charger/bapkyh/usd/MJCF/bapkyh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CheeseDanish(CustomObjects):
    def __init__(self,
                 name="cheese_danish",
                 obj_name="cheese_danish",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cheese_danish/hkdtwp/usd/MJCF/hkdtwp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CheeseTart_1(CustomObjects):
    def __init__(self,
                 name="cheese_tart__1",
                 obj_name="cheese_tart__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cheese_tart/dxluyi/usd/MJCF/dxluyi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CheeseTart_2(CustomObjects):
    def __init__(self,
                 name="cheese_tart__2",
                 obj_name="cheese_tart__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cheese_tart/pyynjg/usd/MJCF/pyynjg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CheeseTart_3(CustomObjects):
    def __init__(self,
                 name="cheese_tart__3",
                 obj_name="cheese_tart__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cheese_tart/rnsdha/usd/MJCF/rnsdha.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cheesecake(CustomObjects):
    def __init__(self,
                 name="cheesecake",
                 obj_name="cheesecake",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cheesecake/epmobi/usd/MJCF/epmobi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Chestnut_1(CustomObjects):
    def __init__(self,
                 name="chestnut__1",
                 obj_name="chestnut__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chestnut/fmomat/usd/MJCF/fmomat.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Chestnut_2(CustomObjects):
    def __init__(self,
                 name="chestnut__2",
                 obj_name="chestnut__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chestnut/gjbnba/usd/MJCF/gjbnba.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Chestnut_3(CustomObjects):
    def __init__(self,
                 name="chestnut__3",
                 obj_name="chestnut__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chestnut/tairrn/usd/MJCF/tairrn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Chestnut_4(CustomObjects):
    def __init__(self,
                 name="chestnut__4",
                 obj_name="chestnut__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chestnut/tulvpb/usd/MJCF/tulvpb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChiaSeedBag(CustomObjects):
    def __init__(self,
                 name="chia_seed_bag",
                 obj_name="chia_seed_bag",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chia_seed_bag/xkixrg/usd/MJCF/xkixrg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChickenBrothCarton(CustomObjects):
    def __init__(self,
                 name="chicken_broth_carton",
                 obj_name="chicken_broth_carton",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chicken_broth_carton/ztripg/usd/MJCF/ztripg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChickenSoupCarton(CustomObjects):
    def __init__(self,
                 name="chicken_soup_carton",
                 obj_name="chicken_soup_carton",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chicken_soup_carton/ooyqcr/usd/MJCF/ooyqcr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChickpeaCan(CustomObjects):
    def __init__(self,
                 name="chickpea_can",
                 obj_name="chickpea_can",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chickpea_can/jeqtzg/usd/MJCF/jeqtzg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Chili_1(CustomObjects):
    def __init__(self,
                 name="chili__1",
                 obj_name="chili__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chili/agecro/usd/MJCF/agecro.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Chili_2(CustomObjects):
    def __init__(self,
                 name="chili__2",
                 obj_name="chili__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chili/pbbkpz/usd/MJCF/pbbkpz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Chili_3(CustomObjects):
    def __init__(self,
                 name="chili__3",
                 obj_name="chili__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chili/rafkbt/usd/MJCF/rafkbt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Chili_4(CustomObjects):
    def __init__(self,
                 name="chili__4",
                 obj_name="chili__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chili/xhbpqh/usd/MJCF/xhbpqh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Chip(CustomObjects):
    def __init__(self,
                 name="chip",
                 obj_name="chip",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chip/obgeiz/usd/MJCF/obgeiz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Chives_1(CustomObjects):
    def __init__(self,
                 name="chives__1",
                 obj_name="chives__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chives/gboofh/usd/MJCF/gboofh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Chives_2(CustomObjects):
    def __init__(self,
                 name="chives__2",
                 obj_name="chives__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chives/vvacxt/usd/MJCF/vvacxt.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Chives_3(CustomObjects):
    def __init__(self,
                 name="chives__3",
                 obj_name="chives__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chives/yifjct/usd/MJCF/yifjct.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChocolateBar_1(CustomObjects):
    def __init__(self,
                 name="chocolate_bar__1",
                 obj_name="chocolate_bar__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chocolate_bar/amutpr/usd/MJCF/amutpr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChocolateBar_2(CustomObjects):
    def __init__(self,
                 name="chocolate_bar__2",
                 obj_name="chocolate_bar__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chocolate_bar/bcfudr/usd/MJCF/bcfudr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChocolateBar_3(CustomObjects):
    def __init__(self,
                 name="chocolate_bar__3",
                 obj_name="chocolate_bar__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chocolate_bar/dxnisi/usd/MJCF/dxnisi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChocolateBar_4(CustomObjects):
    def __init__(self,
                 name="chocolate_bar__4",
                 obj_name="chocolate_bar__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chocolate_bar/eqfznz/usd/MJCF/eqfznz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChocolateBar_5(CustomObjects):
    def __init__(self,
                 name="chocolate_bar__5",
                 obj_name="chocolate_bar__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chocolate_bar/wkjiri/usd/MJCF/wkjiri.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChocolateBiscuit_1(CustomObjects):
    def __init__(self,
                 name="chocolate_biscuit__1",
                 obj_name="chocolate_biscuit__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chocolate_biscuit/fwnyas/usd/MJCF/fwnyas.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChocolateBiscuit_2(CustomObjects):
    def __init__(self,
                 name="chocolate_biscuit__2",
                 obj_name="chocolate_biscuit__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chocolate_biscuit/xhmpht/usd/MJCF/xhmpht.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChocolateChipCookie_1(CustomObjects):
    def __init__(self,
                 name="chocolate_chip_cookie__1",
                 obj_name="chocolate_chip_cookie__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chocolate_chip_cookie/ggpang/usd/MJCF/ggpang.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChocolateChipCookie_2(CustomObjects):
    def __init__(self,
                 name="chocolate_chip_cookie__2",
                 obj_name="chocolate_chip_cookie__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chocolate_chip_cookie/oyhoth/usd/MJCF/oyhoth.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChocolateChipCookie_3(CustomObjects):
    def __init__(self,
                 name="chocolate_chip_cookie__3",
                 obj_name="chocolate_chip_cookie__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chocolate_chip_cookie/xprsse/usd/MJCF/xprsse.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppedLettuce_1(CustomObjects):
    def __init__(self,
                 name="chopped_lettuce__1",
                 obj_name="chopped_lettuce__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopped_lettuce/amarhu/usd/MJCF/amarhu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppedLettuce_2(CustomObjects):
    def __init__(self,
                 name="chopped_lettuce__2",
                 obj_name="chopped_lettuce__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopped_lettuce/bbyzry/usd/MJCF/bbyzry.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppedLettuce_3(CustomObjects):
    def __init__(self,
                 name="chopped_lettuce__3",
                 obj_name="chopped_lettuce__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopped_lettuce/bcxcij/usd/MJCF/bcxcij.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppedLettuce_4(CustomObjects):
    def __init__(self,
                 name="chopped_lettuce__4",
                 obj_name="chopped_lettuce__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopped_lettuce/bqqmxy/usd/MJCF/bqqmxy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppedLettuce_5(CustomObjects):
    def __init__(self,
                 name="chopped_lettuce__5",
                 obj_name="chopped_lettuce__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopped_lettuce/caybcx/usd/MJCF/caybcx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_1(CustomObjects):
    def __init__(self,
                 name="chopping_board__1",
                 obj_name="chopping_board__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/afwefw/usd/MJCF/afwefw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_2(CustomObjects):
    def __init__(self,
                 name="chopping_board__2",
                 obj_name="chopping_board__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/akgegh/usd/MJCF/akgegh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_3(CustomObjects):
    def __init__(self,
                 name="chopping_board__3",
                 obj_name="chopping_board__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/ayegwd/usd/MJCF/ayegwd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_4(CustomObjects):
    def __init__(self,
                 name="chopping_board__4",
                 obj_name="chopping_board__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/bzauwp/usd/MJCF/bzauwp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_5(CustomObjects):
    def __init__(self,
                 name="chopping_board__5",
                 obj_name="chopping_board__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/cptayn/usd/MJCF/cptayn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_6(CustomObjects):
    def __init__(self,
                 name="chopping_board__6",
                 obj_name="chopping_board__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/drjnag/usd/MJCF/drjnag.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_7(CustomObjects):
    def __init__(self,
                 name="chopping_board__7",
                 obj_name="chopping_board__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/gaxhrw/usd/MJCF/gaxhrw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_8(CustomObjects):
    def __init__(self,
                 name="chopping_board__8",
                 obj_name="chopping_board__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/iocgzv/usd/MJCF/iocgzv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_9(CustomObjects):
    def __init__(self,
                 name="chopping_board__9",
                 obj_name="chopping_board__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/ktxcvz/usd/MJCF/ktxcvz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_10(CustomObjects):
    def __init__(self,
                 name="chopping_board__10",
                 obj_name="chopping_board__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/mqsqhl/usd/MJCF/mqsqhl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_11(CustomObjects):
    def __init__(self,
                 name="chopping_board__11",
                 obj_name="chopping_board__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/mwmzzv/usd/MJCF/mwmzzv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_12(CustomObjects):
    def __init__(self,
                 name="chopping_board__12",
                 obj_name="chopping_board__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/ozrzrr/usd/MJCF/ozrzrr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_13(CustomObjects):
    def __init__(self,
                 name="chopping_board__13",
                 obj_name="chopping_board__13",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/psabiv/usd/MJCF/psabiv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_14(CustomObjects):
    def __init__(self,
                 name="chopping_board__14",
                 obj_name="chopping_board__14",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/sygezm/usd/MJCF/sygezm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ChoppingBoard_15(CustomObjects):
    def __init__(self,
                 name="chopping_board__15",
                 obj_name="chopping_board__15",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/chopping_board/uzeftd/usd/MJCF/uzeftd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CinnamonStick_1(CustomObjects):
    def __init__(self,
                 name="cinnamon_stick__1",
                 obj_name="cinnamon_stick__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cinnamon_stick/bmbjdf/usd/MJCF/bmbjdf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CinnamonStick_2(CustomObjects):
    def __init__(self,
                 name="cinnamon_stick__2",
                 obj_name="cinnamon_stick__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cinnamon_stick/cdkjfo/usd/MJCF/cdkjfo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CinnamonStick_3(CustomObjects):
    def __init__(self,
                 name="cinnamon_stick__3",
                 obj_name="cinnamon_stick__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cinnamon_stick/hjhcpm/usd/MJCF/hjhcpm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CinnamonStick_4(CustomObjects):
    def __init__(self,
                 name="cinnamon_stick__4",
                 obj_name="cinnamon_stick__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cinnamon_stick/kdaxdy/usd/MJCF/kdaxdy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CinnamonStick_5(CustomObjects):
    def __init__(self,
                 name="cinnamon_stick__5",
                 obj_name="cinnamon_stick__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cinnamon_stick/qmlyim/usd/MJCF/qmlyim.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CinnamonStick_6(CustomObjects):
    def __init__(self,
                 name="cinnamon_stick__6",
                 obj_name="cinnamon_stick__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cinnamon_stick/qsqvgk/usd/MJCF/qsqvgk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CinnamonStick_7(CustomObjects):
    def __init__(self,
                 name="cinnamon_stick__7",
                 obj_name="cinnamon_stick__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cinnamon_stick/qxpzdm/usd/MJCF/qxpzdm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CinnamonStick_8(CustomObjects):
    def __init__(self,
                 name="cinnamon_stick__8",
                 obj_name="cinnamon_stick__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cinnamon_stick/smfuqz/usd/MJCF/smfuqz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Clamp(CustomObjects):
    def __init__(self,
                 name="clamp",
                 obj_name="clamp",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/clamp/feswhy/usd/MJCF/feswhy.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CleansingBottle(CustomObjects):
    def __init__(self,
                 name="cleansing_bottle",
                 obj_name="cleansing_bottle",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cleansing_bottle/ovjhuf/usd/MJCF/ovjhuf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Clipboard_1(CustomObjects):
    def __init__(self,
                 name="clipboard__1",
                 obj_name="clipboard__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/clipboard/envjqe/usd/MJCF/envjqe.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Clipboard_2(CustomObjects):
    def __init__(self,
                 name="clipboard__2",
                 obj_name="clipboard__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/clipboard/gmxyfo/usd/MJCF/gmxyfo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Clipper(CustomObjects):
    def __init__(self,
                 name="clipper",
                 obj_name="clipper",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/clipper/befwbq/usd/MJCF/befwbq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CloveJar(CustomObjects):
    def __init__(self,
                 name="clove_jar",
                 obj_name="clove_jar",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/clove_jar/cqdioi/usd/MJCF/cqdioi.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Coaster_1(CustomObjects):
    def __init__(self,
                 name="coaster__1",
                 obj_name="coaster__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coaster/arjpcz/usd/MJCF/arjpcz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Coaster_2(CustomObjects):
    def __init__(self,
                 name="coaster__2",
                 obj_name="coaster__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coaster/httqaj/usd/MJCF/httqaj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CocoaPowderJar(CustomObjects):
    def __init__(self,
                 name="cocoa_powder_jar",
                 obj_name="cocoa_powder_jar",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cocoa_powder_jar/cjmtvq/usd/MJCF/cjmtvq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoconutOilJar(CustomObjects):
    def __init__(self,
                 name="coconut_oil_jar",
                 obj_name="coconut_oil_jar",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coconut_oil_jar/phimqa/usd/MJCF/phimqa.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeBeanJar(CustomObjects):
    def __init__(self,
                 name="coffee_bean_jar",
                 obj_name="coffee_bean_jar",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_bean_jar/loduxu/usd/MJCF/loduxu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeCup_1(CustomObjects):
    def __init__(self,
                 name="coffee_cup__1",
                 obj_name="coffee_cup__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_cup/ckkwmj/usd/MJCF/ckkwmj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeCup_2(CustomObjects):
    def __init__(self,
                 name="coffee_cup__2",
                 obj_name="coffee_cup__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_cup/dkxddg/usd/MJCF/dkxddg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeCup_3(CustomObjects):
    def __init__(self,
                 name="coffee_cup__3",
                 obj_name="coffee_cup__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_cup/fgizgn/usd/MJCF/fgizgn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeCup_4(CustomObjects):
    def __init__(self,
                 name="coffee_cup__4",
                 obj_name="coffee_cup__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_cup/ibhhfj/usd/MJCF/ibhhfj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeCup_5(CustomObjects):
    def __init__(self,
                 name="coffee_cup__5",
                 obj_name="coffee_cup__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_cup/nbhcgu/usd/MJCF/nbhcgu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeCup_6(CustomObjects):
    def __init__(self,
                 name="coffee_cup__6",
                 obj_name="coffee_cup__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_cup/nhzrei/usd/MJCF/nhzrei.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeCup_7(CustomObjects):
    def __init__(self,
                 name="coffee_cup__7",
                 obj_name="coffee_cup__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_cup/rixzrk/usd/MJCF/rixzrk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeCup_8(CustomObjects):
    def __init__(self,
                 name="coffee_cup__8",
                 obj_name="coffee_cup__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_cup/rypdvd/usd/MJCF/rypdvd.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeCup_9(CustomObjects):
    def __init__(self,
                 name="coffee_cup__9",
                 obj_name="coffee_cup__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_cup/siksnl/usd/MJCF/siksnl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeCup_10(CustomObjects):
    def __init__(self,
                 name="coffee_cup__10",
                 obj_name="coffee_cup__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_cup/skamgp/usd/MJCF/skamgp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeCup_11(CustomObjects):
    def __init__(self,
                 name="coffee_cup__11",
                 obj_name="coffee_cup__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_cup/xjdyon/usd/MJCF/xjdyon.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeCup_12(CustomObjects):
    def __init__(self,
                 name="coffee_cup__12",
                 obj_name="coffee_cup__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_cup/ykuftq/usd/MJCF/ykuftq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CoffeeGrinder(CustomObjects):
    def __init__(self,
                 name="coffee_grinder",
                 obj_name="coffee_grinder",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/coffee_grinder/bubzvn/usd/MJCF/bubzvn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColaBottle(CustomObjects):
    def __init__(self,
                 name="cola_bottle",
                 obj_name="cola_bottle",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cola_bottle/oyqdtz/usd/MJCF/oyqdtz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_1(CustomObjects):
    def __init__(self,
                 name="colored_pencil__1",
                 obj_name="colored_pencil__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/acumgp/usd/MJCF/acumgp.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_2(CustomObjects):
    def __init__(self,
                 name="colored_pencil__2",
                 obj_name="colored_pencil__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/bkdqwb/usd/MJCF/bkdqwb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_3(CustomObjects):
    def __init__(self,
                 name="colored_pencil__3",
                 obj_name="colored_pencil__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/bnsqcn/usd/MJCF/bnsqcn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_4(CustomObjects):
    def __init__(self,
                 name="colored_pencil__4",
                 obj_name="colored_pencil__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/deuvcx/usd/MJCF/deuvcx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_5(CustomObjects):
    def __init__(self,
                 name="colored_pencil__5",
                 obj_name="colored_pencil__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/egvqng/usd/MJCF/egvqng.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_6(CustomObjects):
    def __init__(self,
                 name="colored_pencil__6",
                 obj_name="colored_pencil__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/jdqvdl/usd/MJCF/jdqvdl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_7(CustomObjects):
    def __init__(self,
                 name="colored_pencil__7",
                 obj_name="colored_pencil__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/jnccuz/usd/MJCF/jnccuz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_8(CustomObjects):
    def __init__(self,
                 name="colored_pencil__8",
                 obj_name="colored_pencil__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/kadvlg/usd/MJCF/kadvlg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_9(CustomObjects):
    def __init__(self,
                 name="colored_pencil__9",
                 obj_name="colored_pencil__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/kgmapz/usd/MJCF/kgmapz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_10(CustomObjects):
    def __init__(self,
                 name="colored_pencil__10",
                 obj_name="colored_pencil__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/kjwoqm/usd/MJCF/kjwoqm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_11(CustomObjects):
    def __init__(self,
                 name="colored_pencil__11",
                 obj_name="colored_pencil__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/nssris/usd/MJCF/nssris.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_12(CustomObjects):
    def __init__(self,
                 name="colored_pencil__12",
                 obj_name="colored_pencil__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/qrxemk/usd/MJCF/qrxemk.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_13(CustomObjects):
    def __init__(self,
                 name="colored_pencil__13",
                 obj_name="colored_pencil__13",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/qwficr/usd/MJCF/qwficr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_14(CustomObjects):
    def __init__(self,
                 name="colored_pencil__14",
                 obj_name="colored_pencil__14",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/qxeydw/usd/MJCF/qxeydw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_15(CustomObjects):
    def __init__(self,
                 name="colored_pencil__15",
                 obj_name="colored_pencil__15",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/tzadtj/usd/MJCF/tzadtj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_16(CustomObjects):
    def __init__(self,
                 name="colored_pencil__16",
                 obj_name="colored_pencil__16",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/vdpmhz/usd/MJCF/vdpmhz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_17(CustomObjects):
    def __init__(self,
                 name="colored_pencil__17",
                 obj_name="colored_pencil__17",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/vtbwvo/usd/MJCF/vtbwvo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_18(CustomObjects):
    def __init__(self,
                 name="colored_pencil__18",
                 obj_name="colored_pencil__18",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/wifbfs/usd/MJCF/wifbfs.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_19(CustomObjects):
    def __init__(self,
                 name="colored_pencil__19",
                 obj_name="colored_pencil__19",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/wmjjvo/usd/MJCF/wmjjvo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ColoredPencil_20(CustomObjects):
    def __init__(self,
                 name="colored_pencil__20",
                 obj_name="colored_pencil__20",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/colored_pencil/zisrpq/usd/MJCF/zisrpq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Comb_1(CustomObjects):
    def __init__(self,
                 name="comb__1",
                 obj_name="comb__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/comb/lqnwhb/usd/MJCF/lqnwhb.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Comb_2(CustomObjects):
    def __init__(self,
                 name="comb__2",
                 obj_name="comb__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/comb/nybyjz/usd/MJCF/nybyjz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Comb_3(CustomObjects):
    def __init__(self,
                 name="comb__3",
                 obj_name="comb__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/comb/yopqrq/usd/MJCF/yopqrq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ComicBook_1(CustomObjects):
    def __init__(self,
                 name="comic_book__1",
                 obj_name="comic_book__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/comic_book/nekxsh/usd/MJCF/nekxsh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ComicBook_2(CustomObjects):
    def __init__(self,
                 name="comic_book__2",
                 obj_name="comic_book__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/comic_book/qkczyc/usd/MJCF/qkczyc.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ComicBook_3(CustomObjects):
    def __init__(self,
                 name="comic_book__3",
                 obj_name="comic_book__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/comic_book/scycof/usd/MJCF/scycof.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ComicBook_4(CustomObjects):
    def __init__(self,
                 name="comic_book__4",
                 obj_name="comic_book__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/comic_book/vzejnl/usd/MJCF/vzejnl.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class ComicBook_5(CustomObjects):
    def __init__(self,
                 name="comic_book__5",
                 obj_name="comic_book__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/comic_book/xmbxfm/usd/MJCF/xmbxfm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CookieCutter_1(CustomObjects):
    def __init__(self,
                 name="cookie_cutter__1",
                 obj_name="cookie_cutter__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cookie_cutter/fvxiun/usd/MJCF/fvxiun.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CookieCutter_2(CustomObjects):
    def __init__(self,
                 name="cookie_cutter__2",
                 obj_name="cookie_cutter__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cookie_cutter/jpscvj/usd/MJCF/jpscvj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CookieCutter_3(CustomObjects):
    def __init__(self,
                 name="cookie_cutter__3",
                 obj_name="cookie_cutter__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cookie_cutter/lqrfzo/usd/MJCF/lqrfzo.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CookieStick(CustomObjects):
    def __init__(self,
                 name="cookie_stick",
                 obj_name="cookie_stick",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cookie_stick/zlhayf/usd/MJCF/zlhayf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CookingOilBottle(CustomObjects):
    def __init__(self,
                 name="cooking_oil_bottle",
                 obj_name="cooking_oil_bottle",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cooking_oil_bottle/cfdond/usd/MJCF/cfdond.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CopperPot(CustomObjects):
    def __init__(self,
                 name="copper_pot",
                 obj_name="copper_pot",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/copper_pot/gqemcq/usd/MJCF/gqemcq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CopperWire(CustomObjects):
    def __init__(self,
                 name="copper_wire",
                 obj_name="copper_wire",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/copper_wire/nzafel/usd/MJCF/nzafel.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cork_1(CustomObjects):
    def __init__(self,
                 name="cork__1",
                 obj_name="cork__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cork/lseuwf/usd/MJCF/lseuwf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cork_2(CustomObjects):
    def __init__(self,
                 name="cork__2",
                 obj_name="cork__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cork/ncxgpe/usd/MJCF/ncxgpe.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cork_3(CustomObjects):
    def __init__(self,
                 name="cork__3",
                 obj_name="cork__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cork/uyceta/usd/MJCF/uyceta.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Corkscrew(CustomObjects):
    def __init__(self,
                 name="corkscrew",
                 obj_name="corkscrew",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/corkscrew/gqocna/usd/MJCF/gqocna.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CornstarchJar(CustomObjects):
    def __init__(self,
                 name="cornstarch_jar",
                 obj_name="cornstarch_jar",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cornstarch_jar/dhseui/usd/MJCF/dhseui.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Crayon_1(CustomObjects):
    def __init__(self,
                 name="crayon__1",
                 obj_name="crayon__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/crayon/coapeh/usd/MJCF/coapeh.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Crayon_2(CustomObjects):
    def __init__(self,
                 name="crayon__2",
                 obj_name="crayon__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/crayon/csglmn/usd/MJCF/csglmn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Crayon_3(CustomObjects):
    def __init__(self,
                 name="crayon__3",
                 obj_name="crayon__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/crayon/cvebde/usd/MJCF/cvebde.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Crayon_4(CustomObjects):
    def __init__(self,
                 name="crayon__4",
                 obj_name="crayon__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/crayon/diiinz/usd/MJCF/diiinz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Crayon_5(CustomObjects):
    def __init__(self,
                 name="crayon__5",
                 obj_name="crayon__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/crayon/gfgsev/usd/MJCF/gfgsev.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Crayon_6(CustomObjects):
    def __init__(self,
                 name="crayon__6",
                 obj_name="crayon__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/crayon/jfqetz/usd/MJCF/jfqetz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Crayon_7(CustomObjects):
    def __init__(self,
                 name="crayon__7",
                 obj_name="crayon__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/crayon/jqodxx/usd/MJCF/jqodxx.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Crayon_8(CustomObjects):
    def __init__(self,
                 name="crayon__8",
                 obj_name="crayon__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/crayon/uqlfwf/usd/MJCF/uqlfwf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Crayon_9(CustomObjects):
    def __init__(self,
                 name="crayon__9",
                 obj_name="crayon__9",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/crayon/uwmrwr/usd/MJCF/uwmrwr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Crayon_10(CustomObjects):
    def __init__(self,
                 name="crayon__10",
                 obj_name="crayon__10",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/crayon/vdkdur/usd/MJCF/vdkdur.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Crayon_11(CustomObjects):
    def __init__(self,
                 name="crayon__11",
                 obj_name="crayon__11",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/crayon/xmysum/usd/MJCF/xmysum.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Crayon_12(CustomObjects):
    def __init__(self,
                 name="crayon__12",
                 obj_name="crayon__12",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/crayon/zajomr/usd/MJCF/zajomr.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CreamCarton(CustomObjects):
    def __init__(self,
                 name="cream_carton",
                 obj_name="cream_carton",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cream_carton/lfjmos/usd/MJCF/lfjmos.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CreamCheeseBox(CustomObjects):
    def __init__(self,
                 name="cream_cheese_box",
                 obj_name="cream_cheese_box",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cream_cheese_box/hfclfn/usd/MJCF/hfclfn.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CreamPitcher_1(CustomObjects):
    def __init__(self,
                 name="cream_pitcher__1",
                 obj_name="cream_pitcher__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cream_pitcher/ompiss/usd/MJCF/ompiss.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CreamPitcher_2(CustomObjects):
    def __init__(self,
                 name="cream_pitcher__2",
                 obj_name="cream_pitcher__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cream_pitcher/wmkwhg/usd/MJCF/wmkwhg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CreditCardTerminal(CustomObjects):
    def __init__(self,
                 name="credit_card_terminal",
                 obj_name="credit_card_terminal",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/credit_card_terminal/bqpanz/usd/MJCF/bqpanz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Croissant_1(CustomObjects):
    def __init__(self,
                 name="croissant__1",
                 obj_name="croissant__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/croissant/hnbnap/usd/MJCF/hnbnap.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Croissant_2(CustomObjects):
    def __init__(self,
                 name="croissant__2",
                 obj_name="croissant__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/croissant/xxsanu/usd/MJCF/xxsanu.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cruet(CustomObjects):
    def __init__(self,
                 name="cruet",
                 obj_name="cruet",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cruet/njqmqv/usd/MJCF/njqmqv.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cucumber(CustomObjects):
    def __init__(self,
                 name="cucumber",
                 obj_name="cucumber",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cucumber/wcvwye/usd/MJCF/wcvwye.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CupHolder(CustomObjects):
    def __init__(self,
                 name="cup_holder",
                 obj_name="cup_holder",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cup_holder/wstfid/usd/MJCF/wstfid.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CupOfYogurt(CustomObjects):
    def __init__(self,
                 name="cup_of_yogurt",
                 obj_name="cup_of_yogurt",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cup_of_yogurt/kihdsj/usd/MJCF/kihdsj.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cupcake_1(CustomObjects):
    def __init__(self,
                 name="cupcake__1",
                 obj_name="cupcake__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cupcake/fabdnw/usd/MJCF/fabdnw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cupcake_2(CustomObjects):
    def __init__(self,
                 name="cupcake__2",
                 obj_name="cupcake__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cupcake/hvyxpw/usd/MJCF/hvyxpw.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cupcake_3(CustomObjects):
    def __init__(self,
                 name="cupcake__3",
                 obj_name="cupcake__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cupcake/mbhweg/usd/MJCF/mbhweg.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cupcake_4(CustomObjects):
    def __init__(self,
                 name="cupcake__4",
                 obj_name="cupcake__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cupcake/outske/usd/MJCF/outske.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cupcake_5(CustomObjects):
    def __init__(self,
                 name="cupcake__5",
                 obj_name="cupcake__5",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cupcake/pfwrlq/usd/MJCF/pfwrlq.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cupcake_6(CustomObjects):
    def __init__(self,
                 name="cupcake__6",
                 obj_name="cupcake__6",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cupcake/rpadye/usd/MJCF/rpadye.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cupcake_7(CustomObjects):
    def __init__(self,
                 name="cupcake__7",
                 obj_name="cupcake__7",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cupcake/sutaow/usd/MJCF/sutaow.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class Cupcake_8(CustomObjects):
    def __init__(self,
                 name="cupcake__8",
                 obj_name="cupcake__8",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cupcake/wdiezm/usd/MJCF/wdiezm.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CuttingBoard_1(CustomObjects):
    def __init__(self,
                 name="cutting_board__1",
                 obj_name="cutting_board__1",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cutting_board/aibvew/usd/MJCF/aibvew.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CuttingBoard_2(CustomObjects):
    def __init__(self,
                 name="cutting_board__2",
                 obj_name="cutting_board__2",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cutting_board/idmcgf/usd/MJCF/idmcgf.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CuttingBoard_3(CustomObjects):
    def __init__(self,
                 name="cutting_board__3",
                 obj_name="cutting_board__3",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cutting_board/jfrbuz/usd/MJCF/jfrbuz.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class CuttingBoard_4(CustomObjects):
    def __init__(self,
                 name="cutting_board__4",
                 obj_name="cutting_board__4",
                 ):
        custom_path = os.path.join(
                str(absolute_path), f"assets/new_objects/cutting_board/nsvnai/usd/MJCF/nsvnai.xml"
            )
        super().__init__(
            custom_path=custom_path,
            name=name,
            obj_name=obj_name,
        )
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None
