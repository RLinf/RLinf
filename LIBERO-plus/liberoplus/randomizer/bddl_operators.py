"""
object change after usd2mjcf:
1. body name change to "object"
2. add <site rgba="0 0 0 0" size="0.005" pos="0 0 -0.06" name="bottom_site" />
       <site rgba="0 0 0 0" size="0.005" pos="0 0 0.06" name="top_site" />
       <site rgba="0 0 0 0" size="0.005" pos="0.015 0.015 0" name="horizontal_radius_site" />
"""
import os
import re
import copy
import uuid
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
from bddl.parsing import scan_tokens, package_predicates
from robosuite.models.objects import MujocoXMLObject
from liberoplus.liberoplus.envs.base_object import register_object
from liberoplus.liberoplus.envs.bddl_utils import get_regions, get_scenes
from liberoplus.liberoplus.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info, get_result
from liberoplus.liberoplus.utils.task_generation_utils import get_suite_generator_func
from liberoplus.liberoplus.utils.mu_utils import InitialSceneTemplates


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
        # make sure custom path is an absolute path
        assert (os.path.isabs(custom_path)), "Custom path must be an absolute path"
        # make sure the custom path is also an xml file
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
class LiberoMug1(CustomObjects):
    def __init__(self,
                 name="libero_mug",
                 obj_name="libero_mug",
                 ):
        super().__init__(
            # custom_path=os.path.abspath(os.path.join(
            #     "./", "custom_assets", "libero_mug", "libero_mug.xml"
            # )),
            custom_path="/home/ps/BEHAVIOR-1K/OmniGibson/omnigibson/data/og_dataset/objects/bottle_of_gin/qzgcdx/usd/MJCF/try.xml",
            name=name,
            obj_name=obj_name,
        )

        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None


def lower_to_camel(name):
    """Convert lower case convention name into CamelCase (PascalCase).
    Examples:
        kitchen_demo_scene -> KitchenDemoScene
        my-scene_name -> MySceneName
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    parts = [p for p in re.split(r'[^0-9a-zA-Z]+', name) if p]
    return "".join(p.capitalize() for p in parts)


# ==== BDDL in Python Data Structure ==========================================
# fmt: off
@dataclass
class TaskInfo:
    language:               str
    objects_of_interest:    List[str]
    goal_states:            List[Iterable[str]]
    # scene name is only used as scene key storage in the libero task generation, it does not save into bddl file.
    scene_name:             Optional[str] = None


@dataclass
class RegionInfo:
    region_centroid_xy:     Tuple[float, float]
    region_name:            str
    target_name:            Optional[str] = None
    region_half_len:        float = 0.02
    yaw_rotation:           Tuple[float, float] = (0.0, 0.0)


@dataclass
class BddlSpec:
    task_info:          TaskInfo
    workspace_name:     str
    region_infos:       List[RegionInfo]
    init_states:        List[Tuple[str, str, str]]
    fixture_num_info:   Dict[str, int]
    object_num_info:    Dict[str, int]

    def __post_init__(self):
        # validation
        scene_name = self.task_info.scene_name
        if scene_name is not None:
            assert scene_name == scene_name.lower(), "Scene name must be lower case"
# fmt: on
# =============================================================================


def bddl_spec2str(
        bddl_spec: BddlSpec
) -> str:
    # =========================================================================
    # Step-1: Dynamically generate a scene clas
    # =========================================================================
    class TemporaryScene(InitialSceneTemplates):
        """
        This bddl scene class is temporary generated and being encoded into a bddl file.
        """

        def __init__(self):
            super().__init__(
                workspace_name=bddl_spec.workspace_name,
                fixture_num_info=bddl_spec.fixture_num_info,
                object_num_info=bddl_spec.object_num_info,
            )

        def define_regions(self):
            for region_info in bddl_spec.region_infos:
                self.regions.update(
                    self.get_region_dict(**asdict(region_info))
                )

            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

        @property
        def init_states(self):
            return bddl_spec.init_states

    # Transform scene name from lower case convention into camel convention
    scene_name = bddl_spec.task_info.scene_name
    if scene_name is not None:
        # scene name is optional for generating bddl file since it does not present in bddl file
        # original libero pipeline requires scene name to be existed for using as key to store task generation info
        scene_name_camel = lower_to_camel(scene_name)

        # Modify class name for libero pipeline
        TemporaryScene.__name__ = scene_name_camel
        TemporaryScene.__qualname__ = __name__  # affects repr/introspection
        TemporaryScene.__module__ = __name__  # helpful for tooling/pickling

    # =========================================================================
    # Step-2: Transform scene class into bddl file.
    # =========================================================================
    scene = TemporaryScene()

    # scene validation
    possible_objects_of_interest = scene.possible_objects_of_interest
    for object_name in bddl_spec.task_info.objects_of_interest:
        if object_name not in possible_objects_of_interest:
            print(f"Error: not having valid objects: {object_name}")
            print(possible_objects_of_interest)
            raise ValueError

    language = bddl_spec.task_info.language

    try:
        result = get_suite_generator_func(scene.workspace_name)(
            language=language,
            xy_region_kwargs_list=scene.xy_region_kwargs_list,
            affordance_region_kwargs_list=scene.affordance_region_kwargs_list,
            fixture_object_dict=scene.fixture_object_dict,
            movable_object_dict=scene.movable_object_dict,
            objects_of_interest=bddl_spec.task_info.objects_of_interest,
            init_states=scene.init_states,
            goal_states=[("And", *bddl_spec.task_info.goal_states)],
        )
        bddl_str: str = get_result(result)

    except Exception as e:
        print(f"Failed to generate bddl: {e}")
        import traceback
        traceback.print_exc()
        raise

    return bddl_str


def _robosuite_parse_problem(bddl_str: str):
    """Modified from: libero.libero.envs.bddl_utils.robosuite_parse_problem"""
    domain_name = "robosuite"
    tokens = scan_tokens(string=bddl_str)
    if isinstance(tokens, list) and tokens.pop(0) == "define":
        problem_name = "unknown"
        objects = {}
        obj_of_interest = []
        initial_state = []
        goal_state = []
        fixtures = {}
        regions = {}
        scene_properties = {}
        language_instruction = ""
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == "problem":
                problem_name = group[-1]
            elif t == ":domain":
                if domain_name != group[-1]:
                    raise Exception("Different domain specified in problem file")
            elif t == ":requirements":
                pass
            elif t == ":objects":
                group.pop(0)
                object_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        objects[group.pop(0)] = object_list
                        object_list = []
                    else:
                        object_list.append(group.pop(0))
                if object_list:
                    if not "object" in objects:
                        objects["object"] = []
                    objects["object"] += object_list
            elif t == ":obj_of_interest":
                group.pop(0)
                while group:
                    obj_of_interest.append(group.pop(0))
            elif t == ":fixtures":
                group.pop(0)
                fixture_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        fixtures[group.pop(0)] = fixture_list
                        fixture_list = []
                    else:
                        fixture_list.append(group.pop(0))
                if fixture_list:
                    if not "fixture" in fixtures:
                        fixtures["fixture"] = []
                    fixtures["fixture"] += fixture_list
            elif t == ":regions":
                get_regions(t, regions, group)
            elif t == ":scene_properties":
                get_scenes(t, scene_properties, group)
            elif t == ":language":
                group.pop(0)
                language_instruction = group

            elif t == ":init":
                group.pop(0)
                initial_state = group
            elif t == ":goal":
                package_predicates(group[1], goal_state, "", "goals")
            else:
                print("%s is not recognized in problem" % t)
        return {
            "problem_name":         problem_name,
            "fixtures":             fixtures,
            "regions":              regions,
            "objects":              objects,
            "scene_properties":     scene_properties,
            "initial_state":        initial_state,
            "goal_state":           goal_state,
            "language_instruction": language_instruction,
            "obj_of_interest":      obj_of_interest,
        }
    else:
        raise NotImplemented


def _problem_name_to_workspace_name(problem_name: str) -> str:
    """Transform problem name into workspace name."""
    problem_name = problem_name.lower()
    if "LIBERO_Tabletop_Manipulation".lower() == problem_name:
        return "main_table"
    elif "LIBERO_Kitchen_Tabletop_Manipulation".lower() == problem_name:
        return "kitchen_table"
    elif "LIBERO_Living_Room_Tabletop_Manipulation".lower() == problem_name:
        return "living_room_table"
    elif "LIBERO_Study_Tabletop_Manipulation".lower() == problem_name:
        return "study_table"
    elif "LIBERO_Coffee_Table_Manipulation".lower() == problem_name:
        return "coffee_table"
    elif "LIBERO_Floor_Manipulation".lower() == problem_name:
        return "floor"  # Note that this does not present in `libero.libero.utils.task_generation_utils.get_suite_generator_func`, I found in file `libero.libero.envs.problems.libero_floor_manipulation.Libero_Floor_Manipulation`
    else:
        raise NotImplementedError(f"Cannot infer workspace name from problem name: {problem_name}")


def bddl_str2spec(bddl_str: str) -> BddlSpec:
    """Transform bddl string into BddlSpec dataclass instance."""
    robosuite_problem = _robosuite_parse_problem(bddl_str)
    problem_name = robosuite_problem["problem_name"]
    workspace_name = _problem_name_to_workspace_name(problem_name)

    region_infos = []
    for k, v in robosuite_problem["regions"].items():
        target_name = v.get("target", None)  # e.g. kitchen_table
        region_name = k.split('' if target_name is None else f"{target_name}_")[-1]  # e.g. kitchen_table_wooden_cabinet_init_region -> wooden_cabinet_init_region

        if v["ranges"] == []:
            # affordance region without range, we don't need to explicitly define it in bddl spec
            continue

        else:
            # xy region have have ranges and must be explicitly defined in bddl spec
            _xmin, _ymin, _xmax, _ymax = v["ranges"][0]
            region_centroid_xy = ((_xmin + _xmax) / 2, (_ymin + _ymax) / 2)
            region_half_len = max((_xmax - _xmin) / 2, (_ymax - _ymin) / 2)
            yaw_rotation = tuple(v.get("yaw_rotation", (0.0, 0.0)))

            region_infos.append(
                RegionInfo(
                    region_centroid_xy=region_centroid_xy,
                    region_name=region_name,
                    target_name=target_name,
                    region_half_len=region_half_len,
                    yaw_rotation=yaw_rotation,
                )
            )

    goal_states = []
    for goal in robosuite_problem["goal_state"]:
        goal = copy.deepcopy(goal)
        goal[0] = goal[0].capitalize()  # e.g. "on" -> "On"
        goal_states.append(tuple(goal))

    init_states = []
    for state in robosuite_problem["initial_state"]:
        state = copy.deepcopy(state)
        state[0] = state[0].capitalize()  # e.g. "on" -> "On"
        init_states.append(tuple(state))

    fixture_num_info = {k: len(v) for k, v in robosuite_problem["fixtures"].items()}
    object_num_info = {k: len(v) for k, v in robosuite_problem["objects"].items()}

    task_info = TaskInfo(
        scene_name=None,  # scene name is not contains in bddl content.
        language=" ".join(robosuite_problem["language_instruction"]),
        objects_of_interest=robosuite_problem["obj_of_interest"],
        goal_states=goal_states,
    )
    bddl_spec = BddlSpec(
        task_info=task_info,
        workspace_name=workspace_name,
        region_infos=region_infos,
        init_states=init_states,
        fixture_num_info=fixture_num_info,
        object_num_info=object_num_info,
    )

    return bddl_spec


def save_bddl_file(
        bddl_spec: BddlSpec,
        bddl_filename: Optional[str] = None,  # If None, we will use the default libero naming convention.
        bddl_dirpath: str = "/tmp/bddl_files",
) -> Tuple[str, str]:
    scene_name = bddl_spec.task_info.scene_name
    language = bddl_spec.task_info.language
    bddl_descriptor = f"({scene_name=}, {language=})"

    if bddl_filename is None:
        bddl_filename = ''
        if scene_name is not None:
            bddl_filename += scene_name.upper() + '_'
            bddl_filename += "_".join(language.lower().split(" ")) + '.bddl'

    bddl_filepath = os.path.join(bddl_dirpath, bddl_filename)

    try:
        bddl_str: str = bddl_spec2str(bddl_spec)
        with open(bddl_filepath, "w") as f:
            f.write(bddl_str)

    except:
        print(f"Failed to generate bddl file: {bddl_descriptor}")
        import traceback
        traceback.print_exc()
        raise

    print(f"Successfully generate bddl file: {bddl_descriptor}")
    return bddl_str, bddl_filepath


def load_bddl_file(bddl_filepath: str) -> BddlSpec:
    with open(bddl_filepath, 'r') as f:
        bddl_str = f.read()
    bddl_spec = bddl_str2spec(bddl_str)

    # scene_name is represented as the FULL CAPITALIZED prefix in bddl filename
    words_in_name = os.path.basename(bddl_filepath).replace('.bddl', '').split('_')
    capitalized_words = []
    for word in words_in_name:
        if word.isupper():
            capitalized_words.append(word.lower())
        else:
            break
    if len(capitalized_words) > 0:
        scene_name = '_'.join(capitalized_words).lower()
        bddl_spec.task_info.scene_name = scene_name

    return bddl_spec


# =========================================================================
# Exemplary Perturbation Utility
# =========================================================================
def perturb_region_info(
        region_info: RegionInfo,
        delta_region_centroid_xy: Tuple[float, float] = (0.0, 0.0),
        delta_region_half_len: float = 0.0,
        delta_yaw_rotation: Tuple[float, float] = (0.0, 0.0),
) -> RegionInfo:
    """Utility function for perturbing region position."""
    new_region_info = copy.deepcopy(region_info)
    new_region_info.region_centroid_xy = (
        region_info.region_centroid_xy[0] + delta_region_centroid_xy[0],
        region_info.region_centroid_xy[1] + delta_region_centroid_xy[1],
    )
    new_region_info.region_half_len = region_info.region_half_len + delta_region_half_len
    new_region_info.yaw_rotation = (
        region_info.yaw_rotation[0] + delta_yaw_rotation[0],
        region_info.yaw_rotation[1] + delta_yaw_rotation[1],
    )
    return new_region_info


if __name__ == '__main__':
    """
    This is an exemplary BddlSpec for illustrating how to use the above functions.
    BddlSpec is the key data structure for converting between bddl string/file and structured task generation info.
    In practice, you can generate BddlSpec from your own task generation pipeline or load it from existing bddl files.
    Also, you can modify the bddl_spec to setup your own tasks.
    """
    import time
    from PIL import Image
    from liberoplus.liberoplus.envs import OffScreenRenderEnv

    bddl_spec_1 = BddlSpec(
        task_info=TaskInfo(
            scene_name="kitchen_demo_scene",
            language="libero demo behaviors",
            objects_of_interest=[],
            goal_states=[
                ("Open", "wooden_cabinet_1_top_region"),
                ("In", "libero_mug_yellow_1", "wooden_cabinet_1_top_region"),
            ]),
        workspace_name="kitchen_table",
        region_infos=[
            RegionInfo(
                region_centroid_xy=[0.0, -0.30],
                region_name="wooden_cabinet_init_region",
                target_name='kitchen_table',
                region_half_len=0.01,
                yaw_rotation=(np.pi, np.pi),
            ),
            RegionInfo(
                region_centroid_xy=[0.0, 0.0],
                region_name="libero_mug_init_region",
                target_name='kitchen_table',
                region_half_len=0.025,
            ),
            RegionInfo(
                region_centroid_xy=[0.2, 0.2],
                region_name="libero_mug_init_region_2",
                target_name='kitchen_table',
                region_half_len=0.025,
            ),
            RegionInfo(
                region_centroid_xy=[-0.1, 0.15],
                region_name="libero_mug_yellow_init_region",
                target_name='kitchen_table',
                region_half_len=0.025,
            )
        ],
        init_states=[
            ("On", "libero_mug_1", "kitchen_table_libero_mug_init_region"),
            ("On", "libero_mug_2", "kitchen_table_libero_mug_init_region_2"),
            ("On", "libero_mug_yellow_1", "kitchen_table_libero_mug_yellow_init_region"),
            ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
        ],
        fixture_num_info={
            "kitchen_table":  1,
            "wooden_cabinet": 1,
        },
        object_num_info={
            "libero_mug":        2,
            "libero_mug_yellow": 1,
        }
    )


    # =========================================================================
    # Utility functions
    # =========================================================================
    def _render_bddl_file_to_image(bddl_filepath: str):
        env_args = {
            "bddl_file_name": bddl_filepath,
            "camera_heights": 256,
            "camera_widths":  256
        }

        env = OffScreenRenderEnv(**env_args)
        env.seed(0)
        obs = env.reset()
        env.close()
        image = Image.fromarray(obs["agentview_image"][::-1])
        return image


    # =========================================================================
    # Testing of the conversion functions
    # =========================================================================
    # Setup directory for experimenting
    EXPERIMENT_DIRECTORY = f"./outputs/{time.strftime('%Y%m%d_%H%M%S')}"
    bddl_dirpath = os.path.join(EXPERIMENT_DIRECTORY, "bddl_files")
    image_dirpath = os.path.join(EXPERIMENT_DIRECTORY, "images")
    os.makedirs(bddl_dirpath)
    os.makedirs(image_dirpath)

    # BddlSpec -> bddl string
    bddl_str_1 = bddl_spec2str(bddl_spec_1)

    # BddlSpec -> bddl file
    _bddl_str_1, bddl_filepath_1 = save_bddl_file(bddl_spec_1, bddl_dirpath=bddl_dirpath)
    print(f"bddl file path: {bddl_filepath_1}")
    # Alternatively, you can also specify the bddl filename instead of letting it be automatically generated.
    _, _ = save_bddl_file(bddl_spec_1, bddl_filename="my_bddl_file.bddl", bddl_dirpath=bddl_dirpath)

    # bddl file -> BddlSpec
    bddl_spec_2 = load_bddl_file(bddl_filepath_1)

    # Check if the saved & loaded bddl spec is the same as the original spec via rendering
    # Note that the bddl string/file may not be exactly the same due to different float number rounding,
    # but they should represent the same task.
    _, bddl_filepath_2 = save_bddl_file(bddl_spec_2, bddl_filename="restored.bddl", bddl_dirpath=bddl_dirpath)
    image_spec1 = _render_bddl_file_to_image(bddl_filepath_1)
    image_spec2 = _render_bddl_file_to_image(bddl_filepath_2)
    image_spec1.save(os.path.join(image_dirpath, "spec1.png"))
    image_spec2.save(os.path.join(image_dirpath, "spec2.png"))
    assert np.allclose(np.array(image_spec1), np.array(image_spec2))

    # =========================================================================
    # Example of modifying bddl spec to setup customized (perturbed) tasks
    # =========================================================================
    bddl_spec_3 = copy.deepcopy(bddl_spec_1)

    # modify language
    bddl_spec_3.task_info.language = "put the yellow mug into the cabinet"

    # modify the first region's (wooden_cabinet's)
    bddl_spec_3.region_infos[0] = perturb_region_info(
        bddl_spec_3.region_infos[0],
        delta_region_centroid_xy=(0.1, 0.0),
        delta_region_half_len=0.0,
        delta_yaw_rotation=(0.5, -0.1),
    )

    # verify via rendering
    _, bddl_filepath_3 = save_bddl_file(bddl_spec_3, bddl_filename="modified.bddl", bddl_dirpath=bddl_dirpath)
    image_spec3 = _render_bddl_file_to_image(bddl_filepath_3)
    image_spec3.save(os.path.join(image_dirpath, "spec3.png"))
    assert not np.allclose(np.array(image_spec2), np.array(image_spec3))
