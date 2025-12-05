import os
from collections import namedtuple

from liberopro.liberopro.utils.mu_utils import get_scene_class
from liberopro.liberopro.utils.bddl_generation_utils import *

TASK_INFO = {}

TaskInfoTuple = namedtuple(
    "TaskInfoTuple", "scene_name language objects_of_interest goal_states"
)


def register_task_info(language, scene_name, objects_of_interest=[], goal_states=[]):

    if scene_name not in TASK_INFO:
        TASK_INFO[scene_name] = []

    scene = get_scene_class(scene_name)()
    possible_objects_of_interest = scene.possible_objects_of_interest
    print(objects_of_interest)
    for object_name in objects_of_interest:
        if object_name not in possible_objects_of_interest:
            
            print(f"Error!! {scene_name} not having valid objects: {object_name}")
            print(possible_objects_of_interest)
            break
    print("remove invalid objects")
    objects_of_interest = []
    task_goal = [("And", *goal_states)]
    TASK_INFO[scene_name].append(
        TaskInfoTuple(scene_name, language, objects_of_interest, task_goal)
    )


def get_task_info(scene_name=None):
    if scene_name is None:
        return TASK_INFO
    else:
        return TASK_INFO[scene_name]


def get_suite_generator_func(workspace_name):
    if workspace_name == "main_table":
        return tabletop_task_suites_generator
    elif workspace_name == "kitchen_table":
        return kitchen_table_task_suites_generator
    elif workspace_name == "living_room_table":
        return living_room_table_task_suites_generator
    elif workspace_name == "study_table":
        return study_table_task_suites_generator
    elif workspace_name == "coffee_table":
        return coffee_table_task_suites_generator
    else:
        return floor_task_suites_generator


def generate_bddl_from_task_info(folder="/tmp/pddl"):
    results = []
    failures = []
    bddl_file_names = []
    os.makedirs(folder, exist_ok=True)

    registered_task_info_dict = get_task_info()
    for scene_name in registered_task_info_dict:
        for task_info_tuple in registered_task_info_dict[scene_name]:
            scene_name = task_info_tuple.scene_name
            language = task_info_tuple.language
            objects_of_interest = task_info_tuple.objects_of_interest
            goal_states = task_info_tuple.goal_states
            scene = get_scene_class(scene_name)()

            try:
                result = get_suite_generator_func(scene.workspace_name)(
                    language=language,
                    xy_region_kwargs_list=scene.xy_region_kwargs_list,
                    affordance_region_kwargs_list=scene.affordance_region_kwargs_list,
                    fixture_object_dict=scene.fixture_object_dict,
                    movable_object_dict=scene.movable_object_dict,
                    objects_of_interest=objects_of_interest,
                    init_states=scene.init_states,
                    goal_states=goal_states,
                )
                result = get_result(result)
                bddl_file_name = save_to_file(
                    result, scene_name=scene_name, language=language, folder=folder
                )
                if bddl_file_name in bddl_file_names:
                    print(bddl_file_name)
                bddl_file_names.append(bddl_file_name)
                results.append(result)
            except Exception as e:
                print(e)
                failures.append((scene_name, language))
    print(f"Succefully generated: {len(results)}")
    return bddl_file_names, failures
