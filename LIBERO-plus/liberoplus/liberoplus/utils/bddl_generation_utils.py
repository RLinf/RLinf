import os
import numpy as np

INDENT = "  "


def print_result(result):
    print("\n".join(result))


def get_result(result):
    return "\n".join(result)


def save_to_file(result, scene_name, language, folder=""):
    file_name = os.path.join(
        folder,
        f"{scene_name}".upper() + "_" + "_".join(language.lower().split(" ")) + ".bddl",
    )
    with open(file_name, "w") as f:
        f.write(result)
    return file_name


class _PDDLDefinition:
    def __init__(self, func, problem_name="", domain="robosuite"):
        self.func = func
        self.problem_name = problem_name
        self.domain = domain

    def __call__(self, *args, **kwargs):
        meta_strings = []
        content_strings = []
        meta_strings.append(f"(define (problem {self.problem_name})")
        content_strings.append(f"(:domain {self.domain})")
        content_strings += self.func(*args, **kwargs)
        content_strings = [INDENT + each_line for each_line in content_strings]
        meta_strings += content_strings
        meta_strings.append(")\n")
        return meta_strings


def PDDLDefinition(func=None, problem_name=""):
    if func:
        return _PDDLDefinition(func)
    else:

        def wrapper(func):
            return _PDDLDefinition(func, problem_name)

        return wrapper


class Language:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        strings = []
        language = kwargs["language"]
        strings.append(f"(:language {language})")
        del kwargs["language"]
        strings += self.func(*args, **kwargs)
        # strings = [INDENT + each_line for each_line in strings]
        return strings


class _LogicalState:
    def __init__(self, func, state_type=""):
        self.func = func
        self.state_type = state_type

    def __call__(self, *args, **kwargs):
        strings = []
        strings.append(f"(:{self.state_type}")
        strings += self.func(*args, **kwargs)
        strings.append(")\n")
        # strings = [INDENT + each_line for each_line in strings]
        return strings


def LogicalState(func=None, state_type="PLACEHOLDER"):
    if func:
        return _LogicalState(func)
    else:

        def wrapper(func):
            return _LogicalState(func, state_type)

        return wrapper


# Region definition


class RegionWrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        strings = []
        strings.append("(:regions")
        strings += self.func(*args, **kwargs)
        strings.append(")\n")
        strings = [INDENT + each_line for each_line in strings]
        return strings


class Region:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        assert "target" in kwargs, "target property has to be defined!"
        assert "region_name" in kwargs, "region name has to be defined"
        strings = []
        region_name = kwargs["region_name"]
        strings.append(f"({region_name}")
        del kwargs["region_name"]
        strings += self.func(*args, **kwargs)
        strings.append(")")
        strings = [INDENT + each_line for each_line in strings]
        return strings


# Objects
class _ObjectDict:
    def __init__(self, func, object_type):
        self.func = func
        self.object_type = object_type

    def __call__(self, *args, **kwargs):
        strings = []
        strings.append(f"(:{self.object_type}")
        strings += self.func(*args, **kwargs)
        strings.append(")\n")
        # strings = [INDENT + each_line for each_line in strings]
        return strings


def ObjectDict(func=None, object_type="objects"):
    if func:
        return _ObjectDict(func)
    else:

        def wrapper(func):
            return _ObjectDict(func, object_type)

        return wrapper


@ObjectDict(object_type="fixtures")
def get_fixtures(**kwargs):
    return get_dict_string(**kwargs)


@ObjectDict(object_type="objects")
def get_objects(**kwargs):
    return get_dict_string(**kwargs)


@ObjectDict(object_type="obj_of_interest")
def get_objects_of_interest(l):
    return get_list_string(l)


def general_get_str_func(v):
    if type(v) is list:
        return get_list_string(v)
    elif type(v) is tuple:
        return get_tuple_string(v)
    elif type(v) is dict:
        return get_dict_string(v)
    elif type(v) is int or type(v) is float:
        return str(v)
    elif type(v) is str:
        return v


def get_dict_string(**kwargs):
    strings = []
    for k, v in kwargs.items():
        assert type(v) is list
        object_names = " ".join(v)
        strings.append(f"{object_names} - {k}")
    strings = [INDENT + each_line for each_line in strings]
    return strings


def get_list_string(l):
    strings = []
    assert type(l) is list
    for v in l:
        strings.append(general_get_str_func(v))
    strings = [INDENT + each_line for each_line in strings]
    return strings


def get_tuple_string(t):
    strings = []
    assert type(t) is tuple
    return "(" + " ".join([general_get_str_func(v) for v in t]) + ")"


def get_logical_expression_string(l):
    strings = []
    assert type(l) is list
    for v in l:
        strings.append(general_get_str_func(v))
    strings = [INDENT + each_line for each_line in strings]
    return strings


def get_property_string(**kwargs):
    strings = []
    for k, v in kwargs.items():
        if type(v) is str:
            strings.append(f"{INDENT}(:{k} {v})")
        else:
            strings.append(f"{INDENT}(:{k} (")
            strings += [INDENT + INDENT + new_v for new_v in general_get_str_func(v)]
            strings.append(f"{INDENT}{INDENT})")
            strings.append(f"{INDENT})")
    strings = [INDENT + each_line for each_line in strings]
    return strings


def get_prediate_string(predicates):
    # Handle init case and goal case
    assert type(predicates)
    strings = []


@LogicalState(state_type="init")
def get_init_state(l):
    return get_logical_expression_string(l)


@LogicalState(state_type="goal")
def get_goal_state(l):
    return get_logical_expression_string(l)


@Region
def get_xy_region(**kwargs):
    assert "ranges" in kwargs
    for v in kwargs["ranges"]:
        assert len(v) == 4
    return get_property_string(**kwargs)


@Region
def get_object_affordance_region(**kwargs):
    new_kwargs = {"target": kwargs["target"]}
    return get_property_string(**new_kwargs)


@RegionWrapper
def region_module(xy_region_kwargs_list=None, affordance_region_kwargs_list=None):
    strings = []
    if xy_region_kwargs_list is not None:
        for fixture_kwargs in xy_region_kwargs_list:
            strings += get_xy_region(**fixture_kwargs)
    if affordance_region_kwargs_list is not None:
        for fixture_kwargs in affordance_region_kwargs_list:
            strings += get_object_affordance_region(**fixture_kwargs)
    return strings


def object_naming_mapping(category_name, object_id):
    if category_name == "table":
        if object_id > 1:
            raise ValueError("Table can only be one for the moment.")
        return "main_table"
    elif category_name == "kitchen_table":
        if object_id > 1:
            raise ValueError("Kitchen table can only be one for the moment.")
        return "kitchen_table"
    elif category_name == "floor":
        if object_id > 1:
            raise ValueError("Floor can only be one.")
        return "floor"
    elif category_name == "coffee_table":
        if object_id > 1:
            raise ValueError("Coffee table can only be one for the moment.")
        return "coffee_table"
    elif category_name == "living_room_table":
        if object_id > 1:
            raise ValueError("Living room table can only be one for the moment.")
        return "living_room_table"
    elif category_name == "study_table":
        if object_id > 1:
            raise ValueError("Study table can only be one for the moment.")
        return "study_table"
    else:
        return f"{category_name}_{object_id}"


def retrieve_fixture_property(category_name):
    """Retrieve fixture property"""
    property_dict = {}
    return property_dict


def get_affordance_region_kwargs_list_from_fixture_info(fixture_info_dict):
    kwargs_list = []
    for k, v in fixture_info_dict.items():
        for item in v:
            kwargs_list.append({"target": k, "region_name": item})
    return kwargs_list


def get_xy_region_kwargs_list_from_regions_info(regions_info_dict):
    kwargs_list = []
    for k, v in regions_info_dict.items():
        assert type(v) is dict
        kwargs = {
            "region_name": k,
        }
        kwargs.update(v)
        kwargs_list.append(kwargs)
    return kwargs_list


def get_object_dict(objects_num_info):
    """
    A dctionary of objects that has "category_name": number_of_objects
    """
    object_dict = {}
    for category_name, num_objects in objects_num_info.items():
        object_dict[category_name] = []
        for object_id in range(1, num_objects + 1):
            object_dict[category_name].append(
                object_naming_mapping(category_name, object_id)
            )
    return object_dict


@PDDLDefinition(problem_name="LIBERO_Tabletop_Manipulation")
@Language
def tabletop_task_suites_generator(
    xy_region_kwargs_list,
    affordance_region_kwargs_list,
    fixture_object_dict,
    movable_object_dict,
    objects_of_interest,
    init_states,
    goal_states,
):
    result = []
    result += region_module(
        xy_region_kwargs_list=xy_region_kwargs_list,
        affordance_region_kwargs_list=affordance_region_kwargs_list,
    )
    result += get_fixtures(**fixture_object_dict)
    result += get_objects(**movable_object_dict)
    result += get_objects_of_interest(objects_of_interest)
    result += get_init_state(init_states)
    result += get_goal_state(goal_states)
    return result


@PDDLDefinition(problem_name="LIBERO_Kitchen_Tabletop_Manipulation")
@Language
def kitchen_table_task_suites_generator(
    xy_region_kwargs_list,
    affordance_region_kwargs_list,
    fixture_object_dict,
    movable_object_dict,
    objects_of_interest,
    init_states,
    goal_states,
):
    result = []
    result += region_module(
        xy_region_kwargs_list=xy_region_kwargs_list,
        affordance_region_kwargs_list=affordance_region_kwargs_list,
    )
    result += get_fixtures(**fixture_object_dict)
    result += get_objects(**movable_object_dict)
    result += get_objects_of_interest(objects_of_interest)
    result += get_init_state(init_states)
    result += get_goal_state(goal_states)
    return result


@PDDLDefinition(problem_name="LIBERO_Floor_Manipulation")
@Language
def floor_task_suites_generator(
    xy_region_kwargs_list,
    affordance_region_kwargs_list,
    fixture_object_dict,
    movable_object_dict,
    objects_of_interest,
    init_states,
    goal_states,
):
    result = []
    result += region_module(
        xy_region_kwargs_list=xy_region_kwargs_list,
        affordance_region_kwargs_list=affordance_region_kwargs_list,
    )
    result += get_fixtures(**fixture_object_dict)
    result += get_objects(**movable_object_dict)
    result += get_objects_of_interest(objects_of_interest)
    result += get_init_state(init_states)
    result += get_goal_state(goal_states)
    return result


@PDDLDefinition(problem_name="LIBERO_Coffee_Table_Manipulation")
@Language
def coffee_table_task_suites_generator(
    xy_region_kwargs_list,
    affordance_region_kwargs_list,
    fixture_object_dict,
    movable_object_dict,
    objects_of_interest,
    init_states,
    goal_states,
):
    result = []
    result += region_module(
        xy_region_kwargs_list=xy_region_kwargs_list,
        affordance_region_kwargs_list=affordance_region_kwargs_list,
    )
    result += get_fixtures(**fixture_object_dict)
    result += get_objects(**movable_object_dict)
    result += get_objects_of_interest(objects_of_interest)
    result += get_init_state(init_states)
    result += get_goal_state(goal_states)
    return result


@PDDLDefinition(problem_name="LIBERO_Study_Tabletop_Manipulation")
@Language
def study_table_task_suites_generator(
    xy_region_kwargs_list,
    affordance_region_kwargs_list,
    fixture_object_dict,
    movable_object_dict,
    objects_of_interest,
    init_states,
    goal_states,
):
    result = []
    result += region_module(
        xy_region_kwargs_list=xy_region_kwargs_list,
        affordance_region_kwargs_list=affordance_region_kwargs_list,
    )
    result += get_fixtures(**fixture_object_dict)
    result += get_objects(**movable_object_dict)
    result += get_objects_of_interest(objects_of_interest)
    result += get_init_state(init_states)
    result += get_goal_state(goal_states)
    return result


@PDDLDefinition(problem_name="LIBERO_Living_Room_Tabletop_Manipulation")
@Language
def living_room_table_task_suites_generator(
    xy_region_kwargs_list,
    affordance_region_kwargs_list,
    fixture_object_dict,
    movable_object_dict,
    objects_of_interest,
    init_states,
    goal_states,
):
    result = []
    result += region_module(
        xy_region_kwargs_list=xy_region_kwargs_list,
        affordance_region_kwargs_list=affordance_region_kwargs_list,
    )
    result += get_fixtures(**fixture_object_dict)
    result += get_objects(**movable_object_dict)
    result += get_objects_of_interest(objects_of_interest)
    result += get_init_state(init_states)
    result += get_goal_state(goal_states)
    return result
