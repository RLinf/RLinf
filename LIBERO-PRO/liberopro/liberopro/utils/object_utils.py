# This is a util file for various functions that retrieve object information
from xml.etree import ElementTree

from liberopro.liberopro.envs.objects import OBJECTS_DICT, get_object_fn

EXCEPTION_DICT = {"flat_stove": "flat_stove_burner"}


def update_exception_dict(object_name, site_name):
    """Update EXCEPTION_DICT information. This is to handle some special case of affordance region naming.

    Args:
        object_name (str): object name
        site_name (str): site name
    """
    EXCEPTION_DICT[object_name] = site_name


def get_affordance_regions(objects, verbose=False):
    """_summary_

    Args:
        objects (MujocoObject): a dictionary of objects
        verbose (bool, optional): Print additional debug information. Defaults to False.

    Returns:
        dict: a dictionary of object names and their affordance regions.
    """
    affordances = {}
    for object_name in objects.keys():
        try:
            obj = get_object_fn(object_name)()
            # print(obj.root.findall(".//site"))
            object_affordance = []
            for site in obj.root.findall(".//site"):
                site_name = site.get("name")
                if "site" not in site_name and (
                    object_name not in EXCEPTION_DICT
                    or object_name in EXCEPTION_DICT
                    and site_name not in EXCEPTION_DICT[object_name]
                ):
                    # print(site_name)
                    # object name is already added as prefix when the object is initialized. remove them for consistency in bddl files
                    object_affordance.append(site_name.replace(f"{object_name}_", ""))
            if len(object_affordance) > 0:
                affordances[object_name] = object_affordance
        except:
            if verbose:
                print(f"Skipping {object_name}")

    return affordances
