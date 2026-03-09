import numpy as np

# Import centralized logging utilities
from yam_realtime.utils.logging_utils import log_data_utils

def get_dict(demo):
    dic = {}
    obs_keys = demo[0].keys()
    for key in obs_keys:
        dic[key] = np.stack([d[key] for d in demo])
    return dic

def shortest_angle(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi

def action_preprocessing(dic, actions):
        # compute actual deltas s_t+1 - s_t (keep gripper actions)
    actions_tmp = actions.copy()
    
    # Ensure shapes are compatible
    if "lowdim_ee" in dic and dic["lowdim_ee"].shape[0] > actions.shape[0]:
        # If lowdim_ee has more timesteps than actions, truncate it
        dic["lowdim_ee"] = dic["lowdim_ee"][:actions.shape[0]]
    elif "lowdim_ee" in dic and dic["lowdim_ee"].shape[0] < actions.shape[0]:
        # If actions has more timesteps than lowdim_ee, truncate actions
        actions_tmp = actions_tmp[:dic["lowdim_ee"].shape[0]]
    
    # Now compute deltas
    if "lowdim_ee" in dic and dic["lowdim_ee"].shape[0] > 1:
        actions_tmp[:-1, ..., :6] = (
            dic["lowdim_ee"][1:, ..., :6] - dic["lowdim_ee"][:-1, ..., :6]
        )
        actions = actions_tmp[:-1]
    else:
        # If no lowdim_ee or only one timestep, return actions as is
        actions = actions_tmp
    

        # compute shortest angle -> avoid wrap around
    actions[..., 3:6] = shortest_angle(actions[..., 3:6])

    # real data source
    #actions[..., [3,4,5]] = actions[..., [4,3,5]]
    #actions[...,4] = -actions[...,4]
    # actions[...,3] = -actions[...,3] this is a bug

    print(f'Action min & max: {actions[...,:6].min(), actions[...,:6].max()}')

    return actions
    log_data_utils("=" * 60, "data_info")