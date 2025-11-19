from typing import List, Optional, Tuple, Union

import numpy as np

from mshab.utils.array import to_numpy


def get_episode_label_and_events(
    task_cfgs, ep_success, ep_infos
) -> Tuple[
    str, List[str], List[Tuple[Union[List[int], int], str, Optional[List[float]]]]
]:
    ep_success = to_numpy(ep_success)
    ep_infos = to_numpy(ep_infos)
    subtask_type = ep_infos["subtask_type"]
    subtask_type = ep_infos["subtask_type"][ep_infos["subtask_type"] <= 10]
    subtask_types = np.unique(subtask_type)
    assert len(subtask_types) == 1
    episode_type = subtask_types.item()
    if episode_type == 0:
        return get_pick_episode_label_and_events(
            task_cfgs["pick"], ep_success, ep_infos
        )
    if episode_type == 1:
        return get_place_episode_label_and_events(
            task_cfgs["place"], ep_success, ep_infos
        )
    if episode_type == 2:
        return get_navigate_episode_label_and_events(
            task_cfgs["navigate"], ep_success, ep_infos
        )
    if episode_type == 3:
        return get_open_episode_label_and_events(
            task_cfgs["open"], ep_success, ep_infos
        )
    if episode_type == 4:
        return get_close_episode_label_and_events(
            task_cfgs["close"], ep_success, ep_infos
        )


def get_pick_episode_label_and_events(pick_cfg, ep_success, ep_infos):
    success: np.ndarray = ep_success
    is_grasped: np.ndarray = ep_infos["is_grasped"]
    contacts: np.ndarray = ep_infos["robot_target_pairwise_force"] > 0
    robot_force: np.ndarray = ep_infos["robot_force"]
    robot_cumulative_force: np.ndarray = ep_infos["robot_cumulative_force"]

    events, events_verbose = [], []

    def append_event(event_step, event):
        if (len(events) == 0 or events[-1] != event) and event in [
            "contact",
            "grasped",
            "dropped",
            "success",
            "exceeded_cumulative_force_limit",
        ]:
            events.append(event)
        if event == "collision":
            if len(events_verbose) == 0 or events_verbose[-1][1] != "collision":
                events_verbose.append(
                    (
                        [event_step],
                        event,
                        [round(robot_force[event_step].item(), 2)],
                    )
                )
            else:
                events_verbose[-1][0].append(event_step)
                events_verbose[-1][2].append(round(robot_force[event_step].item(), 2))
        else:
            events_verbose.append((event_step, event, None))

    for step, (spg, sg, sps, ss, spc, sc, srf, srcf) in enumerate(
        zip(
            [False] + is_grasped.tolist(),
            is_grasped.tolist(),
            [False] + success.tolist(),
            success.tolist(),
            [False] + contacts.tolist(),
            contacts.tolist(),
            robot_force.tolist(),
            robot_cumulative_force.tolist(),
        )
    ):
        if not spc and sc:
            append_event(step, "contact")
        if spc and not sc:
            append_event(step, "lost_contact")
        if not spg and sg:
            append_event(step, "grasped")
        if spg and not sg:
            append_event(step, "dropped")
        if not sps and ss:
            append_event(step, "success")
        if sps and not ss:
            append_event(step, "lost_success")
        if srf > 0:
            append_event(step, "collision")
        if srcf >= pick_cfg.robot_cumulative_force_limit:
            if "exceeded_cumulative_force_limit" not in events:
                append_event(step, "exceeded_cumulative_force_limit")

    if success.any():
        if len(events) == 3:
            label = "straightforward_success"
        elif len(events) > 3 and events[-1] == "success":
            label = "winding_success"
        elif (
            len(events) > 3
            and events[-1] != "success"
            and "exceeded_cumulative_force_limit" not in events
        ):
            label = "success_then_drop"
        # elif (
        #     len(events) > 3
        #     and events[-1] != "success"
        #     and "exceeded_cumulative_force_limit" in events
        # ):
        else:
            label = "success_then_excessive_collisions"
    else:
        if "exceeded_cumulative_force_limit" in events:
            label = "excessive_collision_failure"
        else:
            if len(events) == 0:
                label = "mobility_failure"
            elif len(events) == 1:
                label = "cant_grasp_failure"
            elif "dropped" in events and events[-1] != "grasped":
                label = "drop_failure"
            # elif events[-1] == "grasped":
            else:
                label = "too_slow_failure"

    return label, events, events_verbose


def get_place_episode_label_and_events(place_cfg, ep_success, ep_infos):
    success: np.ndarray = ep_success
    is_grasped: np.ndarray = ep_infos["is_grasped"]
    obj_at_goal: np.ndarray = ep_infos["obj_at_goal"]
    robot_force: np.ndarray = ep_infos["robot_force"]
    robot_cumulative_force: np.ndarray = ep_infos["robot_cumulative_force"]

    events, events_verbose = [], []

    def append_event(event_step, event):
        if (len(events) == 0 or events[-1] != event) and event in [
            "grasped",
            "obj_at_goal",
            "released_at_goal",
            "released_outside_goal",
            "obj_left_goal",
            "success",
            "exceeded_cumulative_force_limit",
        ]:
            events.append(event)
        if event == "collision":
            if len(events_verbose) == 0 or events_verbose[-1][1] != "collision":
                events_verbose.append(
                    (
                        [event_step],
                        event,
                        [round(robot_force[event_step].item(), 2)],
                    )
                )
            else:
                events_verbose[-1][0].append(event_step)
                events_verbose[-1][2].append(round(robot_force[event_step].item(), 2))
        else:
            events_verbose.append((event_step, event, None))

    for step, (spg, sg, sps, ss, spag, sag, srf, srcf) in enumerate(
        zip(
            [False] + is_grasped.tolist(),
            is_grasped.tolist(),
            [False] + success.tolist(),
            success.tolist(),
            [False] + obj_at_goal.tolist(),
            obj_at_goal.tolist(),
            robot_force.tolist(),
            robot_cumulative_force.tolist(),
        )
    ):
        if not spg and sg:
            append_event(step, "grasped")
        if not spag and sag:
            append_event(step, "obj_at_goal")
        if spg and not sg:
            if sag:
                append_event(step, "released_at_goal")
            else:
                append_event(step, "released_outside_goal")
        if spag and not sag:
            append_event(step, "obj_left_goal")
        if not sps and ss:
            append_event(step, "success")
        if sps and not ss:
            append_event(step, "lost_success")
        if srf > 0:
            append_event(step, "collision")
        if srcf >= place_cfg.robot_cumulative_force_limit:
            if "exceeded_cumulative_force_limit" not in events:
                append_event(step, "exceeded_cumulative_force_limit")

    def last_index(lst: list, x):
        if x in lst:
            return len(lst) - lst[::-1].index(x) - 1
        return -1

    if success.any():
        if "exceeded_cumulative_force_limit" in events:
            label = "success_then_excessive_collisions"
        else:
            if last_index(events, "obj_at_goal") < last_index(events, "obj_left_goal"):
                label = "dubious_success"
            elif len(events) <= 4 and ("released_at_goal" in events or obj_at_goal[0]):
                label = "placed_in_goal_success"
            elif len(events) <= 4 and (
                "released_outside_goal" in events or not obj_at_goal[0]
            ):
                label = "dropped_to_goal_success"
            # elif len(events) > 4 and last_index(events, "obj_at_goal") >= last_index(
            #     events, "obj_left_goal"
            # ):
            else:
                label = "winding_success"
    else:
        if "exceeded_cumulative_force_limit" in events:
            label = "excessive_collision_failure"
        else:
            if len(events) == 0:
                label = "didnt_grasp_failure"
            elif len(events) > 0 and "obj_at_goal" not in events:
                label = "didnt_reach_goal_failure"
            elif (
                "obj_at_goal" in events
                and (
                    (len(events) <= 2 and obj_at_goal[0])
                    or (
                        last_index(events, "released_at_goal")
                        > last_index(events, "released_outside_goal")
                        and last_index(events, "released_at_goal")
                        > last_index(events, "grasped")
                    )
                )
                and last_index(events, "obj_left_goal")
                > last_index(events, "obj_at_goal")
            ):
                label = "place_in_goal_failure"
            elif (
                "obj_at_goal" in events
                and (
                    (len(events) <= 2 and not obj_at_goal[0])
                    or (
                        last_index(events, "released_outside_goal")
                        > last_index(events, "released_at_goal")
                        and last_index(events, "released_outside_goal")
                        > last_index(events, "grasped")
                    )
                )
                and last_index(events, "obj_left_goal")
                > last_index(events, "obj_at_goal")
            ):
                label = "drop_to_goal_failure"
            elif (
                "obj_at_goal" in events
                and last_index(events, "grasped")
                > last_index(events, "released_at_goal")
                and last_index(events, "grasped")
                > last_index(events, "released_outside_goal")
            ):
                label = "wont_let_go_failure"
            # elif last_index(events, "obj_at_goal") > last_index(
            #     events, "obj_left_goal"
            # ):
            else:
                label = "too_slow_failure"

    return label, events, events_verbose


def get_navigate_episode_label_and_events(navigate_cfg, ep_success, ep_infos):
    success: np.ndarray = ep_success
    is_grasped: np.ndarray = ep_infos["is_grasped"]
    navigated_close: np.ndarray = ep_infos["navigated_close"]
    robot_force: np.ndarray = ep_infos["robot_force"]
    robot_cumulative_force: np.ndarray = ep_infos["robot_cumulative_force"]

    events, events_verbose = [], []

    def append_event(event_step, event):
        if (len(events) == 0 or events[-1] != event) and event in [
            "grasped",
            "dropped",
            "navigated_close",
            "success",
            "exceeded_cumulative_force_limit",
        ]:
            events.append(event)
        if event == "collision":
            if len(events_verbose) == 0 or events_verbose[-1][1] != "collision":
                events_verbose.append(
                    (
                        [event_step],
                        event,
                        [round(robot_force[event_step].item(), 2)],
                    )
                )
            else:
                events_verbose[-1][0].append(event_step)
                events_verbose[-1][2].append(round(robot_force[event_step].item(), 2))
        else:
            events_verbose.append((event_step, event, None))

    for step, (spg, sg, sps, ss, spn, sn, srf, srcf) in enumerate(
        zip(
            [False] + is_grasped.tolist(),
            is_grasped.tolist(),
            [False] + success.tolist(),
            success.tolist(),
            [False] + navigated_close.tolist(),
            navigated_close.tolist(),
            robot_force.tolist(),
            robot_cumulative_force.tolist(),
        )
    ):
        if not spn and sn:
            append_event(step, "navigated_close")
        if spn and not sn:
            append_event(step, "lost_navigated_close")
        if not spg and sg:
            append_event(step, "grasped")
        if spg and not sg:
            append_event(step, "dropped")
        if not sps and ss:
            append_event(step, "success")
        if sps and not ss:
            append_event(step, "lost_success")
        if srf > 0:
            append_event(step, "collision")
        if srcf >= navigate_cfg.robot_cumulative_force_limit:
            if "exceeded_cumulative_force_limit" not in events:
                append_event(step, "exceeded_cumulative_force_limit")

    if success.any():
        if "exceeded_cumulative_force_limit" in events:
            label = "success_then_excessive_collisions"
        elif "dropped" in events:
            label = "success_then_dropped"
        else:
            label = "navigation_success"
    else:
        if "exceeded_cumulative_force_limit" in events:
            label = "excessive_collision_failure"
        elif "dropped" in events:
            label = "dropped_failure"
        else:
            label = "navigation_failure"

    return label, events, events_verbose


def get_open_episode_label_and_events(open_cfg, ep_success, ep_infos):
    success: np.ndarray = ep_success
    articulation_open: np.ndarray = ep_infos["articulation_open"]
    contacts: np.ndarray = ep_infos["robot_target_pairwise_force"] > 0
    robot_force: np.ndarray = ep_infos["robot_force"]
    robot_cumulative_force: np.ndarray = ep_infos["robot_cumulative_force"]

    handle_active_joint_qpos: np.ndarray = ep_infos["handle_active_joint_qpos"]
    handle_active_joint_qmax = np.mean(ep_infos["handle_active_joint_qmax"])
    handle_active_joint_qmin = np.mean(ep_infos["handle_active_joint_qmin"])
    articulation_slightly_open = handle_active_joint_qpos > (
        (handle_active_joint_qmax - handle_active_joint_qmin) * 0.1
        + handle_active_joint_qmin
    )
    articulation_closed = ~articulation_slightly_open

    events, events_verbose = [], []

    def append_event(event_step, event):
        if (len(events) == 0 or events[-1] != event) and event in [
            "contact",
            "opened",
            "slightly_opened",
            "closed",
            "success",
            "exceeded_cumulative_force_limit",
        ]:
            events.append(event)
        if event == "collision":
            if len(events_verbose) == 0 or events_verbose[-1][1] != "collision":
                events_verbose.append(
                    (
                        [event_step],
                        event,
                        [round(robot_force[event_step].item(), 2)],
                    )
                )
            else:
                events_verbose[-1][0].append(event_step)
                events_verbose[-1][2].append(round(robot_force[event_step].item(), 2))
        else:
            events_verbose.append((event_step, event, None))

    for step, (spo, so, sps, ss, spc, sc, srf, srcf, spso, sso, spcl, scl) in enumerate(
        zip(
            [False] + articulation_open.tolist(),
            articulation_open.tolist(),
            [False] + success.tolist(),
            success.tolist(),
            [False] + contacts.tolist(),
            contacts.tolist(),
            robot_force.tolist(),
            robot_cumulative_force.tolist(),
            [False] + articulation_slightly_open.tolist(),
            articulation_slightly_open.tolist(),
            [True] + articulation_closed.tolist(),
            articulation_closed.tolist(),
        )
    ):
        if not spc and sc:
            append_event(step, "contact")
        if spc and not sc:
            append_event(step, "lost_contact")
        if not spo and so:
            append_event(step, "opened")
        if not sps and ss:
            append_event(step, "success")
        if sps and not ss:
            append_event(step, "lost_success")
        if srf > 0:
            append_event(step, "collision")
        if srcf >= open_cfg.robot_cumulative_force_limit:
            if "exceeded_cumulative_force_limit" not in events:
                append_event(step, "exceeded_cumulative_force_limit")
        if not spso and sso:
            append_event(step, "slightly_opened")
        if not spcl and scl:
            append_event(step, "closed")

    def last_index(lst: list, x):
        if x in lst:
            return len(lst) - lst[::-1].index(x) - 1
        return -1

    if success.any():
        if "exceeded_cumulative_force_limit" in events:
            label = "success_then_excessive_collisions"
        else:
            if last_index(events, "closed") > last_index(events, "opened"):
                label = "dubious_success"
            else:
                label = "open_success"
    else:
        if "exceeded_cumulative_force_limit" in events:
            label = "excessive_collision_failure"
        else:
            if "contact" not in events:
                label = "cant_reach_articulation_failure"
            elif (
                "closed" in events
                and last_index(events, "closed") > last_index(events, "opened")
                and last_index(events, "closed") > last_index(events, "slightly_opened")
            ):
                label = "closed_after_open_failure"
            elif last_index(events, "slightly_opened") > last_index(events, "opened"):
                label = "slightly_open_failure"
            elif "opened" in events:
                label = "too_slow_failure"
            else:
                label = "cant_open_failure"

    return label, events, events_verbose


def get_close_episode_label_and_events(close_cfg, ep_success, ep_infos):
    success: np.ndarray = ep_success
    articulation_closed: np.ndarray = ep_infos["articulation_closed"]
    contacts: np.ndarray = ep_infos["robot_target_pairwise_force"] > 0
    robot_force: np.ndarray = ep_infos["robot_force"]
    robot_cumulative_force: np.ndarray = ep_infos["robot_cumulative_force"]

    handle_active_joint_qpos: np.ndarray = ep_infos["handle_active_joint_qpos"]
    handle_active_joint_qmax = np.mean(ep_infos["handle_active_joint_qmax"])
    handle_active_joint_qmin = np.mean(ep_infos["handle_active_joint_qmin"])
    articulation_slightly_closed = handle_active_joint_qpos < (
        handle_active_joint_qpos[0]
        - (handle_active_joint_qmax - handle_active_joint_qmin) * 0.05
    )
    articulation_open = ~articulation_slightly_closed

    events, events_verbose = [], []

    def append_event(event_step, event):
        if (len(events) == 0 or events[-1] != event) and event in [
            "contact",
            "closed",
            "slightly_closed",
            "opened",
            "success",
            "exceeded_cumulative_force_limit",
        ]:
            events.append(event)
        if event == "collision":
            if len(events_verbose) == 0 or events_verbose[-1][1] != "collision":
                events_verbose.append(
                    (
                        [event_step],
                        event,
                        [round(robot_force[event_step].item(), 2)],
                    )
                )
            else:
                events_verbose[-1][0].append(event_step)
                events_verbose[-1][2].append(round(robot_force[event_step].item(), 2))
        else:
            events_verbose.append((event_step, event, None))

    for step, (
        spcl,
        scl,
        sps,
        ss,
        spc,
        sc,
        srf,
        srcf,
        spscl,
        sscl,
        spo,
        so,
    ) in enumerate(
        zip(
            [False] + articulation_closed.tolist(),
            articulation_closed.tolist(),
            [False] + success.tolist(),
            success.tolist(),
            [False] + contacts.tolist(),
            contacts.tolist(),
            robot_force.tolist(),
            robot_cumulative_force.tolist(),
            [False] + articulation_slightly_closed.tolist(),
            articulation_slightly_closed.tolist(),
            [True] + articulation_open.tolist(),
            articulation_open.tolist(),
        )
    ):
        if not spc and sc:
            append_event(step, "contact")
        if spc and not sc:
            append_event(step, "lost_contact")
        if not spcl and scl:
            append_event(step, "closed")
        if not sps and ss:
            append_event(step, "success")
        if sps and not ss:
            append_event(step, "lost_success")
        if srf > 0:
            append_event(step, "collision")
        if srcf >= close_cfg.robot_cumulative_force_limit:
            if "exceeded_cumulative_force_limit" not in events:
                append_event(step, "exceeded_cumulative_force_limit")
        if not spscl and sscl:
            append_event(step, "slightly_closed")
        if not spo and so:
            append_event(step, "opened")

    def last_index(lst: list, x):
        if x in lst:
            return len(lst) - lst[::-1].index(x) - 1
        return -1

    if success.any():
        if "exceeded_cumulative_force_limit" in events:
            label = "success_then_excessive_collisions"
        else:
            if last_index(events, "opened") > last_index(events, "closed"):
                label = "dubious_success"
            else:
                label = "closed_success"
    else:
        if "exceeded_cumulative_force_limit" in events:
            label = "excessive_collision_failure"
        else:
            if "contact" not in events:
                label = "cant_reach_articulation_failure"
            elif (
                "opened" in events
                and last_index(events, "opened") > last_index(events, "closed")
                and last_index(events, "opened") > last_index(events, "slightly_closed")
            ):
                label = "open_after_closing_failure"
            elif last_index(events, "slightly_closed") > last_index(
                events, "closed"
            ) and last_index(events, "slightly_closed") > last_index(events, "opened"):
                label = "slightly_closed_failure"
            elif "closed" in events:
                label = "too_slow_failure"
            else:
                label = "cant_close_failure"

    return label, events, events_verbose
