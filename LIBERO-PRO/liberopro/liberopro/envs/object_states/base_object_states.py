import robosuite.utils.transform_utils as transform_utils
import numpy as np


class BaseObjectState:
    def __init__(self):
        pass

    def get_geom_state(self):
        raise NotImplementedError

    def check_contact(self, other):
        raise NotImplementedError

    def check_contain(self, other):
        raise NotImplementedError

    def get_joint_state(self):
        raise NotImplementedError

    def is_open(self):
        raise NotImplementedError

    def is_close(self):
        raise NotImplementedError

    def get_size(self):
        raise NotImplementedError

    def check_ontop(self, other):
        raise NotImplementedError


class ObjectState(BaseObjectState):
    def __init__(self, env, object_name, is_fixture=False):
        self.env = env
        self.object_name = object_name
        self.is_fixture = is_fixture
        self.query_dict = (
            self.env.fixtures_dict if self.is_fixture else self.env.objects_dict
        )
        self.object_state_type = "object"
        self.has_turnon_affordance = hasattr(
            self.env.get_object(self.object_name), "turn_on"
        )

    def get_geom_state(self):
        object_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[self.object_name]]
        object_quat = self.env.sim.data.body_xquat[
            self.env.obj_body_id[self.object_name]
        ]
        return {"pos": object_pos, "quat": object_quat}

    def check_contact(self, other):
        object_1 = self.env.get_object(self.object_name)
        object_2 = self.env.get_object(other.object_name)
        return self.env.check_contact(object_1, object_2)

    def check_contain(self, other):
        object_1 = self.env.get_object(self.object_name)
        object_1_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]
        object_2 = self.env.get_object(other.object_name)
        object_2_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[other.object_name]
        ]
        return object_1.in_box(object_1_position, object_2_position)

    def get_joint_state(self):
        # Return None if joint state does not exist
        joint_states = []
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            joint_states.append(self.env.sim.data.qpos[qpos_addr])
        return joint_states

    def check_ontop(self, other):
        this_object = self.env.get_object(self.object_name)
        this_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]
        other_object = self.env.get_object(other.object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[other.object_name]
        ]
        return (
            (this_object_position[2] <= other_object_position[2])
            and self.check_contact(other)
            and (
                np.linalg.norm(this_object_position[:2] - other_object_position[:2])
                < 0.03
            )
        )

    def set_joint(self, qpos=1.5):
        for joint in self.env.get_object(self.object_name).joints:
            self.env.sim.data.set_joint_qpos(joint, qpos)

    def is_open(self):
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if self.env.get_object(self.object_name).is_open(qpos):
                return True
        return False

    def is_close(self):
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if not (self.env.get_object(self.object_name).is_close(qpos)):
                return False
        return True

    def turn_on(self):
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if self.env.get_object(self.object_name).turn_on(qpos):
                return True
        return False

    def turn_off(self):
        for joint in self.env.get_object(self.object_name).joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if not (self.env.get_object(self.object_name).turn_off(qpos)):
                return False
        return True

    def update_state(self):
        if self.has_turnon_affordance:
            self.turn_on()


class SiteObjectState(BaseObjectState):
    """
    This is to make site based objects to have the same API as normal Object State.
    """

    def __init__(self, env, object_name, parent_name, is_fixture=False):
        self.env = env
        self.object_name = object_name
        self.parent_name = parent_name
        self.is_fixture = self.parent_name in self.env.fixtures_dict
        self.query_dict = (
            self.env.fixtures_dict if self.is_fixture else self.env.objects_dict
        )
        self.object_state_type = "site"

    def get_geom_state(self):
        object_pos = self.env.sim.data.get_site_xpos(self.object_name)
        object_quat = transform_utils.mat2quat(
            self.env.sim.data.get_site_xmat(self.object_name)
        )
        return {"pos": object_pos, "quat": object_quat}

    def check_contain(self, other):
        this_object = self.env.object_sites_dict[self.object_name]
        this_object_position = self.env.sim.data.get_site_xpos(self.object_name)
        this_object_mat = self.env.sim.data.get_site_xmat(self.object_name)

        other_object = self.env.get_object(other.object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[other.object_name]
        ]
        return this_object.in_box(
            this_object_position, this_object_mat, other_object_position
        )

    def check_contact(self, other):
        """
        There is no dynamics for site objects, so we return true all the time.
        """
        return True

    def check_ontop(self, other):
        this_object = self.env.object_sites_dict[self.object_name]
        if hasattr(this_object, "under"):
            this_object_position = self.env.sim.data.get_site_xpos(self.object_name)
            this_object_mat = self.env.sim.data.get_site_xmat(self.object_name)
            other_object = self.env.get_object(other.object_name)
            other_object_position = self.env.sim.data.body_xpos[
                self.env.obj_body_id[other.object_name]
            ]
            # print(self.object_name, this_object_position)
            # print(other_object_position)

            parent_object = self.env.get_object(self.parent_name)
            if parent_object is None:
                return this_object.under(
                    this_object_position, this_object_mat, other_object_position
                )
            else:
                return this_object.under(
                    this_object_position, this_object_mat, other_object_position
                ) and self.env.check_contact(parent_object, other_object)
        else:
            return True

    def set_joint(self, qpos=1.5):
        for joint in self.env.object_sites_dict[self.object_name].joints:
            self.env.sim.data.set_joint_qpos(joint, qpos)

    def is_open(self):
        for joint in self.env.object_sites_dict[self.object_name].joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if self.env.get_object(self.parent_name).is_open(qpos):
                return True
        return False

    def is_close(self):
        for joint in self.env.object_sites_dict[self.object_name].joints:
            qpos_addr = self.env.sim.model.get_joint_qpos_addr(joint)
            qpos = self.env.sim.data.qpos[qpos_addr]
            if not (self.env.get_object(self.parent_name).is_close(qpos)):
                return False
        return True
