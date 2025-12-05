import collections
import numpy as np
import os
import robosuite
import xml.etree.ElementTree as ET

from copy import copy

from robosuite.models.objects import MujocoObject


class ObjectPropertySampler:
    """
    Base class of object placement sampler.
    Args:
        name (str): Name of this sampler.
        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models
        ensure_object_boundary_in_range (bool): If True, will ensure that the object is enclosed within a given boundary
            (should be implemented by subclass)
        ensure_valid_placement (bool): If True, will check for correct (valid) object placements
        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur
        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
    ):
        # Setup attributes
        self.name = name
        if mujoco_objects is None:
            self.mujoco_objects = []
        else:
            # Shallow copy the list so we don't modify the inputted list but still keep the object references
            self.mujoco_objects = (
                [mujoco_objects]
                if isinstance(mujoco_objects, MujocoObject)
                else copy(mujoco_objects)
            )

    def add_objects(self, mujoco_objects):
        """
        Add additional objects to this sampler. Checks to make sure there's no identical objects already stored.
        Args:
            mujoco_objects (MujocoObject or list of MujocoObject): single model or list of MJCF object models
        """
        mujoco_objects = (
            [mujoco_objects]
            if isinstance(mujoco_objects, MujocoObject)
            else mujoco_objects
        )
        for obj in mujoco_objects:
            assert (
                obj not in self.mujoco_objects
            ), "Object '{}' already in sampler!".format(obj.name)
            self.mujoco_objects.append(obj)

    def reset(self):
        """
        Resets this sampler. Removes all mujoco objects from this sampler.
        """
        self.mujoco_objects = []

    def sample(self, predicate_name=None):
        """
        Uniformly sample on a surface (not necessarily table surface).
        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)
            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.
            on_top (bool): if True, sample placement on top of the reference object.
        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form
        """
        raise NotImplementedError


class OpenCloseSampler(ObjectPropertySampler):
    def __init__(
        self,
        name,
        state_type,
        mujoco_objects=None,
        joint_ranges=(0.0, 0.0),
    ):
        assert state_type in ["open", "close"]
        self.state_type = state_type
        self.joint_ranges = joint_ranges
        assert self.joint_ranges[0] <= self.joint_ranges[1]
        super().__init__(name, mujoco_objects)

    def sample(self):
        return np.random.uniform(high=self.joint_ranges[1], low=self.joint_ranges[0])


class TurnOnOffSampler(ObjectPropertySampler):
    def __init__(
        self,
        name,
        state_type,
        mujoco_objects=None,
        joint_ranges=(0.0, 0.0),
    ):
        assert state_type in ["turnon", "turnoff"]
        self.state_type = state_type
        self.joint_ranges = joint_ranges
        assert self.joint_ranges[0] <= self.joint_ranges[1]
        super().__init__(name, mujoco_objects)

    def sample(self):
        return np.random.uniform(high=self.joint_ranges[1], low=self.joint_ranges[0])
