import torch

import sapien

from mani_skill.utils.scene_builder.registration import register_scene_builder
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from mani_skill.utils.structs import Actor, Articulation


@register_scene_builder("ReplicaCAD1Cam")
class ReplicaCADSceneBuilderFetch1Cam(ReplicaCADSceneBuilder):

    def initialize(self, env_idx: torch.Tensor):

        # teleport robot away for init
        self.env.agent.robot.set_pose(sapien.Pose([-10, 0, -100]))

        for obj, pose in self._default_object_poses:
            obj.set_pose(pose)
            if isinstance(obj, Articulation):
                # note that during initialization you may only ever change poses/qpos of objects in scenes being reset
                obj.set_qpos(obj.qpos[0] * 0)
                obj.set_qvel(obj.qvel[0] * 0)

        if self.scene.gpu_sim_enabled and len(env_idx) == self.env.num_envs:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene.px.step()
            self.scene._gpu_fetch_all()

        # teleport robot back to correct location
        if self.env.robot_uids in ["fetch", "fetch_1cam"]:
            self.env.agent.reset(self.env.agent.keyframes["rest"].qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1, 0, 0.02]))
        else:
            raise NotImplementedError(self.env.robot_uids)
