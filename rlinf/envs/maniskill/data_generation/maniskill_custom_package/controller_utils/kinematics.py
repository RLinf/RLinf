"""
Code for kinematics utilities on CPU/GPU
"""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from typing import List

try:
    import pytorch_kinematics as pk
except ImportError:
    raise ImportError(
        "pytorch_kinematics_ms not installed. Install with pip install pytorch_kinematics_ms"
    )
import torch
from sapien.wrapper.pinocchio_model import PinocchioModel

from mani_skill.utils import common
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.articulation_joint import ArticulationJoint
from mani_skill.utils.structs.pose import Pose

# currently fast_kinematics has some bugs on some systems so we use the slower pytorch kinematics package instead.
# try:
#     import fast_kinematics
# except:
#     # not all systems support the fast_kinematics package at the moment
#     fast_kinematics = None


class Kinematics:
    def __init__(
        self,
        urdf_path: str,
        end_link_name: str,
        articulation: Articulation,
        active_joint_indices: torch.Tensor,
    ):
        """
        Initialize the kinematics solver. It will be run on whichever device the articulation is on.

        Args:
            urdf_path (str): path to the URDF file
            end_link_name (str): name of the end-effector link
            articulation (Articulation): the articulation object
            active_joint_indices (torch.Tensor): indices of the active joints that can be controlled
        """
        self.urdf_path = urdf_path
        self.end_link = articulation.links_map[end_link_name]
        self.end_link_idx = articulation.links.index(self.end_link)
        self.active_joint_indices = active_joint_indices
        self.articulation = articulation
        self.device = articulation.device
        # note that everything past the end-link is ignored. Any joint whose ancestor is self.end_link is ignored
        cur_link = self.end_link
        active_ancestor_joints: List[ArticulationJoint] = []
        while cur_link is not None:
            if cur_link.joint.active_index is not None:
                active_ancestor_joints.append(cur_link.joint)
            cur_link = cur_link.joint.parent_link
        active_ancestor_joints = active_ancestor_joints[::-1]
        self.active_ancestor_joints = active_ancestor_joints

        # initially self.active_joint_indices references active joints that are controlled.
        # we also make the assumption that the active index is the same across all parallel managed joints
        self.active_ancestor_joint_idxs = [
            (x.active_index[0]).cpu().item() for x in self.active_ancestor_joints
        ]
        self.controlled_joints_idx_in_qmask = [
            self.active_ancestor_joint_idxs.index(idx)
            for idx in self.active_joint_indices
        ]
        if self.device.type == "cuda":
            self.use_gpu_ik = True
            self._setup_gpu()
        else:
            self.use_gpu_ik = False
            self._setup_cpu()

    def _setup_cpu(self):
        """setup the kinematics solvers on the CPU"""
        self.use_gpu_ik = False
        # NOTE (stao): currently using the pinnochio that comes packaged with SAPIEN
        self.qmask = torch.zeros(
            self.articulation.max_dof, dtype=bool, device=self.device
        )
        self.pmodel: PinocchioModel = self.articulation._objs[
            0
        ].create_pinocchio_model()
        self.qmask[self.active_joint_indices] = 1

    def _setup_gpu(self):
        """setup the kinematics solvers on the GPU"""
        self.use_gpu_ik = True
        with open(self.urdf_path, "rb") as f:
            urdf_str = f.read()

        # NOTE (stao): it seems that the pk library currently always outputs some complaints if there are unknown attributes in a URDF. Hide it with this contextmanager here
        @contextmanager
        def suppress_stdout_stderr():
            """A context manager that redirects stdout and stderr to devnull"""
            with open(devnull, "w") as fnull:
                with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
                    yield (err, out)

        with suppress_stdout_stderr():
            self.pk_chain = pk.build_serial_chain_from_urdf(
                urdf_str,
                end_link_name=self.end_link.name,
            ).to(device=self.device)
        lim = torch.tensor(self.pk_chain.get_joint_limits(), device=self.device)
        self.pik = pk.PseudoInverseIK(
            self.pk_chain,
            joint_limits=lim.T,
            early_stopping_any_converged=True,
            max_iterations=200,
            num_retries=1,
        )

        self.qmask = torch.zeros(
            len(self.active_ancestor_joints), dtype=bool, device=self.device
        )
        self.qmask[self.controlled_joints_idx_in_qmask] = 1

    def compute_ik(
        self,
        target_pose: Pose,
        q0: torch.Tensor,
        pos_only: bool = False,
        action=None,
        use_delta_ik_solver: bool = False,
    ):
        """Given a target pose, via inverse kinematics compute the target joint positions that will achieve the target pose

        Args:
            target_pose (Pose): target pose of the end effector in the world frame. note this is not relative to the robot base frame!
            q0 (torch.Tensor): initial joint positions of every active joint in the articulation
            pos_only (bool): if True, only the position of the end link is considered in the IK computation
            action (torch.Tensor): delta action to be applied to the articulation. Used for fast delta IK solutions on the GPU.
            use_delta_ik_solver (bool): If true, returns the target joint positions that correspond with a delta IK solution. This is specifically
                used for GPU simulation to determine which GPU IK algorithm to use.
        """
        if self.use_gpu_ik:
            q0 = q0[:, self.active_ancestor_joint_idxs]
            if not use_delta_ik_solver:
                tf = pk.Transform3d(
                    pos=target_pose.p,
                    rot=target_pose.q,
                    device=self.device,
                )
                self.pik.initial_config = q0  # shape (num_retries, active_ancestor_dof)
                result = self.pik.solve(
                    tf
                )  # produce solutions in shape (B, num_retries/initial_configs, active_ancestor_dof)
                # TODO return mask for invalid solutions. CPU returns None at the moment
                return result.solutions[:, 0, :]
            else:
                jacobian = self.pk_chain.jacobian(q0)
                # code commented out below is the fast kinematics method
                # jacobian = (
                #     self.fast_kinematics_model.jacobian_mixed_frame_pytorch(
                #         self.articulation.get_qpos()[:, self.active_ancestor_joint_idxs]
                #     )
                #     .view(-1, len(self.active_ancestor_joints), 6)
                #     .permute(0, 2, 1)
                # )
                # jacobian = jacobian[:, :, self.qmask]
                if pos_only:
                    jacobian = jacobian[:, 0:3]

                # NOTE (stao): this method of IK is from https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf by Samuel R. Buss
                delta_joint_pos = torch.linalg.pinv(jacobian) @ action.unsqueeze(-1)
                return q0 + delta_joint_pos.squeeze(-1)
        else:
            result, success, error = self.pmodel.compute_inverse_kinematics(
                self.end_link_idx,
                target_pose.sp,
                initial_qpos=q0.cpu().numpy()[0],
                active_qmask=self.qmask,
                max_iterations=100,
            )
            if success:
                return common.to_tensor(
                    [result[self.active_ancestor_joint_idxs]], device=self.device
                )
            else:
                return None

    def compute_fk(self, qpos: torch.Tensor):
        """Given joint positions, compute the end effector pose

        Args:
            qpos (torch.Tensor): joint positions of every active joint in the articulation
        """
        if self.use_gpu_ik:
            qpos = qpos[..., self.active_ancestor_joint_idxs]
            tf_matrix = self.pk_chain.forward_kinematics(qpos.float()).get_matrix()
            pos = tf_matrix[:, :3, 3]
            rot = pk.matrix_to_quaternion(tf_matrix[:, :3, :3])
            return Pose.create_from_pq(pos, rot, device=self.device)
        else:
            self.pmodel.compute_forward_kinematics(qpos[0].cpu().numpy())
            ee_pose = self.pmodel.get_link_pose(self.end_link_idx)
            return Pose.create(ee_pose, device=self.device)
