import time
from typing import List, Optional, Tuple

import mink
import mujoco
import numpy as np


class Kinematics:
    def __init__(self, xml_path: str, site_name: Optional[str]):
        """Initialize the Kinematics object.

        Args:
            xml_path (str): Path to the MuJoCo XML model file.
            site_name (Optional[str]): Name of the site for which to compute the forward kinematics.
        """
        model = mujoco.MjModel.from_xml_path(xml_path)
        self._configuration = mink.Configuration(model)
        self._site_name = site_name

    def fk(self, q: np.ndarray, site_name: Optional[str] = None) -> np.ndarray:
        """Compute the forward kinematics for the given joint configuration.

        Args:
            q (np.ndarray): The joint configuration.
            site_name (Optional[str]): Name of the site for which to compute the forward kinematics.
                       If not provided, the default site name is used.

        Returns:
            (np.ndarray): Site frame in world frame. Shape: (4, 4)
        """
        self._configuration.update(q)
        site_name = site_name or self._site_name
        assert site_name is not None, "site_name must be provided"
        return self._configuration.get_transform_frame_to_world(site_name, "site").as_matrix()

    def ik(
        self,
        target_pose: np.ndarray,
        site_name: str,
        init_q: Optional[np.ndarray] = None,
        limits: Optional[List[mink.Limit]] = None,
        dt: float = 0.01,
        solver: str = "quadprog",
        pos_threshold: float = 1e-4,
        ori_threshold: float = 1e-4,
        damping: float = 1e-4,
        max_iters: int = 200,
        verbose: bool = False,
    ) -> Tuple[bool, np.ndarray]:
        """Differential ik solver, leverging mink.

        Args:
            target_pose (np.ndarray): The target pose to reach.
            site_name (str): Name of the desired site.
            limits (List[mink.Limit]): List of limits to enforce.
            init_q (Optional[np.ndarray]): Initial joint configuration.
            dt (float): Integration timestep in [s].
            solver (str): Quadratic program solver.
            pos_threshold (float): Position threshold for convergence.
            ori_threshold (float): Orientation threshold for convergence.
            damping (float): Levenberg-Marquardt damping.
            max_iters (int): Maximum number of iterations.
            verbose (bool): Whether to print debug information.

        Returns:
            Tuple[bool, np.ndarray]: Success flag and the converged joint configuration.
        """
        if init_q is not None:
            self._configuration.update(init_q)

        end_effector_task = mink.FrameTask(
            frame_name=site_name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

        end_effector_task.set_target(mink.SE3.from_matrix(target_pose))
        tasks = [end_effector_task]

        start_time = time.time()  # Start timing

        for j in range(max_iters):
            vel = mink.solve_ik(self._configuration, tasks, dt, solver, damping=damping, limits=limits)
            self._configuration.integrate_inplace(vel, dt)
            err = end_effector_task.compute_error(self._configuration)

            pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
            if pos_achieved and ori_achieved:
                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time
                if verbose:
                    print(
                        f"Exiting after {j} iterations, configuration: {self._configuration.q}, time taken: {elapsed_time:.4f} seconds"
                    )
                return True, self._configuration.q

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        if verbose:
            print(
                f"Failed to converge after {max_iters} iterations, time taken: {elapsed_time:.4f} seconds, pos_err: {err[:3]}, rot_err: {err[3:]}"
            )
        return False, self._configuration.q


def main() -> None:
    from i2rt.robots.utils import YAM_XML_PATH

    mj_model = Kinematics(YAM_XML_PATH, "grasp_site")
    q = np.zeros(6)
    pose = mj_model.fk(q)
    print(pose)

    pose[0, 3] -= 0.1
    pose[2, 3] += 0.1
    print(pose)
    q_ik = mj_model.ik(pose, "grasp_site")
    print(q_ik)


if __name__ == "__main__":
    main()
