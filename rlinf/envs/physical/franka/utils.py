import numpy as np

def normalize(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    return q / n

# geometry
def quat_slerp(q0, q1, t):
    """
    Slerp q0 and q1 with t steps.
    This is computed by the 4-dim sphere. And it does not depend on the order of "xyzw".
    """
    
    q0 = normalize(q0)
    q1 = normalize(q1)

    dot = np.dot(q0, q1)

    # ensure shortest path
    if dot < 0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if np.isscalar(t):
        t_arr = np.linspace(0, 1, t, dtype=float)
    else:
        t_arr = np.array(t, dtype=float)

    results = []

    # nearly identical → fallback to LERP
    if dot > 0.9995:
        for tt in t_arr:
            q = normalize(q0 + tt * (q1 - q0))
            results.append(q)
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)

        for tt in t_arr:
            theta = theta_0 * tt
            s0 = np.sin(theta_0 - theta) / sin_theta_0
            s1 = np.sin(theta) / sin_theta_0
            q = s0 * q0 + s1 * q1
            results.append(q)

    results = np.stack(results)
    return results