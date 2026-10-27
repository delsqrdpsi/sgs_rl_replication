"""Relative Cartesian OSC on the wrist_3_link pose, after OmniReset's RelCartesianOSCAction.

Each policy step sets a target pose = current pose + scaled action (in the robot base frame). Each physics step
applies tau = J^T (Kp e - Kd J qdot), clamped per joint. J is the geometric Jacobian of the wrist_3_link origin,
built from the arm's joint anchors and axes; the arm's joints are the first six degrees of freedom.
"""

import numpy as np
import warp as wp

ACTION_SCALE = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.2])
KP = np.array([200.0, 200.0, 200.0, 3.0, 3.0, 3.0])
DAMPING_RATIO = np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])
KD = 2.0 * np.sqrt(KP) * DAMPING_RATIO
TORQUE_LIMIT = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

GRIPPER_CLOSED = 255.0  # 2F-85 actuator range is 0 (open) .. 255 (closed)

vec6 = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def rotation_error(target: wp.mat33, current: wp.mat33) -> wp.vec3:
    """Axis-angle vector of target * current^T (world frame)."""
    q = wp.quat_from_matrix(target * wp.transpose(current))
    if q[3] < 0.0:
        q = -q
    axis = wp.vec3()
    angle = float(0.0)
    wp.quat_to_axis_angle(q, axis, angle)
    return axis * angle


@wp.kernel
def osc_torque(
    qvel: wp.array2d(dtype=wp.float32),
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    xanchor: wp.array2d(dtype=wp.vec3),
    xaxis: wp.array2d(dtype=wp.vec3),
    target_pos: wp.array(dtype=wp.vec3),
    target_mat: wp.array(dtype=wp.mat33),
    ee_body: int,
    kp: vec6,
    kd: vec6,
    torque_limit: vec6,
    ctrl: wp.array2d(dtype=wp.float32),
):
    w = wp.tid()
    p = xpos[w, ee_body]
    ep = target_pos[w] - p
    er = rotation_error(target_mat[w], xmat[w, ee_body])

    # Geometric Jacobian columns: linear = axis x (p - anchor), angular = axis.
    v = wp.vec3()
    om = wp.vec3()
    for j in range(6):
        a = xaxis[w, j]
        v += wp.cross(a, p - xanchor[w, j]) * qvel[w, j]
        om += a * qvel[w, j]

    f = vec6()
    for i in range(3):
        f[i] = kp[i] * ep[i] - kd[i] * v[i]
        f[i + 3] = kp[i + 3] * er[i] - kd[i + 3] * om[i]

    for j in range(6):
        a = xaxis[w, j]
        lin = wp.cross(a, p - xanchor[w, j])
        tau = float(0.0)
        for i in range(3):
            tau += lin[i] * f[i] + a[i] * f[i + 3]
        ctrl[w, j] = wp.clamp(tau, -torque_limit[j], torque_limit[j])


def osc_torque_reference(qvel, ee_pos, ee_mat, anchors, axes, target_pos, target_mat):
    """NumPy version of osc_torque for one world, used by tests."""
    from scipy.spatial.transform import Rotation

    jac = np.zeros((6, 6))
    for j in range(6):
        jac[:3, j] = np.cross(axes[j], ee_pos - anchors[j])
        jac[3:, j] = axes[j]
    error = np.concatenate([target_pos - ee_pos, Rotation.from_matrix(target_mat @ ee_mat.T).as_rotvec()])
    force = KP * error - KD * (jac @ qvel[:6])
    return np.clip(jac.T @ force, -TORQUE_LIMIT, TORQUE_LIMIT)


def axis_angle_to_matrix(rotvec):
    """Batched Rodrigues formula (torch), rotvec (..., 3) -> (..., 3, 3)."""
    import torch

    angle = rotvec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = rotvec / angle
    kx, ky, kz = k.unbind(-1)
    zero = torch.zeros_like(kx)
    skew = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], -1).reshape(*k.shape[:-1], 3, 3)
    s, c = angle.sin()[..., None], angle.cos()[..., None]
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand_as(skew)
    return eye + s * skew + (1 - c) * (skew @ skew)
