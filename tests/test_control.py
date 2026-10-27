import mujoco
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from sgs_rod import control, scene


def test_osc_kernel_matches_reference():
    m = scene.build_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    rng = np.random.default_rng(0)
    d.qpos[:6] += rng.uniform(-0.3, 0.3, 6)
    d.qvel[:6] = rng.uniform(-0.5, 0.5, 6)
    mujoco.mj_forward(m, d)
    ee = m.body("wrist_3_link").id
    target_pos = d.xpos[ee] + [0.01, -0.02, 0.005]
    target_mat = Rotation.from_rotvec([0.05, -0.02, 0.1]).as_matrix() @ d.xmat[ee].reshape(3, 3)
    expected = control.osc_torque_reference(d.qvel, d.xpos[ee], d.xmat[ee].reshape(3, 3),
                                            d.xanchor[:6], d.xaxis[:6], target_pos, target_mat)

    f32 = lambda x: np.asarray(x, dtype=np.float32)
    with wp.ScopedDevice("cpu"):
        ctrl = wp.zeros((1, m.nu), dtype=wp.float32)
        wp.launch(control.osc_torque, dim=1, inputs=[
            wp.array(f32(d.qvel[None]), dtype=wp.float32),
            wp.array(f32(d.xpos[None]), dtype=wp.vec3),
            wp.array(f32(d.xmat.reshape(1, -1, 3, 3)), dtype=wp.mat33),
            wp.array(f32(d.xanchor[None]), dtype=wp.vec3),
            wp.array(f32(d.xaxis[None]), dtype=wp.vec3),
            wp.array(f32(target_pos[None]), dtype=wp.vec3),
            wp.array(f32(target_mat[None]), dtype=wp.mat33),
            ee, control.vec6(*control.KP), control.vec6(*control.KD), control.vec6(*control.TORQUE_LIMIT), ctrl])
        got = ctrl.numpy()[0, :6]
    assert np.allclose(got, expected, rtol=1e-3, atol=1e-3)


def test_axis_angle_to_matrix():
    v = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])
    expected = Rotation.from_rotvec(v.numpy()).as_matrix()
    assert np.allclose(control.axis_angle_to_matrix(v).numpy(), expected, atol=1e-6)
