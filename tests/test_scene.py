import mujoco
import numpy as np

from sgs_rod import scene


def _drop(tilt):
    m = scene.build_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    adr = m.joint("rod").qposadr[0]
    d.qpos[adr:adr + 3] = scene.HOLE_POS + [0.002, 0, scene.ROD_HALF_LENGTH + 0.005]
    d.qpos[adr + 3:adr + 7] = [np.cos(tilt / 2), np.sin(tilt / 2), 0, 0]
    for _ in range(400):
        mujoco.mj_step(m, d)
    return d.site("rod_tip").xpos - d.site("hole_seat").xpos


def test_rod_seats_in_bore():
    for tilt in (0.0, 0.05):
        offset = _drop(tilt)
        assert np.linalg.norm(offset) < 1e-3


def test_robot_parts():
    m = scene.build_model()
    assert m.opt.timestep == scene.TIMESTEP
    assert m.nu == 7  # six arm torque motors + gripper
    for i in range(1, m.nbody):
        expected = 0 if m.body(i).name in ("rod", "hole") else 1
        assert m.body_gravcomp[i] == expected
    pad = m.geom("hand/left_pad1")
    assert pad.friction[0] == scene.PAD_FRICTION
