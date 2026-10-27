"""MuJoCo scene: UR5e + Robotiq 2F-85, a 16 mm rod, and a fixed block with a 16.5 mm bore.

See DESIGN.md for where each value comes from.
"""

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[2] / "assets"

TIMESTEP = 0.005  # SGS: 200 Hz physics
TABLE_TOP = 0.0  # robot base and table surface share z = 0

ROD_RADIUS = 0.008  # SGS: 16 mm rod
ROD_HALF_LENGTH = 0.0375  # assumption: 75 mm rod, so 50 mm stands proud when seated, room for the fingers
ROD_MASS = 0.041  # assumption: aluminium, pi * r^2 * L * 2700 kg/m^3

BORE_RADIUS = 0.00825  # 16.5 mm bore: 0.5 mm diametral clearance
HOLE_DEPTH = 0.025  # assumption
HOLE_SIDES = 16  # the bore is a 16-sided prism; its flat faces sit at BORE_RADIUS
HOLE_POS = np.array([-0.13, 0.55, TABLE_TOP])  # assumption: in front of the robot, centre of the bore on the table

ROD_REST = np.array([0.10, 0.50, TABLE_TOP + ROD_HALF_LENGTH])  # standing upright

PAD_FRICTION = 2.0  # SGS: mu = 2 with max combine (MuJoCo combines friction by max)

# Collision bits: robot geoms touch the rod and the static world, never each other;
# the rod touches everything; static geoms never test against each other.
ROBOT, STATIC, ROD = (1, 2), (2, 1), (3, 3)

ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
HOME_QPOS = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])  # Menagerie "home" keyframe


def _set_bits(geom, bits):
    geom.contype, geom.conaffinity = bits


def build_spec() -> mujoco.MjSpec:
    arm = mujoco.MjSpec.from_file(str(ASSETS / "universal_robots_ur5e" / "ur5e.xml"))
    hand = mujoco.MjSpec.from_file(str(ASSETS / "robotiq_2f85" / "2f85.xml"))

    for key in list(arm.keys):
        arm.delete(key)
    # Torque control for OSC: replace the position servos with motors at the same force limits.
    for act in list(arm.actuators):
        limit = act.forcerange[1]
        joint = act.target
        arm.delete(act)
        arm.add_actuator(name=joint.removesuffix("_joint"), target=joint,
                         trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[-limit, limit], ctrllimited=True)

    for geom in hand.geoms:
        if geom.name in ("left_pad1", "left_pad2", "right_pad1", "right_pad2"):
            geom.friction[0] = PAD_FRICTION

    spec = arm
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC  # the 2F-85 model's own settings
    spec.option.impratio = 10
    spec.site("attachment_site").attach_body(hand.body("base_mount"), "hand/", "")

    # OmniReset's robot assets are gravity-free; MuJoCo's equivalent is full gravity compensation.
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            _set_bits(geom, ROBOT)

    world = spec.worldbody
    world.add_light(pos=[0, 0.5, 1.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    table = world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.8, 0.8, 0.02],
                           pos=[0, 0.4, TABLE_TOP - 0.02], rgba=[0.6, 0.55, 0.5, 1])
    _set_bits(table, STATIC)

    block = world.add_body(name="hole", pos=HOLE_POS.tolist())
    block.add_site(name="hole_seat", pos=[0, 0, 0])  # rod bottom centre when fully seated
    step = 2 * np.pi / HOLE_SIDES
    wall = 0.02
    half_width = (BORE_RADIUS + 2 * wall) * np.tan(step / 2) + 1e-3
    for i in range(HOLE_SIDES):
        angle = i * step
        centre = (BORE_RADIUS + wall) * np.array([np.cos(angle), np.sin(angle)])
        geom = block.add_geom(name=f"hole_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                              size=[wall, half_width, HOLE_DEPTH / 2],
                              pos=[centre[0], centre[1], HOLE_DEPTH / 2],
                              quat=[np.cos(angle / 2), 0, 0, np.sin(angle / 2)], rgba=[0.3, 0.35, 0.45, 1])
        _set_bits(geom, STATIC)

    rod = world.add_body(name="rod", pos=ROD_REST.tolist())
    rod.add_freejoint(name="rod")
    geom = rod.add_geom(name="rod", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[ROD_RADIUS, ROD_HALF_LENGTH],
                        mass=ROD_MASS, rgba=[0.75, 0.75, 0.8, 1])
    _set_bits(geom, ROD)
    rod.add_site(name="rod_tip", pos=[0, 0, -ROD_HALF_LENGTH])  # bottom face centre

    spec.add_key(name="home", qpos=np.concatenate([HOME_QPOS, np.zeros(8),
                                                   ROD_REST, [1, 0, 0, 0]]))
    return spec


def build_model() -> mujoco.MjModel:
    return build_spec().compile()
