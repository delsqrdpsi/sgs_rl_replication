"""Generate SGS start configurations on the CPU with MuJoCo.

Each candidate: sample the rod pose and a gripper pose for one family, solve IK for the arm, reject
penetrating placements, let the fingers move to their commanded opening and the rod settle (arm held
still), then validate. Accepted configurations store the settled qpos and the gripper command.
Ranges SGS does not state are marked as assumptions.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import scene
from .bank import FAMILIES

GRASP_DEPTH = 0.020  # pinch point sits this far below the rod's top face (fingers cover the top ~37 mm)
NEAR_GOAL_FRACTION = (0.0, 1.1)  # SGS rod-in-hole: 0 = fully seated, 1 = rod tip at the hole entrance
NEAR_GOAL_XY = 0.0002  # assumption: within the 0.25 mm radial clearance
NEAR_GOAL_OPEN_PROB = 0.5  # assumption: share of Near-Goal configurations with an open hand
STABLE_HEIGHT = (0.01, 0.10)  # assumption: rod tip height above the hole entrance
STABLE_XY = 0.05  # assumption
STABLE_TILT = 0.2  # assumption, rad
REACH_X, REACH_Y = (-0.35, 0.10), (0.38, 0.70)  # assumption: table region the rod may stand in
REACH_CLEAR = 0.05  # keep the rod this far from the bore axis
REACH_OFFSET_UP = (0.02, 0.15)  # assumption: gripper starts this far above the grasp pose
REACH_OFFSET_XY = 0.05
REACH_TILT = 0.2

CLOSE_STEPS = 60  # 0.3 s: fingers move to the commanded opening, rod held for grasped families
SETTLE_STEPS = 60  # 0.3 s: rod released
PENETRATION = 0.0005


@dataclass
class Candidate:
    rod_pos: np.ndarray
    rod_rot: Rotation
    pinch_pos: np.ndarray
    pinch_rot: Rotation
    gripper_ctrl: float
    grasped: bool


def _grasp_pose(rod_pos, rod_rot, yaw):
    """Pinch pose for a top-down grasp: pinch z along -rod axis, any yaw about the axis."""
    pinch_rot = rod_rot * Rotation.from_euler("z", yaw) * Rotation.from_euler("x", np.pi)
    pinch_pos = rod_pos + rod_rot.apply([0, 0, scene.ROD_HALF_LENGTH - GRASP_DEPTH])
    return pinch_pos, pinch_rot


def _random_tilt(rng, max_angle):
    axis = rng.normal(size=2)
    axis = np.append(axis / np.linalg.norm(axis), 0)
    return Rotation.from_rotvec(axis * rng.uniform(0, max_angle))


def sample_candidate(family, rng):
    hole = scene.HOLE_POS
    yaw = rng.uniform(-np.pi, np.pi)
    if family == "near_goal":
        frac = rng.uniform(*NEAR_GOAL_FRACTION)
        xy = hole[:2] + rng.uniform(-NEAR_GOAL_XY, NEAR_GOAL_XY, 2)
        rod_pos = np.array([*xy, hole[2] + frac * scene.HOLE_DEPTH + scene.ROD_HALF_LENGTH])
        rod_rot = Rotation.from_euler("z", rng.uniform(-np.pi, np.pi))
        pinch_pos, pinch_rot = _grasp_pose(rod_pos, rod_rot, yaw)
        open_hand = rng.random() < NEAR_GOAL_OPEN_PROB
        ctrl = rng.uniform(0, 100) if open_hand else 255.0
        return Candidate(rod_pos, rod_rot, pinch_pos, pinch_rot, ctrl, not open_hand)
    if family == "stable_grasp":
        rod_rot = _random_tilt(rng, STABLE_TILT) * Rotation.from_euler("z", rng.uniform(-np.pi, np.pi))
        tip = np.array([*(hole[:2] + rng.uniform(-STABLE_XY, STABLE_XY, 2)),
                        hole[2] + scene.HOLE_DEPTH + rng.uniform(*STABLE_HEIGHT)])
        rod_pos = tip + rod_rot.apply([0, 0, scene.ROD_HALF_LENGTH])
        pinch_pos, pinch_rot = _grasp_pose(rod_pos, rod_rot, yaw)
        return Candidate(rod_pos, rod_rot, pinch_pos, pinch_rot, 255.0, True)
    if family == "reaching":
        while True:
            xy = np.array([rng.uniform(*REACH_X), rng.uniform(*REACH_Y)])
            if np.linalg.norm(xy - hole[:2]) > REACH_CLEAR:
                break
        rod_pos = np.array([*xy, scene.TABLE_TOP + scene.ROD_HALF_LENGTH])
        rod_rot = Rotation.from_euler("z", rng.uniform(-np.pi, np.pi))
        pinch_pos, pinch_rot = _grasp_pose(rod_pos, rod_rot, yaw)
        pinch_pos = pinch_pos + [*rng.uniform(-REACH_OFFSET_XY, REACH_OFFSET_XY, 2), rng.uniform(*REACH_OFFSET_UP)]
        pinch_rot = _random_tilt(rng, REACH_TILT) * pinch_rot
        return Candidate(rod_pos, rod_rot, pinch_pos, pinch_rot, rng.uniform(0, 255), False)
    raise ValueError(family)


class Generator:
    def __init__(self, seed):
        self.m = scene.build_model()
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        m = self.m
        self.pinch = m.site("hand/pinch").id
        self.rod_adr = m.joint("rod").qposadr[0]
        self.rod_vadr = m.joint("rod").dofadr[0]
        self.rod_body = m.body("rod").id
        self.pads = {m.body("hand/left_pad").id, m.body("hand/right_pad").id}
        self.static = {0, m.body("hole").id}
        self.home = scene.HOME_QPOS
        self.lower, self.upper = m.jnt_range[:6, 0], m.jnt_range[:6, 1]

    def ik(self, pos, rot, iters=150):
        m, d = self.m, self.d
        target = rot.as_matrix()
        d.qpos[:6] = self.home + self.rng.uniform(-0.3, 0.3, 6)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            cur = d.site_xmat[self.pinch].reshape(3, 3)
            err = np.concatenate([pos - d.site_xpos[self.pinch], Rotation.from_matrix(target @ cur.T).as_rotvec()])
            if np.linalg.norm(err[:3]) < 2e-4 and np.linalg.norm(err[3:]) < 2e-3:
                return True
            mujoco.mj_jacSite(m, d, jacp, jacr, self.pinch)
            jac = np.vstack([jacp[:, :6], jacr[:, :6]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            d.qpos[:6] = np.clip(d.qpos[:6] + np.clip(dq, -0.2, 0.2), self.lower, self.upper)
        return False

    def _contacts(self):
        m, d = self.m, self.d
        for c in d.contact[:d.ncon]:
            yield m.geom_bodyid[c.geom[0]], m.geom_bodyid[c.geom[1]], c.dist

    def _placement_ok(self, grasped):
        for b1, b2, dist in self._contacts():
            if dist > -PENETRATION:
                continue
            pair = {b1, b2}
            if self.rod_body in pair and grasped and pair & self.pads:
                continue
            return False
        return True

    def _held(self):
        touching = set()
        for b1, b2, _ in self._contacts():
            if self.rod_body in (b1, b2):
                touching |= {b1, b2} & self.pads
        return touching == self.pads

    def try_one(self, family):
        m, d = self.m, self.d
        c = sample_candidate(family, self.rng)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        if not self.ik(c.pinch_pos, c.pinch_rot):
            return None
        arm = d.qpos[:6].copy()
        rod = np.concatenate([c.rod_pos, np.roll(c.rod_rot.as_quat(), 1)])  # wxyz
        d.qpos[self.rod_adr:self.rod_adr + 7] = rod
        mujoco.mj_forward(m, d)
        if not self._placement_ok(c.grasped):
            return None

        d.ctrl[:] = 0
        d.ctrl[6] = c.gripper_ctrl
        for i in range(CLOSE_STEPS + SETTLE_STEPS):
            mujoco.mj_step(m, d)
            d.qpos[:6], d.qvel[:6] = arm, 0
            if c.grasped and i < CLOSE_STEPS:
                d.qpos[self.rod_adr:self.rod_adr + 7] = rod
                d.qvel[self.rod_vadr:self.rod_vadr + 6] = 0
        mujoco.mj_forward(m, d)
        if not np.isfinite(d.qpos).all():
            return None

        rod_pos = d.qpos[self.rod_adr:self.rod_adr + 3]
        axis = Rotation.from_quat(np.roll(d.qpos[self.rod_adr + 3:self.rod_adr + 7], -1)).apply([0, 0, 1])
        moved = np.linalg.norm(rod_pos - c.rod_pos)
        speed = np.linalg.norm(d.qvel[self.rod_vadr:self.rod_vadr + 6])
        if speed > 0.05:
            return None
        if c.grasped and (not self._held() or moved > 0.005):
            return None
        if family == "near_goal" and (axis[2] < 0.95 or np.linalg.norm(rod_pos[:2] - scene.HOLE_POS[:2]) > 0.005):
            return None
        if family == "reaching" and (axis[2] < 0.99 or moved > 0.01):
            return None
        return d.qpos.copy(), c.gripper_ctrl

    def generate(self, family, count, max_tries=None):
        qpos, ctrl, tries = [], [], 0
        while len(qpos) < count:
            tries += 1
            if max_tries and tries > max_tries:
                break
            out = self.try_one(family)
            if out is not None:
                qpos.append(out[0])
                ctrl.append(out[1])
        return np.array(qpos), np.array(ctrl), tries


def generate_chunk(args):
    family, count, seed = args
    qpos, ctrl, tries = Generator(seed).generate(family, count)
    return family, qpos, ctrl, tries


__all__ = ["FAMILIES", "Generator", "generate_chunk"]
