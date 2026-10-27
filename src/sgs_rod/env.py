"""Batched rod-in-hole environment on MuJoCo Warp, with the rsl_rl VecEnv interface."""

from dataclasses import dataclass

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from . import control, scene
from .bank import FAMILIES, Bank

POLICY_DT = 0.1  # SGS: 10 Hz policy
DECIMATION = round(POLICY_DT / scene.TIMESTEP)  # 20 physics steps per action
EPISODE_STEPS = 160  # OmniReset: 16 s episodes

SUCCESS_POSITION = 0.0025  # OmniReset PegHole metadata
SUCCESS_ORIENTATION = 0.025  # rad, |roll| + |pitch| of rod relative to hole
ARM_SPEED_LIMIT = np.pi  # UR5e: 180 deg/s on every joint
WORKSPACE_LOW = np.array([-0.6, 0.15, -0.02])  # assumption: table region in front of the robot
WORKSPACE_HIGH = np.array([0.35, 0.95, 0.5])
POINT_NOISE = 0.001  # SGS rod-in-hole: 1 mm per-step Gaussian noise on the point cloud
POINTS_PER_BODY = 32


@dataclass
class EnvCfg:
    num_envs: int = 10_000
    nconmax: int = 64  # contacts allocated per world (random-action rollouts peak near 28)
    njmax: int = 256  # constraint rows allocated per world (peak near 91)
    seed: int = 0


def _quat_from_matrix(mat):
    """(..., 3, 3) -> (..., 4) wxyz with w >= 0."""
    m = mat
    w = 0.5 * (1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]).clamp_min(1e-12).sqrt()
    x = 0.5 * (1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]).clamp_min(0).sqrt()
    y = 0.5 * (1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2]).clamp_min(0).sqrt()
    z = 0.5 * (1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]).clamp_min(0).sqrt()
    x = x.copysign(m[..., 2, 1] - m[..., 1, 2])
    y = y.copysign(m[..., 0, 2] - m[..., 2, 0])
    z = z.copysign(m[..., 1, 0] - m[..., 0, 1])
    return torch.stack([w, x, y, z], -1)


def rod_tilt_error(rel_mat):
    """|roll| + |pitch| of the rod in the hole frame (XYZ Euler angles, as OmniReset computes them)."""
    q = _quat_from_matrix(rel_mat)
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
    return roll.abs() + pitch.abs()


def local_points(m: mujoco.MjModel):
    """Fixed surface points in each body's frame: the two finger pads, the rod, and the hole block."""
    def box_corners(geom):
        g = m.geom(geom)
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * g.size
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, g.quat)
        return corners @ rot.reshape(3, 3).T + g.pos

    def ring(radius, z, count, phase=0.0):
        a = phase + 2 * np.pi * np.arange(count) / count
        return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(count, z)], -1)

    half = scene.ROD_HALF_LENGTH
    r = scene.ROD_RADIUS
    rod = np.concatenate([ring(r, half, 8), ring(r, -half, 8), ring(r, half / 3, 8, np.pi / 8),
                          ring(r, -half / 3, 8, np.pi / 8)])
    top = scene.HOLE_DEPTH
    board = np.concatenate([ring(scene.BORE_RADIUS, top, 16), ring(scene.BORE_RADIUS, top / 2, 8, np.pi / 8),
                            ring(scene.BORE_RADIUS + 0.02, top, 8)])
    bodies = {
        "hand/left_pad": np.concatenate([box_corners("hand/left_pad1"), box_corners("hand/left_pad2")]),
        "hand/right_pad": np.concatenate([box_corners("hand/right_pad1"), box_corners("hand/right_pad2")]),
        "rod": rod,
        "hole": board,
    }
    for name, pts in bodies.items():
        assert len(pts) == (16 if "pad" in name else POINTS_PER_BODY), name
    return bodies


class RodEnv(VecEnv):
    """Each world restarts from a bank configuration chosen by `sampler` (SGS or uniform)."""

    def __init__(self, bank: Bank, sampler, cfg: EnvCfg, device: str = "cuda:0"):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.num_actions = 7
        self.max_episode_length = EPISODE_STEPS
        self.device = torch.device(device)
        self.bank = bank.to(self.device)
        self.sampler = sampler
        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed)

        mjm = scene.build_model()
        mjd = mujoco.MjData(mjm)
        mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
        mujoco.mj_forward(mjm, mjd)
        self.mjm = mjm
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.wp_device = wp.get_device(str(self.device))
        with wp.ScopedDevice(self.wp_device):
            self.m = mjw.put_model(mjm)
            self.d = mjw.put_data(mjm, mjd, nworld=self.num_envs, nconmax=cfg.nconmax, njmax=cfg.njmax)
            self.target_pos = wp.zeros(self.num_envs, dtype=wp.vec3)
            self.target_mat = wp.zeros(self.num_envs, dtype=wp.mat33)

        t = wp.to_torch
        self.qpos, self.qvel, self.ctrl = t(self.d.qpos), t(self.d.qvel), t(self.d.ctrl)
        self.qacc_warmstart = t(self.d.qacc_warmstart)
        self.xpos, self.xmat = t(self.d.xpos), t(self.d.xmat)
        self.site_xpos, self.site_xmat = t(self.d.site_xpos), t(self.d.site_xmat)
        self._target_pos, self._target_mat = t(self.target_pos), t(self.target_mat)

        self.ee_body = mjm.body("wrist_3_link").id
        self.base_rot = torch.tensor(mjd.body("base").xmat.reshape(3, 3), dtype=torch.float32, device=self.device)
        self.pinch = mjm.site("hand/pinch").id
        self.rod_tip, self.hole_seat = mjm.site("rod_tip").id, mjm.site("hole_seat").id
        self.rod_body = mjm.body("rod").id
        self.rod_qpos = mjm.joint("rod").qposadr[0]
        self.finger_qpos = mjm.joint("hand/right_driver_joint").qposadr[0]
        self.finger_range = mjm.jnt_range[mjm.joint("hand/right_driver_joint").id, 1]

        points = local_points(mjm)
        self.point_bodies = torch.tensor([mjm.body(n).id for n in points], device=self.device)
        self.point_counts = [len(p) for p in points.values()]
        self.points_local = torch.tensor(np.concatenate(list(points.values())), dtype=torch.float32,
                                         device=self.device)
        self.point_owner = torch.repeat_interleave(self.point_bodies,
                                                   torch.tensor(self.point_counts, device=self.device))

        self.action_scale = torch.tensor(control.ACTION_SCALE, dtype=torch.float32, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions = torch.zeros(self.num_envs, 7, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.state_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._graph = None
        everyone = torch.arange(self.num_envs, device=self.device)
        self.reset_worlds(everyone)
        self._run(self._kinematics)
        # Compile (and on CUDA, capture) the physics loop once, then restart every world from the bank.
        self._set_targets(self.actions)
        self._run(self._physics)
        self.reset_worlds(everyone, self.state_ids.clone())
        self._run(self._kinematics)

    # ---------------------------------------------------------------- physics

    def _physics(self):
        for _ in range(DECIMATION):
            mjw.step1(self.m, self.d)
            wp.launch(control.osc_torque, dim=self.num_envs,
                      inputs=[self.d.qvel, self.d.xpos, self.d.xmat, self.d.xanchor, self.d.xaxis,
                              self.target_pos, self.target_mat, self.ee_body,
                              control.vec6(*control.KP), control.vec6(*control.KD),
                              control.vec6(*control.TORQUE_LIMIT), self.d.ctrl])
            mjw.step2(self.m, self.d)
        mjw.kinematics(self.m, self.d)

    def _kinematics(self):
        mjw.kinematics(self.m, self.d)

    def _run(self, fn):
        """Run warp work on torch's stream; on CUDA the physics loop is captured once as a graph."""
        if self.device.type != "cuda":
            with wp.ScopedDevice(self.wp_device):
                fn()
            return
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        with wp.ScopedDevice(self.wp_device), wp.ScopedStream(stream):
            if fn is self._physics:
                if self._graph is None:
                    fn()  # load kernels outside the capture
                    with wp.ScopedCapture() as capture:
                        fn()
                    self._graph = capture.graph
                wp.capture_launch(self._graph)
            else:
                fn()

    # ---------------------------------------------------------------- resets

    def reset_worlds(self, ids, state_ids=None):
        if state_ids is None:
            state_ids = self.sampler.sample(len(ids), self.generator)
        self.state_ids[ids] = state_ids
        self.qpos[ids] = self.bank.qpos[state_ids]
        self.qvel[ids] = 0
        self.qacc_warmstart[ids] = 0
        self.ctrl[ids] = 0
        self.ctrl[ids, 6] = self.bank.gripper_ctrl[state_ids]
        self.episode_length_buf[ids] = 0
        self.actions[ids] = 0
        self.prev_actions[ids] = 0
        return state_ids

    # ---------------------------------------------------------------- observations

    def _point_cloud(self):
        rot = self.xmat[:, self.point_owner]  # (N, P, 3, 3)
        world = self.xpos[:, self.point_owner] + (rot @ self.points_local[None, :, :, None]).squeeze(-1)
        world = world + POINT_NOISE * torch.randn(world.shape, device=self.device, generator=self.generator)
        p, r = self.site_xpos[:, self.pinch], self.site_xmat[:, self.pinch]
        return ((world - p[:, None]) @ r).reshape(self.num_envs, -1)  # gripper frame: r^T (x - p)

    def get_observations(self):
        arm = self.qpos[:, :6]
        fingers = self.qpos[:, self.finger_qpos:self.finger_qpos + 1] / self.finger_range
        ee_pos = self.site_xpos[:, self.pinch] @ self.base_rot  # base frame (base sits at the origin)
        ee_quat = _quat_from_matrix(self.base_rot.T @ self.site_xmat[:, self.pinch])
        obs = torch.cat([arm, fingers, ee_pos, ee_quat, self.prev_actions, self._point_cloud()], -1)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    # ---------------------------------------------------------------- step

    def _set_targets(self, actions):
        scaled = actions[:, :6] * self.action_scale
        ee_pos, ee_mat = self.xpos[:, self.ee_body], self.xmat[:, self.ee_body]
        # Increments are expressed in the robot base frame.
        self._target_pos[:] = ee_pos + scaled[:, :3] @ self.base_rot.T
        delta = self.base_rot @ control.axis_angle_to_matrix(scaled[:, 3:]) @ self.base_rot.T
        self._target_mat[:] = delta @ ee_mat
        self.ctrl[:, 6] = torch.where(actions[:, 6] > 0, control.GRIPPER_CLOSED, 0.0)

    def task_status(self):
        tip, seat = self.site_xpos[:, self.rod_tip], self.site_xpos[:, self.hole_seat]
        hole_rot = self.site_xmat[:, self.hole_seat]
        rel = hole_rot.transpose(-1, -2) @ self.site_xmat[:, self.rod_tip]
        success = ((tip - seat).norm(dim=-1) < SUCCESS_POSITION) & (rod_tilt_error(rel) < SUCCESS_ORIENTATION)
        rod = self.qpos[:, self.rod_qpos:self.rod_qpos + 3]
        low = torch.as_tensor(WORKSPACE_LOW, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(WORKSPACE_HIGH, dtype=torch.float32, device=self.device)
        outside = ((rod < low) | (rod > high)).any(-1)
        too_fast = (self.qvel[:, :6].abs() > 2 * ARM_SPEED_LIMIT).any(-1)
        broken = ~torch.isfinite(self.qpos).all(-1) | ~torch.isfinite(self.qvel).all(-1)
        return success & ~broken, outside | too_fast | broken, broken

    def step(self, actions):
        actions = actions.to(self.device)
        self.prev_actions, self.actions = self.actions, actions.clone()
        self._set_targets(actions)
        self._run(self._physics)
        self.episode_length_buf += 1

        success, failure, broken = self.task_status()
        clamp = lambda x: x.clamp_max(1e4)
        reward = (100.0 * success - failure.float()
                  - 1e-5 * clamp(actions.square().sum(-1))
                  - 1e-4 * clamp((actions - self.prev_actions).square().sum(-1))
                  - 1e-4 * clamp(self.qvel[:, :6].square().sum(-1).nan_to_num(1e4)))
        timeout = self.episode_length_buf >= self.max_episode_length
        done = success | failure | timeout
        extras = {"time_outs": timeout & ~success & ~failure, "log": {}}

        ids = done.nonzero().squeeze(-1)
        if len(ids):
            finished = self.state_ids[ids]
            self.sampler.record(finished, success[ids])
            log = extras["log"]
            log["Episode/success"] = success[ids].float()
            log["Episode/failure"] = failure[ids].float()
            log["Episode/broken"] = broken[ids].float()
            family = self.bank.family[finished]
            for f, name in enumerate(FAMILIES):
                mask = family == f
                if mask.any():
                    log[f"Episode/success_{name}"] = success[ids][mask].float()
            new = self.reset_worlds(ids)
            new_family = self.bank.family[new]
            for f, name in enumerate(FAMILIES):
                log[f"Sampler/share_{name}"] = (new_family == f).float()
            self._run(self._kinematics)
        return self.get_observations(), reward, done.long(), extras
