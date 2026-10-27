"""Render a few configurations per family to a PNG grid for a visual check."""
import argparse

import imageio.v3 as iio
import mujoco
import numpy as np

from sgs_rod import scene
from sgs_rod.bank import FAMILIES, Bank

parser = argparse.ArgumentParser()
parser.add_argument("bank")
parser.add_argument("out")
parser.add_argument("--per-family", type=int, default=4)
args = parser.parse_args()

bank = Bank.load(args.bank)
m = scene.build_model()
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, 320, 320)
cam = mujoco.MjvCamera()
cam.lookat[:] = scene.HOLE_POS + [0.05, -0.05, 0.08]
cam.distance, cam.azimuth, cam.elevation = 0.55, 135, -25
rows = []
for f, name in enumerate(FAMILIES):
    ids = np.flatnonzero(bank.family.numpy() == f)[:args.per_family]
    row = []
    for i in ids:
        d.qpos[:] = bank.qpos[i].numpy()
        d.ctrl[6] = bank.gripper_ctrl[i].item()
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, cam)
        row.append(renderer.render())
    rows.append(np.concatenate(row, 1))
iio.imwrite(args.out, np.concatenate(rows, 0))
