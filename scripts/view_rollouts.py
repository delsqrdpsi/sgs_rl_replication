"""Replay recorded rollouts (scripts/rollout.py) in the browser with viser: python scripts/view_rollouts.py rollouts.npz"""

import argparse
import time

import mujoco
import numpy as np
import trimesh
import viser

from sgs_rod import scene
from sgs_rod.bank import FAMILIES

OUTCOMES = ["timeout", "success", "failure"]


def geom_mesh(m: mujoco.MjModel, g: int):
    size, kind = m.geom_size[g], m.geom_type[g]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = m.geom_dataid[g]
        v0, nv = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        f0, nf = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
        return m.mesh_vert[v0:v0 + nv], m.mesh_face[f0:f0 + nf]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        shape = trimesh.creation.box(extents=2 * size)
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        shape = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=32)
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        shape = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[16, 16])
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        shape = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        shape = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        shape.apply_scale(size)
    elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
        shape = trimesh.creation.box(extents=(1.6, 1.6, 0.002))
        shape.apply_translation((0, 0, -0.001))
    else:
        return None
    return shape.vertices, shape.faces


def geom_color(m: mujoco.MjModel, g: int):
    rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
    return tuple(int(255 * c) for c in rgba[:3]), float(rgba[3])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    data = np.load(args.rollouts)
    qpos, ids, family, outcome, length = (data[k] for k in ("qpos", "state_ids", "family", "outcome", "length"))

    m = scene.build_model()
    d = mujoco.MjData(m)
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    handles = {}
    for g in range(m.ngeom):
        if m.geom_group[g] >= 3:  # collision-only copies of the visual meshes
            continue
        mesh = geom_mesh(m, g)
        if mesh is None:
            continue
        color, alpha = geom_color(m, g)
        if alpha == 0:
            continue
        handles[g] = server.scene.add_mesh_simple(f"/geom_{g}", np.asarray(mesh[0], np.float32),
                                                  np.asarray(mesh[1], np.uint32), color=color,
                                                  opacity=alpha if alpha < 1 else None)

    labels = [f"{i}: {FAMILIES[family[i]]} #{ids[i]} → {OUTCOMES[outcome[i]]} ({length[i]} steps)"
              for i in range(len(ids))]
    episode = server.gui.add_dropdown("Episode", labels)
    frame = server.gui.add_slider("Step (0.1 s)", min=0, max=int(length.max()), step=1, initial_value=0)
    playing = server.gui.add_checkbox("Play", initial_value=True)
    speed = server.gui.add_slider("Speed (×)", min=0.25, max=4.0, step=0.25, initial_value=1.0)

    def show(i, t):
        d.qpos[:] = qpos[i, min(t, length[i])]
        mujoco.mj_kinematics(m, d)
        quat = np.zeros(4)
        with server.atomic():
            for g, h in handles.items():
                mujoco.mju_mat2Quat(quat, d.geom_xmat[g])
                h.position, h.wxyz = d.geom_xpos[g].copy(), quat.copy()

    @episode.on_update
    def _(_):
        frame.value = 0

    @server.on_client_connect
    def _(client):
        client.camera.position = (0.45, -0.15, 0.55)
        client.camera.look_at = (-0.05, 0.5, 0.05)

    shown = None
    while True:
        i = labels.index(episode.value)
        if playing.value:
            frame.value = 0 if frame.value >= length[i] else frame.value + 1
        if shown != (i, frame.value):
            show(i, frame.value)
            shown = (i, frame.value)
        time.sleep((1.0 if frame.value == 0 and playing.value else 0.1) / speed.value)


if __name__ == "__main__":
    main()
