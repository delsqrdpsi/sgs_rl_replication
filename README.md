# sgs-rod

SGS start-state sampling ([arXiv 2610.12465](https://arxiv.org/html/2610.12465v1)) on UR5e + Robotiq 2F-85
rod-in-hole, in MuJoCo Warp. Design, sources and every assumption: [DESIGN.md](DESIGN.md).

## Setup

```sh
uv sync
uv run --no-sync pytest -q
```

On macOS with the project under an iCloud-synced folder, the editable-install `.pth` file gets the `hidden`
flag and Python skips it; tests set `pythonpath` themselves, and scripts need `PYTHONPATH=src`.

Robot models: MuJoCo Menagerie commit in `assets/MENAGERIE_COMMIT`, licenses kept beside each model.

## Status

- [x] Scene: UR5e (torque motors, gravity compensated), 2F-85 (pad μ = 2), 16 mm rod, 16.5 mm bore; runs in
      MuJoCo Warp
- [ ] Relative OSC controller and 10 Hz environment step
- [ ] 32,768-configuration reset bank (Near-Goal, Stable Grasp, Reaching) and a Reaching evaluation set
- [ ] SGS sampler, reward, point-cloud observations, PPO
- [ ] RunPod RTX 5090 run, SGS vs uniform
