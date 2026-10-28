"""Time each phase of a policy step at full scale, optionally with different solver settings.

python scripts/profile_env.py --num-envs 10000 [--iterations 10 --ls-iterations 20]
"""

import argparse
import time

import numpy as np
import torch

from sgs_rod import env as env_module
from sgs_rod import scene
from sgs_rod.bank import Bank
from sgs_rod.env import EnvCfg, RodEnv
from sgs_rod.sampler import SGSCfg, SGSSampler


def timed(fn, sync, repeats):
    sync()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync()
    return (time.perf_counter() - start) / repeats * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=None, help="solver iterations (model default if unset)")
    parser.add_argument("--ls-iterations", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--steps", type=int, default=30, help="policy steps per measurement")
    parser.add_argument("--action-std", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    build = scene.build_model

    def build_model():
        m = build()
        if args.iterations is not None:
            m.opt.iterations = args.iterations
        if args.ls_iterations is not None:
            m.opt.ls_iterations = args.ls_iterations
        if args.tolerance is not None:
            m.opt.tolerance = args.tolerance
        return m

    scene.build_model = env_module.scene.build_model = build_model
    bank = Bank.load("banks/train.npz")
    env = RodEnv(bank, SGSSampler(len(bank), SGSCfg(), device=args.device), EnvCfg(num_envs=args.num_envs),
                 device=args.device)
    m = env.mjm
    print(f"{args.num_envs} envs, solver iterations {m.opt.iterations}, ls {m.opt.ls_iterations}, "
          f"tolerance {m.opt.tolerance:g}")
    sync = torch.cuda.synchronize if env.device.type == "cuda" else (lambda: None)
    actions = lambda: args.action_std * torch.randn(args.num_envs, 7, device=env.device)

    for _ in range(5):  # warm-up: kernels loaded, worlds spread over the episode
        env.step(actions())
    a = actions()
    phases = {
        "set targets": lambda: env._set_targets(a),
        "physics (20 substeps, graph)": lambda: env._run(env._physics),
        "task status + reward": lambda: env.task_status(),
        "kinematics (after resets)": lambda: env._run(env._kinematics),
        "observations": lambda: env.get_observations(),
        "full env.step": lambda: env.step(actions()),
    }
    for name, fn in phases.items():
        print(f"  {name:30s} {timed(fn, sync, args.steps):8.2f} ms")

    full = timed(lambda: env.step(actions()), sync, args.steps)
    print(f"  => {args.num_envs * 1e3 / full:,.0f} policy steps/s, "
          f"{args.num_envs * 20 * 1e3 / full:,.0f} physics steps/s, one PPO iteration (32 steps) "
          f"{32 * full / 1e3:.1f} s")

    d = env.d
    stats = {}
    for field in ("nacon", "nefc", "solver_niter"):
        value = getattr(d, field, None)
        if value is not None:
            stats[field] = value.numpy()
    if "nacon" in stats:
        print(f"  contacts: {stats['nacon'].sum() / args.num_envs:.1f} per world on average")
    if "nefc" in stats:
        nefc = stats["nefc"]
        print(f"  constraint rows per world: mean {nefc.mean():.1f}, max {nefc.max()}")
    if "solver_niter" in stats:
        niter = stats["solver_niter"]
        print(f"  solver iterations per world: mean {niter.mean():.1f}, p99 {np.percentile(niter, 99):.0f}, "
              f"max {niter.max()}")


if __name__ == "__main__":
    main()
