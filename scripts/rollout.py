"""Record a checkpoint's deterministic policy from chosen bank configurations, for replay in view_rollouts.py.

python scripts/rollout.py runs/<name>/model_N.pt out.npz --per-family 4
"""

import argparse
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from sgs_rod.bank import FAMILIES, Bank
from sgs_rod.env import EnvCfg, RodEnv
from sgs_rod.sampler import UniformSampler
from train import ppo_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    parser.add_argument("--bank", default="banks/train.npz")
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    bank = Bank.load(args.bank)
    rng = np.random.default_rng(args.seed)
    family = bank.family.numpy()
    ids = np.concatenate([rng.choice(np.flatnonzero(family == f), args.per_family, replace=False)
                          for f in range(len(FAMILIES))])
    sampler = UniformSampler(len(bank), device=args.device)
    env = RodEnv(bank, sampler, EnvCfg(num_envs=len(ids), seed=args.seed), device=args.device)
    cfg = ppo_cfg(SimpleNamespace(save_interval=100, wandb_project=None))
    runner = OnPolicyRunner(env, cfg, log_dir=tempfile.mkdtemp(), device=args.device)
    runner.load(args.checkpoint, map_location=args.device)
    policy = runner.get_inference_policy(args.device)

    with torch.inference_mode():
        worlds = torch.arange(len(ids), device=env.device)
        env.reset_worlds(worlds, torch.as_tensor(ids, device=env.device))
        env._run(env._kinematics)
        frames = [env.qpos.cpu().numpy().copy()]
        open_ = np.ones(len(ids), bool)
        outcome = np.zeros(len(ids), int)  # 0 timeout, 1 success, 2 failure
        length = np.full(len(ids), env.max_episode_length)
        obs = env.get_observations()
        for step in range(env.max_episode_length):
            actions = policy(obs)
            env.prev_actions, env.actions = env.actions, actions.clone()
            env._set_targets(actions)
            env._run(env._physics)
            env.episode_length_buf += 1
            success, failure, _ = (x.cpu().numpy() for x in env.task_status())
            frames.append(env.qpos.cpu().numpy().copy())
            for i in np.flatnonzero(open_ & (success | failure)):
                outcome[i], length[i] = (1 if success[i] else 2), step + 1
            open_ &= ~(success | failure)
            if not open_.any():
                break
            obs = env.get_observations()
            print(f"step {step + 1}/{env.max_episode_length}", end="\r", flush=True)

    np.savez(args.out, qpos=np.stack(frames, 1), state_ids=ids, family=family[ids], outcome=outcome,
             length=length, dt=0.1)
    names = ["timeout", "success", "failure"]
    for i, s in enumerate(ids):
        print(f"config {s:5d} {FAMILIES[family[s]]:13s} {names[outcome[i]]:8s} after {length[i]} steps")


if __name__ == "__main__":
    main()
