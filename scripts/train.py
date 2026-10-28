"""Train the rod-in-hole teacher with PPO, restarting episodes from SGS (or uniform) bank samples."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from sgs_rod.bank import FAMILIES, Bank
from sgs_rod.env import EnvCfg, RodEnv
from sgs_rod.sampler import SGSCfg, SGSSampler, UniformSampler


def ppo_cfg(args):
    """OmniReset's runner settings (SGS lists none), with a state-independent Gaussian policy."""
    return {
        "num_steps_per_env": 32,
        "save_interval": args.save_interval,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128, 64],
            "activation": "elu",
            "obs_normalization": True,
            # OmniReset's std bounds and log parameterization; state-independent std (plain Gaussian).
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log",
                                 "std_range": [0.001, 2.0]},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128, 64],
            "activation": "elu",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "normalize_advantage_per_mini_batch": False,
            "clip_param": 0.2,
            "entropy_coef": 0.006,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "logger": ({"class_name": "WandbLogWriter", "project_name": args.wandb_project}
                   if args.wandb_project else "tensorboard"),
    }


@torch.inference_mode()
def evaluate(policy, env: RodEnv, bank: Bank):
    """Deterministic policy, one episode from every evaluation configuration (in batches of num_envs)."""
    outcomes = []
    for start in range(0, len(bank), env.num_envs):
        ids = torch.arange(start, min(start + env.num_envs, len(bank)), device=env.device)
        worlds = torch.arange(len(ids), device=env.device)
        env.reset_worlds(worlds, ids)
        env._run(env._kinematics)
        result = torch.zeros(len(ids), dtype=torch.bool, device=env.device)
        open_ = torch.ones_like(result)
        obs = env.get_observations()
        for _ in range(env.max_episode_length):
            actions = policy(obs)
            env.prev_actions, env.actions = env.actions, actions.clone()
            env._set_targets(actions)
            env._run(env._physics)
            env.episode_length_buf += 1
            success, failure, _ = env.task_status()
            result |= open_ & success[:len(ids)]
            open_ &= ~(success | failure)[:len(ids)]
            if not open_.any():
                break
            obs = env.get_observations()
        outcomes.append(result)
    return torch.cat(outcomes).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default="banks/train.npz")
    parser.add_argument("--eval-bank", default="banks/eval.npz")
    parser.add_argument("--sampler", choices=("sgs", "uniform"), default="sgs")
    parser.add_argument("--num-envs", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--eval-envs", type=int, default=1_024)
    parser.add_argument("--eval-size", type=int, default=None, help="evaluate on the first N held-out configurations")
    parser.add_argument("--save-interval", type=int, default=100, help="checkpoint and evaluate every N iterations")
    parser.add_argument("--max-hours", type=float, default=None, help="stop after this much wall time")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    parser.add_argument("--wandb-project", default=None, help="log to W&B (set WANDB_MODE=offline to sync later)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    run_dir = args.run_dir or Path("runs") / f"{args.sampler}-{args.num_envs}-s{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(run_dir.resolve()))

    bank = Bank.load(args.bank)
    eval_bank = Bank.load(args.eval_bank)
    if args.eval_size:
        eval_bank = Bank(eval_bank.qpos[:args.eval_size], eval_bank.gripper_ctrl[:args.eval_size],
                         eval_bank.family[:args.eval_size])
    sampler_cls = SGSSampler if args.sampler == "sgs" else UniformSampler
    sampler = sampler_cls(len(bank), SGSCfg(), device=args.device)
    env = RodEnv(bank, sampler, EnvCfg(num_envs=args.num_envs, seed=args.seed), device=args.device)
    eval_env = RodEnv(eval_bank, UniformSampler(len(eval_bank), device=args.device),
                      EnvCfg(num_envs=min(args.eval_envs, len(eval_bank)), seed=args.seed + 1), device=args.device)

    cfg = ppo_cfg(args)
    (run_dir / "config.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()}, "ppo": cfg, "sgs": asdict(sampler.cfg),
         "env": asdict(env.cfg), "bank_size": len(bank)}, indent=2))
    deadline = None if args.max_hours is None else time.time() + 3600 * args.max_hours
    runner = Runner(env, cfg, log_dir=str(run_dir), device=args.device)
    runner.setup_eval(eval_env, eval_bank, sampler, bank, deadline)
    if args.resume:
        runner.load(str(args.resume))
        runner.current_learning_iteration += 1  # the checkpoint's iteration is already done
        sgs_state = args.resume.with_suffix(".sgs.pt")
        if sgs_state.exists():
            sampler.load_state_dict(torch.load(sgs_state, map_location=args.device))
    try:
        runner.learn(args.iterations)
    except TimeLimit:
        print("time limit reached", flush=True)
        runner.logger.stop_logging_writer()


class TimeLimit(Exception):
    pass


class Runner(OnPolicyRunner):
    """Evaluates on held-out Reaching configurations and saves the sampler state at every checkpoint."""

    def setup_eval(self, eval_env, eval_bank, sampler, bank, deadline):
        self.eval_env, self.eval_bank, self.sampler, self.bank, self.deadline = \
            eval_env, eval_bank, sampler, bank, deadline
        self.start_time = time.time()

    def save(self, path, infos=None):
        it = self.current_learning_iteration
        policy = self.alg.get_policy()
        stats = {"Eval/success_reaching": evaluate(lambda obs: policy(obs), self.eval_env, self.eval_bank)}
        p, tried = self.sampler.success_rate(), self.sampler.count > 0
        stats["SGS/tried_fraction"] = tried.float().mean().item()
        family = self.bank.family.to(p.device)
        for f, name in enumerate(FAMILIES):
            mask = (family == f) & tried
            if mask.any():
                stats[f"SGS/mean_p_{name}"] = p[mask].mean().item()
                stats[f"SGS/solved_{name}"] = (p[mask] >= 0.8).float().mean().item()
        for k, v in stats.items():
            self.logger.writer.add_scalar(k, v, it)
        print(f"[{(time.time() - self.start_time) / 3600:.2f} h] it {it}: "
              + ", ".join(f"{k} {v:.3f}" for k, v in stats.items()), flush=True)
        super().save(path, infos)
        torch.save(self.sampler.state_dict(), Path(path).with_suffix(".sgs.pt"))
        if self.deadline and time.time() > self.deadline:
            raise TimeLimit


if __name__ == "__main__":
    main()
