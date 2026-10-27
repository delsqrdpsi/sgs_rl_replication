"""Generate the training bank (32,768 configurations, three equal families) and a Reaching evaluation set."""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from sgs_rod.bank import FAMILIES, Bank
from sgs_rod.generate import generate_chunk


def build(counts, seed, workers, chunk=256):
    tasks, s = [], seed * 1_000_003
    for family, total in counts.items():
        for start in range(0, total, chunk):
            tasks.append((family, min(chunk, total - start), s))
            s += 1
    qpos, ctrl, family_ids, tries = [], [], [], {f: 0 for f in counts}
    with mp.Pool(workers) as pool:
        for family, q, c, n in pool.imap_unordered(generate_chunk, tasks):
            qpos.append(q)
            ctrl.append(c)
            family_ids.append(np.full(len(q), FAMILIES.index(family)))
            tries[family] += n
    order = np.argsort(np.concatenate(family_ids), kind="stable")
    bank = Bank(torch.tensor(np.concatenate(qpos)[order], dtype=torch.float32),
                torch.tensor(np.concatenate(ctrl)[order], dtype=torch.float32),
                torch.tensor(np.concatenate(family_ids)[order]))
    acceptance = {f: counts[f] / tries[f] for f in counts}
    return bank, acceptance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("banks"))
    parser.add_argument("--size", type=int, default=32_768)
    parser.add_argument("--eval-size", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per = [args.size // 3 + (i < args.size % 3) for i in range(3)]
    for name, counts, seed in (("train", dict(zip(FAMILIES, per)), args.seed),
                               ("eval", {"reaching": args.eval_size}, args.seed + 1)):
        start = time.time()
        bank, acceptance = build(counts, seed, args.workers)
        path = args.out / f"{name}.npz"
        bank.save(path, seed=seed)
        print(f"{path}: {len(bank)} configurations in {time.time() - start:.0f} s, "
              f"acceptance {', '.join(f'{k} {v:.0%}' for k, v in acceptance.items())}", flush=True)


if __name__ == "__main__":
    main()
