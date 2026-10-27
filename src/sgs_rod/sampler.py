"""Start-state samplers: SGS (success-guided sampling) and the uniform baseline."""

from dataclasses import dataclass

import torch


@dataclass
class SGSCfg:
    history: int = 100  # H: latest Boolean outcomes kept per configuration
    target: float = 0.5  # t
    concentration: float = 1.0  # kappa
    epsilon: float = 1e-4
    temperature: float = 2.0  # T; the paper's implementation uses max(T, 1)


class SGSSampler:
    """w_i = (p_i + eps)^(kappa t) (1 - p_i + eps)^(kappa (1 - t)); sample from softmax(log(w_i + eps) / T).

    p_i is the mean of configuration i's latest H outcomes. Untried configurations count as p = 0.
    """

    def __init__(self, size: int, cfg: SGSCfg = SGSCfg(), device="cpu"):
        self.cfg = cfg
        self.outcomes = torch.zeros(size, cfg.history, dtype=torch.bool, device=device)
        self.count = torch.zeros(size, dtype=torch.long, device=device)  # outcomes stored, at most H
        self.cursor = torch.zeros(size, dtype=torch.long, device=device)  # next slot to write

    def success_rate(self):
        return self.outcomes.sum(-1) / self.count.clamp_min(1)

    def probabilities(self):
        c, p = self.cfg, self.success_rate()
        weight = (p + c.epsilon).pow(c.concentration * c.target) * \
                 (1 - p + c.epsilon).pow(c.concentration * (1 - c.target))
        return torch.softmax((weight + c.epsilon).log() / max(c.temperature, 1.0), 0)

    def sample(self, n, generator=None):
        return torch.multinomial(self.probabilities(), n, replacement=True, generator=generator)

    def record(self, ids, success):
        """Append outcomes; several episodes from the same configuration may finish together."""
        H = self.cfg.history
        order = ids.argsort(stable=True)
        ids, success = ids[order], success[order]
        unique, counts = ids.unique_consecutive(return_counts=True)
        first = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        rank = torch.arange(len(ids), device=ids.device) - first
        total = torch.repeat_interleave(counts, counts)
        keep = rank >= total - H  # only the latest H of a large batch survive
        slot = (self.cursor[ids] + rank) % H
        self.outcomes[ids[keep], slot[keep]] = success[keep]
        self.cursor[unique] = (self.cursor[unique] + counts) % H
        self.count[unique] = (self.count[unique] + counts).clamp_max(H)

    def state_dict(self):
        return {"outcomes": self.outcomes, "count": self.count, "cursor": self.cursor}

    def load_state_dict(self, state):
        for k, v in state.items():
            getattr(self, k).copy_(v)


class UniformSampler(SGSSampler):
    """Baseline: every configuration equally likely. Outcomes are still recorded, for logging."""

    def probabilities(self):
        return torch.full((len(self.count),), 1 / len(self.count), device=self.count.device)
