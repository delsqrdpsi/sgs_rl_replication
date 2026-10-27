"""The fixed set of start configurations SGS samples from."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

FAMILIES = ("near_goal", "stable_grasp", "reaching")  # SGS's three reset strategies


@dataclass
class Bank:
    qpos: torch.Tensor  # (N, nq) settled joint and rod positions; velocities start at zero
    gripper_ctrl: torch.Tensor  # (N,) gripper actuator command that holds the stored finger opening
    family: torch.Tensor  # (N,) index into FAMILIES

    def __len__(self):
        return len(self.family)

    def to(self, device):
        return Bank(self.qpos.to(device), self.gripper_ctrl.to(device), self.family.to(device))

    def save(self, path, **metadata):
        np.savez_compressed(path, qpos=self.qpos.cpu().numpy(), gripper_ctrl=self.gripper_ctrl.cpu().numpy(),
                            family=self.family.cpu().numpy(), **metadata)

    @staticmethod
    def load(path):
        data = np.load(Path(path))
        return Bank(torch.tensor(data["qpos"], dtype=torch.float32),
                    torch.tensor(data["gripper_ctrl"], dtype=torch.float32),
                    torch.tensor(data["family"], dtype=torch.long))
