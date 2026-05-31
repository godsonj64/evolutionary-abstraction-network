from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class Episode:
    x: torch.Tensor
    z: torch.Tensor
    y: torch.Tensor | None


class EpisodicMemory:
    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.buffer: deque[Episode] = deque(maxlen=capacity)

    def add(self, x: torch.Tensor, z: torch.Tensor, y: torch.Tensor | None = None) -> None:
        self.buffer.append(Episode(x.detach().cpu(), z.detach().cpu(), None if y is None else y.detach().cpu()))

    def __len__(self) -> int:
        return len(self.buffer)

    def sample_latents(self, n: int) -> torch.Tensor:
        if len(self.buffer) == 0:
            raise RuntimeError("Cannot sample empty memory")
        n = min(n, len(self.buffer))
        idx = torch.randperm(len(self.buffer))[:n].tolist()
        return torch.stack([self.buffer[i].z for i in idx], dim=0)
