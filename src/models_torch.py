"""Top-level torch MLP and adapter factory (picklable, reloadable).

Kept separate from the harness's notebook-oriented builder so that ``build_fn`` and
``train_fn`` are top-level (``functools.partial`` of module functions), which lets
the resulting ``TorchAdapter`` support ``refit`` (faithfulness retrain mode) and
``randomized_copy`` (true weight-reinitialization sanity check) in P4, and lets us
persist/reload models via ``state_dict`` + a small spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch import nn

from xai_eval.models.base import CLASSIFICATION
from xai_eval.models.torch_adapter import TorchAdapter


@dataclass
class MLPSpec:
    in_dim: int
    out_dim: int
    hidden: tuple[int, ...] = (128, 64)
    task: str = CLASSIFICATION


class TabMLP(nn.Module):
    def __init__(self, spec: MLPSpec):
        super().__init__()
        layers: list[nn.Module] = []
        prev = spec.in_dim
        for h in spec.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, spec.out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_module(spec: MLPSpec) -> TabMLP:
    return TabMLP(spec)


def train_module(module, X, y, *, spec: MLPSpec, epochs: int = 150, lr: float = 1e-3,
                 device: str = "cpu"):
    module = module.to(device).train()
    opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=1e-5)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    if spec.task == CLASSIFICATION:
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device).view(-1, 1)
        loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(module(Xt), yt)
        loss.backward()
        opt.step()
    return module.eval()


def make_mlp_adapter(spec: MLPSpec, X, y, feature_names=None, seed: int = 0,
                     epochs: int = 150, lr: float = 1e-3, device: str = "cpu") -> TorchAdapter:
    torch.manual_seed(seed)
    build_fn = partial(build_module, spec)
    train_fn = partial(train_module, spec=spec, epochs=epochs, lr=lr, device=device)
    module = train_fn(build_fn(), np.asarray(X), np.asarray(y))
    n_classes = spec.out_dim if spec.task == CLASSIFICATION else 1
    adapter = TorchAdapter(module, spec.task, n_classes, build_fn, train_fn, device)
    adapter._train_X = np.asarray(X, dtype=np.float32)
    adapter._train_y = np.asarray(y)
    adapter.feature_names = list(feature_names) if feature_names is not None else None
    return adapter
