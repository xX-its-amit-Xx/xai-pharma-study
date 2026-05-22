"""Paper 1 R2: learned-representation (GNN) extension — addresses reviewer critique #2
(no GNNs) and strengthens #3 (true weight-reinitialization sanity on a non-MLP model).

A GIN on molecular graphs (PyG from_smiles) for the safety-critical classification endpoints.
Node-level attributions via occlusion (mask each atom's features, measure predicted-prob drop)
— robust and faithfulness-aligned. Reliability battery (within the GNN representation):
  faithfulness  : comprehensiveness of occlusion attributions vs a random-node null
  sanity        : Adebayo TRUE weight-reinitialization (similarity trained vs random GIN)
Stability is omitted (discrete atom features make L-inf perturbation ill-defined; noted).
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import from_smiles

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import data as data_mod  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
ENDPOINTS = ["DILI", "hERG", "AMES", "BBB"]
FRACS = (0.1, 0.2, 0.3)
torch.manual_seed(0)


class GIN(nn.Module):
    def __init__(self, in_dim=9, hidden=64, out_dim=2, n_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        prev = in_dim
        for _ in range(n_layers):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(prev, hidden), nn.ReLU(),
                                                    nn.Linear(hidden, hidden))))
            prev = hidden
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        h = x.float()
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        return self.head(global_add_pool(h, batch))


def to_graphs(df):
    gs = []
    for smi, y in zip(df["Drug"], df["Y"]):
        g = from_smiles(smi)
        if g.num_nodes and g.edge_index.numel():
            g.x = g.x.float(); g.y = torch.tensor([int(y)]); gs.append(g)
    return gs


def train_gin(graphs, epochs=60):
    model = GIN(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(); loader = DataLoader(graphs, batch_size=64, shuffle=True)
    model.train()
    for _ in range(epochs):
        for b in loader:
            opt.zero_grad()
            loss = loss_fn(model(b.x, b.edge_index, b.batch), b.y)
            loss.backward(); opt.step()
    return model.eval()


@torch.no_grad()
def graph_proba(model, g):
    batch = torch.zeros(g.num_nodes, dtype=torch.long)
    return torch.softmax(model(g.x, g.edge_index, batch), dim=-1)[0].numpy()


@torch.no_grad()
def occlusion_attr(model, g):
    base = graph_proba(model, g); pred = int(base.argmax()); base_q = base[pred]
    attr = np.zeros(g.num_nodes)
    for i in range(g.num_nodes):
        x2 = g.x.clone(); x2[i] = 0.0
        gg = g.clone(); gg.x = x2
        attr[i] = base_q - graph_proba(model, gg)[pred]
    return attr, pred, base_q


@torch.no_grad()
def comprehensiveness(model, g, attr, pred, base_q):
    n = g.num_nodes; order = np.argsort(-np.abs(attr)); drops = []
    for frac in FRACS:
        k = max(1, int(round(frac * n)))
        x2 = g.x.clone(); x2[order[:k]] = 0.0
        gg = g.clone(); gg.x = x2
        drops.append(base_q - graph_proba(model, gg)[pred])
    return float(np.mean(drops))


def main():
    rng = np.random.default_rng(0)
    rows = []
    eps = {e.name: e for e in data_mod.load_selected(seed=0)}
    for name in ENDPOINTS:
        ep = eps[name]
        gtr = to_graphs(ep.splits["scaffold"]["train"])
        gte = to_graphs(ep.splits["scaffold"]["test"])
        model = train_gin(gtr)
        proba = np.array([graph_proba(model, g) for g in gte])
        y = np.array([int(g.y) for g in gte])
        auc = roc_auc_score(y, proba[:, 1]) if len(np.unique(y)) == 2 else np.nan

        rand_model = GIN()  # true weight reinitialization (Adebayo)
        faiths, nulls, sanities = [], [], []
        sub = gte[: min(80, len(gte))]
        for g in sub:
            attr, pred, base_q = occlusion_attr(model, g)
            faiths.append(comprehensiveness(model, g, attr, pred, base_q))
            rnd = rng.standard_normal(g.num_nodes)
            nulls.append(comprehensiveness(model, g, rnd, pred, base_q))
            rattr, _, _ = occlusion_attr(rand_model, g)
            if g.num_nodes > 2 and not np.allclose(attr, attr[0]) and not np.allclose(rattr, rattr[0]):
                s = spearmanr(np.abs(attr), np.abs(rattr)).statistic
                sanities.append(abs(s) if not np.isnan(s) else 0.0)
        faith = float(np.mean(faiths)); null = float(np.mean(nulls))
        sanity = float(np.mean(sanities))
        rows.append({"endpoint": name, "model": "GIN", "test_auroc": auc, "n_explained": len(sub),
                     "faith": faith, "null_faith": null, "beats_null": faith > null,
                     "sanity_sim": sanity, "sanity_passed": sanity < 0.5})
        print(f"{name:6s} GIN AUROC={auc:.3f} faith={faith:+.3f}(null{null:+.3f}) "
              f"beats_null={faith>null} sanity={sanity:.3f} {'PASS' if sanity<0.5 else 'FAIL'}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "gnn_extension.csv"), index=False)
    print(f"\nGNN extension: {df.beats_null.sum()}/{len(df)} beat null on faithfulness; "
          f"{(~df.sanity_passed).sum()}/{len(df)} fail the (true weight-reinit) sanity check")


if __name__ == "__main__":
    main()
