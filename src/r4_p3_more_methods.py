"""R4-1: extend Paper 3's D2 with more attribution methods.

The criticism: "Spearman=1.0 across n=3 methods is meaningless." We add four gradient-based
variants (saliency, gradient*input, SmoothGrad over node features, and IG with more steps),
keeping occlusion and random, and recompute the D1 recovery AUROC + D2 faithfulness-vs-recovery
Spearman over the now n>=6 methods. If gradient methods consistently fail while occlusion
succeeds, the IG failure generalizes; if some gradient variant succeeds, IG is the specific
culprit, not gradient attribution in general.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import from_smiles

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.dirname(__file__))
import data as data_mod  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
N_BITS = 1024; RADIUS = 2; K_BITS = 96; N_MOLS = 700
torch.manual_seed(0)


class MultiBitGIN(nn.Module):
    def __init__(self, in_dim=9, hidden=64, n_out=K_BITS, n_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        prev = in_dim
        for _ in range(n_layers):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(prev, hidden), nn.ReLU(),
                                                    nn.Linear(hidden, hidden))))
            prev = hidden
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x, edge_index, batch):
        h = x.float()
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
        return self.head(global_add_pool(h, batch))


def morgan_with_bitinfo(mol):
    bi = {}; fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS, bitInfo=bi)
    return np.array(fp, dtype=np.float32), bi


def gt_atoms(bi, bit): return {int(a) for a, _ in bi.get(bit, [])}


def saliency(model, x, edge_index, batch, target_bit):
    x_in = x.float().detach().requires_grad_(True)
    out = model(x_in, edge_index, batch)[0, target_bit]
    g = torch.autograd.grad(out, x_in)[0]
    return g.detach().numpy()


def grad_x_input(model, x, edge_index, batch, target_bit):
    return saliency(model, x, edge_index, batch, target_bit) * x.detach().numpy()


def smoothgrad(model, x, edge_index, batch, target_bit, n=10, sigma=0.1):
    grads = []
    for _ in range(n):
        noise = torch.randn_like(x) * sigma
        x_in = (x + noise).detach().requires_grad_(True)
        out = model(x_in, edge_index, batch)[0, target_bit]
        grads.append(torch.autograd.grad(out, x_in)[0].detach().numpy())
    return np.mean(grads, axis=0)


def integrated_gradients(model, x, edge_index, batch, target_bit, n_steps=32):
    baseline = torch.zeros_like(x); grads = []
    for a in torch.linspace(0, 1, n_steps):
        x_a = (baseline + a * (x - baseline)).detach().requires_grad_(True)
        out = model(x_a, edge_index, batch)[0, target_bit]
        grads.append(torch.autograd.grad(out, x_a)[0])
    return (torch.stack(grads).mean(0) * (x - baseline)).detach().numpy()


@torch.no_grad()
def occlusion(model, x, edge_index, batch, target_bit):
    base = torch.sigmoid(model(x, edge_index, batch))[0, target_bit].item()
    out = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        x2 = x.clone(); x2[i] = 0.0
        out[i] = base - torch.sigmoid(model(x2, edge_index, batch))[0, target_bit].item()
    return out


@torch.no_grad()
def comprehensiveness(model, x, edge_index, batch, target_bit, atom_scores, fracs=(0.1, 0.2, 0.3)):
    base = torch.sigmoid(model(x, edge_index, batch))[0, target_bit].item()
    order = np.argsort(-np.abs(atom_scores)); drops = []
    for f in fracs:
        k = max(1, int(round(f * x.shape[0])))
        x2 = x.clone(); x2[order[:k]] = 0.0
        drops.append(base - torch.sigmoid(model(x2, edge_index, batch))[0, target_bit].item())
    return float(np.mean(drops))


METHODS = [
    ("IG", lambda m, x, e, b, j: integrated_gradients(m, x, e, b, j, n_steps=32)),
    ("grad", lambda m, x, e, b, j: saliency(m, x, e, b, j)),
    ("grad*input", lambda m, x, e, b, j: grad_x_input(m, x, e, b, j)),
    ("smoothgrad", lambda m, x, e, b, j: smoothgrad(m, x, e, b, j)),
    ("occlusion", lambda m, x, e, b, j: occlusion(m, x, e, b, j)),
]


def main():
    eps = {e.name: e for e in data_mod.load_selected(seed=0)}
    smis = []
    for name in ["AMES", "LD50", "DILI", "BBB", "Caco2"]:
        smis.extend(eps[name].splits["scaffold"]["train"]["Drug"].tolist())
    smis = list(dict.fromkeys(smis))[:N_MOLS]
    items = []
    for smi in smis:
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        g = from_smiles(smi)
        if g.num_nodes == 0 or g.edge_index.numel() == 0: continue
        fp, bi = morgan_with_bitinfo(m); items.append((g, fp, bi))
    freqs = np.stack([fp for _, fp, _ in items]).sum(0)
    top_bits = np.argsort(-freqs)[:K_BITS].tolist()

    dataset, bis = [], []
    for g, fp, bi in items:
        dataset.append(Data(x=g.x.float(), edge_index=g.edge_index,
                            y=torch.tensor([[fp[b] for b in top_bits]], dtype=torch.float32)))
        bis.append(bi)
    n_tr = int(0.7 * len(dataset))
    tr, te, te_bis = dataset[:n_tr], dataset[n_tr:], bis[n_tr:]

    model = MultiBitGIN(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(); loader = DataLoader(tr, batch_size=64, shuffle=True)
    for _ in range(80):
        model.train()
        for b in loader:
            opt.zero_grad(); loss_fn(model(b.x, b.edge_index, b.batch), b.y).backward(); opt.step()
    model.eval()

    rng = np.random.default_rng(0)
    rows = []
    for mi, (d, bi) in enumerate(zip(te, te_bis)):
        batch = torch.zeros(d.num_nodes, dtype=torch.long)
        active = [j for j, b in enumerate(top_bits) if d.y[0, j].item() > 0.5 and b in bi]
        if d.num_nodes < 3 or not active: continue
        for j in active:
            gt = gt_atoms(bi, top_bits[j])
            y_bin = np.array([1 if a in gt else 0 for a in range(d.num_nodes)])
            if not gt or len(np.unique(y_bin)) < 2: continue
            scores = {name: fn(model, d.x.float(), d.edge_index, batch, j) for name, fn in METHODS}
            scores["random"] = rng.standard_normal(d.num_nodes)
            row = {"mol_idx": mi, "bit": top_bits[j]}
            for name, s in scores.items():
                atom_score = np.abs(s).sum(axis=1) if (s.ndim == 2) else np.abs(s)
                row[f"{name}_auc"] = roc_auc_score(y_bin, atom_score)
                row[f"{name}_faith"] = comprehensiveness(model, d.x.float(), d.edge_index, batch, j, atom_score)
            rows.append(row)

    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "r4_p3_methods.csv"), index=False)
    methods_all = [n for n, _ in METHODS] + ["random"]

    def boot(v):
        v = np.asarray(v); b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
        return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    print(f"\n=== R4-1: D1 with {len(methods_all)} methods (n={len(df)} mol-bit pairs) ===")
    recov, faith = {}, {}
    for name in methods_all:
        a, lo, hi = boot(df[f"{name}_auc"]); recov[name] = a
        f, _, _ = boot(df[f"{name}_faith"]); faith[name] = f
        print(f"  {name:11s} recovery AUROC = {a:.3f} [{lo:.3f}, {hi:.3f}]   faithfulness = {f:+.3f}")

    print(f"\n=== R4-1: D2 over {len(methods_all)} methods ===")
    rec = [recov[m] for m in methods_all]; fai = [faith[m] for m in methods_all]
    rho, p = spearmanr(rec, fai)
    print(f"  Spearman(recovery, faithfulness) = {rho:+.3f}  p={p:.3f}  (n_methods = {len(methods_all)})")
    print(f"  -> harness-faithfulness {'tracks' if rho > 0.5 else 'does NOT track'} ground-truth recovery on a broader method set")

    pd.DataFrame({"method": methods_all, "recovery": rec, "faithfulness": fai}).to_csv(
        os.path.join(RES, "r4_p3_d2_extended.csv"), index=False)


if __name__ == "__main__":
    main()
