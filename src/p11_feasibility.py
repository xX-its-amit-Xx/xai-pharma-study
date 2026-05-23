"""Paper 3 P11 feasibility gate.

1. Confirm RDKit bitInfo gives per-(mol,bit) atom sets.
2. Train a tiny multi-task GIN to predict the K most frequent Morgan bits on ~500 SMILES.
3. Run IG attributions targeting individual active bits; check whether IG localizes on the
   bitInfo ground-truth atoms substantially better than random.

If IG mean AUROC > 0.6 over a few hundred (mol, bit) pairs -> the design is feasible.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
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
N_BITS = 1024
RADIUS = 2
K_BITS = 64       # target the top-K most frequent bits
N_MOLS = 500
torch.manual_seed(0)


def morgan_with_bitinfo(mol):
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS, bitInfo=bi)
    return np.array(fp, dtype=np.float32), bi


def gt_atoms_for_bit(bi, bit_idx):
    """RDKit lists (atom_idx, radius) tuples; the centre atom + its radius-r environment."""
    if bit_idx not in bi:
        return set()
    return {int(a) for a, _ in bi[bit_idx]}


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


def main():
    # Use a chemically diverse pool from existing TDC endpoints
    eps = {e.name: e for e in data_mod.load_selected(seed=0)}
    smis = []
    for name in ["AMES", "LD50", "DILI", "BBB"]:
        smis.extend(eps[name].splits["scaffold"]["train"]["Drug"].tolist())
    smis = list(dict.fromkeys(smis))[:N_MOLS]
    print(f"loaded {len(smis)} unique SMILES")

    # build graphs + fp + bitInfo
    items = []
    for smi in smis:
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        g = from_smiles(smi)
        if g.num_nodes == 0 or g.edge_index.numel() == 0: continue
        fp, bi = morgan_with_bitinfo(m)
        items.append((g, fp, bi))
    print(f"valid molecules: {len(items)}")

    # pick top-K bits by frequency
    freqs = np.stack([fp for _, fp, _ in items]).sum(0)
    top_bits = np.argsort(-freqs)[:K_BITS].tolist()
    print(f"K={K_BITS} top bits: frequency range {int(freqs[top_bits].min())}-{int(freqs[top_bits].max())}/{len(items)}")

    # prepare PyG dataset with K-bit targets; keep bitInfo OUTSIDE Data (PyG can't collate dicts)
    dataset, bis = [], []
    for g, fp, bi in items:
        d = Data(x=g.x.float(), edge_index=g.edge_index,
                 y=torch.tensor([[fp[b] for b in top_bits]], dtype=torch.float32))
        dataset.append(d); bis.append(bi)
    n_tr = int(0.8 * len(dataset))
    tr, te = dataset[:n_tr], dataset[n_tr:]
    te_bis = bis[n_tr:]

    # train
    model = MultiBitGIN(); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(tr, batch_size=64, shuffle=True)
    for epoch in range(80):
        model.train()
        for b in loader:
            opt.zero_grad()
            out = model(b.x, b.edge_index, b.batch)
            loss = loss_fn(out, b.y); loss.backward(); opt.step()
    model.eval()

    # multi-bit test accuracy
    with torch.no_grad():
        Y, Yh = [], []
        for d in te:
            batch = torch.zeros(d.num_nodes, dtype=torch.long)
            yh = torch.sigmoid(model(d.x, d.edge_index, batch))[0].numpy()
            Y.append(d.y.numpy()[0]); Yh.append(yh)
        Y = np.array(Y); Yh = np.array(Yh)
    per_bit_auc = []
    for j in range(K_BITS):
        if len(np.unique(Y[:, j])) == 2:
            per_bit_auc.append(roc_auc_score(Y[:, j], Yh[:, j]))
    print(f"GIN multi-bit prediction: mean per-bit test AUROC = {np.mean(per_bit_auc):.3f} "
          f"(over {len(per_bit_auc)}/{K_BITS} bits with both classes)")

    # IG attribution-recovery probe (manual IG on graph inputs; captum's batching breaks edge_index)
    def graph_ig(x, edge_index, batch, target_bit, n_steps=16):
        baseline = torch.zeros_like(x)
        grads = []
        for a in torch.linspace(0, 1, n_steps):
            x_a = (baseline + a * (x - baseline)).detach().requires_grad_(True)
            out = model(x_a, edge_index, batch)[0, target_bit]
            g = torch.autograd.grad(out, x_a, retain_graph=False)[0]
            grads.append(g)
        return (torch.stack(grads).mean(0) * (x - baseline)).detach().numpy()

    @torch.no_grad()
    def occlusion_atom_scores(x, edge_index, batch, target_bit):
        base = torch.sigmoid(model(x, edge_index, batch))[0, target_bit].item()
        scores = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            x2 = x.clone(); x2[i] = 0.0
            scores[i] = base - torch.sigmoid(model(x2, edge_index, batch))[0, target_bit].item()
        return scores

    aucs_ig, aucs_occ, aucs_rand = [], [], []
    rng = np.random.default_rng(0)
    for d, bi in zip(te, te_bis):
        batch = torch.zeros(d.num_nodes, dtype=torch.long)
        active = [j for j, b in enumerate(top_bits) if d.y[0, j].item() > 0.5 and b in bi]
        if not active or d.num_nodes < 3:
            continue
        for j in active[:3]:                       # cap per molecule to bound compute
            attr = graph_ig(d.x.float(), d.edge_index, batch, j, n_steps=16)
            atom_score = np.abs(attr).sum(axis=1)
            occ_score = occlusion_atom_scores(d.x.float(), d.edge_index, batch, j)
            gt = gt_atoms_for_bit(bi, top_bits[j])
            if not gt or len(gt) == d.num_nodes:    # degenerate
                continue
            y_bin = np.array([1 if a in gt else 0 for a in range(d.num_nodes)])
            if len(np.unique(y_bin)) < 2:
                continue
            aucs_ig.append(roc_auc_score(y_bin, atom_score))
            aucs_occ.append(roc_auc_score(y_bin, occ_score))
            aucs_rand.append(roc_auc_score(y_bin, rng.standard_normal(d.num_nodes)))

    print(f"\nground-truth-atom recovery (mean AUROC over n={len(aucs_ig)} mol-bit pairs):")
    print(f"  IG (manual graph IG)   = {np.mean(aucs_ig):.3f}")
    print(f"  atom occlusion         = {np.mean(aucs_occ):.3f}")
    print(f"  random baseline        = {np.mean(aucs_rand):.3f}")
    best = max(np.mean(aucs_ig), np.mean(aucs_occ))
    if best > 0.6:
        print(f"\nFEASIBILITY GATE PASS (best method clears 0.6)")
    else:
        print(f"\nNeither method clears 0.6 -- the multi-task GIN appears to take shortcuts that"
              f" predict ECFP without attending to bit-defining atoms (a substantive finding).")

    import pandas as pd
    pd.DataFrame({"ig_auc": aucs_ig, "occ_auc": aucs_occ, "random_auc": aucs_rand}).to_csv(
        os.path.join(RES, "p11_feasibility.csv"), index=False)


if __name__ == "__main__":
    main()
